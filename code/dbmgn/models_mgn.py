import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from typing import Tuple, Union, Optional, Callable
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv, SAGPooling, GATv2Conv, GraphConv, GINConv, knn_interpolate
from torch_geometric.nn import Sequential as GeoSequential
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.utils import softmax
from torch.nn import Linear, ReLU, Sequential
import torch.nn.functional as F
from graphdata import PairData
import numpy as np


def get_pooled_sz(full_sz: int, ratio: float, layer: int):
  out_sz = full_sz
  for l in range(layer):
    out_sz = int(-(out_sz // -(1 / ratio)))
  return out_sz


def get_deg(x, edge_index):
  deg = torch.sparse_coo_tensor(edge_index.cpu(),
                                torch.ones((edge_index.size(1),))) @ torch.ones(
                                    (x.size(0), 1))
  return deg


class MLP(nn.Module):

  def __init__(self,
               in_channels: int,
               hid_channels: int,
               out_channels: int,
               act: callable = torch.nn.functional.elu,
               device: str = "cuda:0"):
    super(MLP, self).__init__()
    self.in_channels = in_channels
    self.hid_channels = hid_channels
    self.out_channels = out_channels
    self.act = act

    self.lin_0 = Linear(in_channels, hid_channels).cuda(device)
    # self.lin_1 = Linear(hid_channels, out_channels).cuda(device)
    self.lin_2 = Linear(hid_channels, hid_channels).cuda(device)
    self.lin_3 = Linear(hid_channels, out_channels).cuda(device)

  def forward(self, x):
    out = self.act(self.lin_0(x), inplace=True)
    # skip = self.lin_1(out)
    out = self.act(self.lin_2(out), inplace=True)
    out = self.lin_3(out)
    return out


class SAGPGIN(nn.Module):

  def __init__(self,
               in_channels: int,
               out_channels: int,
               hidden_channels: int,
               act: callable = torch.nn.functional.elu,
               device: str = "cuda:0"):
    super(SAGPGIN, self).__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.act = act

    self.mlp = MLP(
        in_channels, hidden_channels, out_channels, device=device).cuda(device)
    self.gin = GINConv(self.mlp).cuda(device)

  def reset_parameters(self):
    self.gin.reset_parameters()

  def forward(self, x, edge_index, edge_weight=None, size=None):
    out = self.gin(x, edge_index, size)
    return out


class SAGPoolWithPos(SAGPooling):

  def __init__(
      self,
      in_channels: int,
      ratio: Union[float, int] = 0.5,
      GNN: torch.nn.Module = SAGPGIN,
      min_score: Optional[float] = None,
      multiplier: float = 1.0,
      nonlinearity: Union[str, Callable] = 'tanh',
      augmentation: bool = True,
      **kwargs,
  ):
    super(SAGPoolWithPos, self).__init__(in_channels, ratio, GNN, min_score,
                                         multiplier, nonlinearity, **kwargs)
    self.augmentation = augmentation

  def forward(
      self,
      x: Tensor,
      edge_index: Tensor,
      pos: Tensor,
      edge_attr: OptTensor = None,
      batch: OptTensor = None,
      attn: OptTensor = None,
  ) -> Tuple[Tensor, Tensor, Tensor, OptTensor, OptTensor, Tensor, Tensor]:
    if batch is None:
      batch = edge_index.new_zeros(x.size(0))

    attn = torch.cat((x, pos), dim=1) if attn is None else attn
    attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
    if self.augmentation:
      adj = torch.sparse_coo_tensor(edge_index,
                                    torch.ones(edge_index.size(1),).cuda())
      adj_aug = (adj + (adj@adj)).coalesce()
      edge_index_aug = adj_aug.indices()
      score = self.gnn(attn, edge_index_aug).view(-1)
    else:
      score = self.gnn(attn, edge_index).view(-1)

    if self.min_score is None:
      score = self.nonlinearity(score)
    else:
      score = softmax(score, batch)

    perm = topk(score, self.ratio, batch, self.min_score)
    x = x[perm]*score[perm].view(-1, 1)
    x = self.multiplier*x if self.multiplier != 1 else x
    pos = pos[perm]

    batch = batch[perm]
    edge_index, edge_attr = filter_adj(
        edge_index, edge_attr, perm, num_nodes=score.size(0))

    return x, edge_index, pos, edge_attr, batch, perm, score[perm]


class EdgeNet(nn.Module):

  def __init__(self,
               dim: int,
               in_channels,
               hidden_channels,
               out_channels,
               device: str = "cuda:0",
               **kwargs) -> None:
    super(EdgeNet, self).__init__(**kwargs)
    self.dim = dim
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels

    self.mlp = MLP(
        in_channels, hidden_channels, out_channels, device=device).cuda(device)

  def forward(self, x, edge_index, pos):
    edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
    node_i = x[edge_index[0]]
    node_j = x[edge_index[1]]
    inputs = torch.cat((edge_attr, node_i, node_j), dim=1)

    out = self.mlp(inputs)
    return out


class NodeNet(nn.Module):

  def __init__(self,
               dim: int,
               in_channels,
               hidden_channels,
               out_channels,
               device: str = "cuda:0",
               **kwargs) -> None:
    super(NodeNet, self).__init__(**kwargs)
    self.dim = dim
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.device = device

    self.mlp = MLP(
        in_channels, hidden_channels, out_channels, device=device).cuda(device)

  def forward(self, x, edge_index, edge_attr):
    # edge_mat = torch.sparse_coo_tensor(edge_index, edge_attr, (x.size(0), x.size(0))).cuda(device)
    edge_sum = torch.cat([
        torch.sparse_coo_tensor(edge_index, edge_attr[:, i],
                                (x.size(0), x.size(0))).cuda(self.device)
        @ torch.ones((x.size(0), 1)).cuda(self.device)
        for i in range(self.hidden_channels)
    ],
                         dim=1)

    inputs = torch.cat((x, edge_sum), dim=1)
    out = self.mlp(inputs)
    return out


class SizeFieldNet(nn.Module):

  def __init__(self,
               dim: int,
               in_channels,
               hidden_channels,
               device: str = "cuda:0",
               **kwargs) -> None:
    super(SizeFieldNet, self).__init__(**kwargs)
    self.dim = dim
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = dim**2

    self.mlp = MLP(
        in_channels, hidden_channels, self.out_channels,
        device=device).cuda(device)

  def forward(self, x):
    out = self.mlp(x)
    out = torch.reshape(out, (self.dim, self.dim))
    return out


class MGN(nn.Module):

  def __init__(self,
               dim: int,
               in_features: int,
               hidden_channels: int,
               out_features: int,
               device: str = "cuda:0",
               **kwargs) -> None:
    super(MGN, self).__init__(**kwargs)
    self.dim = dim
    self.in_features = in_features
    self.hidden_channels = hidden_channels
    self.edge_net = EdgeNet(dim, dim + 2*self.in_features, hidden_channels,
                            hidden_channels, device).cuda(device)
    self.node_net = NodeNet(dim, self.in_features + hidden_channels,
                            hidden_channels, hidden_channels,
                            device).cuda(device)
    # self.sfn = SizeFieldNet(dim, hidden_channels, hidden_channels)
    self.decoder = MLP(
        hidden_channels, hidden_channels, out_features,
        device=device).cuda(device)

  def forward(self, x, edge_index, pos):
    edge_attr = self.edge_net(x, edge_index, pos)
    node_attr = self.node_net(x, edge_index, edge_attr)
    out = self.decoder(node_attr)

    # size_tensor = self.sfn(node_attr)
    # return out, size_tensor

    return out


# class Remesher(nn.Module):

#   def __init__(self, dim: int, in_channels, hidden_channels, out_channels,
#                *args, **kwargs) -> None:
#     super(Remesher, self).__init__(*args, **kwargs)
#     self.dim = dim
#     self.in_channels = in_channels
#     self.hidden_channels = hidden_channels
#     self.out_channels = out_channels


class DBMGN(nn.Module):

  def __init__(self,
               dim: int,
               init_data: Data,
               hidden_channels: int,
               device: str = "cuda:0",
               **kwargs) -> None:
    super(DBMGN, self).__init__(**kwargs)
    self.dim = dim
    self.in_features = init_data.num_node_features
    self.hidden_channels = hidden_channels

    # self.mgn_0 = MGN(dim, self.in_features, hidden_channels, hidden_channels).cuda(device)
    # self.mgn_1 = MGN(dim, hidden_channels, hidden_channels, self.in_features).cuda(device)

    self.mgn = MGN(dim, self.in_features, hidden_channels, self.in_features,
                   device).cuda(device)

  def forward(self, x, edge_index, pos):
    # Double MGN
    # out = self.mgn_0(x, edge_index, pos)
    # out = self.mgn_1(out, edge_index, pos)

    # Double Residual MGN
    # out = self.mgn_0(x, edge_index, pos)
    # out = self.mgn_1(out, edge_index, pos) + x

    # Single MGN
    # out = self.mgn(x, edge_index, pos)

    # Residual MGN
    out = self.mgn(x, edge_index, pos) + x
    return out
