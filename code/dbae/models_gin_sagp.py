import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from typing import Tuple, Union, Optional, Callable
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv, SAGPooling, GraphConv, GINConv, knn_interpolate
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


def get_deg(x, edge_index, device: str = "cpu"):
  deg = torch.sparse_coo_tensor(
      edge_index,
      torch.ones((edge_index.size(1),)).to(device)).to(device) @ torch.ones(
          (x.size(0), 1)).to(device)
  return deg


def get_edge_attr(edge_index, pos):
  # edge_attr = torch.zeros((edge_index.size(1), 3)).to(self.device)
  # for i, xs in enumerate(edge_index.transpose(0, 1)):
  #   edge_attr[i] = pos[xs[1]] - pos[xs[0]]
  edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
  return edge_attr


class GINMLP(nn.Module):

  def __init__(self,
               in_channels: int,
               hid_channels: int,
               out_channels: int,
               act: callable = torch.nn.functional.elu,
               device: str = "cpu"):
    super(GINMLP, self).__init__()
    self.in_channels = in_channels
    self.hid_channels = hid_channels
    self.out_channels = out_channels
    self.act = act
    self.device = device

    self.lin_0 = Linear(in_channels, hid_channels).to(self.device)
    self.lin_1 = Linear(hid_channels, out_channels).to(self.device)

  def forward(self, x):
    out = self.act(self.lin_0(x), inplace=True)
    out = self.lin_1(out)
    return out


class SAGPGIN(nn.Module):

  def __init__(self,
               in_channels: int,
               out_channels: int,
               hidden_channels: int,
               act: callable = torch.nn.functional.elu,
               device: str = "cpu"):
    super(SAGPGIN, self).__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.act = act
    self.device = device

    self.mlp = GINMLP(
        in_channels, hidden_channels, out_channels, device=device).to(device)
    self.gin = GINConv(self.mlp).to(device)

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
      device: str = "cpu",
      **kwargs,
  ):
    super(SAGPoolWithPos, self).__init__(in_channels, ratio, GNN, min_score,
                                         multiplier, nonlinearity, **kwargs)
    self.augmentation = augmentation
    self.device = device

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
      adj = torch.sparse_coo_tensor(
          edge_index,
          torch.ones(edge_index.size(1),).to(self.device))
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


class Encoder(nn.Module):

  def __init__(self,
               dim: int,
               init_data: Data,
               hidden_channels: int,
               latent_channels: int,
               k_size: int,
               n_pools: int,
               pool_ratio: float,
               device: str = "cpu"):
    super(Encoder, self).__init__()
    self.dim = dim
    self.in_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.k_size = k_size
    self.n_pools = n_pools
    self.pool_ratio = pool_ratio
    self.device = device

    # initial aggr
    self.conv_list = nn.ModuleList([
        GINConv(
            GINMLP(
                self.in_channels,
                hidden_channels,
                hidden_channels,
                device=device)).to(self.device)
    ])
    self.conv_list.append(
        GINConv(
            GINMLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                device=device)).to(self.device))

    # pools
    self.pool_list = nn.ModuleList()
    for _ in range(n_pools):
      self.pool_list.append(
          SAGPoolWithPos(
              hidden_channels + dim,
              pool_ratio,
              hidden_channels=hidden_channels,
              device=device).to(self.device))
      self.conv_list.append(
          GINConv(
              GINMLP(
                  hidden_channels,
                  hidden_channels,
                  hidden_channels,
                  device=device)).to(self.device))

    # latent dense map
    out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    self.conv_list.append(
        GINConv(
            GINMLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                device=device)).to(self.device))

    self.lin_0 = Linear(hidden_channels, latent_channels).to(self.device)
    # self.lin_1 = Linear(latent_channels, latent_channels).to(self.device)

  def forward(self, x, edge_index, pos):
    # edge_attr = get_edge_attr(edge_index, pos)
    x = F.elu(self.conv_list[0](x, edge_index), inplace=True)
    x = F.elu(self.conv_list[1](x, edge_index), inplace=True)

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    for l, pool in enumerate(self.pool_list):
      x, edge_index, pos, _, _, _, _ = pool(x, edge_index, pos)
      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)

      # edge_attr = get_edge_attr(edge_index, pos)
      x = F.elu(self.conv_list[l + 2](x, edge_index), inplace=True)

    x = F.elu(self.conv_list[-1](x, edge_index), inplace=True)
    latent = torch.vmap(self.lin_0, in_dims=(0,))(x)
    # latent = self.lin_0(x.ravel())
    # latent = self.lin_1(x.ravel())
    return latent, pool_edge_list, pool_pos_list


class StructureEncoder(Encoder):

  def __init__(self,
               dim: int,
               init_data: Data,
               hidden_channels: int,
               latent_channels: int,
               k_size: int,
               n_pools: int,
               pool_ratio: float,
               device: str = "cpu"):
    super(StructureEncoder,
          self).__init__(dim, init_data, hidden_channels, latent_channels,
                         k_size, n_pools, pool_ratio, device)
    self.in_channels = self.dim + 1

    # initial aggr
    self.conv_list = nn.ModuleList([
        GINConv(
            GINMLP(
                self.in_channels,
                hidden_channels,
                hidden_channels,
                device=device)).to(self.device)
    ])
    self.conv_list.append(
        GINConv(
            GINMLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                device=device)).to(self.device))

    # pools
    self.pool_list = nn.ModuleList()
    for _ in range(n_pools):
      self.pool_list.append(
          SAGPoolWithPos(
              hidden_channels + dim + 1,
              pool_ratio,
              hidden_channels=hidden_channels,
              device=device).to(self.device))
      self.conv_list.append(
          GINConv(
              GINMLP(
                  hidden_channels,
                  hidden_channels,
                  hidden_channels,
                  device=device)).to(self.device))

    # latent dense map
    out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    self.conv_list.append(
        GINConv(
            GINMLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                device=device)).to(self.device))

  def forward(self, x, edge_index, pos):
    # edge_attr = get_edge_attr(edge_index, pos)
    deg = get_deg(x, edge_index, self.device)
    x = torch.cat([pos, deg], dim=1)
    x = F.elu(self.conv_list[0](x, edge_index), inplace=True)
    x = F.elu(self.conv_list[1](x, edge_index), inplace=True)

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    for l, pool in enumerate(self.pool_list):
      x = torch.cat([x, deg], dim=1)
      x, edge_index, pos, _, _, _, _ = pool(x, edge_index, pos)
      x = x[:, :-1]
      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)
      deg = get_deg(x, edge_index, self.device)

      # edge_attr = get_edge_attr(edge_index, pos)
      x = F.elu(self.conv_list[l + 2](x, edge_index), inplace=True)
    return pool_edge_list, pool_pos_list


class Decoder(nn.Module):

  def __init__(self,
               dim: int,
               init_data: Data,
               hidden_channels: int,
               latent_channels: int,
               k_size: int,
               n_pools: int,
               pool_ratio: float,
               device: str = "cpu"):
    super(Decoder, self).__init__()
    self.dim = dim
    self.out_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.k_size = k_size
    self.n_pools = n_pools
    self.pool_ratio = pool_ratio
    self.device = device

    # latent dense map
    # self.out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    # self.lin_0 = Linear(latent_channels, latent_channels).to(self.device)
    self.lin_1 = Linear(latent_channels, hidden_channels).to(self.device)

    # initial aggr
    self.conv_list = nn.ModuleList([
        GINConv(
            GINMLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                device=device)).to(self.device)
    ])
    self.conv_list.append(
        GINConv(
            GINMLP(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                device=device)).to(self.device))

    # unpools
    for _ in range(n_pools):
      self.conv_list.append(
          GINConv(
              GINMLP(
                  hidden_channels,
                  hidden_channels,
                  hidden_channels,
                  device=device)).to(self.device))

    self.conv_list.append(
        GINConv(
            GINMLP(
                hidden_channels,
                hidden_channels,
                self.out_channels,
                device=device)).to(self.device))

  def forward(self, latent, edge_index_list, pos_list):
    # x = F.elu(self.lin_0(latent), inplace=True)
    # x = torch.reshape(x, (self.out_sz, self.hidden_channels))
    x = F.elu(torch.vmap(self.lin_1, in_dims=(0,))(latent), inplace=True)

    edge_index = edge_index_list[0]
    # pos = pos_list[0]
    # edge_attr = get_edge_attr(edge_index, pos)
    x = F.elu(self.conv_list[0](x, edge_index), inplace=True)
    x = F.elu(self.conv_list[1](x, edge_index), inplace=True)

    for l in range(self.n_pools):
      deg = get_deg(x, edge_index, self.device)
      x = knn_interpolate(x, pos_list[l], pos_list[l + 1], k=int(deg.mean()))
      edge_index = edge_index_list[l + 1]
      # pos = pos_list[l + 1]
      # edge_attr = get_edge_attr(edge_index, pos)

      x = F.elu(self.conv_list[l + 2](x, edge_index), inplace=True)

    out = self.conv_list[-1](x, edge_index)
    return out


class DBA(nn.Module):

  def __init__(self,
               dim: int,
               init_data: PairData,
               hidden_channels: int,
               latent_channels: int,
               k_size: int,
               n_pools: int,
               pool_ratio: float,
               device: str = "cpu"):
    super(DBA, self).__init__()
    self.dim = dim
    self.in_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.k_size = k_size
    self.n_pools = n_pools
    self.pool_ratio = pool_ratio
    self.device = device

    # only used for getting pooling structure
    init_data_3 = Data(
        x=init_data.x_3, edge_index=init_data.edge_index_3, pos=init_data.pos_3)
    self.encoder3D = StructureEncoder(
        dim,
        init_data_3,
        hidden_channels,
        latent_channels,
        k_size,
        n_pools,
        pool_ratio,
        device=device).to(self.device)

    # used for model eval
    init_data_2 = Data(
        x=init_data.x_2, edge_index=init_data.edge_index_2, pos=init_data.pos_2)
    self.encoder2D = Encoder(
        dim,
        init_data_2,
        hidden_channels,
        latent_channels,
        k_size,
        n_pools,
        pool_ratio,
        device=device).to(self.device)
    self.decoder = Decoder(
        dim,
        init_data_3,
        hidden_channels,
        latent_channels,
        k_size,
        n_pools+1,
        pool_ratio,
        device=device).to(self.device)

  def forward(self,
              x_3,
              edge_index_3,
              pos_3,
              x_2,
              edge_index_2,
              pos_2,
              pool_edge_list=None,
              pool_pos_list=None):
    ret = 0
    latent_2, edge_list_2, pos_list_2 = self.encoder2D(
        x_2, edge_index_2, pos_2)
    if pool_edge_list is None and pool_pos_list is None:
      pool_edge_list, pool_pos_list = self.encoder3D(
          x_3, edge_index_3, pos_3)
      ret = 1
      pool_edge_list.insert(0, edge_list_2[0])
      pool_pos_list.insert(0, pos_list_2[0])
    else:
      pool_edge_list.pop(0)
      pool_edge_list.insert(0, edge_list_2[0])

      pool_pos_list.pop(0)
      pool_pos_list.insert(0, pos_list_2[0])

    out = self.decoder(latent_2, pool_edge_list, pool_pos_list)

    if ret == 1:
      return out, pool_edge_list, pool_pos_list
    return out
