import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from typing import Tuple, Union, Optional, Callable
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv, SAGPooling, GATv2Conv, GraphConv, GINConv, PNAConv, knn_interpolate
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import Sequential as GeoSequential
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.utils import softmax
from torch.nn import Linear, ReLU, Sequential
import torch.nn.functional as F
from graphdata import PairData
import numpy as np
import math


def get_pooled_sz(full_sz: int, ratio: float, layer: int):
  out_sz = full_sz
  for l in range(layer):
    out_sz = int(-(out_sz // -(1 / ratio)))
  return out_sz


def get_deg(x, edge_index, device: str = "cpu"):
  deg = torch.sparse_coo_tensor(edge_index, torch.ones(
      (edge_index.size(1),))).to(device) @ torch.ones((x.size(0), 1)).to(device)
  return deg


def get_edge_attr(edge_index, pos):
  edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
  return edge_attr


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
      GNN: torch.nn.Module = GraphConv,
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
    self.device = device

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


class RDB(nn.Module):

  def __init__(self,
               dim: int,
               in_channels: int,
               growth_channels: int,
               device: str = "cpu") -> None:
    super(RDB, self).__init__()
    self.dim = dim
    self.in_channels = in_channels
    self.growth_channels = growth_channels
    self.device = device

    self.conv1 = GraphConv(in_channels, growth_channels)
    self.conv2 = GraphConv(in_channels + growth_channels,
                           growth_channels).to(device)
    self.conv3 = GraphConv(in_channels + 2*growth_channels,
                           in_channels).to(device)

    self.bnorms = nn.ModuleList(
        [BatchNorm(growth_channels).to(device) for i in range(2)])
    self.bnorms.append(BatchNorm(in_channels).to(device))

  def forward(self, x, edge_index):
    x1 = F.elu(self.bnorms[0](self.conv1(x, edge_index)))
    x2 = F.elu(self.bnorms[1](self.conv2(torch.cat((x, x1), 1), edge_index)))
    x3 = F.elu(self.bnorms[2](self.conv3(torch.cat((x, x1, x2), 1),
                                         edge_index)))
    out = x + 0.2*x3
    return out


class ERDB(nn.Module):

  def __init__(self,
               dim: int,
               in_channels: int,
               growth_channels: int,
               device: str = "cpu") -> None:
    super(ERDB, self).__init__()
    self.dim = dim
    self.in_channels = in_channels
    self.growth_channels = growth_channels
    self.device = device

    self.rdb1 = RDB(dim, in_channels, growth_channels, device).to(device)
    self.rdb2 = RDB(dim, in_channels, growth_channels, device).to(device)

  def forward(self, x, edge_index):
    x1 = self.rdb1(x, edge_index)
    x2 = self.rdb2(x + 0.2*x1, edge_index)
    out = x + 0.2*x2
    return out


class Neighborhood():

  def __init__(self, x: torch.Tensor, edge_index: torch.Tensor,
               old_ids: torch.Tensor, pos: torch.Tensor, node_id: int, k: int):
    self.x = x
    self.pos = pos
    self.node_id = node_id
    self.k = k
    self.old_ids = old_ids

    # renumber nodes
    for new, old in enumerate(torch.unique(edge_index)):
      edge_index = torch.where(edge_index == old,
                               torch.full_like(edge_index, new), edge_index)
    self.edge_index = edge_index

    self.pool_val, _ = torch.max(x, 0)


class NeighborhoodPool(nn.Module):

  def __init__(self,
               in_channels: int,
               n_size: int = 8,
               k_max: int = 5,
               gnn: nn.Module = GATv2Conv,
               device: str = "cpu",
               edge_dim: int = 3,
               **kwargs) -> None:
    super(NeighborhoodPool, self).__init__()
    self.n_size = n_size
    self.k_max = k_max
    self.gnn = gnn(
        in_channels=in_channels, out_channels=1, edge_dim=edge_dim,
        **kwargs).to(device)
    self.device = device

  def forward(self, x, edge_index, pos):
    n_list = []
    old_edge_index = edge_index
    old_ids = torch.arange(x.size(0)).to(self.device)
    edge_attr = get_edge_attr(edge_index, pos)
    score = torch.squeeze(
        self.gnn(torch.cat((x, pos), dim=1), edge_index, edge_attr))
    deg = get_deg(x, edge_index, self.device)
    k_hops = min(
        int(-(math.log(self.n_size) // -math.log(min(deg)))), self.k_max)
    # cluster and pool
    while x.size(0) > 0:
      node_id = torch.argmax(score)

      # if node is connected
      if node_id in edge_index:
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1),).to(self.device))
        adj_aug = adj
        for k in range(k_hops):
          adj_aug = (adj_aug @ adj).coalesce()
        edge_aug = adj_aug.indices()
        k_hop_nh = torch.unique(edge_aug[1, :][edge_aug[0, :] == node_id])
        pool_idx = (torch.tensor([i in k_hop_nh for i in edge_index[0, :]])
                    & torch.tensor([i in k_hop_nh for i in edge_index[1, :]
                                   ])).to(self.device)
        pool_edge_index = edge_index[:, pool_idx]
      # else node is isolated
      else:
        k_hop_nh = torch.tensor([node_id]).to(self.device)
        pool_edge_index = torch.tile(k_hop_nh, (2, 1))
      n_list.append(
          Neighborhood(x[k_hop_nh], pool_edge_index, old_ids[k_hop_nh],
                       pos[node_id], len(n_list), k_hops))
      keep_idx = torch.tensor([i not in k_hop_nh for i in range(x.size(0))
                              ]).to(self.device)

      if keep_idx.size(0) == 0:
        break

      score = score[keep_idx]
      x = x[keep_idx, :]
      pos = pos[keep_idx, :]
      old_ids = old_ids[keep_idx]

      # remove edge if a connecting node has been pooled
      keep_idx = (
          torch.tensor([i not in k_hop_nh for i in edge_index[0, :]]).to(
              self.device)
          & torch.tensor([i not in k_hop_nh for i in edge_index[1, :]]).to(
              self.device))

      if keep_idx.sum() > 0:
        edge_index = edge_index[:, keep_idx]
        for new, old in enumerate(torch.unique(edge_index)):
          edge_index = torch.where(edge_index == old,
                                   torch.full_like(edge_index, new), edge_index)
      # if remaining nodes are isolated, pool as one neighborhood
      else:
        edge_index = torch.tile(torch.arange(x.size(0)).to(self.device), (2, 1))
        n_list.append(
            Neighborhood(x, edge_index, old_ids, pos[torch.argmax(score)],
                         len(n_list), k_hops))
        break

    # TODO:streamline and fix this

    # recombine
    edge_index = old_edge_index
    x = torch.stack([n.pool_val for n in n_list])
    pos = torch.stack([n.pos for n in n_list])
    n_ids = torch.tensor([n.node_id for n in n_list])

    # create clustering matrix using n.node_id
    clusters = torch.concat([
        torch.full((n.old_ids.size(0),), n.node_id) for n in n_list
    ]).to(self.device)
    clusters = clusters[torch.concat([n.old_ids for n in n_list])]

    # get edge attr by differencing clustering feature
    edge_attr = get_edge_attr(edge_index, clusters)

    # non-zero edges get saved to list
    nz_edges = edge_index[:, torch.argwhere(edge_attr)]

    # re-index clustering with saved edges to get edges of pooled graph
    pool_edge_index = torch.stack(
        (clusters[nz_edges[0, :]], clusters[nz_edges[1, :]]))
    pool_edge_index = torch.unique(pool_edge_index, dim=1)
    for new, old in enumerate(n_ids):
      pool_edge_index = torch.where(pool_edge_index == old,
                                    torch.full_like(pool_edge_index, new),
                                    pool_edge_index)

    return x, pool_edge_index, pos


class StructureEncoder(nn.Module):

  def __init__(self,
               dim: int,
               init_data: Data,
               hidden_channels: int,
               latent_channels: int,
               k_size: int,
               n_pools: int,
               pool_ratio: float,
               device: str = "cpu"):
    super(StructureEncoder, self).__init__()
    self.dim = dim
    self.in_channels = 1
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.k_size = k_size
    self.n_pools = n_pools
    self.pool_ratio = pool_ratio
    self.device = device

    # initial aggr
    self.conv_list = nn.ModuleList([
        GATv2Conv(self.in_channels + dim, hidden_channels,
                  edge_dim=dim).to(self.device)
    ])
    self.conv_list.append(
        GATv2Conv(hidden_channels + dim, hidden_channels,
                  edge_dim=dim).to(self.device))
    aggs = ["mean", "min", "max", "std"]
    scls = ["identity", "amplification", "attenuation"]
    deg = get_deg(init_data.x, init_data.edge_index)
    # pools
    self.pool_list = nn.ModuleList()
    for _ in range(n_pools):
      # # SAGPOOL
      self.pool_list.append(
          SAGPoolWithPos(
              hidden_channels + dim,
              pool_ratio,
              in_channels=hidden_channels,
              out_channels=hidden_channels,
              edge_dim=dim,
              aggregators=aggs,
              scalers=scls,
              deg=deg).to(self.device))

      # # MY NEIGHBORHOOD POOL
      # self.pool_list.append(
      #     NeighborhoodPool(
      #         hidden_channels + dim, 1 // pool_ratio, device=device).to(device))

      self.conv_list.append(
          GATv2Conv(hidden_channels + self.dim, hidden_channels,
                    edge_dim=dim).to(self.device))

    # latent dense map
    # out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    # self.conv_list.append(
    #     GATv2Conv(hidden_channels,latent_channels).to(self.device))

  def forward(self, x, edge_index, pos):
    x = get_deg(x, edge_index, self.device)
    edge_attr = get_edge_attr(edge_index, pos)
    x = F.elu(
        self.conv_list[0](torch.cat((x, pos), dim=1), edge_index, edge_attr),
        inplace=True)
    x = F.elu(
        self.conv_list[1](torch.cat((x, pos), dim=1), edge_index, edge_attr),
        inplace=True)

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    edge_attr_list = [edge_attr]
    for l, pool in enumerate(self.pool_list):
      # # SAGPOOL
      # x, edge_index, pos, _, _, _, _ = pool(x, edge_index, pos)

      # # MY NEIGHBORHOOD POOL
      x, edge_index, pos = pool(x, edge_index, pos)

      edge_attr = get_edge_attr(edge_index, pos)

      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)
      edge_attr_list.insert(0, edge_attr)

      x = F.elu(
          self.conv_list[l + 2](torch.cat((x, pos), dim=1), edge_index,
                                edge_attr),
          inplace=True)
    return pool_edge_list, pool_pos_list, edge_attr_list


class DBGSR(nn.Module):

  def __init__(self,
               dim: int,
               init_data_2: Data,
               hidden_channels: int,
               device: str = "cpu") -> None:
    super(DBGSR, self).__init__()
    self.dim = dim
    self.in_features = init_data_2.num_node_features
    self.hidden_channels = hidden_channels
    self.device = device

    self.conv1 = GraphConv(self.in_features + dim + 3,
                           hidden_channels).to(device)
    self.erdb1 = ERDB(dim, hidden_channels, -(hidden_channels // -2),
                      device).to(device)
    self.erdb2 = ERDB(dim, hidden_channels, -(hidden_channels // -2),
                      device).to(device)
    self.conv2 = GraphConv(hidden_channels, hidden_channels).to(device)

    # upsample
    self.conv3 = GraphConv(hidden_channels, hidden_channels).to(device)
    self.conv4 = GraphConv(hidden_channels, hidden_channels).to(device)
    self.conv5 = GraphConv(hidden_channels, self.in_features).to(device)

    self.bnorms = nn.ModuleList([BatchNorm(hidden_channels) for i in range(4)])

  def onera_transform(self, pos):
    # adjust x to move leading edge to x=0
    pos[:, 0] = pos[:, 0] - math.tan(math.pi / 6)*pos[:, 1]
    # scale chord to equal root
    # c(y) = r(1 - (1-taper)*(y/s))
    # r = c(y) / (1- (1-taper)*(y/s))
    pos = pos*(1 + (1/0.56 - 1)*(pos[:, 1:2] / 1.1963))
    return pos

  def onera_interp(self, f, pos_x, pos_y):
    return knn_interpolate(
        f, self.onera_transform(pos_x), self.onera_transform(pos_y), k=1)

  def forward(self, x, edge_index_2, edge_index_3, pos_2, pos_3, y):
    x1 = self.bnorms[0](
        self.conv1(
            torch.cat((x, pos_2, y*torch.ones_like(pos_2)), 1), edge_index_2))
    x2 = self.erdb1(x1, edge_index_2)
    x3 = x1 + 0.2*self.erdb2(x2, edge_index_2)
    x = F.elu(self.bnorms[1](self.conv2(x3, edge_index_2)), inplace=True)

    # upsample
    x = self.onera_interp(x3, pos_2, pos_3)
    x = F.elu(self.bnorms[2](self.conv3(x, edge_index_3)), inplace=True)
    x = F.elu(self.bnorms[3](self.conv4(x, edge_index_3)), inplace=True)

    out = self.conv5(x, edge_index_3)
    return out
