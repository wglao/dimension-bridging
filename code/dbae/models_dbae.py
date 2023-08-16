import torch
import math
from torch import Tensor
from torch_geometric.typing import OptTensor
from typing import Tuple, Union, Optional, Callable
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv, SAGPooling, GATv2Conv, GCNConv, GraphConv, GINConv, Linear, knn_interpolate
from torch_geometric.nn import Sequential as GeoSequential
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.nn.pool import max_pool, avg_pool_neighbor_x
from torch_geometric.utils import softmax, add_self_loops, add_remaining_self_loops
from torch.nn import ReLU, Sequential
import torch.nn.functional as F
from graphdata import PairData
import numpy as np
from typing import NamedTuple


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
  edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
  return edge_attr


def get_edge_aug(edge_index, pos, steps: int = 1, device: str = "cpu"):
  # adj = torch.sparse_coo_tensor(edge_index,
  #                               torch.ones(edge_index.size(1),).to(device))
  # adj_aug = adj
  # if steps >= 1:
  #   for _ in range(steps - 1):
  #     adj_aug = (adj_aug @ adj).coalesce()
  #   adj_aug = (adj + adj_aug).coalesce()
  # edge_index_aug = adj_aug.indices()
  # edge_attr_aug = get_edge_attr(edge_index_aug, pos)
  # return edge_index_aug, edge_attr_aug

  edge_index = add_remaining_self_loops(edge_index)[0].unique(dim=1)
  
  # assume symmetric graph
  get_edge_aug = torch.cat(torch.vmap(lambda n: (edge_index[0]==n) & (edge_index[1]>n)))


def onera_transform(pos):
  # adjust x to move leading edge to x=0
  new_x = pos[:, 0] - math.tan(math.pi / 6)*pos[:, 1]
  pos = torch.cat((torch.unsqueeze(new_x, 1), pos[:, 1:]), 1)
  # scale chord to equal root
  # c(y) = r(1 - (1-taper)*(y/s))
  # r = c(y) / (1- (1-taper)*(y/s))
  pos = pos*(1 + (1/0.56 - 1)*(pos[:, 1:2] / 1.1963))
  return pos


def onera_interp(f, pos_x, pos_y):
  return knn_interpolate(f, onera_transform(pos_x), onera_transform(pos_y))


# class Neighborhood():

#   def __init__(self, x: torch.Tensor, old_ids: torch.Tensor, pos: torch.Tensor,
#                node_id: int):
#     self.x = x
#     self.pos = pos
#     self.node_id = node_id
#     self.old_ids = old_ids

#     self.pool_val, _ = torch.max(x, 0)


class NeighborhoodPool(nn.Module):

  def __init__(self,
               dim: int,
               hidden_channels: int,
               k_hops: int = 1,
               gnn: nn.Module = GCNConv,
               device: str = "cpu",
               **kwargs) -> None:
    super(NeighborhoodPool, self).__init__()
    self.dim = dim
    self.k_hops = k_hops
    self.gnn1 = gnn(in_channels=dim, out_channels=hidden_channels, **kwargs).to(device)
    self.gnn2 = gnn(
        in_channels=hidden_channels, out_channels=1, **kwargs).to(device)
    self.device = device

  def forward(self, x, edge_index, pos):
    edge_aug = add_remaining_self_loops(
        get_edge_aug(edge_index, pos, self.k_hops, self.device)[0])[0]
    score = F.selu(self.gnn1(pos, edge_aug))
    score = self.gnn2(score, edge_aug)

    new_order = torch.squeeze(torch.argsort(score, 0))
    old_order = torch.squeeze(torch.argsort(new_order, 0))
    order = new_order

    # reorder nodes
    # x = x[new_order]
    # pos = pos[new_order]
    # for new, old in zip(new_order, old_order):
    #   tmp = torch.argwhere(edge_aug == old)
    #   edge_aug = torch.where(tmp, torch.full_like(edge_aug, new), edge_aug)
    #   edge_pool = torch.where(tmp, torch.full_like(edge_pool, new), edge_pool)

    # remove edges and cluster
    x_pool = None
    pos_pool = None
    n_mask_0 = torch.ones((x.size(0),)).bool().to(self.device)

    cluster = torch.zeros((x.size(0),)).to(self.device)
    n = 0
    while True:
      node = order[0]
      e_mask_1 = edge_aug[0] == node

      n_mask_1 = torch.zeros((x.size(0),)).bool().to(self.device)
      n_mask_1[edge_aug[1, e_mask_1]] = 1
      n_mask_0 = n_mask_0 & ~n_mask_1

      cluster[n_mask_1] = n

      if x_pool is None:
        x_pool = torch.unsqueeze(torch.max(x[n_mask_1], dim=0).values, 0)
        pos_pool = torch.unsqueeze(pos[node], 0)
      else:
        x_pool = torch.cat(
            (x_pool, torch.unsqueeze(torch.max(x[n_mask_1], dim=0).values, 0)))
        pos_pool = torch.cat((pos_pool, torch.unsqueeze(pos[node], dim=0)))

      edge_aug = edge_aug[:, ~(
          edge_aug.ravel().unsqueeze(1) == n_mask_1.argwhere().squeeze()
      ).sum(1).reshape(edge_aug.shape).sum(0).bool()]
      order = new_order[n_mask_0[new_order]]

      n += 1

      if order.size(0) == 0:
        break

    edge_attr = get_edge_attr(edge_index, cluster)
    edge_index = edge_index[:, edge_attr.bool()]
    edge_pool = cluster[edge_index]
    edge_pool = edge_pool.unique(dim=1).int()

    if self.training:
      return x_pool, edge_pool, pos_pool, score
    return x_pool, edge_pool, pos_pool


class Encoder(nn.Module):

  def __init__(self,
               dim: int,
               init_data: Data,
               hidden_channels: int,
               latent_channels: int,
               n_pools: int,
               k_hops: int,
               device: str = "cpu"):
    super(Encoder, self).__init__()
    self.dim = dim
    self.in_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.n_pools = n_pools
    self.k_hops = k_hops
    self.device = device

    # initial aggr
    self.conv_list = nn.ModuleList(
        [GCNConv(
            self.in_channels + dim + 3,
            hidden_channels,
        ).to(self.device)])
    self.bn_list = nn.ModuleList([BatchNorm(hidden_channels)])
    self.conv_list.append(
        GCNConv(
            hidden_channels + dim,
            hidden_channels,
        ).to(self.device))
    self.bn_list.append(BatchNorm(hidden_channels))

    # pools
    self.pool_list = nn.ModuleList()
    for _ in range(n_pools):
      # # MY NEIGHBORHOOD POOL
      self.pool_list.append(
          NeighborhoodPool(
              dim,
              hidden_channels,
              self.k_hops,
              device=device,
          ).to(device))

      self.conv_list.append(
          GCNConv(
              hidden_channels + dim,
              hidden_channels,
          ).to(self.device))
      self.bn_list.append(BatchNorm(hidden_channels))

    # latent
    # out_sz = get_pooled_sz(init_data.num_nodes, k_hops, n_pools)
    self.conv_list.append(
        GCNConv(
            hidden_channels + dim,
            hidden_channels,
        ).to(self.device))
    self.bn_list.append(BatchNorm(hidden_channels))

    self.lin = Linear(hidden_channels, latent_channels).to(self.device)

  def forward(self, x, y, edge_index, pos):
    x = F.selu((self.conv_list[0](torch.cat(
        (x, pos, y*torch.ones_like(pos)), dim=1), edge_index)),
               inplace=True)
    x = F.selu((self.conv_list[1](torch.cat((x, pos), dim=1), edge_index)),
               inplace=True)

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    if self.training:
      score_list = []
    for l, pool in enumerate(self.pool_list):
      if self.training:
        x, edge_index, pos, score = pool(x, edge_index, pos)
      else:
        x, edge_index, pos = pool(x, edge_index, pos)

      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)
      if self.training:
        score_list.insert(0, score)
      x = F.selu((self.conv_list[l + 2](torch.cat(
          (x, pos), dim=1), edge_index)),
                 inplace=True)

    x = F.selu((self.conv_list[-1](torch.cat((x, pos), dim=1), edge_index)),
               inplace=True)
    x = self.lin(x)
    if self.training:
      return x, pool_edge_list, pool_pos_list, score_list
    return x, pool_edge_list, pool_pos_list


class StructureEncoder(Encoder):

  def __init__(self,
               dim: int,
               init_data: Data,
               hidden_channels: int,
               latent_channels: int,
               n_pools: int,
               k_hops: int,
               device: str = "cpu"):
    super(StructureEncoder,
          self).__init__(dim, init_data, hidden_channels, latent_channels,
                         n_pools, k_hops, device)
    self.in_channels = 1

    # pools
    self.pool_list = nn.ModuleList()
    for _ in range(n_pools):
      self.pool_list.append(
          NeighborhoodPool(
              dim,
              hidden_channels,
              self.k_hops,
              device=device,
          ).to(device))

  def forward(self, x, edge_index, pos):

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    if self.training:
      score_list = []
    for l, pool in enumerate(self.pool_list):
      # # MY NEIGHBORHOOD POOL
      if self.training:
        x, edge_index, pos, score = pool(x, edge_index, pos)
      else:
        x, edge_index, pos = pool(x, edge_index, pos)

      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)
      if self.training:
        score_list.insert(0, score)

    if self.training:
      return pool_edge_list, pool_pos_list, score_list
    return pool_edge_list, pool_pos_list


class Decoder(nn.Module):

  def __init__(self,
               dim: int,
               init_data: Data,
               hidden_channels: int,
               latent_channels: int,
               n_pools: int,
               k_hops: int,
               interpolate: callable = onera_interp,
               device: str = "cpu"):
    super(Decoder, self).__init__()
    self.dim = dim
    self.out_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.n_pools = n_pools
    self.k_hops = k_hops
    self.interpolate = interpolate
    self.device = device

    # latent dense map
    # self.out_sz = get_pooled_sz(init_data.num_nodes, k_hops, n_pools)
    self.lin = Linear(latent_channels, hidden_channels).to(self.device)
    self.bn_list = nn.ModuleList([BatchNorm(hidden_channels)])

    # initial aggr
    self.conv_list = nn.ModuleList(
        [GCNConv(
            hidden_channels + dim,
            hidden_channels,
        ).to(self.device)])
    self.bn_list.append(BatchNorm(hidden_channels))

    self.conv_list.append(
        GCNConv(
            hidden_channels + dim,
            hidden_channels,
        ).to(self.device))
    self.bn_list.append(BatchNorm(hidden_channels))

    # # no initial aggr
    # self.conv_list = nn.ModuleList()

    # unpools
    for _ in range(n_pools):
      self.conv_list.append(
          GCNConv(
              hidden_channels + dim,
              hidden_channels,
          ).to(self.device))
      self.bn_list.append(BatchNorm(hidden_channels))

    self.conv_list.append(
        GCNConv(
            hidden_channels + dim,
            self.out_channels,
        ).to(self.device))

  def forward(self, latent, edge_index_list, pos_list):
    # INITIAL AGG
    x = F.selu((self.lin(latent)), inplace=True)
    edge_index = edge_index_list[0]
    pos = pos_list[0]
    x = F.selu((self.conv_list[0](torch.cat((x, pos), dim=1), edge_index)),
               inplace=True)
    x = F.selu((self.conv_list[1](torch.cat((x, pos), dim=1), edge_index)),
               inplace=True)

    # # NO INITIAL AGG
    # x = latent
    for l in range(self.n_pools):
      # deg = get_deg(x, edge_index, self.device)
      x = self.interpolate(x, pos_list[l], pos_list[l + 1])
      edge_index = edge_index_list[l + 1]
      pos = pos_list[l + 1]
      # WITH INITIAL AGG
      x = F.selu((self.conv_list[l + 2](torch.cat(
          (x, pos), dim=1), edge_index)),
                 inplace=True)

      # # # NO INITIAL AGG
      # x = F.selu(
      #     self.conv_list[l](torch.cat((x, pos), dim=1), edge_index),
      #     inplace=True)
    out = self.conv_list[-1](torch.cat((x, pos), dim=1), edge_index)
    if out.isnan().sum():
      breakpoint()
    return out


class DBA(nn.Module):

  def __init__(self,
               dim: int,
               init_data: PairData,
               hidden_channels: int,
               latent_channels: int,
               n_pools: int,
               k_hops: int,
               device: str = "cpu"):
    super(DBA, self).__init__()
    self.dim = dim
    self.in_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.n_pools = n_pools
    self.k_hops = k_hops
    self.device = device

    # only used for getting pooling structure
    init_data_3 = Data(
        x=init_data.x_3, edge_index=init_data.edge_index_3, pos=init_data.pos_3)
    self.encoder3D = StructureEncoder(dim, init_data_3, hidden_channels,
                                      latent_channels, n_pools, k_hops,
                                      device)

    # used for model eval
    init_data_2 = Data(
        x=init_data.x_2, edge_index=init_data.edge_index_2, pos=init_data.pos_2)
    self.encoder2D = Encoder(dim, init_data_2, hidden_channels, latent_channels,
                             n_pools, k_hops, device).to(self.device)
    self.decoder = Decoder(
        dim,
        init_data_3,
        hidden_channels,
        latent_channels,
        n_pools + 1,
        k_hops,
        device=device).to(self.device)

  def forward(self,
              x_3,
              edge_index_3,
              pos_3,
              x_2,
              edge_index_2,
              pos_2,
              y,
              pool_edge_list=None,
              pool_pos_list=None):
    ret = 0
    if self.training:
      latent_2, edge_list_2, pos_list_2, score_list_2 = self.encoder2D(
          x_2, y, edge_index_2, pos_2)
    else:
      latent_2, edge_list_2, pos_list_2 = self.encoder2D(
          x_2, y, edge_index_2, pos_2)

    if pool_edge_list is None and pool_pos_list is None:
      if self.training:
        pool_edge_list, pool_pos_list, score_list_3 = self.encoder3D(
            x_3, edge_index_3,
            pos_3)
      else:
        pool_edge_list, pool_pos_list = self.encoder3D(
            x_3, edge_index_3,
            pos_3)
      ret = 1
      pool_edge_list.insert(0,
                            edge_list_2[0])
      pool_pos_list.insert(0, pos_list_2[0])

      # pool_edge_list = [x.to(self.device) for x in pool_edge_list]
      # pool_pos_list = [x.to(self.device) for x in pool_pos_list]
    else:
      pool_edge_list.pop(0)
      pool_edge_list.insert(0,
                            edge_list_2[0])

      pool_pos_list.pop(0)
      pool_pos_list.insert(0, pos_list_2[0])

      # pool_edge_list = [x.to(self.device) for x in pool_edge_list]
      # pool_pos_list = [x.to(self.device) for x in pool_pos_list]

    out = self.decoder(latent_2, pool_edge_list, pool_pos_list)

    if ret == 1:
      if self.training:
        return (out, edge_list_2, pool_edge_list, pos_list_2, pool_pos_list,
                score_list_2, score_list_3)
      return out, pool_edge_list, pool_pos_list
    if self.training:
      return out, score_list_2
    return out
