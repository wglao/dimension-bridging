import torch
import math
from torch import Tensor
from torch_geometric.typing import OptTensor
from typing import Tuple, Union, Optional, Callable
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv, SAGPooling, GATv2Conv, GraphConv, GINConv, Linear, knn_interpolate
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
  # edge_attr = torch.zeros((edge_index.size(1), 3)).to(self.device)
  # for i, xs in enumerate(edge_index.transpose(0, 1)):
  #   edge_attr[i] = pos[xs[1]] - pos[xs[0]]
  edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
  return edge_attr


def get_edge_aug(edge_index, pos, steps: int = 1, device: str = "cpu"):
  adj = torch.sparse_coo_tensor(edge_index,
                                torch.ones(edge_index.size(1),).to(device))
  adj_aug = adj
  if steps >= 1:
    for _ in range(steps - 1):
      adj_aug = (adj_aug @ adj).coalesce()
    adj_aug = (adj + adj_aug).coalesce()
  edge_index_aug = adj_aug.indices()
  edge_attr_aug = get_edge_attr(edge_index_aug, pos)
  return edge_index_aug, edge_attr_aug


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
      device: str = "cpu",
      **kwargs,
  ):
    super(SAGPoolWithPos, self).__init__(in_channels, ratio, GNN, min_score,
                                         multiplier, nonlinearity, **kwargs)
    self.augmentation = augmentation
    self.device = device
    self.gnn = self.gnn.to(device)

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
    edge_attr = get_edge_attr(edge_index, pos)
    if self.augmentation:
      adj = torch.sparse_coo_tensor(
          edge_index,
          torch.ones(edge_index.size(1),).to(self.device))
      adj_aug = (adj + (adj@adj)).coalesce()
      edge_index_aug = adj_aug.indices()
      edge_attr_aug = get_edge_attr(edge_index_aug, pos)
      score = self.gnn(attn, edge_index_aug, edge_attr_aug).view(-1)
    else:
      # edge_attr = get_edge_attr(edge_index, pos)
      score = self.gnn(attn, edge_index, edge_attr).view(-1)

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


class Neighborhood():

  def __init__(self, x: torch.Tensor, old_ids: torch.Tensor, pos: torch.Tensor,
               node_id: int):
    self.x = x
    self.pos = pos
    self.node_id = node_id
    self.old_ids = old_ids

    self.pool_val, _ = torch.max(x, 0)


class NeighborhoodPool(nn.Module):

  def __init__(self,
               in_channels: int,
               ratio: float = 0.8,
               gnn: nn.Module = GraphConv,
               device: str = "cpu",
               **kwargs) -> None:
    super(NeighborhoodPool, self).__init__()
    self.ratio = ratio
    self.gnn = gnn(
        in_channels=in_channels, out_channels=1,
        **kwargs).to(device)
    self.device = device

  def pool(self, x, edge_index, pos, first: bool):
    n_list = []
    edge_index, _ = add_remaining_self_loops(edge_index)
    old_edge_index = edge_index
    old_ids = torch.arange(x.size(0)).to(self.device)
    score = torch.squeeze(
        self.gnn(torch.cat((x, pos), dim=1), edge_index))
    sort_score, _ = torch.sort(score)
    deg = get_deg(x, edge_index, self.device)
    if first:
      k = int(-(math.log(1 / self.ratio) //
                -math.log(deg.mean() - deg.min()))) + 1
    else:
      k = 0

    # cluster and pool
    n = -1
    remaining = old_ids
    # if first:
    while edge_index.size(1) > 0:
      print("remaining: {:d}".format(remaining.size(0)))
      adj = torch.sparse_coo_tensor(
          edge_index,
          torch.ones(edge_index.size(1),).to(self.device))
      adj_aug = adj.coalesce()
      for _ in range(k):
        adj_aug = (adj_aug @ adj).coalesce()
      edge_aug = adj_aug.indices()

      next_score = sort_score[n]
      node_id = torch.argwhere(score == next_score)[0].item()
      n -= 1

      # prioritize connected nodes
      while node_id not in edge_index:
        next_score = sort_score[n]
        node_id = torch.argwhere(score == next_score)[0].item()
        n -= 1
        if node_id in edge_index:
          break
        if -n > sort_score.size(0):
          break

      # if node is connected
      if node_id in edge_index:
        nodes = torch.unique(edge_aug[1][edge_aug[0] == node_id])
        n_list.append(
            Neighborhood(x[nodes], old_ids[nodes], pos[node_id], len(n_list)))
        keep_idx = torch.tensor([i not in nodes for i in remaining
                                ]).to(self.device)

        if keep_idx.sum() == 0:
          break

        # score = score[keep_idx]
        # x = x[keep_idx, :]
        # pos = pos[keep_idx, :]
        remaining = remaining[keep_idx]

        # remove edge if a connecting node has been pooled
        for node in nodes:
          edge_index = edge_index[:, edge_index[0] != node]
          edge_index = edge_index[:, edge_index[1] != node]

      # if remaining nodes are isolated, pool as one neighborhood
      if (edge_index.size(0) == 0 or
          -n > sort_score.size(0)) and remaining.size(0) > 0:
        edge_index = torch.tile(torch.arange(x.size(0)).to(self.device), (2, 1))
        next_score = sort_score[n]
        node_id = torch.argwhere(score == next_score)[0].item()
        n_list.append(
            Neighborhood(x[remaining], old_ids[remaining], pos[node_id],
                         len(n_list)))
        break

    # recombine
    edge_index = old_edge_index
    x = torch.stack([nbh.pool_val for nbh in n_list])
    pos = torch.stack([nbh.pos for nbh in n_list])
    # n_ids = torch.tensor([n.node_id for n in n_list])

    # create clustering matrix using n.node_id
    clusters = torch.concat([
        torch.full((nbh.old_ids.size(0),), nbh.node_id) for nbh in n_list
    ]).to(self.device)
    clusters = clusters[torch.concat([nbh.old_ids for nbh in n_list])]

    # get edge attr by differencing clustering feature
    edge_attr = get_edge_attr(edge_index, clusters)

    # non-zero edges get saved to list
    nz_edges = edge_index[:, torch.argwhere(edge_attr)]

    # re-index clustering with saved edges to get edges of pooled graph
    edge_index = torch.stack(
        (clusters[nz_edges[0, :]], clusters[nz_edges[1, :]]))
    edge_index = torch.unique(edge_index, dim=1)

    # else:
    #   next_score = sort_score[n]
    #   node_id = torch.argwhere(score == next_score)[0].item()
    #   n -= 1

    #   # prioritize connected nodes
    #   while node_id not in edge_index:
    #     next_score = sort_score[n]
    #     node_id = torch.argwhere(score == next_score)[0].item()
    #     n -= 1
    #     if node_id in edge_index:
    #       break
    #     if -n > sort_score.size(0):
    #       break

    #   # if node is connected
    #   if node_id in edge_index:
    #     nodes = torch.unique(edge_aug[1][edge_aug[0] == node_id])
    #     nbh = Neighborhood(x[nodes], old_ids[nodes], pos[node_id], len(n_list))
    #     keep_idx = torch.tensor([i not in nodes for i in remaining
    #                             ]).to(self.device)

    #   # TODO: THIS

    #   # score = score[keep_idx]
    #   # x = x[keep_idx, :]
    #   # pos = pos[keep_idx, :]
    #   remaining = remaining[keep_idx]

    #   # remove edge if a connecting node has been pooled
    #   for node in nodes:
    #     edge_index = edge_index[:, edge_index[0] != node]
    #     edge_index = edge_index[:, edge_index[1] != node]
    #   # recombine
    #   edge_index = old_edge_index
    #   x = torch.stack([nbh.pool_val for nbh in n_list])
    #   pos = torch.stack([nbh.pos for nbh in n_list])
    #   # n_ids = torch.tensor([n.node_id for n in n_list])

    #   # create clustering matrix using n.node_id
    #   clusters = torch.concat([
    #       torch.full((nbh.old_ids.size(0),), nbh.node_id) for nbh in n_list
    #   ]).to(self.device)
    #   clusters = clusters[torch.concat([nbh.old_ids for nbh in n_list])]

    #   # get edge attr by differencing clustering feature
    #   edge_attr = get_edge_attr(edge_index, clusters)

    #   # non-zero edges get saved to list
    #   nz_edges = edge_index[:, torch.argwhere(edge_attr)]

    #   # re-index clustering with saved edges to get edges of pooled graph
    #   edge_index = torch.stack(
    #       (clusters[nz_edges[0, :]], clusters[nz_edges[1, :]]))
    #   edge_index = torch.unique(edge_index, dim=1)

    return x, edge_index, pos

  def forward(self, x, edge_index, pos):
    target_sz = int((x.size(0)*self.ratio) // 1)
    print("current size: {:d}".format(x.size(0)))
    print("target size: {:d}".format(target_sz))
    x, edge_index, pos = self.pool(x, edge_index, pos, first=True)

    while x.size(0) > target_sz:
      print("current size: {:d}".format(x.size(0)))
      print("target size: {:d}".format(target_sz))
      x, edge_index, pos = self.pool(x, edge_index, pos, first=False)

    return x, edge_index, pos


class NeighborhoodScorePool(nn.Module):

  def __init__(self,
               dim,
               ratio,
               gnn: nn.Module,
               channels: int,
               device: str = "cpu",
               **kwargs) -> None:
    super().__init__()
    self.dim = dim
    self.ratio = ratio
    self.gnn1 = gnn(dim, channels, **kwargs)
    self.gnn2 = gnn(channels + dim, 1, **kwargs)
    self.device = device

  def forward(self, x: Tensor, edge_index: Tensor, pos: Tensor):
    target_size = int(-(x.size(0)*self.ratio // -1))
    score = torch.squeeze(
        self.gnn2(
            torch.cat((F.elu(self.gnn1(pos, edge_index)), pos), 1),
            edge_index))

    order = torch.argsort(score, 0)
    unsort = torch.argsort(order, 0)

    nbh_size = int(-(1 // -self.ratio))
    num_full_nbh = score.size(0) // nbh_size
    # if graph will be too coarse, intentionally use smaller neighborhoods
    if num_full_nbh < target_size:
      # a_full(n) + b(n-1) = T
      # a_full + b = t
      # a_full = T + (1-n)t
      num_full_nbh = score.size(0) + target_size*(1-nbh_size)
      num_small_nbh = target_size - num_full_nbh
    else:
      num_small_nbh = 0

    # pool remaining nodes as one nbh
    # divides = (num_full_nbh*nbh_size + num_small_nbh*(nbh_size-1)) == score.size(0)

    x = x[order]
    a = nbh_size*num_full_nbh
    b = a + (nbh_size-1)*num_small_nbh
    x_full = x[:a].view((num_full_nbh, nbh_size, x.size(1)))
    x_full = torch.vmap(torch.max, in_dims=(0, None))(x_full, 0)[0]
    if num_small_nbh > 0:
      x_small = x[a:b].view((num_small_nbh, nbh_size - 1, x.size(1)))
      x_small = torch.vmap(torch.max, in_dims=(0, None))(x_small, 0)[0]
      x_pool = torch.cat((x_full, x_small), 0)
    # elif not divides:
    #   x_pool = torch.cat(
    #       (x_full,
    #        torch.unsqueeze(torch.max(x[a:], 0)[0], 0)))

    pos = pos[order]
    pos_full = pos[:a].view((num_full_nbh, nbh_size, pos.size(1)))
    pos_std_full, pos_full = torch.vmap(
        torch.std_mean, in_dims=(0, None))(pos_full, 0)
    if num_small_nbh > 0:
      pos_small = pos[a:b].view((num_small_nbh, nbh_size - 1, pos.size(1)))
      if nbh_size > 2:
        pos_small, pos_std_small = torch.vmap(
            torch.std_mean, in_dims=(0, None))(pos_small, 0)
      else:
        pos_small = torch.vmap(torch.mean, in_dims=(0, None))(pos_small, 0)
        pos_std_small = torch.zeros_like(pos_small).to(self.device)
      pos_pool = torch.cat((pos_full, pos_small), 0)
      pos_std_pool = torch.cat((pos_std_full, pos_std_small), 0)
    # elif not divides:
    #   if score.size(0) % nbh_size > 1:
    #     extra_std, extra_pos = torch.std_mean(pos[nbh_size*num_full_nbh:], 0)
    #   else:
    #     extra_pos = pos[-1]
    #     extra_std = torch.zeros_like(extra_pos).to(self.device)
    #   pos_pool = torch.cat((pos_pool, torch.unsqueeze(extra_pos, 0)))
    #   pos_std_pool = torch.cat((pos_std_pool, torch.unsqueeze(extra_std, 0)))

    cluster_full = torch.ravel(
        torch.tile(torch.arange(num_full_nbh),
                   (nbh_size, 1)).transpose(0, 1)).to(self.device)
    if num_small_nbh > 0:
      cluster_small = torch.ravel(
          torch.tile(torch.arange(num_small_nbh),
                     (nbh_size - 1, 1)).transpose(0, 1)).to(self.device)
      cluster = torch.cat((cluster_full, cluster_small))
    # if not divides:
    #   cluster = torch.cat((cluster,
    #                        torch.full((x.size(0) - (nbh_size*num_full_nbh),),
    #                                   num_full_nbh).to(self.device)))

    cluster = cluster[unsort]

    edge_keep = torch.squeeze(
        torch.argwhere(get_edge_attr(edge_index, cluster)))
    nz_edges = edge_index[:, edge_keep]
    edge_pool = torch.stack((cluster[nz_edges[0, :]], cluster[nz_edges[1, :]]))
    edge_pool = torch.unique(edge_pool, dim=1)

    if self.training:
      return x_pool, edge_pool, pos_pool, score, pos_std_pool
    return x_pool, edge_pool, pos_pool


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
        GraphConv(
            self.in_channels + dim + 3,
            hidden_channels,
        ).to(self.device)
    ])
    self.bn_list = nn.ModuleList([BatchNorm(hidden_channels)])
    self.conv_list.append(
        GraphConv(
            hidden_channels + dim,
            hidden_channels,
        ).to(self.device))
    self.bn_list.append(BatchNorm(hidden_channels))

    # pools
    self.pool_list = nn.ModuleList()
    for _ in range(n_pools):
      # # SAGPOOL
      # self.pool_list.append(
      #     SAGPoolWithPos(
      #         hidden_channels + dim,
      #         pool_ratio,
      #         hidden_channels=hidden_channels,
      #         ,
      #         device=self.device).to(self.device))

      # # MY NEIGHBORHOOD POOL
      # self.pool_list.append(
      #     NeighborhoodPool(hidden_channels + dim, pool_ratio,
      #                      device=device).to(device))
      self.pool_list.append(
          NeighborhoodScorePool(
              dim,
              pool_ratio,
              GraphConv,
              hidden_channels,
              device=device,
          ).to(device))

      self.conv_list.append(
          GraphConv(
              hidden_channels + dim,
              hidden_channels,
          ).to(self.device))
      self.bn_list.append(BatchNorm(hidden_channels))

    # latent
    # out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    self.conv_list.append(
        GraphConv(
            hidden_channels + dim,
            hidden_channels,
        ).to(self.device))
    self.bn_list.append(BatchNorm(hidden_channels))

    self.lin = Linear(hidden_channels, latent_channels).to(self.device)

  def forward(self, x, y, edge_index, pos):
    x = F.elu(
        self.bn_list[0](self.conv_list[0](torch.cat(
            (x, pos, y*torch.ones_like(pos)), dim=1), edge_index)),
        inplace=True)
    x = F.elu(
        self.bn_list[1](self.conv_list[1](torch.cat((x, pos), dim=1),
                                          edge_index)),
        inplace=True)

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    if self.training:
      score_list = []
      pos_std_list = []

    for l, pool in enumerate(self.pool_list):
      # # SAGPOOL
      # x, edge_index, pos, _, _, _, _ = pool(x, edge_index, pos)

      # # MY NEIGHBORHOOD POOL
      if self.training:
        x, edge_index, pos, score, pos_std = pool(x, edge_index, pos)
        score_list.append[score]
        pos_std_list.append(pos_std)
      else:
        x, edge_index, pos = pool(x, edge_index, pos)

      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)
      x = F.elu(
          self.bn_list[l + 2](self.conv_list[l + 2](torch.cat((x, pos), dim=1),
                                                    edge_index)),
          inplace=True)

    x = F.elu(
        self.bn_list[-1](self.conv_list[-1](torch.cat((x, pos), dim=1),
                                            edge_index)),
        inplace=True)
    x = torch.vmap(self.lin, in_dims=(0,))(x)
    if self.training:
      return x, pool_edge_list, pool_pos_list, score_list, pos_std_list
    return x, pool_edge_list, pool_pos_list


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
    self.in_channels = 1

    # initial aggr
    self.conv_list = nn.ModuleList(
        [GraphConv(
            self.in_channels + dim,
            hidden_channels,
        ).to(self.device)])
    self.conv_list.append(
        GraphConv(
            hidden_channels + dim,
            hidden_channels,
        ).to(self.device))

    # pools
    self.pool_list = nn.ModuleList()
    for _ in range(n_pools):
      # # SAGPOOL
      # self.pool_list.append(
      #     SAGPoolWithPos(
      #         hidden_channels + dim,
      #         pool_ratio,
      #         hidden_channels=hidden_channels,
      #         ,
      #         device=self.device).to(self.device))

      # # MY NEIGHBORHOOD POOL
      # self.pool_list.append(
      #     NeighborhoodPool(hidden_channels + dim, pool_ratio,
      #                      device=device).to(device))
      self.pool_list.append(
          NeighborhoodScorePool(
              dim,
              pool_ratio,
              GraphConv,
              hidden_channels,
              device=device,
          ).to(device))

      self.conv_list.append(
          GraphConv(
              hidden_channels + self.dim,
              hidden_channels,
          ).to(self.device))

    # latent dense map
    # out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    # self.conv_list.append(
    #     GraphConv(hidden_channels,latent_channels).to(self.device))

  def forward(self, x, edge_index, pos):
    x = get_deg(x, edge_index, self.device)
    x = F.elu(
        self.conv_list[0](torch.cat((x, pos), dim=1), edge_index),
        inplace=True)
    x = F.elu(
        self.conv_list[1](torch.cat((x, pos), dim=1), edge_index),
        inplace=True)

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    if self.training:
      score_list = []
      pos_std_list = []

    for l, pool in enumerate(self.pool_list):
      # # SAGPOOL
      # x, edge_index, pos, _, _, _, _ = pool(x, edge_index, pos)

      # # MY NEIGHBORHOOD POOL
      if self.training:
        x, edge_index, pos, score, pos_std = pool(x, edge_index, pos)
        score_list.append(score)
        pos_std_list.append(pos_std)
      else:
        x, edge_index, pos = pool(x, edge_index, pos)

      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)

      x = F.elu(
          self.conv_list[l + 2](torch.cat((x, pos), dim=1), edge_index),
          inplace=True)
    if self.training:
      return pool_edge_list, pool_pos_list, score_list, pos_std_list
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
               interpolate: callable = onera_interp,
               device: str = "cpu"):
    super(Decoder, self).__init__()
    self.dim = dim
    self.out_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.k_size = k_size
    self.n_pools = n_pools
    self.pool_ratio = pool_ratio
    self.interpolate = interpolate
    self.device = device

    # latent dense map
    # self.out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    self.lin = Linear(latent_channels, hidden_channels).to(self.device)
    self.bn_list = nn.ModuleList([BatchNorm(hidden_channels)])

    # initial aggr
    self.conv_list = nn.ModuleList(
        [GraphConv(
            hidden_channels + dim,
            hidden_channels,
        ).to(self.device)])
    self.bn_list.append(BatchNorm(hidden_channels))

    self.conv_list.append(
        GraphConv(
            hidden_channels + dim,
            hidden_channels,
        ).to(self.device))
    self.bn_list.append(BatchNorm(hidden_channels))

    # # no initial aggr
    # self.conv_list = nn.ModuleList()

    # unpools
    for _ in range(n_pools):
      self.conv_list.append(
          GraphConv(
              hidden_channels + dim,
              hidden_channels,
          ).to(self.device))
      self.bn_list.append(BatchNorm(hidden_channels))

    self.conv_list.append(
        GraphConv(
            hidden_channels + dim,
            self.out_channels,
        ).to(self.device))

  def forward(self, latent, edge_index_list, pos_list):
    # INITIAL AGG
    x = F.elu(self.bn_list[0](self.lin(latent)), inplace=True)
    edge_index = edge_index_list[0]
    pos = pos_list[0]
    x = F.elu(
        self.bn_list[1](self.conv_list[0](torch.cat((x, pos), dim=1),
                                          edge_index)),
        inplace=True)
    x = F.elu(
        self.bn_list[2](self.conv_list[1](torch.cat((x, pos), dim=1),
                                          edge_index)),
        inplace=True)

    # # NO INITIAL AGG
    # x = latent
    for l in range(self.n_pools):
      # deg = get_deg(x, edge_index, self.device)
      x = self.interpolate(x, pos_list[l], pos_list[l + 1])
      edge_index = edge_index_list[l + 1]
      pos = pos_list[l + 1]
      # WITH INITIAL AGG
      x = F.elu(
          self.bn_list[l + 3](self.conv_list[l + 2](torch.cat((x, pos), dim=1),
                                                    edge_index)),
          inplace=True)

      # # # NO INITIAL AGG
      # x = F.elu(
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
    self.encoder3D = StructureEncoder(dim, init_data_3, hidden_channels,
                                      latent_channels, k_size, n_pools,
                                      pool_ratio, device).to(self.device)

    # used for model eval
    init_data_2 = Data(
        x=init_data.x_2, edge_index=init_data.edge_index_2, pos=init_data.pos_2)
    self.encoder2D = Encoder(dim, init_data_2, hidden_channels, latent_channels,
                             k_size, n_pools, pool_ratio,
                             device).to(self.device)
    self.decoder = Decoder(
        dim,
        init_data_3,
        hidden_channels,
        latent_channels,
        k_size,
        n_pools + 1,
        pool_ratio,
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
      latent_2, edge_list_2, pos_list_2, score_list_2, pos_std_list_2 = self.encoder2D(
          x_2, y, edge_index_2, pos_2)
    else:
      latent_2, edge_list_2, pos_list_2 = self.encoder2D(
          x_2, y, edge_index_2, pos_2)
    if pool_edge_list is None and pool_pos_list is None:
      if self.training:
        pool_edge_list, pool_pos_list, score_list_3, pos_std_list_3 = self.encoder3D(
            x_3, edge_index_3, pos_3)
      else:
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
      if self.training:
        return out, edge_list_2, pool_edge_list, pool_pos_list, score_list_2, score_list_3, pos_std_list_2, pos_std_list_3
      return out, pool_edge_list, pool_pos_list
    if self.training:
      return out, score_list_2
    return out
