import os
import math
import torch
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import OptTensor
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import SAGEConv, SAGPooling, GATConv, GATv2Conv, GCNConv, GraphConv, GINConv, Linear, PNAConv, knn_interpolate
from torch_geometric.nn import Sequential as GeoSequential
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import softmax, add_self_loops
from torch.nn import ReLU, Sequential
import torch.nn.functional as F
from graphdata import PairData
import numpy as np


def get_pooled_sz(full_sz: int, ratio: float, layer: int):
  out_sz = full_sz
  for l in range(layer):
    out_sz = int(-(out_sz // -(1 / ratio)))
  return out_sz


def get_deg(x, edge_index, device):
  deg = torch.sparse_coo_tensor(
      edge_index,
      torch.ones((edge_index.size(1),)).to(device)).to(device) @ torch.ones(
          (x.size(0), 1)).to(device)
  return deg


def get_edge_attr(edge_index, y, mode:str="diff"):
  # edge_attr = torch.zeros((edge_index.size(1), 3)).to(self.device)
  # for i, xs in enumerate(edge_index.transpose(0, 1)):
  #   edge_attr[i] = pos[xs[1]] - pos[xs[0]]
  if mode == "diff":
    edge_attr = y[edge_index[1]] - y[edge_index[0]]
  elif mode == "cat":
    edge_attr = y[edge_index[0]]
  elif mode == "cat_both":
    edge_attr = torch.cat((y[edge_index[0]],y[edge_index[1]]),dim=1)
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


class SAGPoolWithPos(SAGPooling):

  def __init__(
      self,
      in_channels: int,
      ratio: Union[float, int] = 0.5,
      GNN: torch.nn.Module = GATv2Conv,
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

  def __init__(self, x: torch.Tensor, edge_index: torch.Tensor,
               pos: torch.Tensor, node_id: int, k: int):
    self.x = x
    self.pos = pos
    self.node_id = node_id
    self.k = k

    # renumber nodes
    old_ids = torch.sort(torch.unique(edge_index))
    self.old_ids = old_ids

    for new, old in enumerate(old_ids):
      edge_index = torch.where(edge_index == old,
                               torch.full_like(edge_index, new), edge_index)
    self.edge_index = edge_index

    self.pool_val = torch.max(x, 0)


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
    self.lin = Linear(in_channels, out_channels=1)
    self.device = device

  def forward(self, x, edge_index, pos):
    n_list = []
    old_edge_index = edge_index
    # cluster and pool
    while x.size(0) > 0:
      edge_aug, edge_attr = get_edge_aug(edge_index, pos, self.k_max,
                                         self.device)
      breakpoint()
      score = torch.squeeze(
          self.gnn(torch.cat((x, pos), dim=1), edge_aug, edge_attr) +
          self.lin(torch.cat((x, pos), dim=1)))
      # edge_attr = get_edge_attr(edge_index, pos)
      # score = softmax(torch.squeeze(self.gnn(x, edge_index, edge_attr)))
      node_id = torch.argmax(score)
      k = 1
      k_hop_nh = torch.unique(edge_index[:, 1][edge_index[:, 0] == node_id])
      if k_hop_nh.size(0) < self.n_size and k_hop_nh.size(0) < x.size(0):
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1),).to(self.device))
        adj_aug = adj
        while k_hop_nh.size(0) < self.n_size and k_hop_nh.size(0) < x.size(
            0) and k < self.k_max:
          k += 1
          adj_aug = (adj_aug @ adj).coalesce()
          edge_aug = adj_aug.indices()
          k_hop_nh = torch.unique(edge_aug[:, 1][edge_aug[:, 0] == node_id])

      n_list.append(
          Neighborhood(x[k_hop_nh], edge_aug[:, edge_aug[:, 0] == node_id],
                       pos[node_id], len(n_list), k))

      keep_idx = torch.tensor([i not in k_hop_nh for i in range(x.size(0))
                              ]).to(self.device)
      x = x[keep_idx, :]
      pos = pos[keep_idx, :]

      keep_idx = (torch.tensor([i not in k_hop_nh for i in edge_index[:, 0]])
                  & torch.tensor([i not in k_hop_nh for i in edge_index[:, 1]
                                 ])).to(self.device)
      edge_index = edge_index[:, keep_idx]
      for new, old in enumerate(torch.sort(torch.unique(edge_index))):
        edge_index = torch.where(edge_index == old,
                                 torch.full_like(edge_index, new), edge_index)

    # recombine
    edge_index = old_edge_index
    x = torch.stack([n.pool_val for n in n_list])
    pos = torch.stack([n.pos for n in n_list])
    n_ids = torch.stack([n.node_id for n in n_list])

    # create clustering matrix using n.node_id
    clusters = torch.concat(
        [torch.full((n.node_ids.size(0),), n.node_id) for n in n_list])
    clusters = clusters[torch.concat([n.old_ids for n in n_list])]

    # get edge attr by differencing clustering feature
    edge_attr = get_edge_attr(edge_index, clusters)

    # non-zero edges get saved to list
    nz_edges = edge_index[:, torch.argwhere(edge_attr)]

    # re-index clustering with saved edges to get edges of pooled graph
    pool_edge_index = torch.stack(
        (clusters[nz_edges[:, 0]], clusters[nz_edges[:, 1]]))
    pool_edge_index = torch.unique(pool_edge_index, dim=1)
    for new, old in enumerate(n_ids):
      pool_edge_index = torch.where(pool_edge_index == old,
                                    torch.full_like(pool_edge_index, new),
                                    pool_edge_index)

    return x, pool_edge_index, pos


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
    self.lin_list = nn.ModuleList(
        [Linear(self.in_channels + dim + 3, hidden_channels).to(device)])

    self.conv_list = nn.ModuleList([
        GATv2Conv(hidden_channels + dim, hidden_channels,
                  edge_dim=dim).to(device)
    ])
    self.lin_list.append(
        Linear(hidden_channels + dim, hidden_channels).to(device))

    self.conv_list.append(
        GATv2Conv(hidden_channels + dim, hidden_channels,
                  edge_dim=dim).to(device))
    self.lin_list.append(
        Linear(hidden_channels + dim, hidden_channels).to(device))

    # pools
    self.pool_list = nn.ModuleList()
    for _ in range(n_pools):
      # # SAGPOOL
      # self.pool_list.append(
      #     SAGPoolWithPos(
      #         hidden_channels + dim,
      #         pool_ratio,
      #         hidden_channels=hidden_channels,
      #         edge_dim=dim,
      #         device=self.device).to(self.device))

      # # MY NEIGHBORHOOD POOL
      self.pool_list.append(
          NeighborhoodPool(
              hidden_channels + dim, 1 // pool_ratio, device=device).to(device))

      self.conv_list.append(
          GATv2Conv(hidden_channels + dim, hidden_channels,
                    edge_dim=dim).to(device))
      self.lin_list.append(
          Linear(hidden_channels + dim, hidden_channels).to(device))

    # latent
    # out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    self.conv_list.append(
        GATv2Conv(hidden_channels + dim, hidden_channels,
                  edge_dim=dim).to(device))
    self.lin_list.append(
        Linear(hidden_channels + dim, hidden_channels).to(device))

    self.lin_list.append(Linear(hidden_channels, latent_channels).to(device))

  def forward(self, x, y, edge_index, pos):
    x = self.lin_list[0](torch.cat((x, pos, y*torch.ones_like(pos)), dim=1))

    # edge_attr = get_edge_attr(edge_index, pos)
    edge_aug, edge_attr = get_edge_aug(edge_index, pos, self.k_size,
                                       self.device)
    x = F.elu(self.conv_list[0](torch.cat((x,
                                           pos), dim=1), edge_aug, edge_attr) +
              self.lin_list[1](torch.cat((x, pos), dim=1)))
    x = F.elu(
        self.conv_list[1](torch.cat(
            (x, pos), dim=1), edge_aug, edge_attr) + self.lin_list[2](torch.cat(
                (x, pos), dim=1)),
        inplace=True)

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    edge_attr_list = [edge_attr]
    for l, pool in enumerate(self.pool_list):
      # # SAGPOOL
      # x, edge_index, pos, _, _, _, _ = pool(x, edge_index, pos)

      # # MY NEIGHBORHOOD POOL
      x, edge_index, pos = pool(x, edge_index, pos)

      # edge_attr = get_edge_attr(edge_index, pos)
      edge_aug, edge_attr = get_edge_aug(edge_index, pos, self.k_size,
                                         self.device)

      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)
      edge_attr_list.insert(0, edge_attr)

      x = F.elu(
          self.conv_list[l + 3](torch.cat(
              (x, pos), dim=1), edge_aug, edge_attr) +
          self.lin_list[l + 2](torch.cat((x, pos), dim=1)),
          inplace=True)

    x = F.elu(
        self.conv_list[-1](torch.cat((x, pos), dim=1), edge_aug, edge_attr) +
        self.lin_list[-2](torch.cat((x, pos), dim=1)),
        inplace=True)
    x = self.lin_list[-1](x)
    return x, pool_edge_list, pool_pos_list, edge_attr_list


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
    self.conv_list = nn.ModuleList([
        GATv2Conv(self.in_channels + dim, hidden_channels,
                  edge_dim=dim).to(self.device)
    ])
    self.conv_list.append(
        GATv2Conv(hidden_channels + dim, hidden_channels,
                  edge_dim=dim).to(self.device))

    # pools
    self.pool_list = nn.ModuleList()
    for _ in range(n_pools):
      self.pool_list.append(
          SAGPoolWithPos(
              hidden_channels + dim,
              pool_ratio,
              hidden_channels=hidden_channels,
              edge_dim=dim,
              device=self.device).to(self.device))
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
      x, edge_index, pos, _, _, _, _ = pool(x, edge_index, pos)
      edge_attr = get_edge_attr(edge_index, pos)

      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)
      edge_attr_list.insert(0, edge_attr)

      x = F.elu(
          self.conv_list[l + 2](torch.cat((x, pos), dim=1), edge_index,
                                edge_attr),
          inplace=True)
    return pool_edge_list, pool_pos_list, edge_attr_list


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
    self.lin_list = nn.ModuleList(
        [Linear(latent_channels, hidden_channels).to(device)])

    # initial aggr
    self.conv_list = nn.ModuleList([
        GATv2Conv(hidden_channels + dim, hidden_channels,
                  edge_dim=dim).to(device)
    ])
    self.lin_list.append(
        Linear(hidden_channels + dim, hidden_channels).to(device))

    self.conv_list.append(
        GATv2Conv(hidden_channels + dim, hidden_channels,
                  edge_dim=dim).to(device))
    self.lin_list.append(
        Linear(hidden_channels + dim, hidden_channels).to(device))

    # # no initial aggr
    # self.conv_list = nn.ModuleList()

    # unpools
    for _ in range(n_pools):
      self.conv_list.append(
          GATv2Conv(hidden_channels + dim, hidden_channels,
                    edge_dim=dim).to(device))
      self.lin_list.append(
          Linear(hidden_channels + dim, hidden_channels).to(device))

    self.conv_list.append(
        GATv2Conv(hidden_channels + dim, hidden_channels,
                  edge_dim=dim).to(device))
    self.lin_list.append(
        Linear(hidden_channels + dim, hidden_channels).to(device))
    self.lin_list.append(
        Linear(hidden_channels + dim, self.out_channels).to(device))

  def forward(self, latent, edge_index_list, pos_list, edge_attr_list):
    # INITIAL AGG
    x = self.lin_list[0](latent)

    edge_index = edge_index_list[0]
    pos = pos_list[0]
    # edge_attr = get_edge_attr(edge_index, pos)
    edge_aug, edge_attr = get_edge_aug(edge_index, pos, self.k_size,
                                       self.device)
    x = F.elu(
        self.conv_list[0](torch.cat(
            (x, pos), dim=1), edge_aug, edge_attr) + self.lin_list[1](torch.cat(
                (x, pos), dim=1)),
        inplace=True)
    x = F.elu(
        self.conv_list[1](torch.cat(
            (x, pos), dim=1), edge_aug, edge_attr) + self.lin_list[2](torch.cat(
                (x, pos), dim=1)),
        inplace=True)

    # # NO INITIAL AGG
    # x = latent

    for l in range(self.n_pools):
      # deg = get_deg(x, edge_index, self.device)
      x = knn_interpolate(x, pos_list[l], pos_list[l + 1])
      edge_index = edge_index_list[l + 1]
      pos = pos_list[l + 1]
      # edge_attr = edge_attr_list[l + 1]
      edge_aug, edge_attr = get_edge_aug(edge_index, pos, self.k_size,
                                         self.device)

      # WITH INITIAL AGG
      x = F.elu(
          self.conv_list[l + 2](torch.cat(
              (x, pos), dim=1), edge_aug, edge_attr) +
          self.lin_list[l + 3](torch.cat((x, pos), dim=1)),
          inplace=True)

      # # # NO INITIAL AGG
      # x = F.elu(
      #     self.conv_list[l](torch.cat((x, pos), dim=1), edge_aug, edge_attr),
      #     inplace=True)

    x = F.elu(
        self.conv_list[-1](torch.cat((x, pos), dim=1), edge_aug, edge_attr) +
        self.lin_list[-2](torch.cat((x, pos), dim=1)),
        inplace=True)
    out = self.lin_list[-1](torch.cat((x, pos), dim=1))
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
    self.decoder = Decoder(dim, init_data_3, hidden_channels, latent_channels,
                           k_size, n_pools + 1, pool_ratio,
                           device).to(self.device)

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
    latent_2, edge_list_2, pos_list_2, edge_attr_2 = self.encoder2D(
        x_2, y, edge_index_2, pos_2)
    if pool_edge_list is None and pool_pos_list is None:
      pool_edge_list, pool_pos_list, edge_attr_list = self.encoder3D(
          x_3, edge_index_3, pos_3)
      ret = 1
      pool_edge_list.insert(0, edge_list_2[0])
      pool_pos_list.insert(0, pos_list_2[0])
      edge_attr_list.insert(0, edge_attr_2[0])
    else:
      pool_edge_list.pop(0)
      pool_edge_list.insert(0, edge_list_2[0])

      pool_pos_list.pop(0)
      pool_pos_list.insert(0, pos_list_2[0])

      edge_attr_list.pop(0)
      edge_attr_list.insert(0, edge_attr_2[0])

    out = self.decoder(latent_2, pool_edge_list, pool_pos_list, edge_attr_list)

    if ret == 1:
      return out, pool_edge_list, pool_pos_list
    return out


class LaplaceLayer(nn.Module):

  def __init__(self, channels, kept_modes: int = 20, device: str = "cpu"):
    super(LaplaceLayer, self).__init__()
    self.channels = channels
    self.kept_modes = kept_modes
    self.device = device

    self.lin = Linear(channels, channels, bias=False).to(device)
    self.kernel = Parameter(torch.Tensor(kept_modes, channels,
                                         channels)).to(device)

    self.reset_parameters()

  def reset_parameters(self):
    self.lin.reset_parameters()
    torch.nn.init.normal_(self.kernel)

  def _tensor_op(self, r, v):
    return torch.sum(r*v)

  def tensor_op(self, R, V):
    return torch.squeeze(
        torch.vmap(
            torch.vmap(self._tensor_op, in_dims=(0, None), out_dims=0),
            in_dims=(0, 0))(R, V))

  def forward(self, x, phi):
    v = torch.linalg.pinv(phi) @ x
    out = F.elu(self.lin(x) + phi @ self.tensor_op(self.kernel, v))
    return out


class LNO(nn.Module):

  def __init__(self,
               dim,
               init_data,
               hidden_channels,
               out_channels,
               kept_modes: int = 20,
               laplace_layers: int = 4,
               device: str = "cpu") -> None:
    super(LNO, self).__init__()
    self.dim = dim
    self.in_channels = init_data.num_node_features + dim + init_data.y.size(0)
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.kept_modes = min(kept_modes, init_data.x.size(0))
    self.laplace_layers = laplace_layers
    self.device = device

    self.lift = Linear(self.in_channels, hidden_channels).to(device)

    self.lapl_list = nn.ModuleList()
    for l in range(laplace_layers):
      self.lapl_list.append(
          LaplaceLayer(hidden_channels, kept_modes, device).to(device))

    self.lower = Linear(hidden_channels, out_channels).to(device)

    vec_path = os.path.join(os.environ["SCRATCH"],
                            "ORNL/dimension-bridging/data/processed/eig")
    self.phi = torch.stack([
        torch.load(os.path.join(vec_path, "vec_{:d}.pt".format(i))).to(device)
        for i in range(kept_modes)
    ],
                           dim=1)

    self.reset_parameters()

  def reset_parameters(self):
    self.lift.reset_parameters()
    self.lower.reset_parameters()

  def forward(self, x, pos, y):
    x = self.lift(
        torch.cat((x, pos, y*torch.ones_like(pos).to(self.device)), dim=1))

    for l in range(self.laplace_layers):
      x = self.lapl_list[l](x, self.phi)

    out = self.lower(x)
    return out


class GKMLP(nn.Module):

  def __init__(self,
               in_channels: int,
               hidden_channels: int,
               out_channels: int,
               device: str = "cpu") -> None:
    super(GKMLP, self).__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels

    self.lift = Linear(in_channels, hidden_channels).to(device)
    self.hidden = Linear(hidden_channels, hidden_channels).to(device)
    self.lower = Linear(hidden_channels, out_channels).to(device)

    self.bnorms = nn.ModuleList([BatchNorm(hidden_channels) for i in range(2)])

    self.reset_parameters()

  def reset_parameters(self):
    self.lift.reset_parameters()
    self.hidden.reset_parameters()
    self.lower.reset_parameters()

  def forward(self, x):
    x = F.elu(self.bnorms[0](self.lift(x)))
    x = F.elu(self.bnorms[1](self.hidden(x)), inplace=True)
    out = self.lower(x)
    return out


class GraphKernelConv(MessagePassing):

  def __init__(self,
               dim: int,
               in_channels: int,
               hidden_channels: int,
               out_channels: int,
               kernel: nn.Module = GKMLP,
               device: str = "cpu",
               **kwargs):
    super(GraphKernelConv, self).__init__()
    self.dim = dim
    self.in_channels = in_channels
    self.out_channels = out_channels

    if kernel == GKMLP:
      self.kernel = kernel(
          dim + dim + in_channels + in_channels,
          hidden_channels,
          hidden_channels,
          device=device,
          **kwargs)
    elif kernel == PNAConv:
      aggs = ["mean", "min", "max", "std"]
      scls = ["identity", "amplification", "attenuation"]
      self.kernel = PNAConv(aggregators=aggs, scalers=scls, **kwargs)
    else:
      self.kernel = kernel(**kwargs)

    self.device = device

  def forward(self, x, edge_index, pos, y):
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    edge_weight = self.kernel(
        torch.cat((pos[edge_index[0, :]], pos[edge_index[1, :]],
                   y[edge_index[0, :]], y[edge_index[1, :]]),
                  dim=1))

    deg = get_deg(x, edge_index, self.device)
    deg = deg[edge_index[1, :]]

    edge_weight = edge_weight / deg

    out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
    return out

  def message(self,
              x_j: Tensor,
              edge_weight: OptTensor,
              deg: OptTensor = None) -> Tensor:
    if deg is None:
      return x_j if edge_weight is None else edge_weight*x_j
    return x_j / deg if edge_weight is None else (edge_weight /
                                                  deg)*x_j


class GKLayer(nn.Module):

  def __init__(self,
               dim: int,
               init_data: Data,
               channels: int,
               k_hops: int,
               device: str = "cpu"):
    super(GKLayer, self).__init__()
    self.dim = dim
    self.channels = channels
    self.k_hops = k_hops
    self.device = device
    self.aug_cache = None

    deg = get_deg(init_data.x, init_data.edge_index, device)

    self.lin = Linear(channels, channels, bias=False).to(device)
    self.gnn = GraphKernelConv(
        dim,
        init_data.num_node_features,
        channels,
        channels,
        kernel=GKMLP,
        device=device).to(device)
    self.bn = BatchNorm(channels)

    self.reset_parameters()

  def reset_parameters(self):
    self.lin.reset_parameters()

  def forward(self, x, edge_index, pos, y):
    # out = F.elu(self.lin(x) + self.gnn(torch.cat((x,pos),dim=1), edge_index, pos))
    # if self.kernel == GKMLP:
    out = F.elu(self.bn(self.lin(x) + self.gnn(x, edge_index, pos, y)))
    # elif self.kernel == PNAConv:
    #   edge_attr = get_edge_attr(edge_index, pos, "cat_both")
    #   out = F.elu(self.lin(x) + self.gnn(x, edge_index, edge_attr))
    return out


class GKNO(nn.Module):

  def __init__(self,
               dim,
               init_data,
               hidden_channels,
               out_channels,
               k_hops: int = 5,
               gk_layers: int = 4,
               device: str = "cpu") -> None:
    super(GKNO, self).__init__()
    self.dim = dim
    self.in_channels = init_data.num_node_features + dim + init_data.y.size(0)
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.k_hops = k_hops
    self.aug_cache = None
    self.gk_layers = gk_layers
    self.device = device

    self.lift = Linear(self.in_channels, hidden_channels).to(device)

    self.gk_list = nn.ModuleList()
    for l in range(gk_layers):
      self.gk_list.append(
          GKLayer(dim, init_data, hidden_channels, k_hops, device).to(device))

    self.lower = Linear(hidden_channels, out_channels).to(device)

    self.reset_parameters()

  def reset_parameters(self):
    self.lift.reset_parameters()
    self.lower.reset_parameters()

  def forward(self, x, edge_index, pos, y):
    out = self.lift(
        torch.cat((x, pos, y*torch.ones_like(pos).to(self.device)), dim=1))

    if self.aug_cache is None:
      edge_aug, _ = get_edge_aug(edge_index, pos, self.k_hops, self.device)
      self.aug_cache = edge_aug
    else:
      edge_aug = self.aug_cache

    for l in range(self.gk_layers):
      out = self.gk_list[l](out, edge_aug, pos, x)

    out = self.lower(out)
    return out