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
  deg = torch.sparse_coo_tensor(edge_index,
                                torch.ones(
                                    (edge_index.size(1),)).cuda()) @ torch.ones(
                                        (x.size(0), 1)).cuda()
  return deg


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
      **kwargs,
  ):
    super(SAGPoolWithPos, self).__init__(in_channels, ratio, GNN, min_score,
                                         multiplier, nonlinearity, **kwargs)
    self.augmentation = augmentation

  def get_edge_attr(self, edge_index, pos):
    edge_attr = torch.zeros((edge_index.size(1), 3)).cuda()
    for i, xs in enumerate(edge_index.transpose(0, 1)):
      edge_attr[i] = pos[xs[1]] - pos[xs[0]]
    return edge_attr

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
      edge_attr = self.get_edge_attr(edge_index_aug, pos)
      score = self.gnn(attn, edge_index_aug, edge_attr).view(-1)
    else:
      edge_attr = self.get_edge_attr(edge_index, pos)
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


class Encoder(nn.Module):

  def __init__(self, dim: int, init_data: Data, hidden_channels: int,
               latent_channels: int, k_size: int, n_pools: int,
               pool_ratio: float):
    super(Encoder, self).__init__()
    self.dim = dim
    self.in_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.k_size = k_size
    self.n_pools = n_pools
    self.pool_ratio = pool_ratio

    # initial aggr
    self.conv_list = [
        GATv2Conv(self.in_channels + dim, hidden_channels).cuda(),
        GATv2Conv(hidden_channels + dim, hidden_channels).cuda(),
    ]

    # pools
    self.pool_list = []
    for _ in range(n_pools):
      self.pool_list.append(
          SAGPoolWithPos(
              hidden_channels + dim,
              pool_ratio,
              hidden_channels=hidden_channels).cuda())
      self.conv_list.append(
          GATv2Conv(hidden_channels + dim, hidden_channels).cuda())

    # latent
    # out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    self.conv_list.append(
        GATv2Conv(hidden_channels + dim, latent_channels).cuda())

  def get_edge_attr(self, edge_index, pos):
    edge_attr = torch.zeros((edge_index.size(1), 3)).cuda()
    for i, xs in enumerate(edge_index.transpose(0, 1)):
      edge_attr[i] = pos[xs[1]] - pos[xs[0]]
    return edge_attr

  def forward(self, x, edge_index, pos):
    edge_attr = self.get_edge_attr(edge_index, pos)
    x = F.elu(
        self.conv_list[0](torch.cat((x, pos), dim=1), edge_index, edge_attr),
        inplace=True)
    x = F.elu(
        self.conv_list[1](torch.cat((x, pos), dim=1), edge_index, edge_attr),
        inplace=True)

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    for l, pool in enumerate(self.pool_list):
      x, edge_index, pos, _, _, _, _ = pool(x, edge_index, pos)
      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)

      edge_attr = self.get_edge_attr(edge_index, pos)
      x = F.elu(
          self.conv_list[l + 2](torch.cat((x, pos), dim=1), edge_index,
                                edge_attr),
          inplace=True)

    x = F.elu(
        self.conv_list[-1](torch.cat((x, pos), dim=1), edge_index, edge_attr),
        inplace=True)
    return x, pool_edge_list, pool_pos_list


class StructureEncoder(Encoder):

  def __init__(self, dim: int, init_data: Data, hidden_channels: int,
               latent_channels: int, k_size: int, n_pools: int,
               pool_ratio: float):
    super(StructureEncoder,
          self).__init__(dim, init_data, hidden_channels, latent_channels,
                         k_size, n_pools, pool_ratio)
    self.in_channels = self.dim + 1

    # initial aggr
    self.conv_list = [
        GATv2Conv(self.in_channels, hidden_channels).cuda(),
        GATv2Conv(hidden_channels, hidden_channels).cuda(),
    ]

    # pools
    self.pool_list = []
    for _ in range(n_pools):
      self.pool_list.append(
          SAGPoolWithPos(
              hidden_channels + dim,
              pool_ratio,
              hidden_channels=hidden_channels).cuda())
      self.conv_list.append(GATv2Conv(hidden_channels, hidden_channels).cuda())

    # latent dense map
    # out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)
    # self.conv_list.append(
    #     GATv2Conv(hidden_channels,latent_channels).cuda())

  def get_edge_attr(self, edge_index, pos):
    edge_attr = torch.zeros((edge_index.size(1), 3)).cuda()
    for i, xs in enumerate(edge_index.transpose(0, 1)):
      edge_attr[i] = pos[xs[1]] - pos[xs[0]]
    return edge_attr

  def forward(self, x, edge_index, pos):
    x = get_deg(x, edge_index)
    edge_attr = self.get_edge_attr(edge_index, pos)
    x = F.elu(
        self.conv_list[0](torch.cat((x, pos), dim=1), edge_index, edge_attr),
        inplace=True)
    x = F.elu(
        self.conv_list[1](torch.cat((x, pos), dim=1), edge_index, edge_attr),
        inplace=True)

    pool_edge_list = [edge_index]
    pool_pos_list = [pos]
    for l, pool in enumerate(self.pool_list):
      x, edge_index, pos, _, _, _, _ = pool(x, edge_index, pos)
      pool_edge_list.insert(0, edge_index)
      pool_pos_list.insert(0, pos)

      edge_attr = self.get_edge_attr(edge_index, pos)
      x = F.elu(
          self.conv_list[l + 2](torch.cat((x, pos), dim=1), edge_index,
                                edge_attr),
          inplace=True)
    return pool_edge_list, pool_pos_list


class Decoder(nn.Module):

  def __init__(self, dim: int, init_data: Data, hidden_channels: int,
               latent_channels: int, k_size: int, n_pools: int,
               pool_ratio: float):
    super(Decoder, self).__init__()
    self.dim = dim
    self.out_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.k_size = k_size
    self.n_pools = n_pools
    self.pool_ratio = pool_ratio

    # latent dense map
    # self.out_sz = get_pooled_sz(init_data.num_nodes, pool_ratio, n_pools)

    # initial aggr
    self.conv_list = [
        GATv2Conv(latent_channels + dim, hidden_channels).cuda(),
        GATv2Conv(hidden_channels + dim, hidden_channels).cuda(),
    ]

    # unpools
    for _ in range(n_pools):
      self.conv_list.append(
          GATv2Conv(hidden_channels + dim, hidden_channels).cuda())

    self.conv_list.append(
        GATv2Conv(hidden_channels + dim, self.out_channels).cuda())

  def get_edge_attr(self, edge_index, pos):
    edge_attr = torch.zeros((edge_index.size(1), 3)).cuda()
    for i, xs in enumerate(edge_index.transpose(0, 1)):
      edge_attr[i] = pos[xs[1]] - pos[xs[0]]
    return edge_attr

  def forward(self, latent, edge_index_list, pos_list):
    edge_index = edge_index_list[0]
    pos = pos_list[0]
    edge_attr = self.get_edge_attr(edge_index, pos)
    x = F.elu(
        self.conv_list[0](torch.cat((latent, pos), dim=1), edge_index,
                          edge_attr),
        inplace=True)
    x = F.elu(
        self.conv_list[1](torch.cat((x, pos), dim=1), edge_index, edge_attr),
        inplace=True)

    for l in range(self.n_pools):
      # deg = get_deg(x, edge_index)
      x = knn_interpolate(x, pos_list[l], pos_list[l + 1])
      edge_index = edge_index_list[l + 1]
      pos = pos_list[l + 1]
      edge_attr = self.get_edge_attr(edge_index, pos)

      x = F.elu(
          self.conv_list[l + 2](torch.cat((x, pos), dim=1), edge_index,
                                edge_attr),
          inplace=True)

    out = self.conv_list[-1](torch.cat((x, pos), dim=1), edge_index, edge_attr)
    return out


class DBA(nn.Module):

  def __init__(self, dim: int, init_data: PairData, hidden_channels: int,
               latent_channels: int, k_size: int, n_pools: int,
               pool_ratio: float):
    super(DBA, self).__init__()
    self.dim = dim
    self.in_channels = init_data.num_node_features
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.k_size = k_size
    self.n_pools = n_pools
    self.pool_ratio = pool_ratio

    # only used for getting pooling structure
    init_data_3 = Data(
        x=init_data.x_3, edge_index=init_data.edge_index_3, pos=init_data.pos_3)
    self.encoder3D = StructureEncoder(dim, init_data_3, hidden_channels,
                                      latent_channels, k_size, n_pools,
                                      pool_ratio).cuda()

    # used for model eval
    init_data_2 = Data(
        x=init_data.x_2, edge_index=init_data.edge_index_2, pos=init_data.pos_2)
    self.encoder2D = Encoder(dim, init_data_2, hidden_channels, latent_channels,
                             k_size, n_pools, pool_ratio).cuda()
    self.decoder = Decoder(dim, init_data_3, hidden_channels, latent_channels,
                           k_size, n_pools, pool_ratio).cuda()

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
    if pool_edge_list is None and pool_pos_list is None:
      pool_edge_list, pool_pos_list = self.encoder3D(x_3, edge_index_3, pos_3)
      ret = 1
    latent_2, _, pos_list_2 = self.encoder2D(x_2, edge_index_2, pos_2)

    pool_pos_list.pop(0)
    pool_pos_list.insert(0, pos_list_2[0])

    out = self.decoder(latent_2, pool_edge_list, pool_pos_list)

    if ret == 1:
      return out, pool_edge_list, pool_pos_list
    return out
