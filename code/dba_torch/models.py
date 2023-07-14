import torch
from torch import nn
from graphdata import GraphDataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import SplineConv, SAGPooling, knn_interpolate
from torch_geometric.nn import Sequential as GeoSequential
from torch.nn import Linear, ReLU, Sequential
# import torch.nn.functional as F
from graphdata import PairData
import numpy as np


def get_pooled_sz(full_sz: int, ratio: float, layer: int):
  out_sz = full_sz
  for l in range(layer):
    out_sz = -(out_sz // -(1 / ratio))
  return out_sz


class Encoder3D(nn.Module):

  def __init__(self, dim: int, init_data: Data, hidden_channels: int,
               latent_channels: int, n_pools: int, pool_ratio: float):
    super.__init__()
    self.dim = dim
    self.in_channels = init_data.x.shape[-1]
    self.hidden_channels = hidden_channels
    self.latent_channels = latent_channels
    self.n_pools = n_pools
    self.pool_ratio = pool_ratio

    # set kernel size to mean of node degree vector
    idx = init_data.coo()
    sz = init_data.num_nodes
    adj = torch.sparse_coo_tensor(idx, np.ones((idx.shape[-1],)), (sz, sz))
    deg = adj.matmul(torch.tensor(np.ones((sz, 1))))
    k_sz = torch.mean(deg)

    # initial aggr
    self.module_list = [
        (SplineConv(self.in_channels, hidden_channels, dim, k_sz,
                    aggr='add'), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (SplineConv(hidden_channels, hidden_channels, dim, k_sz,
                    aggr='add'), 'x, edge_index -> x'),
        ReLU(inplace=True)
    ]

    # pools
    for _ in range(n_pools):
      self.module_list.append(
          (SAGPooling(hidden_channels,
                      pool_ratio), 'x, edge_index -> x, edge_index'))
      self.module_list.append(
          (SplineConv(hidden_channels, hidden_channels, dim, k_sz,
                      aggr='add'), 'x, edge_index -> x'))
      self.module_list.append(ReLU(inplace=True))

    # latent dense map
    out_sz = get_pooled_sz(sz, pool_ratio, n_pools)
    self.module_list.append((SplineConv(out_sz, out_sz, dim, k_sz,
                                        aggr='add'), 'x, edge_index -> x'))
    self.module_list.append(ReLU(inplace=True))

    self.module_list_dense = [(Linear(out_sz*hidden_channels, latent_channels)),
                              (ReLU(inplace=True)),
                              (Linear(latent_channels, latent_channels))]

    self.sequential = GeoSequential('x, edge_index, batch', self.module_list)
    self.sequential_dense = Sequential(self.module_list_dense)
  

  def forward(self, data):
    data = self.sequential(data)
    latent = self.sequential_dense(data.x)
    return latent


class Encoder2D(nn.Module):
  pass


class Decoder3D(nn.Module):
  pass