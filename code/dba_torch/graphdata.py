# import jax.experimental.sparse as jxs
# import jax.numpy as jnp

import os
from typing import List, Tuple, Union
import numpy as np
import torch
from torch.utils import data
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import ClusterData, ClusterLoader
import pyvista as pv
from vtk2ids import v2i, combine, connect
import torch

class PairData(Data):
  def __inc__(self, key, value, *args, **kwargs):
    if key == 'graph_2d':
        return self.graph_2d.x.size(0)
    if key == 'graph_3d':
        return self.graph_3d.x.size(0)
    return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(InMemoryDataset):

  def __init__(self,
               root,
               ma_list,
               re_list,
               aoa_list,
               n_slices: int = 5,
               transform=None):
    self.ma_list = ma_list
    self.re_list = re_list
    self.aoa_list = aoa_list
    self.n_slices = n_slices
    super().__init__(root, transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    raw_names = []
    # for ma in ma_list:
    #   for re in re_list:
    #     for aoa in aoa_list:
    #       raw_names.append("ma-{:g}_re-{:g}_aoa-{:g}.raw".format(ma,re,aoa))
    #       for slc in range(self.n_slices):
    #         raw_names.append("ma-{:g}_re-{:g}_aoa-{:g}_slc-{:d}.raw".format(ma,re,aoa,slc))
    return raw_names

  @property
  def processed_file_names(self):
    return ["data.pt"]

  def process(self):
    graph_list = []
    for ma in ma_list:
      for re in re_list:
        for a in aoa_list:
          data_path = os.path.join(self.root,
                                   "ma_{:g}/re_{:g}/a_{:g}".format(ma, re, a))

          mesh = pv.read(os.path.join(data_path, "flow.vtu"))
          # extract point data from coordinates and conservative fields
          coords = np.array(mesh.points)
          features_3 = np.array([
              mesh.point_data.get_array(i)
              for i in ["Density", "Momentum", "Energy"]
          ])

          idx, _, _ = v2i(mesh)

          graph_3 = Data(features_3, idx, coords)

          slice_idx = []
          slice_data = []
          slice_szs = []
          slice_coords = []
          for s in range(self.n_slices):
            mesh = pv.read(os.path.join(data_path, "slice_{:d}.vtk".format(s)))
            slice_coords.append(np.array(mesh.points))

            idx, _, sz = v2i(mesh)

            slice_data.append([
                mesh.point_data.get_array(i)
                for i in ["Density", "Momentum", "Energy"]
            ])

            slice_idx.append(idx)

            slice_szs.append(sz)
          idx, _, sz = combine(slice_idx,
                               [np.ones((i.shape[-1],)) for i in slice_idx],
                               slice_szs)
          features_2 = np.array(slice_data)
          coords = np.array(slice_coords)

          graph_2 = Data(features_2, idx, coords)
          
          data = PairData(graph_3d=graph_3, graph_2d=graph_2)
          data_list.append(data)

    data, slices = self.collate(data_list)

    torch.save((data, slices), self.processed_paths[0])

  def len(self):
    return len(self.processed_file_names)

  def get(self, idx):
    ma = self.ma_list[idx // (len(self.re_list)*len(self.aoa_list))]
    re = self.re_list[(idx // len(self.aoa_list)) % len(self.re_list)]
    a = self.aoa_list[idx % len(self.aoa_list)]
    graph_3 = torch.load(
        os.path.join(self.processed_dir, "ma-{:g}_re-{:g}_a-{:g}_3d.pt"))
    graph_2 = torch.load(
        os.path.join(self.processed_dir, "ma-{:g}_re-{:g}_a-{:g}_2d.pt"))
    return graph_3, graph_2


if __name__ == "__main__":
  ma_list = [0.2, 0.35, 0.5, 0.65, 0.8]
  re_list = [1e5, 1e6, 1e7, 1e8]
  aoa_list = [0, 2, 4, 6, 8, 10, 12]
  n_slices = 5
  data_path = "data"

  train_dataset = GraphDataset(data_path, ma_list, re_list, aoa_list, n_slices)

  n_samples = len(ma_list)*len(re_list)*len(aoa_list)
  batch_sz = 1
  batches = -(n_samples // -batch_sz)

  train_dataloader = data.DataLoader(train_dataset, batch_sz, shuffle=True)

  print(next(iter(train_dataloader)))
  breakpoint()