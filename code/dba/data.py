import os
import numpy as np
from torch.utils import data
import pyvista as pv
from vtk2adj import v2a
import jax.numpy as jnp


def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  if isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  return np.array(batch)


class NumpyLoader(data.DataLoader):

  def __init__(self,
               dataset,
               batch_size=1,
               shuffle=False,
               sampler=None,
               batch_sampler=None,
               num_workers=0,
               pin_memory=False,
               drop_last=False,
               timeout=0,
               worker_init_fn=None):
    super(self.__class__, self).__init__(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


class GraphDataset(data.Dataset):

  def __init__(self, data_dir, ma_list, re_list, aoa_list, n_slices:int = 5):
    self.data_dir = data_dir
    self.ma_list = ma_list
    self.re_list = re_list
    self.aoa_list = aoa_list
    self.n_slices = n_slices
    self.items = 0

  def __len__(self):
    return self.items

  def __getitem__(self, index):
    ma = self.ma_list[index // (len(self.re_list)*len(self.aoa_list))]
    re = self.re_list[(index // len(self.aoa)) % len(self.re_list)]
    a = self.aoa_list[index % len(self.aoa_list)]
    data_path = os.path.join(self.data_dir,
                             "ma_{:g}/re_{:g}/a_{:g}".format(ma, re, a))

    mesh = pv.read(os.path.join(data_path, "flow.vtu"))
    # extract point data from coordinates and conservative fields
    coords = jnp.array(mesh.points)
    train_data = jnp.column_stack([coords] + [
        mesh.point_data.get_array(i)
        for i in ["Density", "Momentum", "Energy"]
    ])
    # [mesh.point_data.get_array(i) for i in range(mesh.n_arrays)]))
    train_adj_3 = v2a(mesh)

    for s in range(self.n_slices):
      mesh = pv.read(os.path.join(data_path, "slice_{:d}.vtk".format(s)))
      