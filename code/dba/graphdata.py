import os
import numpy as np
from torch.utils import data
import pyvista as pv
from vtk2adj import v2a, combineAdjacency
import jax.numpy as jnp
import jax.experimental.sparse as jxs

# def _jaxTyoeCheck(item):
#   import pdb; pdb.set_trace()
#   if isinstance(batch[0], jnp.ndarray):
#     return jnp.stack(batch)
#   if isinstance(batch[0], jxs.BCOO):
#     return batch
#   if isinstance(batch[0], (tuple, list)):
#     transposed = zip(*batch)
#     return [jxsSpCollate(samples) for samples in transposed]
#   return batch


def jxsSpCollate(batch):
  if isinstance(batch[0], jnp.ndarray):
    return jnp.stack(batch)
  if isinstance(batch[0], np.ndarray):
    return jnp.stack(batch)
  if isinstance(batch[0], jxs.BCOO):
    return batch
  if isinstance(batch[0], jxs.BCSR):
    return batch
  if isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [jxsSpCollate(samples) for samples in transposed]
  return batch


class SpLoader(data.DataLoader):

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
        collate_fn=jxsSpCollate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)


class GraphDataset(data.Dataset):

  def __init__(self, data_dir, ma_list, re_list, aoa_list, n_slices: int = 5):
    self.data_dir = data_dir
    self.ma_list = ma_list
    self.re_list = re_list
    self.aoa_list = aoa_list
    self.n_slices = n_slices
    self.items = len(ma_list)*len(re_list)*len(aoa_list)

  def __len__(self):
    return self.items

  def __getitem__(self, index):
    ma = self.ma_list[index // (len(self.re_list)*len(self.aoa_list))]
    re = self.re_list[(index // len(self.aoa_list)) % len(self.re_list)]
    a = self.aoa_list[index % len(self.aoa_list)]
    data_path = os.path.join(self.data_dir,
                             "ma_{:g}/re_{:g}/a_{:g}".format(ma, re, a))

    mesh = pv.read(os.path.join(data_path, "flow.vtu"))
    # extract point data from coordinates and conservative fields
    coords = np.array(mesh.points)
    train_data_3 = np.column_stack([coords] + [
        mesh.point_data.get_array(i)
        # for i in ["Density", "Momentum", "Energy"]
        for i in ["Density"]  # Density only for Memory
    ])
    # [mesh.point_data.get_array(i) for i in range(mesh.n_arrays)]))
    train_adj_3 = v2a(mesh)

    slice_data = []
    slice_adj = []
    for s in range(self.n_slices):
      mesh = pv.read(os.path.join(data_path, "slice_{:d}.vtk".format(s)))
      coords = np.array(mesh.points)
      slice_data.append(
          np.column_stack([coords] + [
              mesh.point_data.get_array(i)
              # for i in ["Density", "Momentum", "Energy"]
              for i in ["Density"]  # Density only for Memory
          ]))
      slice_adj.append(v2a(mesh))
    train_data_2 = np.concatenate(slice_data, axis=0)
    train_adj_2 = combineAdjacency(slice_adj)
    return train_data_3, train_data_2, train_adj_3, train_adj_2


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