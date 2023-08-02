# import jax.numpy as jnp
# import jax.scipy as jsp
# from jax import jit, vmap
# from jax.lax import scan
# from jax.lax import dynamic_slice_in_dim as dsd
# from jax.lax import dynamic_index_in_dim as did
# import jax.experimental.sparse as jxs

import numpy as np
import pandas as pd
import pyvista as pv
import torch
from torch_geometric.data import Data


def connect(edge):
  indices = np.row_stack((edge, np.flip(edge)))
  return indices


def v2i(mesh):
  n_nodes = len(mesh.points)
  edges = mesh.extract_all_edges().lines
  n_edges = len(edges) // 3
  edges = np.reshape(edges, (n_edges, 3))[:, 1:]

  indices = np.array([connect(edge) for edge in edges])
  indices = np.concatenate(indices, axis=-1)
  main_diag = np.row_stack((np.arange(n_nodes), np.arange(n_nodes)))
  indices = np.concatenate((indices, main_diag), axis=-1)

  data = np.ones((indices.shape[0],))

  return indices, data, n_nodes


def combine(idx_list, dat_list, n_list):
  in_szs = np.array([n for n in n_list])
  out_sz = int(np.sum(in_szs))
  
  buffer = np.concatenate((np.array([0]), np.cumsum(in_szs)[:-1]), axis=None)
  indices = np.concatenate([a + b for a, b in zip(idx_list, buffer)],
                           axis=-1)
  
  data = np.concatenate(dat_list, axis=None)
  
  return indices, data, out_sz


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--file", "-f", default="slice_0.vtk", type=str)
  args = parser.parse_args()

  adj_list = []
  for s in range(5):
    fname = "data/ma_0.2/re_1e+08/a_0/slice_{:d}.vtk".format(s)
    mesh = pv.read(fname)
    adj_list.append(v2a(mesh))
  idx, data = combine(adj_list)