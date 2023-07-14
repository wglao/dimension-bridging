import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from jax.lax import scan
from jax.lax import dynamic_slice_in_dim as dsd
from jax.lax import dynamic_index_in_dim as did
import numpy as np
import pandas as pd
import jax.experimental.sparse as jxs
import pyvista as pv


def connect(edge):
  indices = jnp.meshgrid(edge, edge)
  indices = jnp.column_stack((indices[0].ravel(), indices[1].ravel()))
  return indices


def v2a(mesh):
  n_nodes = len(mesh.points)
  edges = mesh.extract_all_edges().lines
  n_edges = len(edges) // 3
  edges = jnp.reshape(edges, (n_edges, 3))[:, 1:]

  indices = vmap(connect)(edges)
  indices = jnp.concatenate(indices, axis=0)
  main_diag = jnp.column_stack((jnp.arange(n_nodes), jnp.arange(n_nodes)))
  indices = jnp.concatenate((indices, main_diag), axis=0)

  data = jnp.ones((indices.shape[0],))

  adjacency = jxs.BCOO((data, indices), shape=(n_nodes, n_nodes))
  return adjacency.sum_duplicates()


def combineAdjacency(adjs):
  in_szs = jnp.array([a.shape[0] for a in adjs])
  out_sz = int(jnp.sum(in_szs))
  buffer = jnp.concatenate((jnp.array([0]), jnp.cumsum(in_szs)[:-1]), axis=None)
  indices = jnp.concatenate([a.indices + b for a, b in zip(adjs, buffer)],
                            axis=0)
  data = jnp.concatenate([a.data for a in adjs], axis=None)
  adjacency = jxs.BCOO((data, indices),
                       shape=(int(out_sz), int(out_sz)))
  return adjacency.sum_duplicates()


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
  a = combineAdjacency(adj_list)