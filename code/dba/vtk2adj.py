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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", "-f", default="slice_0.vtk", type=str)
args = parser.parse_args()


def connect(carry, edge):
  indices = jnp.meshgrid(edge, edge)
  indices = jnp.column_stack((indices[0].ravel(), indices[1].ravel()))
  return None, indices


def v2a(mesh):
  n_nodes = len(mesh.points)
  edges = mesh.extract_all_edges().lines
  n_edges = len(edges) // 3
  edges = jnp.reshape(edges, (n_edges, 3))[:, 1:]

  adjacency = jxs.eye(n_nodes)
  _, indices = scan(connect, None, edges)
  indices = jnp.concatenate(indices, axis=0)
  adjacency.indices = jnp.concatenate((adjacency.indices, indices), axis=0)

  z_pad = jnp.ones((len(adjacency.indices) - len(adjacency.data),))
  adjacency.data = jnp.concatenate((adjacency.data, z_pad), axis=None)
  adjacency = adjacency.sum_duplicates()
  adjacency.data = jnp.ones_like(adjacency.data)

  return adjacency


def combineAdjacency(*adjs):
  in_szs = jnp.array([a.shape[0] for a in adjs])
  out_sz = jnp.sum(in_szs)
  adjacency = jxs.eye(out_sz)

  buffer = jnp.cumsum(in_szs)
  adjacency.indices = jnp.concatenate(
      [a.indices + jnp.array([b, b]) for a, b in zip(adjs, buffer)], axis=None)
  adjacency.data = jnp.concatenate([jnp.zeros_like(adjacency.data)]+[a.data for a in adjs], axis=None)
  adjacency = adjacency.sum_duplicates()
  # adjacency.data = jnp.ones((adjacency.indices.shape[0],))

  return adjacency


if __name__ == "__main__":
  adj_list = []
  for s in range(5):
    fname = "data/ma_0.2/re_1e+08/a_0/slice_{:d}.vtk".format(s)
    mesh = pv.read(fname)
    adj_list.append(v2a(mesh))
  a = combineAdjacency(*adj_list)