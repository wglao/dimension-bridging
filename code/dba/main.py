import os
import sys
from functools import partial
import argparse

import flax.core.frozen_dict as fd
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrn
import jax.tree_util as jtr
import optax
from jax import grad, jit, value_and_grad, vmap
from jax.lax import dynamic_slice_in_dim as dsd
from jax.lax import scan
import jax.experimental.sparse as jxs
import pyvista as pv

from models import GraphDecoder, GraphEncoder
from vtk2adj import v2a

parser = argparse.ArgumentParser()

parser.add_argument("--channels", "-c", default=100, type=int)
parser.add_argument("--latent-sz", "-s", default=50, type=int)
parser.add_argument("--pooling-layers", "-p", default=1, type=int)
parser.add_argument("--lambda-2d", "-l2d", default=1, type=float)
parser.add_argument("--lambda-dp", "-ldp", default=1, type=float)
parser.add_argument("--wandb", "-w", default=0, type=int)

args = parser.parse_args()
wandb_upload = bool(args.wandb)

# loop through folders and load data
ma_list = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1]
re_list = [1e5, 1e6, 1e7, 1e8]
a_list = [0, 2, 4, 6, 8, 10, 12]
n_slices = 5

train_data_3 = []
train_data_2 = []
train_adj_3 = []
train_adj_2 = []

for ma in ma_list:
  for re in re_list:
    for a in a_list:
      path = "/scratch1/07169/wgl/ORNL/2d3d/data/ma_{:g}/re_{:g}/a_{:g}".format(
          ma, re, a)
      mesh_3 = pv.read(os.path.join(path, "flow.vtu"))

      # extract point data from coordinates and conservative fields
      coords_3 = jnp.array(mesh_3.points)
      train_data_3.append(
          jnp.column_stack([
              coords_3,
              [mesh_3.point_data.get_array(i) for i in range(mesh_3.n_arrays)]
          ]))
      train_adj_3.append(v2a(mesh_3))

      slice_data = []
      slice_adj = []
      for i in range(n_slices):
        mesh_2 = pv.read(os.path.join(path, "slice_{:g}.vtk".format(i)))
        coords_2 = jnp.array(mesh_2.points)
        slice_data.append(
            jnp.column_stack([
                coords_2,
                [
                    mesh_2.point_data.get_array(i)
                    for i in range(mesh_2.n_arrays)
                ]
            ]))
        slice_adj.append(v2a(mesh_2))
      train_data_2.append(slice_data)
      train_adj_2.append(slice_adj)

del mesh_3
del mesh_2

# train_data_3 = jnp.array(train_data_3)    #[Batches, Nodes, Fields]
# train_data_2 = jnp.array(train_data_2)    #[Batches, Slices, Nodes, Fields]

test_path = "/scratch1/07169/wgl/ORNL/2d3d/data/ma_{:g}/re_{:g}/a_{:g}".format(
    0.8395, 1.172e7, 3.06)
test_mesh_3 = pv.read(os.path.join(test_path, "flow.vtu"))
test_data_3 = jnp.column_stack([[
    test_mesh_3.point_data.get_array(i) for i in range(test_mesh_3.n_arrays)
]])
# test_adj_3 = v2a(test_mesh_3)

test_data_2 = []
test_adj_2 = []
for i in range(n_slices):
  test_mesh_2 = pv.read(os.path.join(test_path, "slice_{:g}.vtk".format(i)))
  coords_2 = jnp.array(test_mesh_2.points)
  test_data_2.append(
      jnp.column_stack([
          coords_2,
          [
              test_mesh_2.point_data.get_array(i)
              for i in range(test_mesh_2.n_arrays)
          ]
      ]))
  test_adj_2.append(v2a(test_mesh_2))

del test_mesh_3
del test_mesh_2

n_samples = len(ma_list)*len(re_list)*len(a_list)
batch_sz = 10
batches = -(n_samples // -batch_sz)
test_sz = 1

rng = jrn.PRNGKey(1)

n_pools = 1
ge_3 = GraphEncoder(n_pools, dim=3)
ge_2 = GraphEncoder(n_pools, dim=2)
gd = GraphDecoder(n_pools, dim=3)

pe_3 = ge_3.init(rng, train_data_3[0], train_adj_3[0])['params']
pe_2 = ge_2.init(rng, train_data_2[0], train_adj_2[0])['params']
f_latent, a, c, s = ge_3.apply({'params': pe_3}, train_data_3[0],
                               train_adj_3[0])
pd = gd.init(rng, f_latent, a, c, s)['params']
params = [pe_3, pe_2, pd]
tx = optax.adam(1e-3)

n_epochs = 100000

eps = 1e-15


@jit
def train_step(params,
               features_3,
               features_2,
               adjacency_3,
               adjacency_2,
               opt: optax.OptState,
               lam_2: float = 1,
               lam_dp: float = 1):

  def loss_fn(params, features_3, features_2, adjacency_3, adjacency_2):
    loss = 0
    for fb3, fb2 in zip(features_3, features_2):
      fl3, a, c, s = ge_3.apply({'params': params[0]}, fb3, adjacency_3)
      fl2, _, _, _ = ge_2.apply({'params': params[1]}, fb2, adjacency_2)
      f = gd.apply({'params': params[2]}, fl3, a, c, s)
      loss_ae = jnp.mean(jnp.square(f[:, 3:] - fb3[:, 3:]))
      loss_2 = jnp.mean(jnp.square(fl2 - fl3))
      loss_lp = jnp.mean(
          jnp.array(
              jtr.tree_map(
                  lambda a, s: jnp.sqrt(jnp.sum(jnp.square(a - s @ s.T))),
                  a[:-1], s)))
      loss_e = jnp.mean(
          jnp.array(
              jtr.tree_map(
                  lambda s: jnp.mean(jnp.sum(-s*jnp.exp(s + eps), axis=-1)),
                  s)))
      loss = loss + (loss_ae + lam_2*loss_2 + lam_dp*
                     (loss_e+loss_lp)) / batch_sz
    return loss

  def get_acs(params, features_3):
    a_list = []
    c_list = []
    s_list = []
    for fb3 in jnp.concatenate(features_3):
      _, a, c, s = ge_3.apply({'params': params[0]}, fb3, adjacency_3)
      if a_list == []:
        a_list = jtr.tree_map(lambda a: a / batches / batch_sz, a)
        c_list = jtr.tree_map(lambda c: c / batches / batch_sz, c)
        s_list = jtr.tree_map(lambda s: s / batches / batch_sz, s)
      else:
        a_list = jtr.tree_map(lambda a, a_new: a + a_new/batches/batch_sz,
                              a_list, a)
        c_list = jtr.tree_map(lambda c, c_new: c + c_new/batches/batch_sz,
                              c_list, c)
        s_list = jtr.tree_map(lambda s, s_new: s + s_new/batches/batch_sz,
                              s_list, s)
    return a, c, s

  loss = 0
  for batch_3, batch_2 in zip(features_3, features_2):
    batch_loss, grads = value_and_grad(loss_fn)(params, batch_3, batch_2,
                                                adjacency_3, adjacency_2)
    # grads = grad(loss_fn)(params, features, adjacency)
    updates, opt = tx.update(grads, opt, params)
    params = optax.apply_updates(params, updates)

    # ensure covariances are always positive semi-definite
    for i in range(len(params)):
      p = fd.unfreeze(params[i])
      for layer in params[i].keys():
        if 'MoNetLayer' in layer:
          tmp = p[layer]['sigma']
          tmp = jnp.where(tmp > eps, tmp, eps)
          p[layer]['sigma'] = tmp
        else:
          for sublayer in params[i][layer].keys():
            if 'MoNetLayer' in sublayer:
              tmp = p[layer][sublayer]['sigma']
              tmp = jnp.where(tmp > eps, tmp, eps)
              p[layer][sublayer]['sigma'] = tmp
      params[i] = fd.freeze(p)

    loss = loss + batch_loss/batches
  a, c, s = get_acs(params, features_3)
  return loss, params, opt, a, c, s


@jit
def test_step(params, features_3, features_2, adjacency_2, adjacency_list,
              coordinates, selection):

  def loss_fn(params, features_3, features_2, adjacency_2, adjacency_list,
              coordinates, selection):
    loss = 0
    for fb3, fb2 in zip(features_3, features_2):
      fl2, _, _, _ = ge_2.apply({'params': params[1]}, fb2, adjacency_2)
      f = gd.apply({'params': params[2]}, fl2, adjacency_list, coordinates,
                   selection)
      loss_ae = jnp.mean(jnp.square(f[:, 3:] - fb3[:, 3:]))
      loss = loss + loss_ae/test_sz
    return loss

  test_err = loss_fn(params, features_3, features_2, adjacency_2,
                     adjacency_list, coordinates, selection)
  return test_err


def getBatchIndices(indices, i):
  batch_indices = dsd(indices, i, batch_sz)
  return indices, batch_indices


def main(params, n_epochs):
  opt = tx.init(params)
  indices = jnp.arange(len(train_data_3))
  for epoch in range(n_epochs):
    # shuffle batches
    batch_indices = scan(getBatchIndices,
                         jrn.shuffle(jrn.PRNGKey(epoch), indices),
                         jnp.arange(batches))
    loss, params, opt, a, c, s = train_step(
        params, [train_data_3[i] for i in batch_indices],
        [train_data_2[i] for i in batch_indices],
        [train_adj_3[i] for i in batch_indices],
        [train_adj_2[i] for i in batch_indices], opt, args.lambda_2d,
        args.lambda_dp)
    test_err = test_step(params, test_data_3, test_data_2, test_adj_2, a, c, s)
    if epoch % 100 == 0 or epoch == n_epochs - 1:
      if wandb_upload:
        wandb.log({
            "Loss": loss,
            "Error": test_err,
            "Epoch": epoch,
        })
      else:
        print("Loss: {:g}, Error {:g}, Epoch {:g}".format(
            loss, test_err, epoch))


if __name__ == "__main__":
  if wandb_upload:
    import wandb
    wandb.init(project="DB-GNN", entity="wglao", name="graph autoencoder")
  main(params, n_epochs)