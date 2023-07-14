import os
import sys
from datetime import date
import shutil
from functools import partial
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--case-name", default="dba", type=str, help="Architecture Name")
parser.add_argument(
    "--channels", default=10, type=int, help="Aggregation Channels")
parser.add_argument(
    "--latent-sz", default=50, type=int, help="Latent Space Dimensionality")
parser.add_argument(
    "--pooling-layers", default=1, type=int, help="Number of Pooling Layers")
parser.add_argument(
    "--lambda-2d", default=0.01, type=float, help="2D Loss Weight")
parser.add_argument(
    "--lambda-dp", default=1, type=float, help="DiffPool Loss Weight")
parser.add_argument("--wandb", default=0, type=int, help="wandb upload")
parser.add_argument('--gpu-id', default=0, type=int, help="GPU index")

args = parser.parse_args()
wandb_upload = bool(args.wandb)
today = date.today()
case_name = "_".join([
    str(key) + "-" + str(value) for key, value in list(vars(args).items())[:-1]
])[10:]
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

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
from torch.utils.data import DataLoader
import orbax.checkpoint as orb

from models import GraphEncoder, GraphDecoder, GraphEncoderNoPooling, GraphDecoderNoPooling
from graphdata import GraphDataset, GraphLoader
from vtk2adj import v2a, combineAdjacency

if not wandb_upload:
  print('Load')

# loop through folders and load data
# ma_list = [0.2, 0.35, 0.5, 0.65, 0.8]
ma_list = [0.2, 0.35, 0.5] if wandb_upload else [0.2]
# ma_list = [0.2]
# re_list = [1e6, 2e6, 5e6, 1e7, 2e7]
re_list = [1e6, 2e6, 1e7, 2e7] if wandb_upload else [2e6]
# re_list = [1e6]
# aoa_list = [0, 3, 6, 9, 12]
aoa_list = [3, 6, 9, 12] if wandb_upload else [3, 6]
# aoa_list = [3]
n_slices = 5
data_path = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data")

train_dataset = GraphDataset(data_path, ma_list, re_list, aoa_list, n_slices)
test_dataset = GraphDataset(data_path, ma_list, [5e6], aoa_list, n_slices)

n_samples = len(ma_list)*len(re_list)*len(aoa_list)
batch_sz = jnp.min(jnp.array([10, n_samples])).astype(int)
batches = -(n_samples // -batch_sz)
n_test = len(ma_list)*1*len(aoa_list)
test_sz = jnp.min(jnp.array([10, n_test])).astype(int)
test_batches = -(n_test // -test_sz)

train_dataloader = GraphLoader(train_dataset, n_samples, shuffle=True)
test_dataloader = GraphLoader(test_dataset, n_test, shuffle=True)

rng = jrn.PRNGKey(1)
n_pools = args.pooling_layers

mesh = pv.read(
    os.path.join(
        data_path, "ma_{:g}/re_{:g}/a_{:g}".format(ma_list[0], re_list[0],
                                                   aoa_list[0]), "flow.vtu"))
adj_3 = v2a(mesh)

n_slices = 5
slice_adj = []
for s in range(n_slices):
  mesh = pv.read(
      os.path.join(
          data_path, "ma_{:g}/re_{:g}/a_{:g}".format(ma_list[0], re_list[0],
                                                     aoa_list[0]),
          "slice_{:d}.vtk".format(s)))
  slice_adj.append(v2a(mesh))
adj_2 = combineAdjacency(slice_adj)

train_data_3, train_data_2 = next(iter(train_dataloader))
test_data_3, test_data_2 = next(iter(test_dataloader))
init_data_3 = test_data_3[0]
init_data_2 = test_data_2[0]

if not wandb_upload:
  print('Init')

# ge_3 = GraphEncoder(n_pools, args.latent_sz, args.channels, dim=3)
# ge_2 = GraphEncoder(n_pools, args.latent_sz, args.channels, dim=2)
ge_3 = GraphEncoderNoPooling(n_pools, args.latent_sz, args.channels, dim=3)
ge_2 = GraphEncoderNoPooling(
    n_pools, args.latent_sz, args.channels, dim=3)  # slices have 3d coords
final_sz = init_data_3.shape[-1] - 3
# gd = GraphDecoder(n_pools, final_sz, args.channels, dim=3)
gd = GraphDecoderNoPooling(n_pools, final_sz, args.channels, dim=3)

pe_3 = ge_3.init(rng, init_data_3, adj_3)['params']
pe_2 = ge_2.init(rng, init_data_2, adj_2)['params']
f_latent, a, c, s = ge_3.apply({'params': pe_3}, init_data_3, adj_3)
pd = gd.init(rng, f_latent, a, c, s)['params']
params = [pe_3, pe_2, pd]

check_path = os.path.join(data_path, "models_save", case_name,
                          today.strftime("%d%m%y"))
if os.path.exists(check_path):
  shutil.rmtree(check_path)
options = orb.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
ckptr = orb.CheckpointManager(
    check_path, {
        "params": orb.PyTreeCheckpointer(),
        "state": orb.PyTreeCheckpointer()
    },
    options=options)

# ckptr.save(check_path, params)

tx = optax.adam(1e-3)

n_epochs = 50000

eps = 1e-15

# @jit
# def get_acs(params, features_3, adjacency_3):
#   a_list = []
#   c_list = []
#   s_list = []
#   for fb3 in jnp.concatenate(features_3):
#     _, a, c, s = ge_3.apply({'params': params[0]}, fb3, adjacency_3)
#     if a_list == []:
#       a_list = jtr.tree_map(lambda a: a / batches / batch_sz, a)
#       c_list = jtr.tree_map(lambda c: c / batches / batch_sz, c)
#       s_list = jtr.tree_map(lambda s: s / batches / batch_sz, s)
#     else:
#       a_list = jtr.tree_map(lambda a, a_new: a + a_new/batches/batch_sz,
#                             a_list, a)
#       c_list = jtr.tree_map(lambda c, c_new: c + c_new/batches/batch_sz,
#                             c_list, c)
#       s_list = jtr.tree_map(lambda s, s_new: s + s_new/batches/batch_sz,
#                             s_list, s)
#   return a, c, s


@jit
def train_step(params, opt: optax.OptState, lam_2, lam_dp, data_3, data_2):
  loss = 0

  def loss_fn(params, data_3, data_2):
    loss = 0
    fl3, a, c, s = ge_3.apply({'params': params[0]}, data_3, adj_3)
    fl2, _, _, _ = ge_2.apply({'params': params[1]}, data_2, adj_2)

    # # 3->3 isprimary ae loss
    # f = gd.apply({'params': params[2]}, fl3, a, c, s)
    # loss_ae = jnp.mean(jnp.square(f[:, 3:] - data_3[:, 3:]))
    # loss_2 = jnp.mean(jnp.square(fl2 - fl3))

    # 2->3 isprimary ae loss
    f = gd.apply({'params': params[2]}, fl2, a, c, s)
    loss_ae = jnp.mean(jnp.square(f[:, 3:] - data_3[:, 3:]))
    loss_2 = jnp.mean(jnp.square(fl2 - fl3))

    # # WITH POOLING
    # loss_lp = jnp.mean(
    #     jnp.array(
    #         jtr.tree_map(
    #             lambda a, s: jnp.sqrt(jnp.sum(jnp.square(a - s @ s.T))),
    #             a[:-1], s)))
    # loss_e = jnp.mean(
    #     jnp.array(
    #         jtr.tree_map(
    #             lambda s: jnp.mean(jnp.sum(-s*jnp.exp(s + eps), axis=-1)),
    #             s)))

    # NO POOLING:
    loss_lp = 0
    loss_e = 0

    loss = loss + (loss_ae + lam_2*loss_2 + lam_dp*(loss_e+loss_lp)) / batch_sz
    return loss

  data_3_batched = vmap(
      dsd, in_axes=(None, 0, None))(data_3, jnp.arange(batches)*batch_sz,
                                    batch_sz)
  data_2_batched = vmap(
      dsd, in_axes=(None, 0, None))(data_2, jnp.arange(batches)*batch_sz,
                                    batch_sz)
  for (batch_3, batch_2) in zip(data_3_batched, data_2_batched):
    sample_loss = vmap(loss_fn, in_axes=(None, 0, 0))(params, batch_3, batch_2)
    batch_loss = jnp.mean(sample_loss, axis=0)
    loss = loss + batch_loss/batches

    grads = vmap(grad(loss_fn), in_axes=(None, 0, 0))(params, batch_3, batch_2)
    grads = jtr.tree_map(lambda g: jnp.mean(g, axis=0), grads)
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

  _, a, c, s = ge_3.apply({'params': params[0]}, data_3[0], adj_3)

  return loss, params, opt, a, c, s


@jit
def test_step(params, adj_list, coordinates, selection, data_3, data_2):
  test_err = 0

  def loss_fn(params, adj_list, coordinates, selection, data_3, data_2):
    fl2, _, _, _ = ge_2.apply({'params': params[1]}, data_2, adj_2)
    f = gd.apply({'params': params[2]}, fl2, adj_list, coordinates, selection)
    loss_ae = jnp.mean(jnp.square(f[:, 3:] - data_3[:, 3:]))
    return loss_ae

  data_3_batched = vmap(
      dsd, in_axes=(None, 0, None))(data_3, jnp.arange(test_batches)*test_sz,
                                    test_sz)
  data_2_batched = vmap(
      dsd, in_axes=(None, 0, None))(data_2, jnp.arange(test_batches)*test_sz,
                                    test_sz)

  for (batch_3, batch_2) in zip(data_3_batched, data_2_batched):
    sample_loss = vmap(
        loss_fn,
        in_axes=(None, None, None, None, 0, 0))(params, adj_list, coordinates,
                                                selection, batch_3, batch_2)
    batch_loss = jnp.mean(sample_loss, axis=0)
    test_err = test_err + batch_loss/test_batches
  return test_err


# def getBatchIndices(indices, i):
#   batch_indices = dsd(indices, i, batch_sz)
#   return indices, batch_indices

# # shuffle batches
# batch_indices = scan(getBatchIndices,
#                      jrn.shuffle(jrn.PRNGKey(epoch), indices),
#                      jnp.arange(batches))


def main(params, n_epochs):
  opt = tx.init(params)
  # indices = train_dataset.items
  min_err = jnp.inf
  for epoch in range(n_epochs):
    if not wandb_upload:
      print("Train")
    loss, params, opt, a, c, s = train_step(params, opt, args.lambda_2d,
                                            args.lambda_dp, train_data_3,
                                            train_data_2)

    a = [a.sum_duplicates() for a in a]
    s = [s.sum_duplicates() for s in s]
    if not wandb_upload:
      print("Test")
    test_err = test_step(params, a, c, s, test_data_3, test_data_2)
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
      if test_err < min_err or epoch == n_epochs - 1:
        min_err = test_err
        ckptr.save(epoch, {
            "params": params,
            "state": {
                "Loss": loss,
                "Error": test_err
            }
        })


if __name__ == "__main__":
  if wandb_upload:
    import wandb
    wandb.init(project="DB-GNN", entity="wglao", name=case_name)
  main(params, n_epochs)