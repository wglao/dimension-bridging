import os
import sys
from functools import partial
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--case-name", default="dba", type=str, help="Architecture Name")
parser.add_argument(
    "--channels", default=10, type=int, help="Aggregation Channels")
parser.add_argument(
    "--latent-sz", default=10, type=int, help="Latent Space Dimensionality")
parser.add_argument(
    "--pooling-layers", default=1, type=int, help="Number of Pooling Layers")
parser.add_argument(
    "--lambda-0", default=0.0005, type=float, help="Feature Smoothness Weight")
parser.add_argument(
    "--lambda-1", default=0.1, type=float, help="Log Barrier Weight")
parser.add_argument(
    "--lambda-2", default=0.01, type=float, help="Adjacency Norm Weight")
parser.add_argument("--wandb", default=0, type=int, help="wandb upload")
parser.add_argument('--gpu-id', default=0, type=int, help="GPU index")

args = parser.parse_args()
wandb_upload = bool(args.wandb)
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

from models import GraphEncoder, GraphDecoder, GSLEncoder, GSLDecoder, GraphEncoderNoPooling, GraphDecoderNoPooling
from graphdata import GraphDataset, SpLoader
from vtk2adj import v2a, combineAdjacency

# loop through folders and load data
# ma_list = [0.2, 0.35, 0.5, 0.65, 0.8]
ma_list = [0.35, 0.5, 0.65]
# ma_list = [0.5]
re_list = [1e5, 1e6, 1e7, 1e8]
# re_list = [1e5]
aoa_list = [0, 2, 4, 6, 8, 10, 12]
# aoa_list = [0]
n_slices = 5
data_path = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data")

train_dataset = GraphDataset(data_path, ma_list, re_list, aoa_list, n_slices)
test_dataset = GraphDataset(data_path, [0.2, 0.8], re_list, aoa_list, n_slices)

n_samples = len(ma_list)*len(re_list)*len(aoa_list)
batch_sz = 1
batches = -(n_samples // -batch_sz)
test_sz = 1

train_dataloader = SpLoader(train_dataset, batch_sz, shuffle=True)
test_dataloader = SpLoader(test_dataset, test_sz, shuffle=True)

rng = jrn.PRNGKey(1)
n_pools = args.pooling_layers

# init_data_3, init_adj_3, init_data_2, init_adj_2 = [i[0] for i in next(iter(train_dataloader))]
init_data_3, init_data_2, init_adj_3, init_adj_2 = [
    i[0] for i in next(iter(test_dataloader))
]

# slices have 3d coords

# ge_3 = GraphEncoder(n_pools, args.latent_sz, args.channels, dim=3)
# ge_2 = GraphEncoder(n_pools, args.latent_sz, args.channels, dim=3)

ge_3 = GSLEncoder(n_pools, args.latent_sz, args.channels, dim=3)
ge_2 = GSLEncoder(n_pools, args.latent_sz, args.channels, dim=3)

# ge_3 = GraphEncoderNoPooling(n_pools, args.latent_sz, args.channels, dim=3)
# ge_2 = GraphEncoderNoPooling(
#     n_pools, args.latent_sz, args.channels, dim=3)

final_sz = init_data_3.shape[-1] - 3

# gd = GraphDecoder(n_pools, final_sz, args.channels, dim=3)
gd = GSLDecoder(n_pools, final_sz, args.channels, dim=3)
# gd = GraphDecoderNoPooling(n_pools, final_sz, args.channels, dim=3)

pe_3 = ge_3.init(rng, init_data_3, init_adj_3)['params']
pe_2 = ge_2.init(rng, init_data_2, init_adj_2)['params']
f_latent, a, c, s = ge_3.apply({'params': pe_3}, init_data_3, init_adj_3)
pd = gd.init(rng, f_latent, a, c, s)['params']
params = [pe_3, pe_2, pd]

check = orb.PyTreeCheckpointer()
check.save(os.path.join(data_path, "models", case_name + "_init"), params)

tx = optax.adam(1e-3)

n_epochs = 100000

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
def train_step(params, opt: optax.OptState, lam_0, lam_1, lam_2, lam_2d, data_3,
               data_2, adj_3, adj_2):

  def get_loss_f(feats, adj_sp):
    degr = jnp.diag(adj_sp @ jnp.ones((adj_sp.shape[-1], 1)))
    lapl = degr - adj_sp
    return jnp.sum(jnp.diag(feats.T @ lapl @ feats))

  def get_loss_p(adj_sp):
    p1 = jnp.ones((1, adj_sp.shape[0])) @ jnp.log(adj_sp @ jnp.ones(
        (adj_sp.shape[-1], 1)))
    p2 = jnp.sqrt(jnp.sum(jnp.square(adj_sp.data)))
    return -lam_1*p1 + lam_2*p2/2

  def loss_fn(params, data_3, data_2, adj_3, adj_2):
    loss = 0
    for fb3, fb2, adj3, adj2 in zip(data_3, data_2, adj_3, adj_2):

      # # DIFFPOOL
      # fl3, a, c, s = ge_3.apply({'params': params[0]}, fb3, adj3)
      # fl2, _, _, _ = ge_2.apply({'params': params[1]}, fb2, adj2)
      # f = gd.apply({'params': params[2]}, fl3, a, c, s)
      # loss_ae = jnp.mean(jnp.square(f[:, 3:] - fb3[:, 3:]))
      # loss_2 = jnp.mean(jnp.square(fl2 - fl3))
      #
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
      # loss = loss + (loss_ae + lam_2*loss_2 + lam_dp*
      #                (loss_e+loss_lp)) / batch_sz

      # # GSL POOL
      fl3, a, as3, c, s3, fg3 = ge_3.apply({'params': params[0]}, fb3, adj3)
      fl2, _, as2, _, _, fg2 = ge_2.apply({'params': params[1]}, fb2, adj2)
      f = gd.apply({'params': params[2]}, fl3, a, c, s3)
      loss_ae = jnp.mean(jnp.square(f[:, 3:] - fb3[:, 3:]))
      loss_2d = jnp.mean(jnp.square(fl2 - fl3))

      loss_f = jnp.mean(jnp.array(jtr.tree_map(get_loss_f, fg3, as3)))
      loss_f = loss_f + jnp.mean(jnp.array(jtr.tree_map(get_loss_f, fg2, as2)))

      loss_p = jnp.mean(jnp.array(jtr.tree_map(get_loss_p, as3)))
      loss_p = loss_p + jnp.mean(jnp.array(jtr.tree_map(get_loss_p, as2)))
      loss = loss + (loss_ae + lam_2d*loss_2d + lam_0*loss_f +
                     loss_p) / batch_sz

      # NO POOLING:
      # loss = loss + (loss_ae + lam_2*loss_2) / batch_sz
    return loss

  loss = 0
  a_list = []
  c_list = []
  s_list = []
  sample_loss = loss_fn(params, data_3, data_2, adj_3, adj_2)
  grads = grad(loss_fn)(params, data_3, data_2, adj_3, adj_2)
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

  loss = loss + sample_loss/batch_sz
  for d3, a3 in zip(data_3, adj_3):
    _, a, c, s = ge_3.apply({'params': params[0]}, d3, a3)
    if a_list == []:
      a_list = [
          jxs.BCSR((a_i.data / batch_sz, a_i.indices, a_i.indptr),
                   shape=a_i.shape) for a_i in a
      ]
      c_list = jtr.tree_map(lambda c_i: c_i / batch_sz, c)
      s_list = [
          jxs.BCSR((s_i.data / batch_sz, s_i.indices, s_i.indptr),
                   shape=s_i.shape) for s_i in s
      ]
    else:
      a_list = [
          jxs.BCSR((a_i.data + a_new.data / batch_sz, a_i.indices, a_i.indptr),
                   shape=a_i.shape) for a_i, a_new in zip(a_list, a)
      ]
      c_list = jtr.tree_map(lambda c_i, c_new: c_i + c_new/batch_sz, c_list, c)
      s_list = [
          jxs.BCSR((s_i.data + s_new.data / batch_sz, s_i.indices, s_i.indptr),
                   shape=s_i.shape) for s_i, s_new in zip(s_list, s)
      ]

  return loss, params, opt, a_list, c_list, s_list


@jit
def test_step(params, adj_list, coordinates, selection, data_3, data_2, adj_2):
  # @jit
  def loss_fn(params, data_3, data_2, adj_2, adj_list, coordinates, selection):
    loss = 0
    for fb3, fb2, adj2 in zip(data_3, data_2, adj_2):
      fl2, _, _, _ = ge_2.apply({'params': params[1]}, fb2, adj2)
      f = gd.apply({'params': params[2]}, fl2, adj_list, coordinates, selection)
      loss_ae = jnp.mean(jnp.square(f[:, 3:] - fb3[:, 3:]))
      loss = loss + loss_ae/test_sz
    return loss

  test_err = loss_fn(params, data_3, data_2, adj_2, adj_list, coordinates,
                     selection)
  return test_err


def main(params, n_epochs):
  opt = tx.init(params)
  # indices = train_dataset.items
  for epoch in range(n_epochs):
    a_list = []
    c_list = []
    s_list = []
    for batch in range(batches):
      data_3, data_2, adj_3, adj_2 = next(iter(train_dataloader))
      loss, params, opt, a, c, s = train_step(params, opt, args.lambda_2d,
                                              args.lambda_dp, data_3, data_2,
                                              adj_3, adj_2)
      if a_list == []:
        a_list = [
            jxs.BCSR((a_i.data / batches, a_i.indices, a_i.indptr),
                     shape=a_i.shape) for a_i in a
        ]
        c_list = jtr.tree_map(lambda c_i: c_i / batches, c)
        s_list = [
            jxs.BCSR((s_i.data / batches, s_i.indices, s_i.indptr),
                     shape=s_i.shape) for s_i in s
        ]
      else:
        a_list = [
            jxs.BCSR((a_i.data + a_new.data / batches, a_i.indices, a_i.indptr),
                     shape=a_i.shape) for a_i, a_new in zip(a_list, a)
        ]
        c_list = jtr.tree_map(lambda c_i, c_new: c_i + c_new/batches, c_list, c)
        s_list = [
            jxs.BCSR((s_i.data + s_new.data / batches, s_i.indices, s_i.indptr),
                     shape=s_i.shape) for s_i, s_new in zip(s_list, s)
        ]
    data_3, data_2, adj_3, adj_2 = next(iter(test_dataloader))
    test_err = test_step(params, a, c, s, data_3, data_2, adj_2)
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
      check.save(
          os.path.join(data_path, "models",
                       case_name + "_ep-{:g}".format(epoch)), params)


if __name__ == "__main__":
  if wandb_upload:
    import wandb
    wandb.init(project="DB-GNN", entity="wglao", name="graph autoencoder")
  main(params, n_epochs)