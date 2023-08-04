import os
import sys
from datetime import date
import shutil
from functools import partial
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--case-name", default="dba-gsl", type=str, help="Architecture Name")
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
parser.add_argument("--lambda-2d", default=1, type=float, help="2D Loss Weight")
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

from models import DBA
from graphdata import GraphDataset, GraphLoader

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
deg_f = jnp.column_stack((init_data_3[:, :3], adj_3 @ jnp.ones(
    (adj_3.shape[-1], 1))))

# pr<0.8 for memory
pool_ratio = 0.01
# if args.pooling_layers > 0:
#   pool_szs_3 = [int(-(adj_3.shape[-1] // -(1 / pool_ratio)))]
#   pool_szs_2 = [int(-(adj_2.shape[-1] // -(1 / pool_ratio)))]
# else:
#   pool_szs_3 = []
#   pool_szs_2 = []
# for pool in range(args.pooling_layers - 1):
#   pool_szs_3.append(int(-(pool_szs_3[-1] // -(1 / pool_ratio))))
#   pool_szs_2.append(int(-(pool_szs_2[-1] // -(1 / pool_ratio))))

# pool_szs_3 = jnp.array(pool_szs_3)
# pool_szs_2 = jnp.array(pool_szs_2)

# slices have 3d coords

if not wandb_upload:
  print('Init')

model = DBA(3, )

pe_3 = ge_3.init(rng, deg_f, adj_3, pool_ratio)['params']
pe_2 = ge_2.init(rng, init_data_2, adj_2, pool_ratio)['params']
f_latent, a, _, c, s, _ = ge_3.apply({'params': pe_3}, deg_f, adj_3, pool_ratio)
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

tx = optax.adam(1e-3)

n_epochs = 10000

eps = 1e-15


# set kernel size to mean of node degree vector
idx = init_data.coo()
sz = init_data.num_nodes
adj = torch.sparse_coo_tensor(idx, np.ones((init_data.num_edges,)),
                              (sz, sz))
deg = adj.matmul(torch.tensor(np.ones((sz, 1))))



@partial(jit, static_argnums=(8))
def train_step(params, opt: optax.OptState, lam_0, lam_1, lam_2, lam_2d, data_3,
               data_2, pool_ratio):
  loss = 0

  def get_loss_f(feats, adj_sp):
    degr = jxs.BCOO(
        (jnp.squeeze(adj_sp @ jnp.ones((adj_sp.shape[-1], 1))),
         jnp.column_stack(
             (jnp.arange(adj_sp.shape[-1]), jnp.arange(adj_sp.shape[-1])))),
        shape=(adj_sp.shape[-1], adj_sp.shape[-1]))
    lapl = degr - adj_sp
    return jnp.sum(jnp.diag(feats.T @ lapl @ feats))

  def get_loss_p(adj_sp):
    p1 = jnp.ones((1, adj_sp.shape[0])) @ jnp.log(adj_sp @ jnp.ones(
        (adj_sp.shape[-1], 1)))
    p2 = jnp.sqrt(jnp.sum(jnp.square(adj_sp.data)))
    return -lam_1*p1 + lam_2*p2/2

  def loss_fn(params, data_3, data_2):
    # GSL POOL

    # # 3->3 is primary ae loss
    # fl3, a, as3, c, s3, fg3 = ge_3.apply({'params': params[0]}, data_3, adj_3)
    # fl2, _, as2, _, _, fg2 = ge_2.apply({'params': params[1]}, data_2, adj_2)
    # f = gd.apply({'params': params[2]}, fl3, a, c, s3)
    # loss_ae = jnp.mean(jnp.square(f[:, 3:] - data_3[:, 3:]))
    # loss_2d = jnp.mean(jnp.square(fl2 - fl3))

    # # 2->3 is primary ae loss
    _, a, as3, c, s, fg3 = ge_3.apply({'params': params[0]}, deg_f, adj_3,
                                      pool_ratio)
    fl2, _, as2, _, _, fg2 = ge_2.apply({'params': params[1]}, data_2, adj_2,
                                        pool_ratio)
    f = gd.apply({'params': params[2]}, fl2, a, c, s)
    loss_ae = jnp.mean(jnp.square(f[:, 3:] - data_3[:, 3:]))
    # loss_2d = jnp.mean(jnp.square(fl2 - fl3))

    loss_f = jnp.mean(
        jnp.array([get_loss_f(fg_i, as_i) for fg_i, as_i in zip(fg3, as3)]))
    loss_f = loss_f + lam_2d*jnp.mean(
        jnp.array([get_loss_f(fg_i, as_i) for fg_i, as_i in zip(fg2, as2)]))

    loss_p = jnp.mean(jnp.array([get_loss_p(as_i) for as_i in as3]))
    loss_p = loss_p + lam_2d*jnp.mean(
        jnp.array([get_loss_p(as_i) for as_i in as2]))
    # loss = (loss_ae + lam_2d*loss_2d + lam_0*loss_f + loss_p) / batch_sz
    loss = (loss_ae + lam_0*loss_f + loss_p) / batch_sz

    # # NO POOLING:
    # loss = loss + (loss_ae + lam_2d*loss_2d) / batch_sz
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

  # a_list = []
  # c_list = []
  # s_list = []

  # for d3 in data_3:
  #   _, a, c, s = ge_3.apply({'params': params[0]}, d3, adj_3)
  #   if a_list == []:
  #     a_list = [a_i / batch_sz for a_i in a]
  #     c_list = c
  #     s_list = s
  #   else:
  #     a_list = [a_i + a_new/batch_sz for a_i, a_new in zip(a_list, a)]
  #     c_list = jtr.tree_map(
  #         lambda c_i, c_new: jnp.concatenate((c_i, c_new), axis=0), c_list, c)
  #     s_list = jtr.tree_map(
  #         lambda s_i, s_new: jnp.concatenate((s_i, s_new), axis=None), s_list,
  #         s)

  return loss, params, opt  #, a_list, c_list, s_list


@partial(jit, static_argnums=(3))
def test_step(params, data_3, data_2, pool_ratio):
  test_err = 0
  _, a, _, c, s, _ = ge_3.apply({'params': params[0]}, deg_f, adj_3, pool_ratio)

  def loss_fn(params, data_3, data_2):
    fl2, _, _, _, _, _ = ge_2.apply({'params': params[1]}, data_2, adj_2,
                                    pool_ratio)
    f = gd.apply({'params': params[2]}, fl2, a, c, s)
    loss_ae = jnp.mean(jnp.square(f[:, 3:] - data_3[:, 3:]))
    loss = loss + loss_ae/test_sz
    return loss

  data_3_batched = vmap(
      dsd, in_axes=(None, 0, None))(data_3, jnp.arange(test_batches)*test_sz,
                                    test_sz)
  data_2_batched = vmap(
      dsd, in_axes=(None, 0, None))(data_2, jnp.arange(test_batches)*test_sz,
                                    test_sz)

  for (batch_3, batch_2) in zip(data_3_batched, data_2_batched):
    sample_loss = vmap(loss_fn, in_axes=(None, 0, 0))(params, batch_3, batch_2)
    batch_loss = jnp.mean(sample_loss, axis=0)
    test_err = test_err + batch_loss/test_batches

  return test_err


def main(params, n_epochs):
  opt = tx.init(params)
  min_err = jnp.inf
  # indices = train_dataset.items
  for epoch in range(n_epochs):
    # loss, params, opt, a, c, s = train_step(params, opt, args.lambda_0,
    #                                         args.lambda_1, args.lambda_2,
    #                                         args.lambda_2d, train_data_3,
    #                                         train_data_2)
    if not wandb_upload:
      print('Train')
    loss, params, opt = train_step(params, opt, args.lambda_0, args.lambda_1,
                                   args.lambda_2, args.lambda_2d, train_data_3,
                                   train_data_2, pool_ratio)

    # a = [a_i.sum_duplicates() for a_i in a]
    # c = [jnp.unique(c_i, axis=0) for c_i in c]
    # s = [jnp.unique(s_i, axis=None) for s_i in s]
    if not wandb_upload:
      print('Test')
    # test_err = test_step(params, a, c, s, test_data_3, test_data_2)
    test_err = test_step(params, test_data_3, test_data_2, pool_ratio)
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