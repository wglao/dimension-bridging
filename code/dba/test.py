import sys
from functools import partial

import flax.core.frozen_dict as fd
import flax.linen as nn
import jax.numpy as jnp
import jax.random as jrn
import jax.tree_util as jtr
import optax
from jax import grad, jit, value_and_grad, vmap
import jax.experimental.sparse as jxs

from models import (DiffPoolLayer, GraphDecoder, GraphEncoder, MoNetLayer,
                    TransAggLayer)

batches = 10
batch_sz = 1
test_sz = 10
nodes = 16
adjacency = jnp.eye(nodes) + jnp.diag(jnp.ones(
    (nodes - 1,)), 1) + jnp.diag(jnp.ones((nodes - 1,)), -1)
adjacency = jxs.bcoo_fromdense(adjacency)

rng = jtr.tree_map(lambda seed: jrn.PRNGKey(seed),
                   list(jnp.arange(batches*batch_sz + test_sz)))


@partial(jit, static_argnums=(1,))
def get_feats(rng, n):
  coords = jrn.normal(rng, (n, 3))
  coords = coords[jnp.argsort(coords[:, 0])]
  features = 1 / jnp.sum(coords**2, axis=-1)
  features = jnp.column_stack((coords, features))
  return features


b_list_3 = []
b_list_2 = []
for b in range(batches):
  f_list_3 = []
  f_list_2 = []
  for i in range(batch_sz):
    f_list_3.append(get_feats(rng[b*batch_sz + i], nodes))
    no_z = f_list_3[-1][:, :2]
    f_list_2.append(jnp.column_stack((no_z, 1 / jnp.sum(no_z**2, axis=-1))))
  batch_3 = jnp.stack(f_list_3)
  batch_2 = jnp.stack(f_list_2)
  b_list_3.append(batch_3)
  b_list_2.append(batch_2)
features_3 = jnp.stack(b_list_3)
features_2 = jnp.stack(b_list_2)

f_list_3 = []
f_list_2 = []
for i in range(test_sz):
  f_list_3.append(get_feats(rng[i + batches*batch_sz], nodes))
  no_z = f_list_3[-1][:, :2]
  f_list_2.append(jnp.column_stack((no_z, 1 / jnp.sum(no_z**2, axis=-1))))
test_batch_3 = jnp.stack(f_list_3)
test_batch_2 = jnp.stack(f_list_2)

n_pools = 1
ge_3 = GraphEncoder(n_pools, dim=3)
ge_2 = GraphEncoder(n_pools, dim=2)
gd = GraphDecoder(n_pools, dim=3)

pe_3 = ge_3.init(rng[-1], features_3[0, 0], adjacency)['params']
pe_2 = ge_2.init(rng[-1], features_2[0, 0], adjacency)['params']
f_latent, a, c, s = ge_3.apply({'params': pe_3}, features_3[0, 0], adjacency)
pd = gd.init(rng[-1], f_latent, a, c, s)['params']
params = [pe_3, pe_2, pd]
tx = optax.adam(1e-3)

n_epochs = 100000

eps = 1e-15


# @jit
def train_step(params,
               features_3,
               features_2,
               adjacency,
               opt: optax.OptState,
               lam_2: float = 1,
               lam_dp: float = 1):

  def loss_fn(params, features_3, features_2, adjacency):
    loss = 0
    for fb3, fb2 in zip(features_3, features_2):
      fl3, a, c, s = ge_3.apply({'params': params[0]}, fb3, adjacency)
      fl2, _, _, _ = ge_2.apply({'params': params[1]}, fb2, adjacency)
      f = gd.apply({'params': params[2]}, fl3, a, c, s)
      loss_ae = jnp.mean(jnp.square(f[:, 3:] - fb3[:, 3:]))
      loss_2 = jnp.mean(jnp.square(fl2 - fl3))
      loss_lp = jnp.mean(
          jnp.array([
              jnp.sqrt(
                  jnp.sum(
                      jnp.square(
                          jxs.bcoo_sum_duplicates(
                              a_i - jxs.bcoo_fromdense(s_i @ s_i.T)).data)))
              for a_i, s_i in zip(a[:-1], s)
          ]))
      loss_e = jnp.mean(
          jnp.array(
              jtr.tree_map(
                  lambda s: jnp.mean(jnp.sum(-s*jnp.log(s + eps), axis=-1)),
                  s)))
      loss = loss + (loss_ae + lam_2*loss_2 + lam_dp*
                     (loss_e+loss_lp)) / batch_sz
    return loss

  def get_acs(params, features_3):
    a_list = []
    c_list = []
    s_list = []
    for fb3 in jnp.concatenate(features_3):
      _, a, c, s = ge_3.apply({'params': params[0]}, fb3, adjacency)
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
                                                adjacency)
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


# @jit
def test_step(params, features_3, features_2, adjacency, coordinates,
              selection):

  def loss_fn(params, features_3, features_2, adjacency, coordinates,
              selection):
    loss = 0
    for fb3, fb2 in zip(features_3, features_2):
      fl2, _, _, _ = ge_2.apply({'params': params[1]}, fb2, adjacency[0])
      f = gd.apply({'params': params[2]}, fl2, adjacency, coordinates,
                   selection)
      loss_ae = jnp.mean(jnp.square(f[:, 3:] - fb3[:, 3:]))
      loss = loss + loss_ae/test_sz
    return loss

  test_err = loss_fn(params, features_3, features_2, adjacency, coordinates,
                     selection)
  return test_err


def main(params, n_epochs):
  opt = tx.init(params)

  for epoch in range(n_epochs):
    loss, params, opt, a, c, s = train_step(params, features_3, features_2,
                                            adjacency, opt)
    test_err = test_step(params, test_batch_3, test_batch_2, a, c, s)
    if epoch % 100 == 0 or epoch == n_epochs - 1:
      if "-wandb" in sys.argv:
        wandb.log({
            "Loss": loss,
            "Error": test_err,
            "Epoch": epoch,
        })
      else:
        print("Loss: {:g}, Error {:g}, Epoch {:g}".format(
            loss, test_err, epoch))


if __name__ == "__main__":
  if "-wandb" in sys.argv:
    import wandb
    wandb.init(project="DB-GNN", entity="wglao", name="graph autoencoder")
  main(params, n_epochs)