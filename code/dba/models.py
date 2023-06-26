from typing import Any

import flax.linen as nn
import jax.image as jim
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
import jax.experimental.sparse as jxs


class MitsuMatsuCNN(nn.Module):
  act: callable = nn.relu

  @nn.compact
  def __call__(self, input_field) -> Any:
    x = self.act(nn.Conv(32, (3, 3))(input_field))
    x = self.act(nn.Conv(32, (3, 3))(x))

    x = nn.max_pool(x, (2, 2))

    x = self.act(nn.Conv(16, (3, 3))(x))
    x = self.act(nn.Conv(20, (3, 3))(x))

    x = jnp.reshape(x, (128, 64, 20, 1))

    x = self.act(nn.Conv(16, (3, 3, 3))(x))
    x = self.act(nn.Conv(16, (3, 3, 3))(x))

    x = jim.resize(x, x.shape*jnp.array([1, 2, 2, 8]), "bilinear")

    x = self.act(nn.Conv(32, (3, 3, 3))(x))
    x = self.act(nn.Conv(32, (3, 3, 3))(x))

    x_final = nn.Conv(3, (3, 3, 3))(x)
    return x_final


class MoNetLayer(nn.Module):
  channels: int = 32
  dim: int = 3
  r: int = 3
  act: callable = nn.tanh

  def _get_weight_i(self, mu, sig, u_j):
    return jnp.exp(-0.5*
                   ((u_j - mu).T @ jnp.linalg.inv(jnp.diag(sig)) @ (u_j-mu)))

  get_weights = vmap(
      _get_weight_i, in_axes=(None, None, None, 0))  # map over edge

  def sig_init(self, rng, shape):
    return jnp.squeeze(jnp.abs(nn.initializers.lecun_normal()(rng, shape)))

  def aggregate(self, weights, adjacency, features):
    attention = jxs.BCSR(
        (adjacency.data*weights, adjacency.indices, adjacency.indptr),
        shape=adjacency.shape)
    out = attention @ features
    return out

  @nn.compact
  def __call__(self, features, adjacency: jxs.BCSR):
    n_nodes = adjacency.shape[-1]
    # take first `dim` elements to be the node coordinates
    node_coords = features[:, :self.dim]
    # assume node coords are 3d
    monet_xi = jnp.concatenate(
        scan(lambda c, x: (c, x[0]*jnp.ones((x[1] - x[2], 3))))(
            None, (node_coords, adjacency.indptr[1:], adjacency.indptr[:-1]))[1],
        axis=0)
    monet_xj = node_coords[adjacency.indices]
    monet_u = monet_xj - monet_xi
    # learned coordinates from monet paper
    monet_u = jnp.expand_dims(self.act(nn.Dense(self.r)(monet_u)), axis=-1)

    mu = self.param('mu', nn.initializers.lecun_normal(), (self.r, 1))
    sig = self.param('sigma', self.sig_init, (self.r, 1))
    weights = jnp.squeeze(self.get_weights(mu, sig, monet_u))
    out = self.aggregate(weights, adjacency, features[:, self.dim:])
    out = vmap(nn.Dense(self.channels))(out)
    out = jnp.column_stack((node_coords, out))
    return out


class DiffPoolLayer(nn.Module):
  pool_factor: float = 2.
  dim: int = 3
  act: callable = nn.softmax

  @nn.compact
  def __call__(self, features, adjacency: jxs.BCSR):
    # take ceil of n/pf**dim
    n_clusters = int(-(adjacency.shape[-1] // -(self.pool_factor**self.dim)))

    gnn_s = MoNetLayer(n_clusters, self.dim)

    s = gnn_s(features, adjacency)
    s = jxs.BCSR.fromdense(self.act(s[:, self.dim:], axis=-1))
    # s = (s.T[~jnp.all(s.T == 0, axis=-1)]).T

    f = s.T @ features
    # change coordiantes to reflect barycenter of neighborhood
    f = f.at[:, :self.dim].set(f[:, :self.dim] /
                               jnp.expand_dims(jnp.sum(s, axis=0).T, -1))
    a = s.T @ adjacency @ s
    return s, f, a


class TransAggLayer(nn.Module):
  channels: int = 32
  dim: int = 3

  @nn.compact
  def __call__(self, features, node_coords, adjacency, selection):
    # build new graph combining parent and children nodes
    # by appending selection (clustering) adjacency
    # to true graph adjacency
    #TODO: switch a to bcsr
    p_nodes = selection.shape[-1]
    c_nodes = adjacency.shape[-1]
    x_ind = vmap(lambda i: c_nodes + i)(jnp.arange(1, p_nodes + 1))
    a_ind = jnp.concatenate((vmap(lambda i, j, k, l: jnp.concatenate(
        (adjacency.indices[i:j], selection.indices[k:l] + c_nodes), axis=None))(
            adjacency.indptr[:-1], adjacency.indptr[1:], selection.indptr[:-1],
            selection.indptr[1:]), x_ind),
                            axis=None)

    s_ptr = jnp.concatenate((jnp.zeros(
        (1,)), selection.indptr[1:-1] - selection.indptr[:-2]),
                            axis=None)
    x_ptr = jnp.arange(adjacency.nse + selection.nse,
                       adjacency.nse + selection.nse + p_nodes + 1)
    a_ptr = jnp.concatenate(((adjacency.indptr + s_ptr), x_ptr), axis=None)

    s_data = jnp.concatenate(selection.T, axis=None)
    a_data = jnp.concatenate((adjacency.data, s_data, jnp.ones((p_nodes,))),
                             axis=None)
    a = jxs.BCOO((a_data, a_ind), shape=(c_nodes + p_nodes, c_nodes + p_nodes))
    f = jnp.row_stack((jnp.column_stack(
        (node_coords,
         jnp.zeros(
             (adjacency.shape[-1], features.shape[-1] - self.dim)))), features))
    out = MoNetLayer(self.channels, self.dim)(f, a)

    # remove parent nodes from graph
    out = out[:-p_nodes]
    return out


class GraphEncoder(nn.Module):
  n_pools: int = 1
  n_latent_variables: int = 50
  n_hidden_variables: int = 200
  dim: int = 3
  act: callable = nn.elu

  def act_no_coords(self, features):
    out = features.at[:, self.dim:].set(self.act(features[:, self.dim:]))
    return out

  @nn.compact
  def __call__(self, features, adjacency):
    a = []
    c = []
    s = []
    a.append(adjacency)
    c.append(features[:, :self.dim])
    f = features
    f = self.act_no_coords(
        MoNetLayer(self.n_hidden_variables, self.dim)(f, a[-1]))
    # f = self.act_no_coords(MoNetLayer(self.n_hidden_variables,self.dim)(f, a[-1]))

    for l in range(self.n_pools):
      # ideally pf=2, pf>2 due to memory constraints
      selection, f, a_coarse = DiffPoolLayer(
          pool_factor=2, dim=self.dim)(f, a[-1])
      a.append(jxs.bcoo_fromdense(a_coarse))
      c.append(f[:, :self.dim])
      s.append(selection)

      f = self.act_no_coords(
          MoNetLayer(self.n_hidden_variables, self.dim)(f, a[-1]))
      # f = self.act_no_coords(MoNetLayer(self.n_hidden_variables,self.dim)(f, a[-1]))

    f_latent = self.act(nn.Dense(self.n_hidden_variables)(f.ravel()))
    f_latent = nn.Dense(self.n_latent_variables)(f_latent)
    return f_latent, a, c, s


class GraphEncoderNoPooling(GraphEncoder):

  @nn.compact
  def __call__(self, features, adjacency):
    a = []
    c = []
    a.append(adjacency)
    c.append(features[:, :self.dim])

    f = features
    f = self.act_no_coords(
        MoNetLayer(self.n_hidden_variables, self.dim)(f, a[-1]))

    f_latent = self.act(nn.Dense(self.n_hidden_variables)(f.ravel()))
    f_latent = nn.Dense(self.n_latent_variables)(f_latent)
    return f_latent, a, c, None


class GraphDecoder(nn.Module):
  n_upsamples: int = 1
  n_final_variables: int = 1
  n_hidden_variables: int = 200
  dim: int = 3
  act: callable = nn.elu

  def act_no_coords(self, features):
    out = features.at[:, self.dim:].set(self.act(features[:, self.dim:]))
    return out

  @nn.compact
  def __call__(self, f_latent, a_list, c_list, s_list):
    n_nodes = a_list[-1].shape[-1]

    f = nn.Dense(self.n_hidden_variables)(f_latent)
    f = self.act(nn.Dense(self.n_hidden_variables*n_nodes)(f))
    f = jnp.reshape(f, (n_nodes, len(f) // n_nodes))
    f = jnp.column_stack((c_list[-1], f))

    for l in range(self.n_upsamples):
      f = self.act_no_coords(
          MoNetLayer(self.n_hidden_variables, self.dim)(f, a_list[-l - 1]))
      # f = self.act_no_coords(
      #     MoNetLayer(self.n_hidden_variables, self.dim)(f, a_list[-l - 1]))

      f = TransAggLayer(self.n_hidden_variables,
                        self.dim)(f, c_list[-l - 2], a_list[-l - 2],
                                  s_list[-l - 1])

    f = self.act_no_coords(
        MoNetLayer(self.n_final_variables, self.dim)(f, a_list[0]))
    # f = MoNetLayer(self.n_final_variables)(f, a_list[0])
    return f


class GraphDecoderNoPooling(GraphDecoder):

  @nn.compact
  def __call__(self, f_latent, a_list, c_list, s_list):
    n_nodes = a_list[-1].shape[-1]

    f = nn.Dense(self.n_hidden_variables)(f_latent)
    f = self.act(nn.Dense(self.n_hidden_variables*n_nodes)(f))
    f = jnp.reshape(f, (n_nodes, len(f) // n_nodes))
    f = jnp.column_stack((c_list[-1], f))

    f = self.act_no_coords(
        MoNetLayer(self.n_final_variables, self.dim)(f, a_list[0]))
    return f


class GraphAutoEncoder(nn.Module):
  n_pools: int = 1
  n_latent_variables: int = 32
  n_hidden_variables: int = 64
  dim: int = 3
  act: callable = nn.elu

  def act_no_coords(self, features):
    out = features.at[:, self.dim:].set(self.act(features[:, self.dim:]))
    return out

  @nn.compact
  def __call__(self, features, adjacency):
    f, a, c, s = GraphEncoder(self.n_pools, self.n_latent_variables,
                              self.n_hidden_variables, self.dim,
                              self.act)(features, adjacency)
    f = GraphDecoder(self.n_pools, features.size[-1], self.n_hidden_variables,
                     self.dim, self.act)(f, a, c, s)
    return f