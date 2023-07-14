import os, sys, shutil
from datetime import date
import pyvista as pv
import flax.linen as nn
import orbax.checkpoint as orb
import numpy as np
import jax.numpy as jnp
import jax.random as jrn
from models import GraphEncoderNoPooling, GraphDecoderNoPooling
from graphdata import GraphDataset, GraphLoader
from vtk2adj import v2a, combineAdjacency

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
    "--lambda-dp", default=1.0, type=float, help="DiffPool Loss Weight")
parser.add_argument("--wandb", default=1, type=int, help="wandb upload")
parser.add_argument('--gpu-id', default=0, type=int, help="GPU index")
parser.add_argument("--mach", default=0.8395, type=float, help="Mach Number")
parser.add_argument(
    "--reynolds", default=1.172e7, type=float, help="Reynolds Number")
parser.add_argument("--aoa", default=3.06, type=float, help="Angle of Attack")
parser.add_argument(
    "--date", default="040723", type=str, help="Date of run in ddmmyy")
parser.add_argument("--epoch", default=100, type=int, help="Checkpoint Epoch")

args = parser.parse_args()
case_name = "_".join([
    str(key) + "-" + str(value) for key, value in list(vars(args).items())[:-6]
])[10:]
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

data_path = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data")


def main(check_path):
  n_slices = 5
  dataset = GraphDataset(data_path, [args.mach], [args.reynolds], [args.aoa],
                         n_slices)
  dataloader = GraphLoader(dataset)

  rng = jrn.PRNGKey(1)
  n_pools = args.pooling_layers
  mesh_3 = pv.read(
      os.path.join(
          data_path, "ma_{:g}/re_{:g}/a_{:g}".format(args.mach, args.reynolds,
                                                     args.aoa), "flow.vtu"))
  adj_3 = v2a(mesh_3)

  slice_adj = []
  for s in range(n_slices):
    mesh = pv.read(
        os.path.join(
            data_path, "ma_{:g}/re_{:g}/a_{:g}".format(args.mach, args.reynolds,
                                                       args.aoa),
            "slice_{:d}.vtk".format(s)))
    slice_adj.append(v2a(mesh))
  adj_2 = combineAdjacency(slice_adj)

  data_3, data_2 = [d[0] for d in next(iter(dataloader))]

  ge_3 = GraphEncoderNoPooling(n_pools, args.latent_sz, args.channels, dim=3)
  ge_2 = GraphEncoderNoPooling(
      n_pools, args.latent_sz, args.channels, dim=3)  # slices have 3d coords

  final_sz = data_3.shape[-1] - 3
  gd = GraphDecoderNoPooling(n_pools, final_sz, args.channels, dim=3)

  pe_3 = ge_3.init(rng, data_3, adj_3)['params']
  pe_2 = ge_2.init(rng, data_2, adj_2)['params']
  f_latent, a, c, s = ge_3.apply({'params': pe_3}, data_3, adj_3)
  pd = gd.init(rng, f_latent, a, c, s)['params']
  params = [pe_3, pe_2, pd]

  options = orb.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
  ckptr = orb.CheckpointManager(
      check_path, {
          "params": orb.PyTreeCheckpointer(),
          "state": orb.PyTreeCheckpointer()
      },
      options=options)
  if os.path.exists(check_path):
    params = ckptr.restore(args.epoch)["params"]
    # params = restored["params"]
    print("Model found.")
  else:
    print(
        "Model with requested architecture and epoch not saved.\nUsing random initialization."
    )

  _, a, c, s = ge_3.apply({'params': params[0]}, data_3, adj_3)
  f_latent, _, _, _ = ge_2.apply({'params': params[1]}, data_2, adj_2)
  f_recon = gd.apply({'params': params[2]}, f_latent, a, c, s)[:, 3:]

  mse_rho = jnp.mean(jnp.square(f_recon[:, 0] - data_3[:, 3]))
  mse_u = jnp.mean(jnp.square(f_recon[:, 1:4] - data_3[:, 4:7]))
  mse_e = jnp.mean(jnp.square(f_recon[:, 4] - data_3[:, 7]))
  print("Reconstruction MSE: {:g} (density), {:g} (velocity), {:g} (energy)"
        .format(mse_rho, mse_u, mse_e))
  
  abs_err = jnp.abs(f_recon - data_3[:, 3:])
  rel_err = jnp.abs(abs_err / jnp.where(data_3[:, 3:]>1e-15, data_3[:, 3:], 1e-15))

  for idx, field in zip([jnp.s_[:, 0], jnp.s_[:, 1:4], jnp.s_[:, 4]],
                        ["Density", "Momentum", "Energy"]):
    #   for field in ["Density"]:
    mesh_3.point_data.set_array(f_recon[idx], field)
    mesh_3.point_data.set_array(abs_err[idx], field + "_Abs_Err")
    mesh_3.point_data.set_array(rel_err[idx], field + "_Rel_Err")

  save_path = os.path.join(
      data_path, "ma_{:g}/re_{:g}/a_{:g}".format(args.mach, args.reynolds,
                                                 args.aoa), "reconstruct.vtu")
  if os.path.exists(save_path):
    if os.path.isfile(save_path):
      os.remove(save_path)
    else:
      shutil.rmtree(save_path)
  mesh_3.save(save_path)
  return mesh_3


def plot(mesh):
  pass


if __name__ == "__main__":
  check_path = os.path.join(os.environ["SCRATCH"],
                            "ORNL/dimension-bridging/data/models_save",
                            case_name, args.date)
  mesh_recon = main(check_path)
  plot(mesh_recon)