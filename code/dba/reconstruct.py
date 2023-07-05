import os, sys
from datetime import date
import pyvista as pv
import flax.linen as nn
import orbax.checkpoint as orb
import numpy as np
import jax.numpy as jnp
import jax.random as jrn
from models import GraphEncoderNoPooling, GraphDecoderNoPooling
from vtk2adj import v2a, combineAdjacency

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
parser.add_argument("--lambda-2d", default=1, type=float, help="2D Loss Weight")
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

data_path = os.path.join(
    os.environ["SCRATCH"], "ORNL/dimension-bridging/data",
    "ma_{:g}/re_{:g}/a_{:g}".format(args.mach, args.reynolds, args.aoa))


def main(check_path):
  mesh_3 = pv.read(os.path.join(data_path, "flow.vtu"))
  # extract point data from coordinates and conservative fields
  coords = jnp.array(mesh_3.points)
  data_3 = jnp.column_stack([coords] + [
      mesh_3.point_data.get_array(i)
      # for i in ["Density", "Momentum", "Energy"]
      for i in ["Density"]  # Density only for Memory
  ])
  # [mesh_3.point_data.get_array(i) for i in range(mesh_3.n_arrays)]))
  adj_3 = v2a(mesh_3)

  slice_data = []
  slice_adj = []
  n_slices = 5
  for s in range(n_slices):
    mesh_2 = pv.read(os.path.join(data_path, "slice_{:d}.vtk".format(s)))
    coords = jnp.array(mesh_2.points)
    slice_data.append(
        jnp.column_stack([coords] + [
            mesh_2.point_data.get_array(i)
            # for i in ["Density", "Momentum", "Energy"]
            for i in ["Density"]  # Density only for Memory
        ]))
    slice_adj.append(v2a(mesh_2))
  data_2 = jnp.concatenate(slice_data, axis=0)
  adj_2 = combineAdjacency(slice_adj)

  ge_3 = GraphEncoderNoPooling(
      args.pooling_layers, args.latent_sz, args.channels, dim=3)
  ge_2 = GraphEncoderNoPooling(
      args.pooling_layers, args.latent_sz, args.channels, dim=3)
  gd = GraphDecoderNoPooling(
      args.pooling_layers, data_3.shape[-1] - 3, args.channels, dim=3)

  rng = jrn.PRNGKey(1)
  pe_3 = ge_3.init(rng, data_3, adj_3)['params']
  pe_2 = ge_2.init(rng, data_2, adj_2)['params']
  f_latent, a, c, s = ge_3.apply({'params': pe_3}, data_3, adj_3)
  pd = gd.init(rng, f_latent, a, c, s)['params']
  params = [pe_3, pe_2, pd]

  check = orb.PyTreeCheckpointer()
  if os.path.exists(check_path):
    params = check.restore(check_path, item=params)
  else:
    print(
        "Model with requested architecture and epoch not saved.\nUsing random initialization."
    )

  _, a, c, s = ge_3.apply({'params': params[0]}, data_3, adj_3)
  f_latent, _, _, _ = ge_2.apply({'params': params[1]}, data_2, adj_2)
  f_recon = gd.apply({'params': params[2]}, f_latent, a, c, s)[:, 3:]

  # for field in ["Density", "Momentum", "Energy"]:
  for field in ["Density"]:
    mesh_3.point_data.set_array(f_recon, field)

  mesh_3.save(os.path.join(data_path, "reconstruct.vtu"))
  return mesh_3


def plot(mesh):
  pass


if __name__ == "__main__":
  check_path = os.path.join(os.environ["SCRATCH"],
                            "ORNL/dimension-bridging/data/models_save",
                            case_name + "_ep-{:d}".format(args.epoch),
                            args.date)
  mesh_recon = main(check_path)
  plot(mesh_recon)