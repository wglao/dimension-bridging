import os, sys, shutil, glob
from datetime import date
import pyvista as pv
import torch
from torch_geometric.loader import DataLoader
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--modes", default=10, type=int, help="Number of Modes to Plot")
args = parser.parse_args()

data_path = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data")
save_path = os.path.join(data_path, "models_save")


def main(save_path):
  mesh = pv.read(
      os.path.join(data_path, "ma_{:g}/re_{:g}/a_{:g}".format(0.3, 3e6, 3),
                   "flow.vtu"))
  vec_path = os.path.join(data_path, "processed/eig")
  for i in range(args.modes):
    save = os.path.join(vec_path,"vec_{:d}.pt".format(i))
    mode = torch.load(save)
    mesh.point_data.set_array(mode.cpu().detach().numpy(), "Laplacian Mode {:d}".format(i + 1))

  save_path = os.path.join(data_path, "modes.vtu")
  if os.path.exists(save_path):
    if os.path.isfile(save_path):
      os.remove(save_path)
    else:
      shutil.rmtree(save_path)
  mesh.save(save_path)
  return mesh


def plot(mesh):
  pass


if __name__ == "__main__":
  mesh_modes = main(save_path)
  plot(mesh_modes)