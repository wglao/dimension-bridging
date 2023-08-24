import os
import sys
import math
import glob
from datetime import date
import shutil
from functools import partial
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--case-name", default="dblno", type=str, help="Architecture Name")
parser.add_argument(
    "--channels", default=10, type=int, help="Aggregation Channels")
parser.add_argument(
    "--kept-modes",
    default=20,
    type=int,
    help="Number of Eigenfunctions to Retain")
parser.add_argument(
    "--learning-rate", default=1e-3, type=float, help="Learning Rate")
parser.add_argument(
    "--decay", default=0.99, type=float, help="Exponential LR Decay Rate")
parser.add_argument("--wandb", default=0, type=int, help="wandb upload")
parser.add_argument("--debug", default=0, type=bool, help="debug prints")
parser.add_argument('--gpu-id', default=0, type=int, help="GPU index")

args = parser.parse_args()
wandb_upload = bool(args.wandb)
debug = True if not wandb_upload else args.debug
today = date.today()
case_name = "_".join([
    str(key) + "-" + str(value) for key, value in list(vars(args).items())[:-2]
])[10:]
device = "cuda:{:d}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"

import numpy as np
import torch
from torch_geometric import compile
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import knn_interpolate

from models_no import LNO, LaplaceLayer
from graphdata import PairData, PairDataset

if debug:
  print('Load')

torch.manual_seed(0)

# loop through folders and load data
# ma_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
ma_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# re_list = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
re_list = [2e6, 3e6, 5e6, 6e6, 8e6, 9e6]
# aoa_list = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
aoa_list = [-9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9]
n_slices = 5
data_path = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data")

if wandb_upload:
  train_dataset = PairDataset(data_path, ma_list, re_list, aoa_list, "train",
                              n_slices)
  test_dataset = PairDataset(data_path, ma_list, [4e6, 7e6, 1e7], aoa_list,
                             "test", n_slices)
else:
  # train_dataset = PairDataset(data_path, [0.3, 0.4], [3e6, 4e6], [3, 4],
  #                             "idev-train", n_slices)
  train_dataset = PairDataset(data_path, [0.3], [3e6], [3], "recon", n_slices)
  test_dataset = PairDataset(data_path, [0.5, 0.6], [5e6, 6e6], [5, 6],
                             "idev-test", n_slices)

n_samples = len(train_dataset)
batch_sz = int(np.min(np.array([1, n_samples])))
batches = -(n_samples // -batch_sz)
n_test = len(test_dataset)
test_sz = int(np.min(np.array([1, n_test])))
test_batches = -(n_test // -test_sz)

train_loader = DataLoader(train_dataset, batch_sz, follow_batch=['x_3', 'x_2'])
test_loader = DataLoader(test_dataset, test_sz, follow_batch=['x_3', 'x_2'])

init_pair = next(iter(test_loader))
init_pair = init_pair[0].to(device)
init_data = Data(
    init_pair.x_3, init_pair.edge_index_3, y=init_pair.y, pos=init_pair.pos_3)

transform_path = os.path.join(os.environ["SCRATCH"],
                              "ORNL/dimension-bridging/data/processed/eig")
if not os.path.exists(transform_path):
  os.makedirs(transform_path, exist_ok=True)

n_eigvecs = len(glob.glob(os.path.join(transform_path, "*.pt")))

if n_eigvecs < args.kept_modes:

  def get_deg(x, edge_index):
    deg = torch.sparse_coo_tensor(edge_index, torch.ones(
        (edge_index.size(1),))) @ torch.ones((x.size(0), 1))
    return deg

  def get_laplacian(x, edge_index):
    n_nodes = x.size(0)
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1),))
    deg_idx = torch.stack((torch.arange(n_nodes), torch.arange(n_nodes)), dim=0)
    deg = torch.squeeze(get_deg(x, edge_index))
    sqrt_deg = torch.sparse_coo_tensor(deg_idx, 1 / deg, (n_nodes, n_nodes))
    lapl = (torch.eye(n_nodes).to_sparse_coo() -
            sqrt_deg @ (adj@sqrt_deg)).coalesce()
    return lapl

  def get_basis(x, edge_index, kept_modes):
    lapl = get_laplacian(x, edge_index).to_dense()
    # # LAPL TOO BIG FOR TORCH EIG
    # _, eigvec = torch.linalg.eigh(lapl)

    # # USE LOBPCG
    vals, basis = torch.lobpcg(lapl, kept_modes, niter=-1)
    breakpoint()
    return vals, basis

  # move to cpu for memory
  init_data.cpu().detach()
  vals, basis = get_basis(init_data.x, init_data.edge_index, args.kept_modes)
  torch.save(vals, os.path.join(transform_path, "vals.pt"))
  for i, vec in enumerate(basis.transpose(0, 1)):
    save = os.path.join(transform_path, "vec_{:d}.pt".format(i))
    if os.path.exists(save):
      os.remove(save)
    torch.save(vec, save)

if debug:
  print('Init')

n_epochs = 10000
eps = 1e-15

# 3 DIM: X, Y, Z
# 5 OUT CHANNELS: RHO, Px, Py, Pz, E
model = LNO(
    3, init_data, args.channels, 5, args.kept_modes, device=device).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
sch = torch.optim.lr_scheduler.LinearLR(opt,1,1e-1,1000)
# sch = torch.optim.lr_scheduler.ExponentialLR(opt, args.decay)
# plat = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=1.25, patience = 20)
loss_fn = torch.nn.MSELoss()

save_path = os.path.join(data_path, "models_save", case_name,
                         today.strftime("%d%m%y"))
if not os.path.exists(save_path):
  os.makedirs(save_path)


def onera_transform(pos):
  # adjust x to move leading edge to x=0
  pos[:,0] = pos[:,0] - math.tan(math.pi/6)*pos[:,1]
  # scale chord to equal root
  # c(y) = r(1 - (1-taper)*(y/s))
  # r = c(y) / (1- (1-taper)*(y/s))
  pos = pos/(1-0.44*(pos[:,1:2]/1.1963))
  return pos

def interpolate(f, pos_x, pos_y):
  return knn_interpolate(f, onera_transform(pos_x), onera_transform(pos_y))


def train_step():
  model.train()
  loss = 0
  for pair_batch in train_loader:
    opt.zero_grad()
    pair_batch = pair_batch.to(device)
    x = interpolate(pair_batch.x_2, pair_batch.pos_2, pair_batch.pos_3)
    out = model(x, pair_batch.pos_3, pair_batch.y)

    batch_loss = loss_fn(out, pair_batch.x_3)
    batch_loss.backward()
    opt.step()

    loss += batch_loss
  loss /= batches
  return loss


def test_step():
  model.eval()
  with torch.no_grad():
    test_err = 0
    for pair_batch in test_loader:
      pair_batch = pair_batch.to(device)
      x = interpolate(pair_batch.x_2, pair_batch.pos_2, pair_batch.pos_3)
      out = model(x, pair_batch.pos_3, pair_batch.y)
      batch_loss = loss_fn(out, pair_batch.x_3)
      test_err += batch_loss
    test_err /= test_batches
  return test_err


def main(n_epochs):
  min_err = torch.inf
  save = os.path.join(save_path, "model_init")
  epl = os.path.join(save_path, "epl")
  # indices = train_dataset.items
  if debug:
    print('Train')
  for epoch in range(n_epochs):
    lr = sch._last_lr[0]
    # lr = args.learning_rate
    loss = train_step()
    test_err = test_step()
    if lr > 1e-5:
      sch.step()
    # plat.step(loss)

    if debug:
      print("Loss {:g}, Error {:g}, Epoch {:g}, LR {:g},".format(
          loss, test_err, epoch, lr))
    if epoch % 100 == 0 or epoch == n_epochs - 1:
      if wandb_upload:
        wandb.log({
            "Loss": loss,
            "Error": test_err,
            "Epoch": epoch,
        })
    if test_err < min_err or epoch == n_epochs - 1:
      min_err = test_err if test_err < min_err else min_err
      if epoch < n_epochs - 1 and epoch > 0:
        old_save = save
        os.remove(old_save)
      save = os.path.join(
          save_path,
          "model_ep-{:d}_L-{:g}_E-{:g}.pt".format(epoch, loss, test_err))
      torch.save(model.state_dict(), save)

      # if test_err == min_err:
      #   torch.save((edge_list, pos_list), epl + "_min.pt")
      # else:
      #   torch.save((edge_list, pos_list), epl + "_final.pt")


if __name__ == "__main__":
  if wandb_upload:
    import wandb
    case_name = "debug_" + case_name if debug else case_name
    wandb.init(
        project="DB-GNN",
        entity="wglao",
        name=case_name,
        settings=wandb.Settings(_disable_stats=True))
  main(n_epochs)