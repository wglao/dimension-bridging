import os
import sys
from datetime import date
import shutil
from functools import partial
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--case-name", default="dba-gcn-nbhp", type=str, help="Architecture Name")
parser.add_argument(
    "--channels", default=10, type=int, help="Aggregation Channels")
parser.add_argument(
    "--latent-sz", default=10, type=int, help="Latent Space Dimensionality")
parser.add_argument(
    "--pooling-layers", default=1, type=int, help="Number of Pooling Layers")
parser.add_argument(
    "--k-hops", default=1, type=int, help="Pooling Neighborhood Span Distance")
parser.add_argument(
    "--learning-rate", default=1e-3, type=float, help="Learning Rate")
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
# device = "cuda:{:d}".format(2*args.gpu_id) if args.gpu_id >= 0 else "cpu"
# device_2 = "cuda:{:d}".format(2*args.gpu_id + 1) if args.gpu_id >= 0 else "cpu"

import numpy as np
import torch
from torch_geometric import compile
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.pool import avg_pool_neighbor_x
import torch.nn.functional as F

from models_dbae import DBA, Encoder, StructureEncoder, Decoder
from graphdata import PairData, PairDataset

if debug:
  # torch.autograd.detect_anomaly(True)
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
  train_dataset = PairDataset(data_path, [0.3, 0.4], [3e6, 4e6], [3, 4],
                              "idev-train", n_slices)
  test_dataset = PairDataset(data_path, [0.5, 0.6], [5e6, 6e6], [5, 6],
                             "idev-test", n_slices)
  # train_dataset = PairDataset(data_path, [0.3], [3e6], [3], "recon", n_slices)
  # test_dataset = PairDataset(data_path, [0.3], [3e6], [3], "recon", n_slices)

n_samples = len(train_dataset)
batch_sz = int(np.min(np.array([1, n_samples])))
batches = -(n_samples // -batch_sz)
n_test = len(test_dataset)
test_sz = int(np.min(np.array([1, n_test])))
test_batches = -(n_test // -test_sz)

train_loader = DataLoader(train_dataset, batch_sz, follow_batch=['x_3', 'x_2'])
test_loader = DataLoader(test_dataset, test_sz, follow_batch=['x_3', 'x_2'])

init_data = next(iter(test_loader))
init_data = init_data[0].to(device)

k_hops = args.k_hops

if debug:
  print('Init')

n_epochs = 10000
eps = 1e-15

model = DBA(3, init_data, args.channels, args.latent_sz, args.pooling_layers,
            k_hops, device).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
sch = torch.optim.lr_scheduler.LinearLR(opt, 1, 1e-2, 1000)
# sch = torch.optim.lr_scheduler.ExponentialLR(opt,args.decay)
# plat = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5)
loss_fn = torch.nn.MSELoss()

save_path = os.path.join(data_path, "models_save", case_name,
                         today.strftime("%d%m%y"))
if not os.path.exists(save_path):
  os.makedirs(save_path)

lambda_2 = 0.1
lambda_3 = 0.1


def get_edge_attr(edge_index, pos):
  edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
  return edge_attr


def train_step():
  model.train()
  loss = 0
  err = 0
  for pair_batch in train_loader:
    opt.zero_grad()
    pair_batch = pair_batch.to(device)
    out, pool_edge_list_2, pool_edge_list_3, _, _, score_list_2, score_list_3 = model(
        pair_batch.x_3, pair_batch.edge_index_3, pair_batch.pos_3,
        pair_batch.x_2, pair_batch.edge_index_2, pair_batch.pos_2)

    data_loss = loss_fn(out, pair_batch.x_3)

    score_loss_2 = sum([
        F.selu(-get_edge_attr(edge_index_2, score_2).square(),
               inplace=True).sum()
        for score_2, edge_index_2 in zip(score_list_2, pool_edge_list_2[:-1])
    ])

    score_loss_3 = sum([
        F.selu(-get_edge_attr(edge_index_3, score_3).square(),
               inplace=True).sum()
        for score_3, edge_index_3 in zip(score_list_3, pool_edge_list_3[:-1])
    ])

    batch_loss = data_loss + lambda_2*score_loss_2 + lambda_3*score_loss_3

    batch_loss.backward()
    # print(batch_loss)
    # print(sum([score_2.max() - score_2.min() for score_2 in score_list_2]))
    # print(sum([score_3.max() - score_3.min() for score_3 in score_list_3]))

    opt.step()

    loss += batch_loss
    err += data_loss
  loss /= batches
  err /= batches
  return loss, err


def test_step():
  model.eval()
  with torch.no_grad():
    test_err = 0
    for pair_batch in train_loader:
      pair_batch = pair_batch.to(device)
      out, _, _ = model(pair_batch.x_3, pair_batch.edge_index_3,
                        pair_batch.pos_3, pair_batch.x_2,
                        pair_batch.edge_index_2, pair_batch.pos_2)
      batch_loss = loss_fn(out, pair_batch.x_3)
      test_err += batch_loss
    test_err /= test_batches
  return test_err


def main(n_epochs):
  min_err = torch.inf
  save = os.path.join(save_path, "model_init")
  # indices = train_dataset.items
  if debug:
    print('Train')
  for epoch in range(n_epochs):
    lr = sch._last_lr[0]
    loss, train_err = train_step()
    test_err = test_step()
    sch.step()

    if debug:
      print("Loss {:g}, Error {:g} / {:g}, Epoch {:g}, LR {:g},".format(
          loss, train_err, test_err, epoch, lr))
    if epoch % 10 == 0 or epoch == n_epochs - 1:
      if wandb_upload:
        wandb.log({
            "Loss": loss,
            "Train Error": train_err,
            "Test Error": test_err,
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