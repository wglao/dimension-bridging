import os
import sys
from datetime import date
import shutil
from functools import partial
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--case-name", default="dba-gin-sagp", type=str, help="Architecture Name")
parser.add_argument(
    "--channels", default=10, type=int, help="Aggregation Channels")
parser.add_argument(
    "--latent-sz", default=10, type=int, help="Latent Space Dimensionality")
parser.add_argument(
    "--pooling-layers", default=1, type=int, help="Number of Pooling Layers")
parser.add_argument(
    "--pooling-ratio", default=0.125, type=float, help="Pooling Ratio")
parser.add_argument(
    "--learning-rate", default=1e-3, type=float, help="Learning Rate")
parser.add_argument("--wandb", default=0, type=int, help="wandb upload")
parser.add_argument('--gpu-id', default=0, type=int, help="GPU index")

args = parser.parse_args()
wandb_upload = bool(args.wandb)
today = date.today()
case_name = "_".join([
    str(key) + "-" + str(value) for key, value in list(vars(args).items())[:-1]
])[10:]
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

import numpy as np
import torch
from torch_geometric import compile
from torch_geometric.loader import DataLoader

from models_gin_sagp import DBA
from graphdata import PairData, PairDataset

if not wandb_upload:
  print('Load')

torch.manual_seed(0)

# loop through folders and load data
# ma_list = [0.2, 0.35, 0.5, 0.65, 0.8]
ma_list = [0.2, 0.35, 0.5] if wandb_upload else [0.2]
# ma_list = [0.2]
# re_list = [1e6, 2e6, 5e6, 1e7, 2e7]
re_list = [1e6, 2e6, 1e7, 2e7] if wandb_upload else [1e6, 2e6]
# re_list = [1e6]
# aoa_list = [0, 3, 6, 9, 12]
aoa_list = [3, 6, 9, 12] if wandb_upload else [3, 6]
# aoa_list = [3]
n_slices = 5
data_path = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data")

if wandb_upload:
  train_dataset = PairDataset(data_path, ma_list, re_list, aoa_list, "train", n_slices)
  test_dataset = PairDataset(data_path, ma_list, [5e6], aoa_list, "test", n_slices)
else:
  train_dataset = PairDataset(data_path, ma_list, re_list, aoa_list, "idev-train", n_slices)
  test_dataset = PairDataset(data_path, ma_list, [5e6], aoa_list, "idev-test", n_slices)

n_samples = len(ma_list)*len(re_list)*len(aoa_list)
batch_sz = int(np.min(np.array([1, n_samples])))
batches = -(n_samples // -batch_sz)
n_test = len(ma_list)*1*len(aoa_list)
test_sz = int(np.min(np.array([1, n_test])))
test_batches = -(n_test // -test_sz)

train_loader = DataLoader(train_dataset, batch_sz, follow_batch=['x_3', 'x_2'])
test_loader = DataLoader(test_dataset, test_sz, follow_batch=['x_3', 'x_2'])

init_data = next(iter(test_loader))
init_data = init_data[0].cuda()

# set kernel size to mean of node degree vector
idx = init_data.edge_index_3
sz = init_data.x_3.shape[0]
con = torch.ones((init_data.num_edges,)).cuda()
adj = torch.sparse_coo_tensor(idx, con, (sz, sz))
deg = adj.matmul(torch.ones((sz, 1)).cuda())
k_size = int(torch.ceil(torch.mean(deg) / 3))
del idx, sz, adj, deg

# pr<0.8 for memory
# pool_ratio = 0.125
pool_ratio = args.pooling_ratio

if not wandb_upload:
  print('Init')

model = DBA(3, init_data, args.channels, args.latent_sz, k_size,
            args.pooling_layers, pool_ratio).cuda()
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
loss_fn = torch.nn.MSELoss()

save_path = os.path.join(data_path, "models_save", case_name,
                         today.strftime("%d%m%y"))
if not os.path.exists(save_path):
  os.makedirs(save_path)

n_epochs = 10000
eps = 1e-15


def train_step():
  model.train()
  loss = 0
  for pair_batch in train_loader:
    pair_batch = pair_batch.cuda()
    out, edge_list, pos_list = model(
        pair_batch.x_3, pair_batch.edge_index_3, pair_batch.pos_3,
        pair_batch.x_2, pair_batch.edge_index_2, pair_batch.pos_2)

    batch_loss = loss_fn(out, pair_batch.x_3)
    batch_loss.backward()
    opt.step()

    loss = loss + batch_loss/batches
  return loss, edge_list, pos_list


def test_step(edge_list, pos_list):
  model.eval()
  test_err = 0
  for pair_batch in test_loader:
    pair_batch = pair_batch.cuda()
    out = model(None, None, None, pair_batch.x_2,
                          pair_batch.edge_index_2, pair_batch.pos_2, edge_list,
                          pos_list)
    batch_loss = loss_fn(out, pair_batch.x_3)
    test_err = test_err + batch_loss/test_batches
  return test_err


def main(n_epochs):
  min_err = torch.inf
  save = os.path.join(save_path, "model_init")
  epl = os.path.join(save_path, "epl")
  # indices = train_dataset.items
  for epoch in range(n_epochs):
    if not wandb_upload:
      print('Train')
    loss, edge_list, pos_list = train_step()

    if not wandb_upload:
      print('Test')
    test_err = test_step(edge_list, pos_list)

    if not wandb_upload:
      print("Loss: {:g}, Error {:g}, Epoch {:g}".format(loss, test_err, epoch))
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
      if test_err == min_err:
        torch.save((edge_list, pos_list), epl + "_min.pt")
      else:
        torch.save((edge_list, pos_list), epl + "_final.pt")


if __name__ == "__main__":
  if wandb_upload:
    import wandb
    wandb.init(
        project="DB-GNN",
        entity="wglao",
        name=case_name,
        settings=wandb.Settings(_disable_stats=True))
  main(n_epochs)