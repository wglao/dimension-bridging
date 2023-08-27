import os
import sys
import glob
import scipy.sparse as sp
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
  # train_dataset = PairDataset(data_path, [0.3, 0.4], [3e6, 4e6], [3, 4],
  #                             "idev-train", n_slices)
  # test_dataset = PairDataset(data_path, [0.5, 0.6], [5e6, 6e6], [5, 6],
  #                            "idev-test", n_slices)
  train_dataset = PairDataset(data_path, [0.3], [3e6], [3], "recon", n_slices)
  test_dataset = PairDataset(data_path, [0.3], [3e6], [3], "recon", n_slices)

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

pool_path = os.path.join(os.environ["SCRATCH"],
                         "ORNL/dimension-bridging/data/processed/pool")
if not os.path.exists(pool_path):
  os.makedirs(pool_path, exist_ok=True)

saved_pool = bool(
    len(
        glob.glob(
            os.path.join(pool_path,
                         "pool*{:d}layers.pt".format(args.pooling_layers)))))

if not saved_pool:
  with torch.no_grad():

    def get_deg(x, edge_index):
      deg = sp.csc_matrix((torch.ones((edge_index.size(1),)), edge_index),
                          (x.size(0), x.size(0))) @ torch.ones((x.size(0), 1))
      return deg

    def get_laplacian(x, edge_index):
      n_nodes = x.size(0)
      adj = sp.csc_matrix((torch.ones(edge_index.size(1),), edge_index),
                          (n_nodes, n_nodes))
      deg_idx = torch.stack((torch.arange(n_nodes), torch.arange(n_nodes)),
                            dim=0)
      deg = np.squeeze(get_deg(x, edge_index))
      sqrt_deg = sp.csc_matrix((1 / deg, deg_idx), (n_nodes, n_nodes))
      lapl = sp.csc_matrix((deg, deg_idx), (n_nodes, n_nodes)) - adj
      lapl_norm = (sp.eye(n_nodes) - sqrt_deg.dot(adj.dot(sqrt_deg)))
      return lapl, lapl_norm

    def get_basis(x, edge_index, kept_modes, lapl_norm=None):
      if lapl_norm is None:
        _, lapl_norm = get_laplacian(x, edge_index).to_dense()

      # # LAPL TOO BIG FOR TORCH EIG
      # _, eigvec = torch.linalg.eigh(lapl)

      # # USE LOBPCG
      vals, basis = torch.lobpcg(
          torch.tensor(lapl_norm.todense()), kept_modes, niter=-1)
      return vals, basis

    def get_edge_list(adj):
      """Get the edge list from a CSC sparse matrix.

      Parameters
      ----------
      adj : scipy.sparse.csc_matrix
          The CSC sparse matrix.

      Returns
      -------
      [2,m] tensor of edges (senders, receivers)
          The edge list.
      """

      rows = adj.indptr
      cols = adj.indices
      data = adj.data

      edges = []
      for i in range(len(rows) - 1):
        for j in range(rows[i], rows[i + 1]):
          edges.append((cols[j], data[j]))

      return torch.tensor(edges).transpose(0, 1).int()

    def decimation_pool(x, edge_index, pos):
      n_nodes = x.size(0)
      adj = sp.csc_matrix((torch.ones(edge_index.size(1),), edge_index),
                          (n_nodes, n_nodes))
      adj_list = [adj]
      pos_list = [pos]
      keep_list = []

      for l in range(args.pooling_layers):
        if debug:
          print("Layer {:d}".format(l + 1))
        lapl, lapl_norm = get_laplacian(x, edge_index)
        _, vec = get_basis(x, edge_index, 1, lapl_norm)
        z_vec = torch.where(vec >= 0, 1, -1)
        gamma = z_vec.transpose(0, 1) @ (lapl@z_vec) / 2*edge_index.size(1)
        # if gamma is worse than random cut
        if gamma < 0.5:
          # random cut with z in {-1,1}
          z_vec = torch.where(torch.randint(0, 1, z_vec.size()) > 0, 1, -1)
        keep_idx = torch.argwhere(z_vec.squeeze() > 0).squeeze().int()
        drop_idx = torch.argwhere(z_vec.squeeze() < 0).squeeze().int()
        keep_list.append(keep_idx)

        lapl_keep = lapl[np.ix_(keep_idx, keep_idx)]
        lapl_in_out = lapl[np.ix_(keep_idx, drop_idx)]
        lapl_out_in = lapl[np.ix_(drop_idx, keep_idx)]
        lapl_drop = lapl[np.ix_(drop_idx, drop_idx)]

        try:
          lapl_new = lapl_keep - lapl_in_out.dot(
              sp.linalg.spsolve(lapl_drop, lapl_out_in))
        except RuntimeError:
          # If lapl_drop is exactly singular, damp the inversion with
          # Marquardt-Levenberg coefficient ml_c
          ml_c = sp.csc_matrix(sp.eye(lapl_drop.shape[0])*1e-6)
          lapl_new = lapl_keep - lapl_in_out.dot(
              sp.linalg.spsolve(ml_c + lapl_drop, lapl_out_in))

        # Make the laplacian symmetric if it is almost symmetric
        if np.abs(lapl_new -
                  lapl_new.T).sum() < np.spacing(1)*np.abs(lapl_new).sum():
          lapl_new = (lapl_new + lapl_new.T) / 2.

        adj_new = -lapl_new
        adj_new.setdiag(0)
        adj_new.eliminate_zeros()
        adj_list.append(adj_new)

        pos = pos[keep_idx]
        pos_list.append(pos)

      for adj in adj_list[1:]:
        adj = adj*np.abs(adj) > 1e-1

      edge_index_list = [get_edge_list(adj) for adj in adj_list]
      return edge_index_list, pos_list, keep_list

    # move to cpu for memory
    init_data.cpu().detach()
    if debug:
      print("Pooling 2D")
    edge_index_list_2, pos_list_2, keep_list_2 = decimation_pool(
        init_data.x_2, init_data.edge_index_2, init_data.pos_2)
    if debug:
      print("Pooling 3D")
    edge_index_list_3, pos_list_3, keep_list_3 = decimation_pool(
        init_data.x_3, init_data.edge_index_3, init_data.pos_3)

    pool_structures = {
        "ei2": edge_index_list_2,
        "ei3": edge_index_list_3,
        "p2": pos_list_2,
        "p3": pos_list_3,
        "k2": keep_list_2,
        "k3": keep_list_3
    }

    torch.save(
        pool_structures,
        os.path.join(pool_path,
                     "pool_{:d}layers.pt".format(args.pooling_layers)))
else:
  if debug:
    print("Loading pooled graphs")
  pool_structures = torch.load(
      os.path.join(pool_path, "pool_{:d}layers.pt".format(args.pooling_layers)))

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
    out = model(pair_batch.x_3, pair_batch.edge_index_3, pair_batch.pos_3,
                pair_batch.x_2, pair_batch.edge_index_2, pair_batch.pos_2,
                [a.to(device) for a in pool_structures["ei2"]],
                [a.to(device) for a in pool_structures["ei3"]],
                [a.to(device) for a in pool_structures["p2"]],
                [a.to(device) for a in pool_structures["p3"]],
                [a.to(device) for a in pool_structures["k2"]],
                [a.to(device) for a in pool_structures["k3"]])

    data_loss = loss_fn(out, pair_batch.x_3)

    # score_loss_2 = sum([
    #     F.selu(-get_edge_attr(edge_index_2, score_2).square(),
    #            inplace=True).sum()
    #     for score_2, edge_index_2 in zip(score_list_2, pool_edge_list_2[:-1])
    # ])

    # score_loss_3 = sum([
    #     F.selu(-get_edge_attr(edge_index_3, score_3).square(),
    #            inplace=True).sum()
    #     for score_3, edge_index_3 in zip(score_list_3, pool_edge_list_3[:-1])
    # ])

    batch_loss = data_loss  # + lambda_2*score_loss_2 + lambda_3*score_loss_3

    batch_loss.backward()
    # print(batch_loss)
    # print(sum([score_2.max() - score_2.min() for score_2 in score_list_2]))
    # print(sum([score_3.max() - score_3.min() for score_3 in score_list_3]))

    opt.step()

    loss += batch_loss
    err += data_loss
    del pair_batch, out, data_loss, batch_loss
  loss /= batches
  err /= batches
  return loss, err


def test_step():
  model.eval()
  with torch.no_grad():
    test_err = 0
    for pair_batch in train_loader:
      pair_batch = pair_batch.to(device)
      out = model(pair_batch.x_3, pair_batch.edge_index_3,
                        pair_batch.pos_3, pair_batch.x_2,
                        pair_batch.edge_index_2, pair_batch.pos_2,
                        [a.to(device) for a in pool_structures["ei2"]],
                        [a.to(device) for a in pool_structures["ei3"]],
                        [a.to(device) for a in pool_structures["p2"]],
                        [a.to(device) for a in pool_structures["p3"]],
                        [a.to(device) for a in pool_structures["k2"]],
                        [a.to(device) for a in pool_structures["k3"]])
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