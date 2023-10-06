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
    "--case-name", default="dbed-siren", type=str, help="Architecture Name"
)
parser.add_argument("--channels", default=64, type=int, help="Aggregation Channels")
parser.add_argument(
    "--latent-sz", default=16, type=int, help="Latent Space Dimensionality"
)
parser.add_argument(
    "--siren-layers", default=1, type=int, help="Number of SIREN Layers"
)
parser.add_argument("--omega", default=1.0, type=float, help="Omega0 for SIREN Layers")
parser.add_argument(
    "--pooling-layers", default=1, type=int, help="Number of Pooling Layers"
)
parser.add_argument("--batch-sz", default=1, type=int, help="Batch Size")
parser.add_argument("--learning-rate", default=1e-4, type=float, help="Learning Rate")
parser.add_argument("--coarse", default=1, type=int, help="Coarse or Fine")
parser.add_argument("--slices", default=5, type=int, help="Number of Input 2D slices")
parser.add_argument("--wandb", default=0, type=int, help="wandb upload")
parser.add_argument("--debug", default=0, type=bool, help="debug prints")
parser.add_argument("--gpu-id", default=0, type=int, help="GPU index")

args = parser.parse_args()
wandb_upload = bool(args.wandb)
debug = True if not wandb_upload else args.debug
today = date.today()
case_name = "_".join(
    [str(key) + "-" + str(value) for key, value in list(vars(args).items())[:-2]]
)[10:]
device = "cuda:{:d}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
# device = "cuda:{:d}".format(2*args.gpu_id) if args.gpu_id >= 0 else "cpu"
# device_2 = "cuda:{:d}".format(2*args.gpu_id + 1) if args.gpu_id >= 0 else "cpu"

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn.pool import avg_pool_neighbor_x
import torch.nn.functional as F

from models_dbed import DBED
from graphdata import PairData, PairDataset
# from pool_and_part import pnp

if debug:
    # torch.autograd.detect_anomaly(True)
    print("Load")

torch.manual_seed(0)

# loop through folders and load data
# ma_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
ma_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# re_list = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
re_list = [2e6, 3e6, 5e6, 6e6, 8e6, 9e6]
# aoa_list = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
aoa_list = [-9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9]
n_slices = args.slices
if args.coarse:
    coarse_fine = "coarse"
else:
    coarse_fine = "fine"
data_path = os.path.join(
    os.environ["SCRATCH"], "ORNL/dimension-bridging/data"
)
if wandb_upload:
    train_dataset = PairDataset(
        data_path, ma_list, re_list, aoa_list, "train_" + coarse_fine + "_{:d}s".format(n_slices), n_slices
    )
    test_dataset = PairDataset(
        data_path, ma_list, [4e6, 7e6, 1e7], aoa_list, "test_" + coarse_fine + "_{:d}s".format(n_slices), n_slices
    )
else:
    train_dataset = PairDataset(
        data_path, [0.3, 0.4], [3e6, 4e6], [3, 4], "idev-train_" + coarse_fine + "_{:d}s".format(n_slices), n_slices
    )
    test_dataset = PairDataset(
        data_path, [0.5, 0.6], [5e6, 6e6], [5, 6], "idev-test_" + coarse_fine + "_{:d}s".format(n_slices), n_slices
    )
    # train_dataset = PairDataset(data_path, [0.3], [3e6], [3], "recon3_" + coarse_fine + "_{:d}s".format(n_slices), n_slices)
    # test_dataset = PairDataset(data_path, [0.3], [3e6], [3], "recon3_" + coarse_fine + "_{:d}s".format(n_slices), n_slices)


n_samples = len(train_dataset)
batch_sz = int(np.min(np.array([args.batch_sz, n_samples])))
batches = -(n_samples // -batch_sz)
n_test = len(test_dataset)
test_sz = int(np.min(np.array([args.batch_sz, n_test])))
test_batches = -(n_test // -test_sz)

train_loader = DataLoader(train_dataset, batch_sz, follow_batch=["x_3", "x_2"])
test_loader = DataLoader(test_dataset, test_sz, follow_batch=["x_3", "x_2"])

init_data = next(iter(test_loader))
init_data = init_data[0].to(device)

pool_path = os.path.join(
    os.environ["SCRATCH"], "ORNL/dimension-bridging/data/processed/pool"
)
if not os.path.exists(pool_path):
    os.makedirs(pool_path, exist_ok=True)

saved_pool = bool(
    len(
        glob.glob(
            os.path.join(
                pool_path,
                "pool_{:d}layers_".format(args.pooling_layers) + coarse_fine + ".pt",
            )
        )
    )
)

if not saved_pool:
    # pool_structures = pnp(init_data)
    print("NO POOL SAVED")
else:
    if debug:
        print("Loading pooled graphs")
    pool_structures = torch.load(
        os.path.join(
            pool_path,
            "pool_{:d}layers_".format(args.pooling_layers) + coarse_fine + ".pt",
        )
    )
    for key in pool_structures.keys():
        item = pool_structures[key]
        if type(item) == torch.Tensor:
            pool_structures[key] = item.to(device)
        elif type(item) == list:
            pool_structures[key] = [i.to(device) for i in item]

if debug:
    print("Init")

n_epochs = 10000
eps = 1e-15

with torch.no_grad():
    init_data = init_data.to(device)
    # get scaling values
    x_min = torch.zeros_like(init_data.x_2[0]).to(device)
    x_max = torch.zeros_like(init_data.x_2[0]).to(device)

    for d in iter(train_loader):
        x = d.x_2.to(device)
        x_min = torch.where(x.min(dim=0).values < x_min, x.min(dim=0).values, x_min).to(device)
        x_max = torch.where(x.max(dim=0).values > x_max, x.max(dim=0).values, x_max).to(device)

    x_init = 2 * (init_data.x_2 - x_min) / (x_max - x_min) - 1

    # # outputs linearly dependent
    out_szs = [5,]
    # # outputs linearly independent
    # out_szs = [1,1,1,1,1]

    model = torch.jit.trace(
        DBED(
            init_data,
            args.channels,
            args.latent_sz,
            out_szs,
            args.siren_layers,
            args.pooling_layers,
            args.omega,
            device,
        ).to(device),
        (x_init, init_data.edge_index_2, init_data.pos_2, pool_structures),
    )
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
sch = torch.optim.lr_scheduler.LinearLR(opt, 1, 1e-2, 1000)
opt_l = torch.optim.LBFGS(model.parameters(), lr=0.01, line_search_fn="strong_wolfe")
sch_l = torch.optim.lr_scheduler.LinearLR(opt_l, 1, 1e-2, 1000)
# sch = torch.optim.lr_scheduler.ExponentialLR(opt,args.decay)
# plat = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5)
loss_fn = torch.nn.MSELoss()

save_path = os.path.join(data_path, "models_save", case_name, today.strftime("%d%m%y"))
if not os.path.exists(save_path):
    os.makedirs(save_path)


def get_edge_attr(edge_index, pos):
    edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
    return edge_attr


def train_step(opt, opt_l, lbfgs:bool = False):
    model.train()
    loss = 0
    err = 0
    for pair_batch in train_loader:
        pair_batch = pair_batch.to(device)
        if not lbfgs:
            opt.zero_grad()

            x_in = 2 * (pair_batch.x_2 - x_min) / (x_max - x_min) - 1
            out = model(
                x_in,
                pair_batch.edge_index_2,
                pair_batch.pos_2,
                pool_structures,
            )

            # in case of additional loss terms
            data_loss = loss_fn(out, pair_batch.x_3)
            batch_loss = data_loss  # + lambda_2*score_loss_2 + lambda_3*score_loss_3
            batch_loss.backward()
            opt.step()

            loss += batch_loss
            err += data_loss
            del pair_batch, out, data_loss, batch_loss
        else:
            params = model.state_dict()
            def closure():
                if torch.is_grad_enabled():
                    opt_l.zero_grad()

                x_in = 2 * (pair_batch.x_2 - x_min) / (x_max - x_min) - 1
                out = model(
                    x_in,
                    pair_batch.edge_index_2,
                    pair_batch.pos_2,
                    pool_structures,
                )

                # in case of additional loss terms
                batch_loss = loss_fn(out, pair_batch.x_3)
                # batch_loss = data_loss  # + lambda_2*score_loss_2 + lambda_3*score_loss_3
                if batch_loss.requires_grad:
                    batch_loss.backward()
                return batch_loss
            
            with torch.no_grad():
                batch_loss = closure()
                loss += batch_loss
                err += batch_loss
            
            opt_l.step(closure)
            for param_name, param in model.state_dict().items():
                if torch.isnan(param).any():
                    model.load_state_dict(params)
                    opt_l = torch.optim.LBFGS(model.parameters(), lr=0.01)
                    return train_step(lbfgs=False)

            x_in = 2 * (pair_batch.x_2 - x_min) / (x_max - x_min) - 1
            out = model(
                x_in,
                pair_batch.edge_index_2,
                pair_batch.pos_2,
                pool_structures,
            )

            # in case of additional loss terms
            data_loss = loss_fn(out, pair_batch.x_3)
            batch_loss = data_loss  # + lambda_2*score_loss_2 + lambda_3*score_loss_3
            batch_loss.backward()
            opt.step()
            del pair_batch, batch_loss

    loss /= batches
    err /= batches
    return loss, err, opt, opt_l


def test_step():
    model.eval()
    with torch.no_grad():
        test_err = 0
        for pair_batch in train_loader:
            pair_batch = pair_batch.to(device)
            x_in = 2 * (pair_batch.x_2 - x_min) / (x_max - x_min) - 1
            out = model(
                x_in,
                pair_batch.edge_index_2,
                pair_batch.pos_2,
                pool_structures,
            )
            batch_loss = loss_fn(out, pair_batch.x_3)
            test_err += batch_loss
        test_err /= test_batches
    return test_err


def main(n_epochs, opt, opt_l):
    min_err = torch.inf
    save = os.path.join(save_path, "model_init")
    lbfgs = False
    first = True
    if debug:
        print("Train")
    for epoch in range(n_epochs):
        # lr = sch._last_lr[0]
        loss, train_err, opt, opt_l = train_step(opt, opt_l, lbfgs)
        lbfgs = loss < 0.1
            
        test_err = test_step()
        sch.step()
        if lbfgs:
            if not first:
                sch_l.step()
            else:
                first = False

        if debug:
            print(
                "Loss {:g}, Error {:g} / {:g}, Epoch {:g},".format(
                    loss, train_err, test_err, epoch
                )
            )
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            if wandb_upload:
                wandb.log(
                    {
                        "Loss": loss,
                        "Train Error": train_err,
                        "Test Error": test_err,
                        "Epoch": epoch,
                    }
                )
        if test_err < min_err or epoch == n_epochs - 1:
            if test_err < min_err:
                min_err = test_err
            if epoch < n_epochs - 1 and epoch > 0:
                old_save = save
                os.remove(old_save)
            save = os.path.join(
                save_path,
                "model_ep-{:d}_L-{:g}_E-{:g}.pt".format(epoch, loss, test_err),
            )
            torch.save(model.state_dict(), save)


if __name__ == "__main__":
    if wandb_upload:
        import wandb

        case_name = "debug_" + case_name if debug else case_name
        wandb.init(
            project="DB-GNN",
            entity="wglao",
            name=case_name,
            settings=wandb.Settings(_disable_stats=True),
        )
    main(n_epochs, opt, opt_l)
