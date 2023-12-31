import os
import sys
import math
from datetime import date
import shutil
from functools import partial
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--case-name", default="dbsr-mods", type=str, help="Architecture Name"
)
parser.add_argument("--channels", default=10, type=int, help="Aggregation Channels")
# parser.add_argument(
#     "--latent-sz", default=10, type=int, help="Latent Space Dimensionality")
parser.add_argument("--layers", default=3, type=int, help="Number of Pooling Layers")
parser.add_argument("--omega", default=30.0, type=float, help="Pooling Ratio")
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning Rate")
parser.add_argument("--coarse", default=1, type=int, help="Coarse (1) or Fine (0)")
parser.add_argument("--wandb", default=0, type=bool, help="wandb upload")
parser.add_argument("--debug", default=0, type=bool, help="debug")
parser.add_argument("--gpu-id", default=0, type=int, help="GPU index")

args = parser.parse_args()
wandb_upload = bool(args.wandb)
debug = True if not wandb_upload else bool(args.debug)
today = date.today()
case_name = "_".join(
    [str(key) + "-" + str(value) for key, value in list(vars(args).items())[:-1]]
)[10:]
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = "cuda:{:d}".format(args.gpu_id)
import numpy as np
import torch
from torch_geometric import compile
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import knn_interpolate

from models_sr import DBGSR, ModSIRENSR
from graphdata import PairData, PairDataset

if debug:
    print("Load")

torch.manual_seed(0)

# loop through folders and load data
# ma_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
ma_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# re_list = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7]
re_list = [2e6, 3e6, 5e6, 6e6, 8e6, 9e6]
# aoa_list = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
aoa_list = [-9, -8, -7, -6, -5, -4, -3, 3, 4, 5, 6, 7, 8, 9]
n_slices = 5
if args.coarse:
    path_ext = "coarse"
else:
    path_ext = "fine"
data_path = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data", path_ext)


if wandb_upload:
    train_dataset = PairDataset(
        data_path, ma_list, re_list, aoa_list, "train_" + path_ext, n_slices
    )
    test_dataset = PairDataset(
        data_path, ma_list, [4e6, 7e6, 1e7], aoa_list, "test_" + path_ext, n_slices
    )
else:
    train_dataset = PairDataset(
        data_path, [0.3, 0.4], [3e6, 4e6], [3, 4], "idev-train_" + path_ext, n_slices
    )
    test_dataset = PairDataset(
        data_path, [0.5, 0.6], [5e6, 6e6], [5, 6], "idev-test_" + path_ext, n_slices
    )
    # train_dataset = PairDataset(data_path, [0.3], [3e6], [3], "recon3_" + path_ext, n_slices)
    # test_dataset = PairDataset(data_path, [0.3], [3e6], [3], "recon3_" + path_ext, n_slices)

n_samples = len(train_dataset)
batch_sz = int(np.min(np.array([1, n_samples])))
batches = -(n_samples // -batch_sz)
n_test = len(test_dataset)
test_sz = int(np.min(np.array([1, n_test])))
test_batches = -(n_test // -test_sz)

train_loader = DataLoader(train_dataset, batch_sz, follow_batch=["x_3", "x_2"])
test_loader = DataLoader(test_dataset, test_sz, follow_batch=["x_3", "x_2"])

init_pair = next(iter(test_loader))
init_data_2 = Data(init_pair.x_2, init_pair.edge_index_2, pos=init_pair.pos_2).to(
    device
)
init_data_3 = Data(init_pair.x_3, init_pair.edge_index_3, pos=init_pair.pos_3).to(
    device
)
if debug:
    print("Init")


def onera_transform(pos):
    # adjust x to move leading edge to x=0
    new_x = pos[:, 0] - math.tan(math.pi / 6) * pos[:, 1]
    pos = torch.cat((torch.unsqueeze(new_x, 1), pos[:, 1:]), 1)
    # scale chord to equal root
    # c(y) = r(1 - (1-taper)*(y/s))
    # r = c(y) / (1- (1-taper)*(y/s))
    pos = pos * (1 + (1 / 0.56 - 1) * (pos[:, 1:2] / 1.1963))
    return pos


def onera_interp(f, pos_x, pos_y, device: str = "cpu"):
    # in_idx = (pos_x[:, 1] < 1.1963, pos_y[:, 1] < 1.1963)
    # out_idx = (pos_x[:, 1] > 1.1963, pos_y[:, 1] > 1.1963)
    # inboard = knn_interpolate(f[in_idx[0]], onera_transform(pos_x[in_idx[0]]),
    #                           onera_transform(pos_y[in_idx[1]]))

    # outboard = knn_interpolate(f, pos_x, pos_y)[out_idx[1]]
    out = torch.where(
        (pos_y[:, 1] < 1.1963).unsqueeze(1).tile((1, f.size(1))),
        knn_interpolate(f, onera_transform(pos_x), onera_transform(pos_y)),
        knn_interpolate(f, pos_x, pos_y),
    )
    return out


# model = DBGSR(3, init_data_2, args.channels, device).to(device)
# x_init = onera_interp(init_data_2.x, init_data_2.pos, init_data_3.pos)
# model = ModSIRENSR(init_data_3, args.channels, args.layers, args.omega, device).to(device)
model = torch.jit.trace(
    ModSIRENSR(init_data_3, args.channels, args.layers, args.omega, device).to(device),
    (init_data_2.x, init_data_2.pos, init_data_3.pos),
)
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
sch = torch.optim.lr_scheduler.LinearLR(opt, 1, 1e-1, 1000)
loss_fn = torch.nn.MSELoss()

del init_pair, init_data_2, init_data_3

save_path = os.path.join(data_path, "models_save", case_name, today.strftime("%d%m%y"))
if not os.path.exists(save_path):
    os.makedirs(save_path)

n_epochs = 10000
eps = 1e-15


def train_step():
    model.train()
    loss = 0
    for pair_batch in train_loader:
        opt.zero_grad()
        pair_batch = pair_batch.to(device)
        # out = model(x_in, pair_batch.edge_index_3, pair_batch.pos_3)
        # out = model(pair_batch.x_2, pair_batch.edge_index_2,
        #           pair_batch.edge_index_3, pair_batch.pos_2, pair_batch.pos_3)
        # x_in = onera_interp(pair_batch.x_2, pair_batch.pos_2, pair_batch.pos_3)
        out = model(pair_batch.x_2, pair_batch.pos_2, pair_batch.pos_3)

        batch_loss = loss_fn(out, pair_batch.x_3)
        batch_loss.backward()
        opt.step()
        del out

        loss = loss + batch_loss / batches
    return loss


# train_jit = torch.jit.script(train_step)


def test_step():
    # model.eval()
    with torch.no_grad():
        test_err = 0
        for pair_batch in test_loader:
            pair_batch = pair_batch.to(device)
            # out = model(x_in, pair_batch.edge_index_3, pair_batch.pos_3)
            # out = model(pair_batch.x_2, pair_batch.edge_index_2,
            #           pair_batch.edge_index_3, pair_batch.pos_2, pair_batch.pos_3)
            # x_in = onera_interp(pair_batch.x_2, pair_batch.pos_2, pair_batch.pos_3)
            out = model(pair_batch.x_2, pair_batch.pos_2, pair_batch.pos_3)

            batch_loss = loss_fn(out, pair_batch.x_3)
            test_err = test_err + batch_loss / test_batches
        return test_err


# test_jit = torch.jit.script(test_step)


def main(n_epochs):
    min_err = torch.inf
    save = os.path.join(save_path, "model_init")
    # indices = train_dataset.items
    if debug:
        print("Train")
    for epoch in range(n_epochs):
        loss = train_step()
        test_err = test_step()
        # loss = train_jit()
        # test_err = test_jit()
        sch.step()

        if debug:
            print(
                "Loss: {:g}, Error {:g}, Epoch {:g}".format(
                    loss,
                    test_err,
                    epoch,
                    # args.omega,
                    # args.omega,
                    # model.mods1.omega.item(),
                    # model.mods2.sin0.omega,
                )
            )
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            if wandb_upload:
                wandb.log(
                    {
                        "Loss": loss,
                        "Error": test_err,
                        "Epoch": epoch,
                    }
                )
        if debug:
            if loss < min_err or epoch == n_epochs - 1:
                min_err = test_err if test_err < min_err else min_err
                if epoch < n_epochs - 1 and epoch > 0:
                    old_save = save
                    os.remove(old_save)
                save = os.path.join(
                    save_path,
                    "model_ep-{:d}_L-{:g}_E-{:g}.pt".format(epoch, loss, test_err),
                )
                torch.save(model.state_dict(), save)
        else:
            if test_err < min_err or epoch == n_epochs - 1:
                min_err = test_err if test_err < min_err else min_err
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

        wandb.init(
            project="DB-GNN",
            entity="wglao",
            name=case_name,
            settings=wandb.Settings(_disable_stats=True),
        )
    main(n_epochs)
