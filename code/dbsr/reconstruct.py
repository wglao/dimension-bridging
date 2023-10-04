import os, sys, shutil, glob
from datetime import date
import pyvista as pv
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import knn_interpolate
import numpy as np
from models_sr import DBMGN, DBGSR, get_deg, ModSIRENSR, MAReModSIREN
from graphdata import PairDataset

today = date.today()

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--case-name", default="mare-mods", type=str, help="Architecture Name"
)
parser.add_argument("--channels", default=128, type=int, help="Aggregation Channels")
parser.add_argument("--latent-sz", default=8, type=int, help="Latent Size")
parser.add_argument("--layers", default=1, type=int, help="Aggregation Channels")
parser.add_argument("--omega", default=11.0, type=float, help="Initial frequence")
parser.add_argument("--learning-rate", default=1e-5, type=float, help="Learning Rate")
parser.add_argument("--coarse", default=1, type=int, help="Coarse (1) or Fine (0)")
parser.add_argument("--wandb", default=1, type=int, help="wandb upload")
parser.add_argument("--debug", default=0, type=int, help="debug printing")
parser.add_argument("--gpu-id", default=0, type=int, help="GPU index")
parser.add_argument("--mach", default=0.7, type=float, help="Mach Number")
parser.add_argument("--reynolds", default=7e6, type=float, help="Reynolds Number")
parser.add_argument("--aoa", default=7.0, type=float, help="Angle of Attack")
parser.add_argument("--date", default="280923", type=str, help="Date of run in ddmmyy")
parser.add_argument("--epoch", default=4815, type=int, help="Checkpoint Epoch")

args = parser.parse_args()
case_name = "_".join(
    [str(key) + "-" + str(value) for key, value in list(vars(args).items())[:-6]]
)[10:]
device = "cuda:{:d}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
if args.coarse:
    coarse_fine = "coarse"
else:
    coarse_fine = "fine"
data_path = os.path.join(
    os.environ["SCRATCH"], "ORNL/dimension-bridging/data", coarse_fine
)
save_path = os.path.join(data_path, "models_save")


def main(save_path):
    n_slices = 5
    recon_dataset = PairDataset(
        data_path,
        [args.mach],
        [args.reynolds],
        [args.aoa],
        "recon7_" + coarse_fine,
        n_slices,
    )
    # recon_dataset = PairDataset(data_path, [args.mach], [args.reynolds],
    #                             [args.aoa], "train", n_slices)
    #   recon_dataset = PairDataset(data_path, [args.mach], [args.reynolds],
    #                               [args.aoa], "test", n_slices)
    #   recon_dataset = PairDataset(data_path, [args.mach], [args.reynolds],
    #                               [args.aoa], "all", n_slices)
    recon_loader = DataLoader(recon_dataset)

    pair = next(iter(recon_loader))[0].to(device)
    init_data = Data(pair.x_3, pair.edge_index_3, pos=pair.pos_3)

    model = MAReModSIREN(
        1,
        args.latent_sz,
        args.omega,
        init_data,
        args.channels,
        5,
        args.layers,
        args.omega,
        device,
    ).to(device)

    # get save
    run_path = os.path.join(save_path, case_name, args.date)
    save = glob.glob(os.path.join(run_path, "model_ep-{:d}*.pt".format(args.epoch)))
    save = save[0] if save != [] else None
    if save is not None:
        print("Model found.")
        state_dict = torch.load(save, device)
        model.load_state_dict(state_dict)
    else:
        print(
            "Model with requested architecture {:s} and epoch {:d} not saved.\nExiting.".format(
                case_name, args.epoch
            )
        )
        return

    model.eval()
    with torch.no_grad():
        mare_mat = torch.ones((pair.x_3.size(0), pair.mare.size(1))).to(device) * pair.mare
        f_recon = model(mare_mat, pair.pos_3)
        mse_rho = torch.nn.MSELoss()(f_recon[:, 0], pair.x_3[:, 0])
        mse_u = torch.nn.MSELoss()(f_recon[:, 1:4], pair.x_3[:, 1:4])
        mse_e = torch.nn.MSELoss()(f_recon[:, 4], pair.x_3[:, 4])
        print(
            "Reconstruction MSE: {:g} (density), {:g} (velocity), {:g} (energy)".format(
                mse_rho, mse_u, mse_e
            )
        )

        abs_rho = torch.abs(f_recon[:, 0] - pair.x_3[:, 0])
        abs_u = torch.abs(f_recon[:, 1:4] - pair.x_3[:, 1:4])
        abs_e = torch.abs(f_recon[:, 4] - pair.x_3[:, 4])

        rel_rho = torch.abs(
            abs_rho
            / torch.where(
                torch.abs(pair.x_3[:, 0]) > 1e-15,
                pair.x_3[:, 0],
                torch.sign(pair.x_3[:, 0]) * 1e-15,
            )
        )
        rel_u = torch.abs(
            abs_u
            / torch.where(
                torch.abs(pair.x_3[:, 1:4]) > 1e-15,
                pair.x_3[:, 1:4],
                torch.sign(pair.x_3[:, 1:4]) * 1e-15,
            )
        )
        rel_e = torch.abs(
            abs_e
            / torch.where(
                torch.abs(pair.x_3[:, 4]) > 1e-15,
                pair.x_3[:, 4],
                torch.sign(pair.x_3[:, 3]) * 1e-15,
            )
        )

        mesh = pv.read(
            os.path.join(
                data_path,
                "ma_{:g}/re_{:g}/a_{:g}".format(args.mach, args.reynolds, args.aoa),
                "flow.vtu",
            )
        )

        for recon, abs_arr, rel_arr, field in zip(
            [f_recon[:, 0], f_recon[:, 1:4], f_recon[:, 4]],
            [abs_rho, abs_u, abs_e],
            [rel_rho, rel_u, rel_e],
            ["Density", "Momentum", "Energy"],
        ):
            #   for field in ["Density"]:
            mesh.point_data.set_array(
                recon.squeeze().cpu().detach().numpy(), field + "_Recon"
            )
            mesh.point_data.set_array(
                abs_arr.squeeze().cpu().detach().numpy(), field + "_Abs_Err"
            )
            mesh.point_data.set_array(
                rel_arr.squeeze().cpu().detach().numpy(), field + "_Rel_Err"
            )

        save_path = os.path.join(
            data_path,
            "ma_{:g}/re_{:g}/a_{:g}".format(args.mach, args.reynolds, args.aoa),
            "reconstruct.vtu",
        )
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
    mesh_recon = main(save_path)
    plot(mesh_recon)
