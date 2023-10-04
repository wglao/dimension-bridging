import os
import sys
import glob
import scipy.sparse as sp
from datetime import date
import shutil
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--part", default=0, type=int, help="Partition (1) or Pool Only (0)")
parser.add_argument(
    "--pooling-layers", default=1, type=int, help="Number of Pooling Layers"
)
parser.add_argument(
    "--k-hops", default=3, type=int, help="Pooling Neighborhood Span Distance"
)
parser.add_argument("--coarse", default=1, type=int, help="Coarse or Fine")
parser.add_argument("--wandb", default=0, type=int, help="wandb upload")
parser.add_argument("--debug", default=0, type=bool, help="debug prints")
parser.add_argument("--gpu-id", default=0, type=int, help="GPU index")

args = parser.parse_args()
wandb_upload = bool(args.wandb)
debug = True if not wandb_upload else args.debug
part = bool(args.part)
today = date.today()
case_name = "_".join(
    [str(key) + "-" + str(value) for key, value in list(vars(args).items())[:-2]]
)[10:]
device = "cuda:{:d}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
# device = "cuda:{:d}".format(2*args.gpu_id) if args.gpu_id >= 0 else "cpu"
# device_2 = "cuda:{:d}".format(2*args.gpu_id + 1) if args.gpu_id >= 0 else "cpu"

import numpy as np
import torch
from torch_geometric import compile
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn.pool import avg_pool_neighbor_x
import torch.nn.functional as F

from models_dbae import DBA, Encoder, StructureEncoder, Decoder
from graphdata import PairData, PairDataset

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

if args.coarse:
    coarse_fine = "_coarse"
else:
    coarse_fine = "_fine"

test_dataset = PairDataset(
    data_path, [0.3], [3e6], [3], "recon3" + coarse_fine, n_slices
)

test_sz = 1
test_loader = DataLoader(test_dataset, test_sz, follow_batch=["x_3", "x_2"])

init_data = next(iter(test_loader))
init_data = init_data[0].to(device)

k_hops = args.k_hops

pool_path = os.path.join(
    os.environ["SCRATCH"], "ORNL/dimension-bridging/data/processed/pool"
)
if not os.path.exists(pool_path):
    os.makedirs(pool_path, exist_ok=True)

with torch.no_grad():

    def get_edge_aug(edge_index, steps: int = 1, device: str = "cpu"):
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(
                edge_index.size(1),
            ).to(device),
        )
        adj_aug = adj
        if steps >= 1:
            for _ in range(steps - 1):
                adj_aug = (adj_aug @ adj).coalesce()
            adj_aug = (adj + adj_aug).coalesce()
        edge_index_aug = adj_aug.indices()
        return edge_index_aug

    def get_deg(x, edge_index):
        deg = sp.csc_matrix(
            (torch.ones((edge_index.size(1),)), edge_index), (x.size(0), x.size(0))
        ) @ torch.ones((x.size(0), 1))
        return deg

    def get_laplacian(x, edge_index):
        n_nodes = x.size(0)
        adj = sp.csc_matrix(
            (
                torch.ones(
                    edge_index.size(1),
                ),
                edge_index,
            ),
            (n_nodes, n_nodes),
        )
        deg_idx = torch.stack((torch.arange(n_nodes), torch.arange(n_nodes)), dim=0)
        deg = np.squeeze(get_deg(x, edge_index))
        sqrt_deg = sp.csc_matrix((1 / deg, deg_idx), (n_nodes, n_nodes))
        lapl = sp.csc_matrix((deg, deg_idx), (n_nodes, n_nodes)) - adj
        lapl_norm = sp.eye(n_nodes) - sqrt_deg.dot(adj.dot(sqrt_deg))
        return lapl, lapl_norm

    def get_basis(x, edge_index, kept_modes, lapl_norm=None):
        if lapl_norm is None:
            _, lapl_norm = get_laplacian(x, edge_index).to_dense()

        # # LAPL TOO BIG FOR TORCH EIG
        # _, eigvec = torch.linalg.eigh(lapl)

        # # USE LOBPCG
        vals, basis = torch.lobpcg(
            torch.tensor(lapl_norm.todense()), kept_modes, niter=-1
        )
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
                edges.append((i, cols[j]))

        return torch.tensor(edges).transpose(0, 1).int()

    def decimation_pool(x, edge_index, pos):
        n_nodes = x.size(0)
        adj = sp.csc_matrix(
            (
                torch.ones(
                    edge_index.size(1),
                ),
                edge_index,
            ),
            (n_nodes, n_nodes),
        )
        adj_list = [adj]
        pos_list = [pos]
        keep_list = []

        for l in range(args.pooling_layers):
            if debug:
                print("Layer {:d}".format(l + 1))
            lapl, lapl_norm = get_laplacian(x, edge_index)
            _, vec = get_basis(x, edge_index, 1, lapl_norm)
            z_vec = torch.where(vec >= 0, 1, -1)
            gamma = z_vec.transpose(0, 1) @ (lapl @ z_vec) / 2 * edge_index.size(1)
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
                    sp.linalg.spsolve(lapl_drop, lapl_out_in)
                )
            except RuntimeError:
                # If lapl_drop is exactly singular, damp the inversion with
                # Marquardt-Levenberg coefficient ml_c
                ml_c = sp.csc_matrix(sp.eye(lapl_drop.shape[0]) * 1e-6)
                lapl_new = lapl_keep - lapl_in_out.dot(
                    sp.linalg.spsolve(ml_c + lapl_drop, lapl_out_in)
                )

            # Make the laplacian symmetric if it is almost symmetric
            if (
                np.abs(lapl_new - lapl_new.T).sum()
                < np.spacing(1) * np.abs(lapl_new).sum()
            ):
                lapl_new = (lapl_new + lapl_new.T) / 2.0

            adj_new = -lapl_new
            adj_new.setdiag(0)
            adj_new.eliminate_zeros()
            adj_list.append(adj_new)

            pos = pos[keep_idx]
            pos_list.append(pos)

        sp_list = [adj_list[0]]
        for adj in adj_list[1:]:
            adj = (adj * np.abs(adj)) > 1e-1
            sp_list.append(adj)

        edge_index_list = [get_edge_list(adj) for adj in sp_list]
        return edge_index_list, pos_list, keep_list

    def partition(
        x, edge_index, pos, ghost: bool = True, maxit: int = 10, tol: float = 0.01
    ):
        it = 0
        err = 1
        y_cut = 2
        while it < maxit and err > tol:
            parts = torch.zeros((x.size(0)))
            parts = torch.where(pos[:, 1] > y_cut, parts + 2, parts)
            parts = torch.where((pos[:, 2]) > 0, parts + 1, parts)

            part_graphs = []
            if not ghost:
                part_idx = parts[edge_index]
                for i in range(4):
                    x_i = x[parts == i]
                    edge_index_i = edge_index[
                        :, (part_idx[0] == i) & (part_idx[1] == i)
                    ]
                    # renumber edges
                    old_idx = edge_index_i.unique()
                    re_idx = torch.zeros((old_idx.max()))
                    re_idx[old_idx] = torch.arange(old_idx.size(0))
                    edge_index_i = re_idx[edge_index_i]
                    pos_i = pos[parts == i]
                    part_graphs.append(
                        Data(x_i, edge_index_i, pos=pos_i, old_idx=old_idx)
                    )
            else:
                edge_index = get_edge_aug(edge_index, args.k_hops)
                part_idx = parts[edge_index]
                for i in range(4):
                    x_i = x[parts == i]
                    edge_index_i = edge_index[
                        :, (part_idx[0] == i) & (part_idx[1] == i)
                    ]
                    # renumber edges
                    old_idx = edge_index_i.unique()
                    re_idx = torch.zeros((old_idx.max()+1)).int()
                    re_idx[old_idx] = torch.arange(old_idx.size(0)).int()
                    edge_index_i = re_idx[edge_index_i]
                    pos_i = pos[parts == i]
                    part_graphs.append(
                        Data(x_i, edge_index_i, pos=pos_i, old_idx=old_idx)
                    )
            it += 1
            edge_counts = torch.tensor([pg.edge_index.size(1) for pg in part_graphs])
            wing_denser = (
                True
                if (torch.argmax(edge_counts) == 0) or (torch.argmax(edge_counts) == 1)
                else False
            )
            if wing_denser:
                y_cut -= 1 / (it + 1)
            else:
                y_cut += 1 / (it + 1)

            err = torch.abs(edge_counts.max() - edge_counts.min()).float() / edge_counts.float().mean()
        return part_graphs

    
    def pool(data):
        # move to cpu for memory
        data = data.cpu().detach()
        if debug:
            print("Pooling 2D")
        edge_index_list_2, pos_list_2, keep_list_2 = decimation_pool(
            data.x_2, data.edge_index_2, data.pos_2
        )
        if debug:
            print("Pooling 3D")
        edge_index_list_3, pos_list_3, keep_list_3 = decimation_pool(
            data.x_3, data.edge_index_3, data.pos_3
        )

        pool_structures = {
            "ei2": edge_index_list_2,
            "ei3": edge_index_list_3,
            "p2": pos_list_2,
            "p3": pos_list_3,
            "k2": keep_list_2,
            "k3": keep_list_3,
        }

        torch.save(
            pool_structures,
            os.path.join(
                pool_path,
                "pool_{:d}layers".format(args.pooling_layers) + coarse_fine + ".pt",
            ),
        )
        return pool_structures


    def pnp(data, pool_stuctures=None):
      # move to cpu for memory
      data = data.cpu().detach()
    
      if pool_structures is None:
        pool_structures = pool(data)

      partitions = []
      for edge, pos in zip(pool_structures["ei3"], pool_structures["p3"]):
          partitions.append(partition(data.x_3, edge, pos))


      torch.save(
          pool_structures,
          os.path.join(
              pool_path,
              "pnp_{:d}layers".format(args.pooling_layers) + coarse_fine + ".pt",
          ),
      )

      return pool_structures
        

if __name__ == "__main__":
    if part:
        pnp(init_data)
    else:
        pool(init_data)