# import jax.experimental.sparse as jxs
# import jax.numpy as jnp

import os, shutil
from typing import List, Tuple, Union
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
import pyvista as pv
from vtk2ids import combine, connect, v2i
import torch


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_3":
            return torch.tensor([self.x_3.size(0)])
        if key == "edge_index_2":
            return torch.tensor([self.x_2.size(0)])
        return super().__inc__(key, value, *args, **kwargs)


class PairDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        ma_list,
        re_list,
        aoa_list,
        dset_name: str = "test",
        n_slices: int = 10,
        transform=None,
    ):
        self.ma_list = ma_list
        self.re_list = re_list
        self.aoa_list = aoa_list
        self.dset_name = dset_name
        self.n_slices = n_slices
        self.coarse_fine = "coarse" if "coarse" in self.dset_name else "fine"
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_names = []
        # for ma in ma_list:
        #   for re in re_list:
        #     for aoa in aoa_list:
        #       raw_names.append("ma-{:g}_re-{:g}_aoa-{:g}.raw".format(ma,re,aoa))
        #       for slc in range(self.n_slices):
        #         raw_names.append("ma-{:g}_re-{:g}_aoa-{:g}_slc-{:d}.raw".format(ma,re,aoa,slc))
        return raw_names

    @property
    def processed_file_names(self):
        return ["_".join([self.dset_name, "data.pt"])]

    def process(self):
        data_list = []
        mu = 0.0000178929762604
        gam = 1.4
        r_air = 287
        t_inf = 288.15
        c = np.sqrt(gam * r_air * t_inf)
        l = 0.64607
        for ma in self.ma_list:
            for re in self.re_list:
                for a in self.aoa_list:
                    data_path = os.path.join(
                        self.root,
                        self.coarse_fine,
                        "ma_{:g}/re_{:g}/a_{:g}".format(ma, re, a),
                    )

                    mesh = pv.read(os.path.join(data_path, "flow.vtu"))
                    # extract point data from coordinates and conservative fields
                    coords = np.array(mesh.points)
                    fields = ["Density", "Momentum", "Energy"]
                    point_data_list = [mesh.point_data.get_array(i) for i in fields]
                    point_data_list = [
                        np.expand_dims(a, 1) if i not in ["Momentum"] else a
                        for a, i in zip(point_data_list, fields)
                    ]
                    features_3 = np.concatenate(point_data_list, axis=1)

                    idx, _, _ = v2i(mesh)

                    graph_3 = Data(
                        torch.tensor(features_3, dtype=torch.float32),
                        torch.tensor(idx),
                        pos=torch.tensor(coords, dtype=torch.float32),
                    )

                    slice_idx = []
                    slice_data = []
                    slice_szs = []
                    slice_coords = []

                    for s in range(self.n_slices):
                        idx = int((s*10)//self.n_slices)
                        mesh = pv.read(
                            os.path.join(data_path, "slice_{:d}.vtk".format(idx))
                        )
                        slice_coords.append(np.array(mesh.points))

                        idx, _, sz = v2i(mesh)

                        point_data_list = [mesh.point_data.get_array(i) for i in fields]
                        point_data_list = [
                            np.expand_dims(a, 1) if i not in ["Momentum"] else a
                            for a, i in zip(point_data_list, fields)
                        ]
                        slice_data.append(np.concatenate(point_data_list, axis=1))

                        slice_idx.append(idx)

                        slice_szs.append(sz)

                    # add slice with freestream condition everywhere
                    slice_idx.append(slice_idx[-1])
                    slice_szs.append(slice_szs[-1])

                    fs_slice_coords = slice_coords[-1]
                    fs_slice_coords[:,1] = 1.2
                    slice_coords.append(fs_slice_coords)

                    u_inf = ma*c
                    rho_inf = mu * re / (u_inf * l)
                    p_inf = rho_inf*r_air*t_inf
                    nrg_inf = (p_inf / (gam-1) / rho_inf) + 0.5*u_inf**2

                    fs_data = rho_inf * np.array([1,u_inf,0,0,nrg_inf])
                    fs_data = np.tile(fs_data,(slice_szs[-1],1))

                    slice_data.append(fs_data)

                    idx, features_2, sz = combine(slice_idx, slice_data, slice_szs)
                    coords = np.concatenate(slice_coords, axis=0)

                    graph_2 = Data(
                        torch.tensor(features_2, dtype=torch.float32),
                        torch.tensor(idx),
                        pos=torch.tensor(coords, dtype=torch.float32),
                    )

                    data = PairData(
                        x_3=graph_3.x,
                        edge_index_3=graph_3.edge_index,
                        pos_3=graph_3.pos,
                        x_2=graph_2.x,
                        edge_index_2=graph_2.edge_index,
                        pos_2=graph_2.pos,
                        mare=torch.tensor([[ma,re,a]]),
                    )

                    data_list.append(data)

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    ma_list = [0.2]
    re_list = [1e6]
    aoa_list = [3, 6, 9]
    n_slices = 5
    root = os.path.join(os.environ["SCRATCH"], "ORNL/dimension-bridging/data")

    train_dataset = PairDataset(root, ma_list, re_list, aoa_list, n_slices)

    n_samples = len(ma_list) * len(re_list) * len(aoa_list)
    batch_sz = 2
    batches = -(n_samples // -batch_sz)

    with DataLoader(
        train_dataset, batch_sz, follow_batch=["x_3", "x_2"]
    ) as train_dataloader:
        print(next(iter(train_dataloader)))
