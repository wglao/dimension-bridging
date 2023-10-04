import torch
import math
from functools import partial
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from torch import nn
from torch import vmap
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import (
    SAGEConv,
    SAGPooling,
    GATv2Conv,
    GCNConv,
    GraphConv,
    GINConv,
    Linear,
)
from torch_geometric.nn import MessagePassing, knn_interpolate
from torch_geometric.nn import Sequential as GeoSequential
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.nn.pool import max_pool, avg_pool_neighbor_x, max_pool_neighbor_x
from torch_geometric.utils import softmax, add_self_loops, add_remaining_self_loops
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.nn.inits import zeros
from torch.nn import Parameter
import torch.nn.functional as F
from graphdata import PairData
import numpy as np
from typing import NamedTuple


def get_pooled_sz(full_sz: int, ratio: float, layer: int):
    out_sz = full_sz
    for l in range(layer):
        out_sz = int(-(out_sz // -(1 / ratio)))
    return out_sz


def get_deg(x, edge_index, device: str = "cpu"):
    deg = torch.sparse_coo_tensor(
        edge_index, torch.ones((edge_index.size(1),)).to(device)
    ).to(device) @ torch.ones((x.size(0), 1)).to(device)
    return deg


def get_edge_attr(edge_index, pos):
    edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
    return edge_attr


def get_edge_aug(edge_index, pos, steps: int = 1, device: str = "cpu"):
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
    edge_attr_aug = get_edge_attr(edge_index_aug, pos)
    return edge_index_aug, edge_attr_aug

    # edge_index = add_remaining_self_loops(edge_index)[0].unique(dim=1)

    # # assume symmetric graph
    # get_edge_aug = torch.cat(torch.vmap(lambda n: (edge_index[0]==n) & (edge_index[1]>n)))


def onera_transform(pos):
    # adjust x to move leading edge to x=0
    new_x = pos[:, 0] - math.tan(math.pi / 6) * pos[:, 1]
    pos = torch.cat((torch.unsqueeze(new_x, 1), pos[:, 1:]), 1)
    # scale chord to equal root
    # c(y) = r(1 - (1-taper)*(y/s))
    # r = c(y) / (1- (1-taper)*(y/s))
    pos = pos * (1 + (1 / 0.56 - 1) * (pos[:, 1:2] / 1.1963))
    return pos


def onera_interp(f, pos_x, pos_y):
    out = torch.where(
        (pos_y[:, 1] < 1.1963).unsqueeze(1).tile((1, f.size(1))),
        knn_interpolate(f, onera_transform(pos_x), onera_transform(pos_y)),
        knn_interpolate(f, pos_x, pos_y),
    )
    return out


class KernelMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        device: str = "cpu",
    ) -> None:
        super(KernelMLP, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.device = device

        # in size is [n, dim + 1] for channel number
        self.lin0 = Linear(dim + 1, hidden_channels, weight_initializer="glorot").to(
            device
        )
        self.lin1 = Linear(
            hidden_channels, hidden_channels, weight_initializer="glorot"
        ).to(device)
        self.lin2 = Linear(
            hidden_channels, in_channels, weight_initializer="glorot"
        ).to(device)
        self.omega=0.1

        self.reset_parameters()

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, rel_pos, channel: int = 0):
        # # Convert to spherical [rho, theta, phi] = [r, az, elev]
        rho = torch.norm(rel_pos, dim=1)
        theta = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        phi = torch.asin(rel_pos[:, 2] / rho)
        theta = torch.where(phi.isnan(), torch.zeros_like(theta), theta)
        phi = torch.where(phi.isnan(), torch.zeros_like(phi), phi)

        rel_pos = torch.stack((rho, theta / torch.pi, phi / torch.pi), dim=1)
        out = torch.sin(self.omega*
            self.lin0(
                torch.cat((rel_pos, torch.full_like(rho, channel).unsqueeze(1)), 1)
            )
        )
        out = torch.sin(self.omega*self.lin1(out))
        out = self.lin2(out)
        # out = out.reshape((self.out_channels, out.size(0), self.in_channels))
        return out


class GraphKernelConv(MessagePassing):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        k_net: nn.Module = KernelMLP,
        device: str = "cpu",
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(GraphKernelConv, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.device = device
        self.k_net = k_net(dim, in_channels, hidden_channels, out_channels, device).to(
            device
        )
        # self.omega=0.1
        # self.lin0 = Linear(in_channels, out_channels).to(device)
        # self.lin1 = Linear(hidden_channels, out_channels).to(device)

        self.bias = Parameter(torch.Tensor(1, out_channels)).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.k_net = self.k_net.to(self.k_net.device)
        self.k_net.reset_parameters()
        # self.lin0.reset_parameters()
        # self.lin1.reset_parameters()
        self.bias = self.bias.to(self.device)
        zeros(self.bias)
        # self._cached_edge_index = None

    def forward(self, x, edge_index, pos):
        # x = F.selu(self.lin0(x), inplace=True)
        edge_index, _ = add_remaining_self_loops(edge_index)
        rel_pos = get_edge_attr(edge_index, pos)

        out = self.propagate(edge_index, x=x, edge_weight=rel_pos, size=None)
        out = out.squeeze() + self.bias
        return out

    def message_calc(self, x_j: Tensor, rel_pos: Tensor, channel: int = 0):
        msg = self.k_net(rel_pos, channel) * x_j
        return msg.sum(1).to(self.device)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            msg = x_j
        else:
            msg = []
            for i in range(self.out_channels):
                msg.append(self.message_calc(x_j, edge_weight, i))
        return torch.stack(msg, 1)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class NeighborhoodPool(nn.Module):
    def __init__(
        self,
        dim: int,
        channels: int,
        k_hops: int = 1,
        gnn: nn.Module = GraphKernelConv,
        device: str = "cpu",
        **kwargs
    ) -> None:
        super(NeighborhoodPool, self).__init__()
        self.dim = dim
        self.k_hops = k_hops
        self.gnn1 = gnn(
            dim,
            in_channels=dim,
            hidden_channels=channels,
            out_channels=channels,
            device=device,
            **kwargs
        ).to(device)
        self.gnn2 = gnn(
            dim,
            in_channels=channels,
            hidden_channels=channels,
            out_channels=1,
            device=device,
            **kwargs
        ).to(device)
        self.device = device

    def forward(self, x, edge_index, pos):
        edge_aug = add_remaining_self_loops(
            get_edge_aug(edge_index, pos, self.k_hops, self.device)[0]
        )[0]
        score = F.selu(self.gnn1(pos, edge_aug, pos))
        score = self.gnn2(score, edge_aug, pos)

        new_order = torch.squeeze(torch.argsort(score, 0))
        old_order = torch.squeeze(torch.argsort(new_order, 0))
        order = new_order

        # reorder nodes
        # x = x[new_order]
        # pos = pos[new_order]
        # for new, old in zip(new_order, old_order):
        #   tmp = torch.argwhere(edge_aug == old)
        #   edge_aug = torch.where(tmp, torch.full_like(edge_aug, new), edge_aug)
        #   edge_pool = torch.where(tmp, torch.full_like(edge_pool, new), edge_pool)

        # remove edges and cluster
        x_pool = None
        pos_pool = None
        n_mask_0 = torch.ones((x.size(0),)).bool().to(self.device)

        cluster = torch.zeros((x.size(0),)).to(self.device)
        n = 0
        while True:
            node = order[0]
            e_mask_1 = edge_aug[0] == node

            n_mask_1 = torch.zeros((x.size(0),)).bool().to(self.device)
            n_mask_1[edge_aug[1, e_mask_1]] = 1
            n_mask_0 = n_mask_0 & ~n_mask_1

            cluster[n_mask_1] = n

            if x_pool is None:
                x_pool = torch.unsqueeze(torch.max(x[n_mask_1], dim=0).values, 0)
                pos_pool = torch.unsqueeze(pos[node], 0)
            else:
                x_pool = torch.cat(
                    (x_pool, torch.unsqueeze(torch.max(x[n_mask_1], dim=0).values, 0))
                )
                pos_pool = torch.cat((pos_pool, torch.unsqueeze(pos[node], dim=0)))

            edge_aug = edge_aug[
                :,
                ~(edge_aug.ravel().unsqueeze(1) == n_mask_1.argwhere().squeeze())
                .sum(1)
                .reshape(edge_aug.shape)
                .sum(0)
                .bool(),
            ]
            order = new_order[n_mask_0[new_order]]

            n += 1

            if order.size(0) == 0:
                break

        edge_attr = get_edge_attr(edge_index, cluster)
        edge_index = edge_index[:, edge_attr.bool()]
        edge_pool = cluster[edge_index]
        edge_pool = edge_pool.unique(dim=1).int()

        if self.training:
            return x_pool, edge_pool, pos_pool, score
        return x_pool, edge_pool, pos_pool


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        init_data: Data,
        hidden_channels: int,
        latent_channels: int,
        n_pools: int,
        k_hops: int,
        device: str = "cpu",
    ):
        super(Encoder, self).__init__()
        self.dim = dim
        self.in_channels = init_data.num_node_features
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.n_pools = n_pools
        self.k_hops = k_hops
        self.device = device
        self.omega = 0.01

        # initial aggr
        self.lin0 = Linear(self.in_channels, hidden_channels).to(self.device)
        self.lin1 = Linear(hidden_channels, latent_channels).to(self.device)

        self.conv_list = nn.ModuleList(
            [
                GraphKernelConv(
                    dim,
                    hidden_channels + dim,
                    hidden_channels,
                    hidden_channels,
                    device=device,
                ).to(self.device)
            ]
        )
        # self.conv_list.append(
        #     GraphKernelConv(
        #         dim,
        #         hidden_channels + dim,
        #         hidden_channels,
        #         hidden_channels,
        #         device=device).to(self.device))

        # pools
        self.pool_list = nn.ModuleList()
        for _ in range(n_pools):
            # MY NEIGHBORHOOD POOL
            # self.pool_list.append(
            #     NeighborhoodPool(
            #         dim,
            #         hidden_channels,
            #         self.k_hops,
            #         device=device,
            #     ).to(device))

            self.conv_list.append(
                GraphKernelConv(
                    dim,
                    hidden_channels + dim,
                    hidden_channels,
                    hidden_channels,
                    device=device,
                ).to(self.device)
            )

        # latent
        # out_sz = get_pooled_sz(init_data.num_nodes, k_hops, n_pools)
        # self.conv_list.append(
        #     GraphKernelConv(
        #         dim,
        #         hidden_channels + dim,
        #         hidden_channels,
        #         hidden_channels,
        #         device=device).to(self.device))

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()

        reset_list = [mod.to(mod.device) for mod in self.conv_list]
        self.conv_list = nn.ModuleList(reset_list)
        for conv in self.conv_list:
            conv.reset_parameters()

    def max_pool(self, x, edge_index, keep_idx):
        data = Data(x, edge_index)
        data = max_pool_neighbor_x(data)
        out = data.x[keep_idx]
        return out

    def forward(
        self,
        x,
        edge_index,
        pos,
        pool_edge_index=None,
        pool_pos=None,
        pool_keep_idx=None,
    ):
        x = torch.sin(self.omega*self.lin0(x))
        x = torch.sin(self.omega*(self.conv_list[0](torch.cat((x, pos), dim=1), edge_index, pos)))

        # for l, pool in enumerate(self.pool_list):
        for l in range(self.n_pools):
            keep_idx = pool_keep_idx[l]
            x = self.max_pool(x, edge_index, keep_idx)
            edge_index = pool_edge_index[l + 1]
            pos = pool_pos[l + 1]

            x = torch.sin(self.omega*
                (self.conv_list[l + 1](torch.cat((x, pos), dim=1), edge_index, pos))
            )

        x = torch.sin(self.omega*self.lin1(x))
        return x


class StructureEncoder(Encoder):
    def __init__(
        self,
        dim: int,
        init_data: Data,
        hidden_channels: int,
        latent_channels: int,
        n_pools: int,
        k_hops: int,
        device: str = "cpu",
    ):
        super(StructureEncoder, self).__init__(
            dim, init_data, hidden_channels, latent_channels, n_pools, k_hops, device
        )
        self.in_channels = 1

        # pools
        self.pool_list = nn.ModuleList()
        for _ in range(n_pools):
            self.pool_list.append(
                NeighborhoodPool(
                    dim,
                    hidden_channels,
                    self.k_hops,
                    device=device,
                ).to(device)
            )

    def forward(self, x, edge_index, pos):
        pool_edge_list = [edge_index]
        pool_pos_list = [pos]
        if self.training:
            score_list = []
        for l, pool in enumerate(self.pool_list):
            # # MY NEIGHBORHOOD POOL
            if self.training:
                x, edge_index, pos, score = pool(x, edge_index, pos)
            else:
                x, edge_index, pos = pool(x, edge_index, pos)

            pool_edge_list.insert(0, edge_index)
            pool_pos_list.insert(0, pos)
            if self.training:
                score_list.insert(0, score)

        if self.training:
            return pool_edge_list, pool_pos_list, score_list
        return pool_edge_list, pool_pos_list


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        init_data: Data,
        hidden_channels: int,
        latent_channels: int,
        n_pools: int,
        k_hops: int,
        interpolate: callable = onera_interp,
        device: str = "cpu",
    ):
        super(Decoder, self).__init__()
        self.dim = dim
        self.out_channels = init_data.num_node_features
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.n_pools = n_pools
        self.k_hops = k_hops
        self.interpolate = interpolate
        self.device = device
        self.omega = 0.01

        # latent dense map
        # self.out_sz = get_pooled_sz(init_data.num_nodes, k_hops, n_pools)
        self.lin0 = Linear(latent_channels, hidden_channels).to(self.device)
        self.lin1 = Linear(hidden_channels, self.out_channels).to(self.device)

        # initial aggr
        self.conv_list = nn.ModuleList(
            [
                GraphKernelConv(
                    dim,
                    hidden_channels + dim,
                    hidden_channels,
                    hidden_channels,
                    device=device,
                ).to(self.device)
            ]
        )

        # self.conv_list.append(
        #     GraphKernelConv(
        #         dim,
        #         hidden_channels + dim,
        #         hidden_channels,
        #         hidden_channels,
        #         device=device).to(self.device))

        # # no initial aggr
        # self.conv_list = nn.ModuleList()

        # unpools
        for _ in range(n_pools):
            self.conv_list.append(
                GraphKernelConv(
                    dim,
                    hidden_channels + dim,
                    hidden_channels,
                    hidden_channels,
                    device=device,
                ).to(self.device)
            )

        # self.conv_list.append(
        #     GraphKernelConv(
        #         dim,
        #         hidden_channels + dim,
        #         hidden_channels,
        #         hidden_channels,
        #         device=device).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin0 = self.lin0.to(self.device)
        self.lin0.reset_parameters()
        self.lin1 = self.lin1.to(self.device)
        self.lin1.reset_parameters()

        reset_list = [mod.to(mod.device) for mod in self.conv_list]
        self.conv_list = nn.ModuleList(reset_list)
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, latent, edge_index_list, pos_list_scale, pos_list):
        # INITIAL AGG
        x = torch.sin(self.omega*(self.lin0(latent)))
        edge_index = edge_index_list[0]
        pos = pos_list_scale[0]
        x = torch.sin(self.omega*
            (self.conv_list[0](torch.cat((x, pos), dim=1), edge_index, pos)))
        
        for l in range(self.n_pools):
            # deg = get_deg(x, edge_index, self.device)
            x = self.interpolate(x, pos_list[l], pos_list[l + 1])
            edge_index = edge_index_list[l + 1]
            pos = pos_list_scale[l + 1]

            x = torch.sin(self.omega*
                (self.conv_list[l](torch.cat((x, pos), dim=1), edge_index, pos)))

        out = self.lin1(x)
        # if out.isnan().sum():
        #   breakpoint()
        return out


class DBA(nn.Module):
    def __init__(
        self,
        dim: int,
        init_data: PairData,
        hidden_channels: int,
        latent_channels: int,
        n_pools: int,
        k_hops: int,
        device: str = "cpu",
    ):
        super(DBA, self).__init__()
        self.dim = dim
        self.in_channels = init_data.num_node_features
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.n_pools = n_pools
        self.k_hops = k_hops
        self.device = device

        init_data_3 = Data(
            x=init_data.x_3, edge_index=init_data.edge_index_3, pos=init_data.pos_3
        )

        # used for model eval
        init_data_2 = Data(
            x=init_data.x_2, edge_index=init_data.edge_index_2, pos=init_data.pos_2
        )
        self.encoder2D = Encoder(
            dim, init_data_2, hidden_channels, latent_channels, n_pools, k_hops, device
        ).to(self.device)
        self.decoder = Decoder(
            dim,
            init_data_3,
            hidden_channels,
            latent_channels,
            n_pools + 1,
            k_hops,
            device=device,
        ).to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder2D = self.encoder2D.to(self.encoder2D.device)
        self.encoder2D.reset_parameters()

        self.decoder = self.decoder.to(self.decoder.device)
        self.decoder.reset_parameters()

    def forward(
        self,
        x_2,
        edge_index_2,
        pos_2,
        pos_3,
        pool_edge_list_2=None,
        pool_edge_list_3=None,
        pool_pos_list_2=None,
        pool_pos_list_3=None,
        pool_keep_list_2=None,
    ):
        # scale data
        x_2 = (x_2 - x_2.min(dim=0).values) / (
            x_2.max(dim=0).values - x_2.min(dim=0).values
        )
        pos_2 = (pos_2 - pos_2.min(dim=0).values) / (
            pos_2.max(dim=0).values - pos_2.min(dim=0).values
        )
        pos_3 = (pos_3 - pos_3.min(dim=0).values) / (
            pos_3.max(dim=0).values - pos_3.min(dim=0).values
        )
        pool_pos_list_2_scaled = [
            (pos - pos.min(dim=0).values)
            / (pos.max(dim=0).values - pos.min(dim=0).values)
            for pos in pool_pos_list_2
        ]
        pool_pos_list_3_scaled = [
            (pos - pos.min(dim=0).values)
            / (pos.max(dim=0).values - pos.min(dim=0).values)
            for pos in pool_pos_list_3
        ]

        latent_2 = self.encoder2D(
            x_2,
            edge_index_2,
            pos_2,
            pool_edge_list_2,
            pool_pos_list_2_scaled,
            pool_keep_list_2,
        )

        pool_edge_list_3.append(pool_edge_list_2[-1])
        pool_edge_list_3 = [a for a in reversed(pool_edge_list_3)]

        pool_pos_list_3_scaled.append(pool_pos_list_2_scaled[-1])
        pool_pos_list_3_scaled = [a for a in reversed(pool_pos_list_3_scaled)]

        pool_pos_list_3.append(pool_pos_list_2[-1])
        pool_pos_list_3 = [a for a in reversed(pool_pos_list_3)]

        out = self.decoder(latent_2, pool_edge_list_3, pool_pos_list_3_scaled, pool_pos_list_3)
        return out
