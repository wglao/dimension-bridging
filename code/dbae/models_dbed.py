import math
import torch
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


def onera_interp(f, pos_x, pos_y, k: int = 1):
    out = torch.where(
        (pos_y[:, 1] < 1.1963).unsqueeze(1).tile((1, f.size(1))),
        knn_interpolate(f, onera_transform(pos_x), onera_transform(pos_y), k=k),
        knn_interpolate(f, pos_x, pos_y, k=k),
    )
    return out


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega=30):
        super().__init__()
        self.omega = omega

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.omega = omega

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)

    def reset_parameters(self):
        self.init_weights()

    def forward(self, input):
        return torch.sin(self.omega * self.linear(input))


class ModulateMLP(nn.Module):
    def __init__(self, in_sz, hidden_sz, layers, omega, device: str = "cpu"):
        super().__init__()
        self.in_sz = in_sz
        self.hidden_sz = hidden_sz
        self.layers = layers
        self.omega = omega
        self.device = device

        # self.sin0 = SineLayer(in_sz, hidden_sz, omega=omega).to(device)
        # self.sin_list = nn.ModuleList()
        self.lin_list = nn.ModuleList()
        for l in range(layers):
            # self.sin_list.append(SineLayer(hidden_sz, hidden_sz, omega=omega).to(device))
            self.lin_list.append(Linear(hidden_sz, hidden_sz).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        # self.sin0.reset_parameters()
        # for s in self.sin_list:
        #     s.reset_parameters()
        for l in self.lin_list:
            l.reset_parameters()

    def forward(self, z):
        # z = self.sin0(z)
        out = []

        # for s in self.sin_list:
        #     z = s(z)
        #     out.append(z)
        for l in self.lin_list:
            z = F.selu(l(z))
            out.append(z)
        if len(out) > 1:
            return torch.stack(out, 1).to(self.device)
        return out[0]


class KernelMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        in_sz: int,
        hidden_sz: int,
        layers: int,
        out_sz: int,
        omega: float = 30,
        device: str = "cpu",
    ) -> None:
        super(KernelMLP, self).__init__()
        self.dim = dim
        self.in_sz = in_sz
        self.hidden_sz = hidden_sz
        self.layers = layers
        self.out_sz = out_sz
        self.omega = omega
        self.device = device

        self.sin0 = SineLayer(dim + 1, hidden_sz, omega=omega)
        self.sin_list = nn.ModuleList([])
        for l in range(layers):
            self.sin_list.append(SineLayer(hidden_sz, hidden_sz, omega=omega))
        self.lin = Linear(hidden_sz, out_sz)

        self.reset_parameters()

    def reset_parameters(self):
        self.sin0.reset_parameters()
        for s in self.sin_list:
            s.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, rel_pos, channel: int = 0):
        # # Convert to spherical [rho, theta, phi] = [r, az, elev]
        rho = torch.norm(rel_pos, dim=1).to(self.device)
        theta = torch.atan2(rel_pos[:, 1], rel_pos[:, 0]).to(self.device)
        phi = torch.asin(rel_pos[:, 2] / rho).to(self.device)
        theta = torch.where(phi.isnan(), torch.zeros_like(theta), theta)
        phi = torch.where(phi.isnan(), torch.zeros_like(phi), phi)

        # # scale [rho, theta, phi] to [-1,1]^3
        # graphs are static, so can scale here instead of outside the loop
        rho = (
            2
            * (rho - rho.min(dim=0).values)
            / (rho.max(dim=0).values - rho.min(dim=0).values)
            - 1
        )
        theta = theta / torch.pi
        phi = 2 * phi / torch.pi
        rel_pos = torch.stack(
            (rho, theta, phi, torch.full_like(rho, 2 * channel / self.in_sz - 1)), dim=1
        )
        out = self.sin0(rel_pos)
        for s in self.sin_list:
            out = s(out)
        out = self.lin(out)
        return out


class GraphKernelConv(MessagePassing):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        k_net: nn.Module = KernelMLP,
        k_net_layers: int = 1,
        omega: float = 30,
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
        self.omega = omega
        self.k_net = k_net(
            dim,
            in_channels,
            hidden_channels,
            k_net_layers,
            out_channels,
            omega=omega,
            device=device,
        ).to(device)

        self.bias = Parameter(torch.Tensor(1, out_channels)).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.k_net.reset_parameters()
        self.bias = self.bias.to(self.device)
        zeros(self.bias)

    def forward(self, x, edge_index, pos):
        # x = F.selu(self.lin0(x), inplace=True)
        edge_index, _ = add_remaining_self_loops(edge_index)
        rel_pos = get_edge_attr(edge_index, pos)
        out = self.propagate(edge_index, x=x, edge_weight=rel_pos, size=None)
        if len(out.shape) > 2:
            out = out.squeeze(2)
        out += self.bias
        return out

    def message_calc(self, x_j: Tensor, rel_pos: Tensor, channel: int = 0):
        msg = self.k_net(rel_pos, channel) * x_j
        return msg.sum(1)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            msg = x_j
        else:
            if self.out_channels > 1:
                msg = []
                for i in range(self.out_channels):
                    msg.append(self.message_calc(x_j, edge_weight, i))
                return torch.stack(msg, 1).to(self.device)
            return self.message_calc(x_j, edge_weight, 1).unsqueeze(1)

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
        in_channels: int,
        hidden_channels: int,
        knet_width: int,
        latent_channels: int,
        n_pools: int,
        omega: float,
        # k_net_layers = k_net_layers,
        device: str = "cpu",
    ):
        super(Encoder, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.knet_width = knet_width
        self.latent_channels = latent_channels
        self.n_pools = n_pools
        self.omega = omega
        # self.k_net_layers = k_net_layers
        self.device = device

        # initial aggr
        self.sin0 = SineLayer(self.in_channels, hidden_channels, omega=omega).to(
            self.device
        )

        self.conv0 = GraphKernelConv(
            dim,
            hidden_channels,
            knet_width,
            hidden_channels,
            # k_net = KernelMLP,
            # k_net_layers = k_net_layers,
            omega=omega,
            device=device,
        ).to(self.device)

        self.conv_list = nn.ModuleList([])
        # pools
        self.pool_list = nn.ModuleList()
        # self.conv_list.append(
        #         GraphKernelConv(
        #             dim,
        #             hidden_channels,
        #             hidden_channels,
        #             hidden_channels,
        #             # k_net = KernelMLP,
        #             # k_net_layers = k_net_layers,
        #             omega=omega,
        #             device=device,
        #         ).to(self.device)
        #     )
        for _ in range(n_pools):
            self.conv_list.append(
                GraphKernelConv(
                    dim,
                    hidden_channels,
                    knet_width,
                    hidden_channels,
                    # k_net = KernelMLP,
                    # k_net_layers = k_net_layers,
                    omega=omega,
                    device=device,
                ).to(self.device)
            )

        # latent
        # self.sin1 = SineLayer(hidden_channels, latent_channels, omega=omega)
        self.lin1 = Linear(hidden_channels, latent_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.sin0.reset_parameters()
        # self.sin1.reset_parameters()
        self.lin1.reset_parameters()

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
        pool_edge_index,
        pool_pos,
        pool_keep_idx,
    ):
        x = self.sin0(x)
        x = torch.sin(self.omega * (self.conv0(x, edge_index, pos)))

        # for l, pool in enumerate(self.pool_list):
        # for l in range(self.n_pools):
        for l, conv in enumerate(self.conv_list):
            keep_idx = pool_keep_idx[l]
            x = self.max_pool(x, edge_index, keep_idx)
            edge_index = pool_edge_index[l + 1]
            pos = pool_pos[l + 1]
            x = torch.sin(self.omega * (conv(x, edge_index, pos)))

        x = self.lin1(x)
        x = x.max(dim=0).values
        return x


class ModSIREN(nn.Module):
    def __init__(
        self,
        dim,
        in_sz,
        hidden_sz,
        out_sz: int,
        layers,
        omega: float = 30.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.in_sz = in_sz
        self.hidden_sz = hidden_sz
        # self.out_sz = init_data.x.size(1)
        self.out_szs = out_sz
        self.layers = layers
        self.omega = omega
        self.device = device

        self.mod = ModulateMLP(self.in_sz, hidden_sz, layers, omega, device).to(device)
        self.sin0 = SineLayer(self.dim, hidden_sz, omega=omega).to(device)
        self.sin_list = nn.ModuleList([])
        for l in range(layers):
            self.sin_list.append(
                SineLayer(hidden_sz, hidden_sz, omega=omega).to(device)
            )
        self.out_lin = Linear(hidden_sz, out_sz).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.mod.reset_parameters()
        self.sin0.reset_parameters()
        for sin in self.sin_list:
            sin.reset_parameters()
        self.out_lin.reset_parameters()

    def forward(self, z, pos):
        alphas = self.mod(z)
        out = self.sin0(pos)
        for l, s in enumerate(self.sin_list):
            out = alphas[:, l] * s(out)
        out = self.out_lin(out)
        return out


class DBED(nn.Module):
    def __init__(
        self,
        channels,
        hidden_sz,
        latent_sz,
        out_szs,
        layers,
        n_pools,
        omega: float = 30.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = 3
        self.in_sz = 5
        self.channels = channels
        self.hidden_sz = hidden_sz
        self.latent_sz = latent_sz
        self.out_szs = out_szs
        self.layers = layers
        self.n_pools = n_pools
        self.omega = omega
        self.device = device

        # self.enc = Encoder(self.dim,self.in_sz,hidden_sz,latent_sz,n_pools,omega,device)
        self.enc = Encoder(
            self.dim, self.in_sz, channels, channels * 2, latent_sz, 1, omega, device
        )
        self.dec = ModSIREN(
            self.dim, latent_sz, hidden_sz, out_szs, layers, omega, device
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.dec.reset_parameters()

    def forward(self, x, edge_index, pos, pool_structures):
        pos_3 = pool_structures["p3"][0]

        # process 2D slice first
        z = self.enc(
            x,
            edge_index,
            pos,
            pool_structures["ei2"],
            pool_structures["p2"],
            pool_structures["k2"],
        )

        # z = onera_interp(z, pool_structures["p2"][-1], pos_3)
        z = z*torch.ones((pos_3.size(0),self.latent_sz)).to(self.device)

        # for part in parts:
        #     z[part.old_idx] = knn_interpolate(z,pos,part.pos,k=1)

        # upsample before processing
        # x_in = torch.zeros((pos_3.size(0),x.size(1)))
        # z = torch.zeros((pos_3.size(0),self.latent_sz)).to(self.device)
        # for part, pool in zip(parts, pnp):
        #     x_in = onera_interp(x,pos,part.pos,3)
        #     z[part.old_idx] = self.enc(
        #         x_in,
        #         part.edge_index,
        #         part.pos,
        #         pool["ei"],
        #         pool["p"],
        #         pool["k"]
        #     )
        pos_in = (
            2
            * (pos_3 - pos_3.min(dim=0).values)
            / (pos_3.max(dim=0).values - pos_3.min(dim=0).values)
            - 1
        )
        x = self.dec(z, pos_in)
        return x


class DBEDParts(nn.Module):
    def __init__(
        self,
        channels,
        hidden_sz,
        latent_sz,
        out_szs,
        layers,
        n_pools,
        omega: float = 30.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = 3
        self.in_sz = 5
        self.channels = channels
        self.hidden_sz = hidden_sz
        self.latent_sz = latent_sz
        self.out_szs = out_szs
        self.layers = layers
        self.n_pools = n_pools
        self.omega = omega
        self.device = device

        # self.enc = Encoder(self.dim,self.in_sz,hidden_sz,latent_sz,n_pools,omega,device)
        self.enc = Encoder(
            self.dim, self.in_sz, channels, channels * 2, latent_sz, 1, omega, device
        )
        self.dec = ModSIREN(
            self.dim, latent_sz, hidden_sz, out_szs, layers, omega, device
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.enc.reset_parameters()
        self.dec.reset_parameters()

    def forward(self, x, edge_index, pos, pool_structures, parts, pnp):
        pos_3 = pool_structures["p3"][0]

        # process 2D slice first
        z = self.enc(
            x,
            edge_index,
            pos,
            pool_structures["ei2"],
            pool_structures["p2"],
            pool_structures["k2"],
        )

        # z = onera_interp(z,pool_structures["p2"][-1],pos_3)

        x = torch.zeros((pool_structures["p3"][0].size(0), 5)).to(self.device)
        for part in parts:
            z_p = knn_interpolate(z, pool_structures["p2"][-1], part.pos, k=1)
            pos_in = (
                2
                * (part.pos - part.pos.min(dim=0).values)
                / (part.pos.max(dim=0).values - part.pos.min(dim=0).values)
                - 1
            )
            x[part.old_idx] = self.dec(z_p, pos_in)

        # upsample before processing
        # x_in = torch.zeros((pos_3.size(0),x.size(1)))
        # z = torch.zeros((pos_3.size(0),self.latent_sz)).to(self.device)
        # for part, pool in zip(parts, pnp):
        #     x_in = onera_interp(x,pos,part.pos,3)
        #     z[part.old_idx] = self.enc(
        #         x_in,
        #         part.edge_index,
        #         part.pos,
        #         pool["ei"],
        #         pool["p"],
        #         pool["k"]
        #     )

        # x = self.dec(z, pos_3)
        return x
