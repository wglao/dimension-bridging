import torch
from torch import Tensor
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from typing import Tuple, Union, Optional, Callable
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.nn import (
    SAGEConv,
    SAGPooling,
    GATv2Conv,
    GraphConv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    knn_interpolate,
    Linear,
)
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import Sequential as GeoSequential
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch_geometric.utils import softmax, add_self_loops, add_remaining_self_loops
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.nn.inits import zeros, ones, constant, uniform
from torch_geometric.nn.pool import (
    global_mean_pool,
    max_pool,
    avg_pool_neighbor_x,
    max_pool_neighbor_x,
)
from torch.nn import Parameter
from torch.nn import ReLU, Sequential
import torch.nn.functional as F
from graphdata import PairData
import numpy as np
import math


def get_pooled_sz(full_sz: int, ratio: float, layer: int):
    out_sz = full_sz
    for l in range(layer):
        out_sz = int(-(out_sz // -(1 / ratio)))
    return out_sz


def get_deg(x, edge_index, device: str = "cpu"):
    deg = torch.sparse_coo_tensor(edge_index, torch.ones((edge_index.size(1),))).to(
        device
    ) @ torch.ones((x.size(0), 1)).to(device)
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


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        out_channels: int,
        act: callable = torch.nn.functional.elu,
        device: str = "cuda:0",
    ):
        super(MLP, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.act = act

        self.lin_0 = Linear(in_channels, hid_channels).cuda(device)
        # self.lin_1 = Linear(hid_channels, out_channels).cuda(device)
        self.lin_2 = Linear(hid_channels, hid_channels).cuda(device)
        self.lin_3 = Linear(hid_channels, out_channels).cuda(device)

    def forward(self, x):
        out = self.act(self.lin_0(x), inplace=True)
        # skip = self.lin_1(out)
        out = self.act(self.lin_2(out), inplace=True)
        out = self.lin_3(out)
        return out


class SAGPGIN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        act: callable = torch.nn.functional.elu,
        device: str = "cuda:0",
    ):
        super(SAGPGIN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.act = act

        self.mlp = MLP(in_channels, hidden_channels, out_channels, device=device).cuda(
            device
        )
        self.gin = GINConv(self.mlp).cuda(device)

    def reset_parameters(self):
        self.gin.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, size=None):
        out = self.gin(x, edge_index, size)
        return out


class KernelMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        in_size: int,
        hidden_size: int,
        out_size: int,
        omega: float = 100.0,
        device: str = "cpu",
    ) -> None:
        super(KernelMLP, self).__init__()
        self.dim = dim
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.omega = omega
        self.device = device

        # in size is [n, dim + 1] for channel number
        # self.lin0 = Linear(dim + 1, hidden_size, weight_initializer="glorot").to(device)
        self.sin0 = SineLayer(dim + 1, hidden_size, omega=omega).to(device)
        # self.lin1 = Linear(
        #     hidden_size, hidden_size,
        #     weight_initializer='glorot').to(device)
        self.sin1 = SineLayer(hidden_size, hidden_size, omega=omega).to(device)
        self.sin2 = SineLayer(hidden_size, hidden_size, omega=omega).to(device)
        self.lin = Linear(hidden_size, in_size).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.sin0.reset_parameters()
        self.sin1.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, rel_pos, channel: int = 1):
        # # Convert to spherical [rho, theta, phi] = [r, az, elev]
        rho = torch.norm(rel_pos, dim=1)
        theta = torch.atan2(rel_pos[:, 1], rel_pos[:, 0])
        phi = torch.asin(rel_pos[:, 2] / rho)
        theta = torch.where(phi.isnan(), torch.zeros_like(theta).to(self.device), theta)
        phi = torch.where(phi.isnan(), torch.zeros_like(phi).to(self.device), phi)
        rel_pos = torch.stack((rho, theta / torch.pi, phi / torch.pi), dim=1)

        out = self.sin0(
            torch.concatenate(
                (rel_pos, torch.full((rel_pos.size(0), 1), channel).to(self.device)), 1
            )
        )
        out = self.sin1(out)
        out = self.sin2(out)
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
        omega: float = 100.0,
        device: str = "cpu",
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GraphKernelConv, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.device = device
        self.k_net = k_net(
            dim, in_channels, hidden_channels, out_channels, omega, device
        ).to(device)
        # self.lin0 = Linear(in_channels, out_channels).to(device)
        # self.lin1 = Linear(hidden_channels, out_channels).to(device)

        self.bias = Parameter(torch.zeros(1, out_channels).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.k_net.reset_parameters()
        # self.lin0.reset_parameters()
        # self.lin1.reset_parameters()
        zeros(self.bias)
        # self._cached_edge_index = None

    def forward(self, x, edge_index, pos):
        # x = F.selu(self.lin0(x), inplace=True)
        edge_index, _ = add_remaining_self_loops(edge_index)
        rel_pos = get_edge_attr(edge_index, pos)

        x = self.propagate(edge_index, x=x, edge_weight=rel_pos, size=None)
        # out = self.lin1(out)
        out = x + self.bias

        return out

    def message_calc(self, x_j: Tensor, rel_pos: Tensor, channel: int = 0):
        msg = self.k_net(rel_pos, channel) * x_j
        return msg.sum(1)

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


class SAGPoolWithPos(SAGPooling):
    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: torch.nn.Module = GraphConv,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = "tanh",
        augmentation: bool = True,
        **kwargs,
    ):
        super(SAGPoolWithPos, self).__init__(
            in_channels, ratio, GNN, min_score, multiplier, nonlinearity, **kwargs
        )
        self.augmentation = augmentation

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        pos: Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        attn: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, OptTensor, OptTensor, Tensor, Tensor]:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = torch.cat((x, pos), dim=1) if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        if self.augmentation:
            adj = torch.sparse_coo_tensor(
                edge_index,
                torch.ones(
                    edge_index.size(1),
                ).cuda(),
            )
            adj_aug = (adj + (adj @ adj)).coalesce()
            edge_index_aug = adj_aug.indices()
            score = self.gnn(attn, edge_index_aug).view(-1)
        else:
            score = self.gnn(attn, edge_index).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x
        pos = pos[perm]

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0)
        )

        return x, edge_index, pos, edge_attr, batch, perm, score[perm]


class EdgeNet(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels,
        hidden_channels,
        out_channels,
        device: str = "cuda:0",
        **kwargs,
    ) -> None:
        super(EdgeNet, self).__init__(**kwargs)
        self.dim = dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.mlp = MLP(in_channels, hidden_channels, out_channels, device=device).cuda(
            device
        )

    def forward(self, x, edge_index, pos):
        edge_attr = pos[edge_index[1]] - pos[edge_index[0]]
        node_i = x[edge_index[0]]
        node_j = x[edge_index[1]]
        inputs = torch.cat((edge_attr, node_i, node_j), dim=1)

        out = self.mlp(inputs)
        return out


class NodeNet(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels,
        hidden_channels,
        out_channels,
        device: str = "cuda:0",
        **kwargs,
    ) -> None:
        super(NodeNet, self).__init__(**kwargs)
        self.dim = dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.device = device

        self.mlp = MLP(in_channels, hidden_channels, out_channels, device=device).cuda(
            device
        )

    def forward(self, x, edge_index, edge_attr):
        # edge_mat = torch.sparse_coo_tensor(edge_index, edge_attr, (x.size(0), x.size(0))).cuda(device)
        edge_sum = torch.cat(
            [
                torch.sparse_coo_tensor(
                    edge_index, edge_attr[:, i], (x.size(0), x.size(0))
                ).cuda(self.device)
                @ torch.ones((x.size(0), 1)).cuda(self.device)
                for i in range(self.hidden_channels)
            ],
            dim=1,
        )

        inputs = torch.cat((x, edge_sum), dim=1)
        out = self.mlp(inputs)
        return out


class SizeFieldNet(nn.Module):
    def __init__(
        self, dim: int, in_channels, hidden_channels, device: str = "cuda:0", **kwargs
    ) -> None:
        super(SizeFieldNet, self).__init__(**kwargs)
        self.dim = dim
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = dim**2

        self.mlp = MLP(
            in_channels, hidden_channels, self.out_channels, device=device
        ).cuda(device)

    def forward(self, x):
        out = self.mlp(x)
        out = torch.reshape(out, (self.dim, self.dim))
        return out


class MGN(nn.Module):
    def __init__(
        self,
        dim: int,
        in_features: int,
        hidden_channels: int,
        out_features: int,
        device: str = "cuda:0",
        **kwargs,
    ) -> None:
        super(MGN, self).__init__(**kwargs)
        self.dim = dim
        self.in_features = in_features
        self.hidden_channels = hidden_channels
        self.edge_net = EdgeNet(
            dim, dim + 2 * self.in_features, hidden_channels, hidden_channels, device
        ).cuda(device)
        self.node_net = NodeNet(
            dim,
            self.in_features + hidden_channels,
            hidden_channels,
            hidden_channels,
            device,
        ).cuda(device)
        # self.sfn = SizeFieldNet(dim, hidden_channels, hidden_channels)
        self.decoder = MLP(
            hidden_channels, hidden_channels, out_features, device=device
        ).cuda(device)

    def forward(self, x, edge_index, pos):
        edge_attr = self.edge_net(x, edge_index, pos)
        node_attr = self.node_net(x, edge_index, edge_attr)
        out = self.decoder(node_attr)

        # size_tensor = self.sfn(node_attr)
        # return out, size_tensor

        return out


class DBMGN(nn.Module):
    def __init__(
        self,
        dim: int,
        init_data: Data,
        hidden_channels: int,
        device: str = "cuda:0",
        **kwargs,
    ) -> None:
        super(DBMGN, self).__init__(**kwargs)
        self.dim = dim
        self.in_features = init_data.num_node_features
        self.hidden_channels = hidden_channels
        self.device = device

        # self.mgn_0 = MGN(dim, self.in_features, hidden_channels, hidden_channels).cuda(device)
        # self.mgn_1 = MGN(dim, hidden_channels, hidden_channels, self.in_features).cuda(device)

        self.mgn = MGN(
            dim, self.in_features, hidden_channels, self.in_features, device
        ).cuda(device)

    def forward(self, x, edge_index, pos):
        # Double MGN
        # out = self.mgn_0(x, edge_index, pos)
        # out = self.mgn_1(out, edge_index, pos)

        # Double Residual MGN
        # out = self.mgn_0(x, edge_index, pos)
        # out = self.mgn_1(out, edge_index, pos) + x

        # Single MGN
        # out = self.mgn(x, edge_index, pos)

        # Residual MGN
        out = self.mgn(x, edge_index, pos) + x
        return out


class RDB(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        growth_channels: int,
        device: str = "cpu",
        beta: float = 0.2,
    ) -> None:
        super(RDB, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.growth_channels = growth_channels
        self.device = device
        self.beta = beta

        self.conv1 = GraphKernelConv(
            dim, in_channels, in_channels, growth_channels, device=device
        ).to(device)
        self.conv2 = GraphKernelConv(
            dim,
            in_channels + growth_channels,
            in_channels,
            growth_channels,
            device=device,
        ).to(device)
        self.conv3 = GraphKernelConv(
            dim,
            in_channels + 2 * growth_channels,
            in_channels,
            in_channels,
            device=device,
        ).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x, edge_index, pos):
        x1 = F.selu(self.conv1(x, edge_index, pos), inplace=True)
        x2 = F.selu(self.conv2(torch.cat((x, x1), 1), edge_index, pos), inplace=True)
        x3 = F.selu(
            self.conv3(torch.cat((x, x1, x2), 1), edge_index, pos), inplace=True
        )
        out = x + self.beta * x3
        return out


class ERDB(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        growth_channels: int,
        device: str = "cpu",
        beta: float = 0.2,
    ) -> None:
        super(ERDB, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.growth_channels = growth_channels
        self.device = device
        self.beta = beta

        self.rdb1 = RDB(dim, in_channels, growth_channels, device).to(device)
        self.rdb2 = RDB(dim, in_channels, growth_channels, device).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.rdb1.reset_parameters()
        self.rdb2.reset_parameters()

    def forward(self, x, edge_index, pos):
        x1 = self.rdb1(x, edge_index, pos)
        x2 = self.rdb2(x + self.beta * x1, edge_index, pos)
        out = x + self.beta * x2
        return out


class NeighborhoodPool(nn.Module):
    def __init__(
        self,
        dim: int,
        channels: int,
        k_hops: int = 1,
        gnn: nn.Module = GraphKernelConv,
        device: str = "cpu",
        **kwargs,
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
            **kwargs,
        ).to(device)
        self.gnn2 = gnn(
            dim,
            in_channels=channels,
            hidden_channels=channels,
            out_channels=1,
            device=device,
            **kwargs,
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
        hidden_channels: int = 64,
        layers: int = 3,
        latent_sz: int = 32,
        omega: float = 100,
        device: str = "cpu",
    ):
        super(Encoder, self).__init__()
        self.dim = dim
        self.in_channels = init_data.num_node_features
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.latent_sz = latent_sz
        self.device = device
        self.omega = omega

        # initial aggr
        self.lin0 = Linear(self.in_channels, hidden_channels).to(self.device)

        self.conv_list = nn.ModuleList(
            [
                GraphKernelConv(
                    dim,
                    hidden_channels + dim,
                    hidden_channels,
                    hidden_channels,
                    omega=omega,
                    device=device,
                ).to(self.device)
            ]
        )
        self.pool_list = nn.ModuleList()
        for _ in range(self.layers):
            self.conv_list.append(
                GraphKernelConv(
                    dim,
                    hidden_channels + dim,
                    hidden_channels,
                    hidden_channels,
                    omega=omega,
                    device=device,
                ).to(self.device)
            )

        # latent
        self.lin1 = Linear(hidden_channels, latent_sz).to(self.device)
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
        # scale to [-1,1]
        x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
        pos_max = pos.max(dim=0).values
        pos_min = pos.min(dim=0).values
        pos = (pos - pos.min(dim=0).values) / (
            pos.max(dim=0).values - pos.min(dim=0).values
        )
        x = torch.sin(self.omega * self.lin0(x))
        x = torch.sin(
            self.omega
            * (self.conv_list[0](torch.cat((x, pos), dim=1), edge_index, pos))
        )

        for l in range(self.layers):
            keep_idx = pool_keep_idx[l]
            x = self.max_pool(x, edge_index, keep_idx)
            edge_index = pool_edge_index[l + 1]
            pos = pool_pos[l + 1]
            pos = (pos - pos_min) / (pos_max - pos_min)

            x = torch.sin(
                self.omega
                * (self.conv_list[l + 1](torch.cat((x, pos), dim=1), edge_index, pos))
            )

        z = self.lin1(x)
        z = z.mean(0)
        return z


class DBGSR(nn.Module):
    def __init__(
        self,
        dim: int,
        init_data_2: Data,
        hidden_channels: int,
        device: str = "cpu",
        beta: float = 0.2,
    ) -> None:
        super(DBGSR, self).__init__()
        self.dim = dim
        self.in_features = init_data_2.num_node_features
        self.hidden_channels = hidden_channels
        self.device = device
        self.beta = beta

        self.lin0 = Linear(self.in_features + dim, hidden_channels).to(device)

        self.conv1 = GraphKernelConv(
            dim, hidden_channels, hidden_channels, hidden_channels, device=device
        ).to(device)
        self.erdb1 = ERDB(dim, hidden_channels, -(hidden_channels // -2), device).to(
            device
        )
        self.erdb2 = ERDB(dim, hidden_channels, -(hidden_channels // -2), device).to(
            device
        )
        self.conv2 = GraphKernelConv(
            dim, hidden_channels, hidden_channels, hidden_channels, device=device
        ).to(device)

        # upsample
        self.conv3 = GraphKernelConv(
            dim, hidden_channels, hidden_channels, hidden_channels, device=device
        ).to(device)
        self.erdb3 = ERDB(dim, hidden_channels, -(hidden_channels // -2), device).to(
            device
        )
        self.erdb4 = ERDB(dim, hidden_channels, -(hidden_channels // -2), device).to(
            device
        )
        self.conv4 = GraphKernelConv(
            dim, hidden_channels, hidden_channels, hidden_channels, device=device
        ).to(device)

        # self.conv5 = GraphKernelConv(
        #     dim, hidden_channels, hidden_channels, self.in_features,
        #     device=device).to(device)
        self.lin1 = Linear(hidden_channels, self.in_features).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()

        self.erdb1.reset_parameters()
        self.erdb2.reset_parameters()
        self.erdb3.reset_parameters()
        self.erdb4.reset_parameters()

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()

    def onera_transform(self, pos):
        # adjust x to move leading edge to x=0
        new_x = pos[:, 0] - math.tan(math.pi / 6) * pos[:, 1]
        pos = torch.cat((torch.unsqueeze(new_x, 1), pos[:, 1:]), 1)
        # scale chord to equal root
        # c(y) = r(1 - (1-taper)*(y/s))
        # r = c(y) / (1- (1-taper)*(y/s))
        pos = pos * (1 + (1 / 0.56 - 1) * (pos[:, 1:2] / 1.1963))
        return pos

    def onera_interp(self, f, pos_x, pos_y, device: str = "cpu"):
        # in_idx = (pos_x[:, 1] < 1.1963, pos_y[:, 1] < 1.1963)
        # out_idx = (pos_x[:, 1] > 1.1963, pos_y[:, 1] > 1.1963)
        # inboard = knn_interpolate(f[in_idx[0]], onera_transform(pos_x[in_idx[0]]),
        #                           onera_transform(pos_y[in_idx[1]]))

        # outboard = knn_interpolate(f, pos_x, pos_y)[out_idx[1]]
        out = torch.where(
            (pos_y[:, 1] < 1.1963).reshape(pos_y.shape),
            knn_interpolate(
                f, self.onera_transform(pos_x), self.onera_transform(pos_y)
            ),
            knn_interpolate(f, pos_x, pos_y),
        )
        return out

    def forward(self, x, edge_index_2, edge_index_3, pos_2, pos_3):
        # scale data
        x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)
        pos_2 = (pos_2 - pos_2.min(dim=0).values) / (
            pos_2.max(dim=0).values - pos_2.min(dim=0).values
        )
        pos_3 = (pos_3 - pos_3.min(dim=0).values) / (
            pos_3.max(dim=0).values - pos_3.min(dim=0).values
        )

        x = F.selu(self.lin0(torch.cat((x, pos_2), 1)), inplace=True)
        x1 = F.selu(self.conv1(x, edge_index_2, pos_2), inplace=True)
        x2 = self.erdb1(x1, edge_index_2, pos_2)
        x = x1 + self.beta * self.erdb2(x2, edge_index_3, pos_3)
        x = F.selu(self.conv2(x, edge_index_2, pos_2), inplace=True)

        # upsample
        x = self.onera_interp(x, pos_2, pos_3)

        x1 = F.selu(self.conv3(x, edge_index_3, pos_3), inplace=True)
        x2 = self.erdb3(x1, edge_index_3, pos_3)
        x = x1 + self.beta * self.erdb4(x2, edge_index_3, pos_3)
        x = F.selu(self.conv4(x, edge_index_3, pos_3), inplace=True)

        # out = self.conv5(x, edge_index_3, pos_3)
        out = self.lin1(x)
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
    def __init__(self, in_sz, hidden_sz, layers, out_size, omega, device: str = "cpu"):
        super().__init__()
        self.in_sz = in_sz
        self.hidden_sz = hidden_sz
        self.layers = layers
        self.out_size = out_size
        self.omega = omega
        self.device = device

        self.sin0 = SineLayer(in_sz, hidden_sz).to(device)
        self.sin_list = nn.ModuleList()
        for l in range(layers):
            self.sin_list.append(SineLayer(hidden_sz, hidden_sz).to(device))
        self.lin = Linear(hidden_sz, out_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        for sin in self.sin_list:
            sin.reset_parameters()

    def forward(self, z):
        z = self.sin0(z)
        for s in self.sin_list:
            z = s(z)
        out = self.lin(z)
        return out


class MAReSIREN(nn.Module):
    def __init__(
        self,
        in_sz,
        hidden_sz,
        out_sz,
        layers,
        omega: float = 30.0,
        device: str = "cpu",
    ):
        super(MAReSIREN, self).__init__()
        self.in_sz = in_sz
        self.hidden_sz = hidden_sz
        self.out_sz = out_sz
        self.layers = layers
        self.omega = omega
        self.device = device

        self.sin0 = SineLayer(self.in_sz, hidden_sz, omega=omega).to(device)
        self.sin_list = nn.ModuleList()
        for l in range(layers):
            self.sin_list.append(
                SineLayer(hidden_sz, hidden_sz, omega=omega).to(device)
            )
        self.sin_fin = SineLayer(hidden_sz, self.out_sz, omega=omega).to(device)

        self.mare_min = torch.tensor([0.2, 2e6, -9]).to(self.device)
        self.mare_max = torch.tensor([0.8, 1e7, 9]).to(self.device)

        self.reset_parameters()

    def reset_parameters(self):
        self.sin0.reset_parameters()
        for sin in self.sin_list:
            sin.reset_parameters()
        self.sin_fin.reset_parameters()

    def forward(self, mare):
        # scale inputs to [-1, 1]
        # (ma,re,a) \in ([0.2,0.8],[2e6,1e7],[-9,9])
        mare = (mare - self.mare_min) / (self.mare_max - self.mare_min)

        out = self.sin0(mare)
        for l in range(self.layers):
            out = self.sin_list[l](out)
        out = self.sin_fin(out)
        return out


class ModSIREN(nn.Module):
    def __init__(
        self,
        dim,
        in_sz,
        hidden_sz,
        out_sz,
        layers,
        omega: float = 30.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.in_sz = in_sz
        self.hidden_sz = hidden_sz
        # self.out_sz = init_data.x.size(1)
        self.out_sz = out_sz
        self.layers = layers
        self.omega = omega
        self.device = device

        self.mod = ModulateMLP(self.in_sz, hidden_sz, layers, layers, device).to(device)
        self.sin0 = SineLayer(self.dim, hidden_sz, omega=omega).to(device)
        self.sin_list = nn.ModuleList()
        for l in range(layers):
            self.sin_list.append(
                SineLayer(hidden_sz, hidden_sz, omega=omega).to(device)
            )
        self.lin = Linear(hidden_sz, self.out_sz).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.mod.reset_parameters()
        self.sin0.reset_parameters()
        for sin in self.sin_list:
            sin.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, z, pos):
        # scale inputs to [-1, 1]
        # z = (z - z.min(dim=0).values) / (z.max(dim=0).values - z.min(dim=0).values)
        pos = (pos - pos.min(dim=0).values) / (
            pos.max(dim=0).values - pos.min(dim=0).values
        )
        alphas = self.mod(z)
        out = self.sin0(pos)
        for l in range(self.layers):
            out = self.sin_list[l](alphas[:, l : l + 1] * out)
        out = self.lin(out)
        return out


class ModSIRENSR(nn.Module):
    def __init__(self, init_data, hidden_sz, layers, omega, device: str = "cpu"):
        super().__init__()
        self.dim = init_data.pos.size(1)
        self.in_sz = init_data.x.size(1)
        self.hidden_sz = hidden_sz
        self.out_sz = init_data.x.size(1)
        self.layers = layers
        self.omega = omega
        self.device = device

        self.mods1 = ModSIREN(
            self.dim, self.in_sz, 2 * hidden_sz + 1, hidden_sz, layers, omega, device
        ).to(device)
        self.mods2 = ModSIREN(
            self.dim,
            self.hidden_sz,
            2 * self.in_sz + 1,
            self.in_sz,
            layers,
            omega,
            device,
        ).to(device)
        # self.mods2 = ModSIREN(
        #     self.dim, self.in_sz, self.hidden_sz, self.in_sz, layers, omega, device
        # ).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.mods1.reset_parameters()
        self.mods2.reset_parameters()

    def forward(self, x, pos_2, pos_3):
        # pos_2_scale = (pos_2 - pos_2.min(dim=0).values) / (
        #     pos_2.max(dim=0).values - pos_2.min(dim=0).values)
        # pos_3_scale = (pos_3 - pos_3.min(dim=0).values) / (
        #     pos_3.max(dim=0).values - pos_3.min(dim=0).values)
        x = self.mods1(x, pos_2)

        # upsample
        x = onera_interp(x, pos_2, pos_3)
        # scale inputs to [-1, 1]
        # x = (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values)

        out = self.mods2(x, pos_3)

        return out


class DBModSIREN(nn.Module):
    def __init__(
        self,
        # Encoder
        init_data,
        enc_layers: int,
        latent_size,
        # ModSIREN
        hidden_size,
        out_size,
        siren_layers: int,
        omega: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super(DBModSIREN, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.out_size = out_size
        self.omega = omega
        self.device = device

        self.dim = init_data.pos.size(1)
        self.enc_layers = enc_layers
        self.siren_layers = siren_layers

        # Encoder
        self.encoder = Encoder(
            self.dim, init_data, hidden_size, enc_layers, latent_size, omega, device
        ).to(device)

        # Modulated SIREN
        self.mods = ModSIREN(
            self.dim,
            latent_size,
            hidden_size,
            out_size,
            siren_layers,
            omega,
            device,
        ).to(device)

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.mods.reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        pos_2,
        pos_3,
        pool_edge_index=None,
        pool_pos=None,
        pool_keep_idx=None,
    ):
        z = self.encoder(x, edge_index, pos_2, pool_edge_index, pool_pos, pool_keep_idx)
        out = self.mods(z, pos_3)
        return out


class MAReModSIREN(nn.Module):
    def __init__(
        self,
        # Encoder
        enc_layers: int,
        latent_size,
        enc_omega,
        # ModSIREN
        init_data,
        hidden_size,
        out_size,
        siren_layers: int,
        omega: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super(MAReModSIREN, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.out_size = out_size
        self.enc_omega = enc_omega
        self.omega = omega
        self.device = device

        self.dim = init_data.pos.size(1)
        self.enc_layers = enc_layers
        self.siren_layers = siren_layers

        # Encoder
        self.mare_encoder = MAReSIREN(
            3, hidden_size, latent_size, enc_layers, enc_omega, device
        ).to(device)

        # 1 Modulated SIREN per output
        self.mods_list = nn.ModuleList([])
        for o in range(out_size):
            self.mods_list.append(
                ModSIREN(
                    self.dim,
                    latent_size,
                    hidden_size,
                    1,
                    siren_layers,
                    omega,
                    device,
                ).to(device)
            )

        self.reset_parameters()

    def reset_parameters(self):
        self.mare_encoder.reset_parameters()
        for mod in self.mods_list:
            mod.reset_parameters()

    def forward(self, mare, pos):
        z = self.mare_encoder(mare)
        out = torch.concatenate([mods(z, pos) for mods in self.mods_list], 1).to(
            self.device
        )
        return out
