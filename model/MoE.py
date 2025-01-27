from torch_geometric.nn import GCNConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGEConv
from model import SAGENorm, GCNNorm


# 定义专家网络（GCN）
class PTNorm(MessagePassing):
    def __init__(self, in_channel, out_channel, dropout, type):
        super(PTNorm, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.type = type
        self.norm = nn.LayerNorm(in_channel)
        if self.type == 'SAGE':
            self.convs.append(SAGENorm(in_channel, out_channel))
        elif self.type == 'GCN':
            self.convs.append(GCNNorm())
        elif self.type == 'GAT':
            self.convs.append(GATConv(in_channel, out_channel // 4, 4))
        self.convs.append(nn.Linear(in_channel, out_channel))
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight, res):
        if self.type == 'SAGE':
            x = self.convs[0](x, edge_index)
        elif self.type == 'GCN':
            x = self.convs[0](x, edge_index, edge_weight)
        elif self.type == 'GAT':
            x = self.convs[0](x, edge_index)
        x = self.convs[1](x)
        F.dropout(F.relu(x), p=self.dropout, training=self.training)
        # x = self.norm((1 - self.alpha) * x + self.alpha * res)
        x = self.norm(x)
        return x

    def message(self, x):
        return x


class TPNorm(MessagePassing):
    def __init__(self, in_channel, out_channel, dropout, type):
        super(TPNorm, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.type = type
        self.norm = nn.LayerNorm(in_channel)
        self.convs.append(nn.Linear(in_channel, out_channel))
        if self.type == 'SAGE':
            self.convs.append(SAGENorm(in_channel, out_channel))
        elif self.type == 'GCN':
            self.convs.append(GCNNorm())
        elif self.type == 'GAT':
            self.convs.append(GATConv(in_channel, out_channel // 4, 4))
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight, res):
        x = self.convs[0](x)
        F.dropout(F.relu(x), p=self.dropout, training=self.training)
        if self.type == 'SAGE':
            x = self.convs[1](x, edge_index)
        elif self.type == 'GCN':
            x = self.convs[1](x, edge_index, edge_weight)
        elif self.type == 'GAT':
            x = self.convs[1](x, edge_index)
        # x = self.norm((1 - self.alpha) * x + self.alpha * res)
        x = self.norm(x)
        return x

    def message(self, x):
        return x


class TTNorm(MessagePassing):
    def __init__(self, in_channel, out_channel, dropout, type):
        super(TTNorm, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.type = type
        self.convs.append(nn.Linear(in_channel, out_channel))
        self.convs.append(nn.Linear(in_channel, out_channel))
        self.norm = nn.LayerNorm(in_channel)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight, res):
        x = self.convs[0](x)
        F.dropout(F.relu(x), p=self.dropout, training=self.training)
        x = self.convs[1](x)
        F.dropout(F.relu(x), p=self.dropout, training=self.training)
        # x = self.norm((1 - self.alpha) * x + self.alpha * res)
        x = self.norm(x)
        return x

    def message(self, x):
        return x


class PPNorm(MessagePassing):
    def __init__(self, in_channel, out_channel, dropout, type):
        super(PPNorm, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        self.type = type
        self.norm = nn.LayerNorm(in_channel)
        if self.type == 'SAGE':
            self.convs.append(SAGEConv(in_channel, out_channel))
            self.convs.append(SAGEConv(in_channel, out_channel))
        elif self.type == 'GCN':
            self.convs.append(GCNNorm())
            self.convs.append(GCNNorm())
        elif self.type == 'GAT':
            self.convs.append(GATConv(in_channel, out_channel // 4, 4))
            self.convs.append(GATConv(in_channel, out_channel // 4, 4))
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight, res):
        if self.type == 'SAGE':
            x = self.convs[0](x, edge_index)
            x = self.convs[1](x, edge_index)
        elif self.type == 'GCN':
            x = self.convs[0](x, edge_index, edge_weight)
            x = self.convs[1](x, edge_index, edge_weight)
        elif self.type == 'GAT':
            x = self.convs[0](x, edge_index)
            x = self.convs[1](x, edge_index)
        # x = self.norm((1 - self.alpha) * x + self.alpha * res)
        x = self.norm(x)
        return x

    def message(self, x):
        return x

class GCNExpert(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dataset):
        super(GCNExpert, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# 定义门控网络
class GatingNetwork(nn.Module):
    def __init__(self, in_channels, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, num_experts),
            nn.Softmax(dim=-1)  # 使用Softmax得到专家的权重
        )

    def reset_parameters(self):
        for layer in self.gate:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        return self.gate(x)


class GLULayer(nn.Module):
    def __init__(self, hidden, activate):
        super(GLULayer, self).__init__()
        self.w1 = nn.Linear(hidden, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.w3 = nn.Linear(hidden, hidden)
        self.activate = activate

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()

    def forward(self, x):
        if self.activate == 'SwishGLU':
            x = self.w2(F.silu(self.w1(x)) * self.w3(x))  # SwishGLU
        elif self.activate == 'GEGLU':
            x = self.w2(F.gelu(self.w1(x)) * self.w3(x))  # gelu
        elif self.activate == 'ReGLU':
            x = self.w2(F.relu(self.w1(x)) * self.w3(x))  # relu
        return x


class MoELayer(nn.Module):
    def __init__(self, data, args):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList(
            [PTNorm(args.hidden, args.hidden, args.dropout, args.type),
             TPNorm(args.hidden, args.hidden, args.dropout, args.type),
             TTNorm(args.hidden, args.hidden, args.dropout, args.type),
             PPNorm(args.hidden, args.hidden, args.dropout, args.type)])
        self.gating = GatingNetwork(args.hidden, args.num_experts)
        self.mlpA = nn.Linear(data.x.shape[0], args.hidden)
        self.gamma = args.gamma


    def reset_parameters(self):
        for expert in self.experts:
            expert.reset_parameters()
        self.gating.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        # 残差
        res = x
        # 门控网络
        gating_weights = self.gating(x)
        # 加权求和
        expert_outputs = [expert(x, edge_index, edge_weight, res) for expert in self.experts]
        combined_output = torch.zeros_like(expert_outputs[0])
        for i, output in enumerate(expert_outputs):
            combined_output += gating_weights[:, i].unsqueeze(1) * output

        return combined_output


class MoEFFNLayer(nn.Module):
    def __init__(self, data, args):
        super(MoEFFNLayer, self).__init__()
        self.GLUList = nn.ModuleList(
            [GLULayer(args.hidden, 'SwishGLU'),
             GLULayer(args.hidden, 'GEGLU'),
             GLULayer(args.hidden, 'ReGLU')])
        # FFNGate
        self.FFNGate = nn.Linear(args.hidden, 3)

    def reset_parameters(self):
        self.FFNGate.reset_parameters()
        for layer in self.GLUList:
            layer.reset_parameters()

    def forward(self, x):
        gate = F.gumbel_softmax(self.FFNGate(x), tau=1, hard=True)
        x = torch.stack([layer(x) for layer in self.GLUList], dim=1)
        x = torch.sum(x * gate.unsqueeze(-1), dim=1)
        return x


# 定义MoE网络
class MoE(nn.Module):
    def __init__(self, data, dataset, args):
        super(MoE, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.is_A = False
        self.MoELayers = nn.ModuleList(
            [MoELayer(data, args) for _ in range(args.n_layers)])
        self.MoEFFNLayers = MoEFFNLayer(data, args)
        self.start = nn.Linear(dataset.num_node_features, args.hidden)
        self.mlpA = nn.Linear(data.x.shape[0], args.hidden)
        if args.dataset in ['minesweeper', 'tolokers', 'questions', 'genius']:
            self.end = nn.Linear(args.hidden, 1)
        else:
            self.end = nn.Linear(args.hidden, dataset.num_classes)
        self.theta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.norm = nn.LayerNorm(args.hidden)
        self.w1 = nn.Linear(args.hidden, args.hidden)
        self.w2 = nn.Linear(args.hidden, args.hidden)
        self.w3 = nn.Linear(args.hidden, args.hidden)

    def reset_parameters(self):
        self.start.reset_parameters()
        self.mlpA.reset_parameters()
        self.end.reset_parameters()
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()
        for layer in self.MoELayers:
            layer.reset_parameters()
        self.MoEFFNLayers.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        # Start-Linear
        x = self.start(x)
        initial_x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        res = initial_x

        # MoE 网络层
        for i, layer in enumerate(self.MoELayers):
            if i == 0:
                x = layer(initial_x, edge_index, edge_weight)
            else:
                x = layer(x, edge_index, edge_weight)

        # MoE FFN层
        x = self.MoEFFNLayers(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm((1 - self.theta) * x + self.theta * res)

        # End-Linear
        x = self.end(x)
        if self.args.dataset in ['minesweeper', 'tolokers', 'questions', 'genius']:
            x = F.sigmoid(x).squeeze()
        return x
