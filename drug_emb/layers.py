import torch as th
import torch.nn as nn


class graph_conv(nn.Module):
    def __init__(self, in_features, out_feature_list, edges, dropout):
        super(graph_conv, self).__init__()
        self.in_features = in_features
        self.out_feature_list = out_feature_list

        self.linear1 = nn.Linear(in_features, out_feature_list[0])
        self.linear2 = nn.Linear(out_feature_list[0], out_feature_list[1])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj, activation=None):
        hidden = th.stack([self.linear1(input) for _ in range(adj.size(1))], 1)
        hidden = th.einsum("bijk,bikl->bijl", (adj, hidden))
        hidden = th.sum(hidden, 1) + self.linear1(input)
        hidden = activation(hidden) if activation is not None else hidden
        hidden = self.dropout(hidden)

        output = th.stack([self.linear2(hidden) for _ in range(adj.size(1))], 1)
        output = th.einsum("bijk,bikl->bijl", (adj, output))
        output = th.sum(output, 1) + self.linear2(hidden)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class graph_aggr(nn.Module):
    def __init__(self, in_features, out_features, edges, dropout):
        super(graph_aggr, self).__init__()
        self.sigmoid_linear = nn.Sequential(
            nn.Linear(in_features + edges, out_features), nn.Sigmoid()
        )
        self.tanh_linear = nn.Sequential(
            nn.Linear(in_features + edges, out_features), nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, activation):
        print("input shape")
        print(input.shape)
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = th.sum(th.mul(i, j), 1)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output
