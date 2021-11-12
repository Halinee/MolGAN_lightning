import torch as th
import torch.nn as nn

from layers import graph_conv, graph_aggr


class generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dims, z_dim, vertex, nodes, edges, dropout):
        super(generator, self).__init__()

        self.vertexes = vertex
        self.edges = edges
        self.nodes = nodes

        layers = []
        for c0, c1 in zip([z_dim] + conv_dims[:-1], conv_dims):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=dropout, inplace=True))
        self.layers = nn.Sequential(*layers)

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertex * vertex)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertex * nodes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = self.layers(x)
        edges_logits = self.edges_layer(output).view(
            -1, self.edges, self.vertexes, self.vertexes
        )
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim, nodes, edges, dropout):
        super(discriminator, self).__init__()

        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = graph_conv(nodes, graph_conv_dim, edges, dropout)
        self.agg_layer = graph_aggr(graph_conv_dim[-1], aux_dim, edges, dropout)

        # multi dense layer
        layers = []
        for c0, c1 in zip([aux_dim] + linear_dim[:-1], linear_dim):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Dropout(dropout))
        self.linear_layer = nn.Sequential(*layers)

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, act=None):
        print("init adj shape")
        print(adj.shape)
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        print("preprocess adj shape")
        print(adj.shape)
        annotations = th.cat((hidden, node), -1) if hidden is not None else node
        print("gcn_layer annotations shape")
        print(annotations.shape)
        h = self.gcn_layer(annotations, adj)
        annotations = th.cat((h, hidden, node) if hidden is not None else (h, node), -1)
        print("agg_layer annotations shape")
        print(annotations.shape)
        h = self.agg_layer(annotations, th.tanh)
        h = self.linear_layer(h)

        output = self.output_layer(h)
        output = act(output) if act is not None else output

        return output, h
