import torch as th
import torch.nn as nn
from layers import *


class generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dims, z_dim, vertex, nodes, edges, dropout_rate):
        super(generator, self).__init__()
        self.activation_f = nn.Tanh()
        self.multi_dense_layer = multi_dense_layer(z_dim, conv_dims, self.activation_f)

        self.vertexes = vertex
        self.edges = edges
        self.nodes = nodes

        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertex * vertex)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertex * nodes)
        self.dropoout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.multi_dense_layer(x)
        edges_logits = self.edges_layer(output).view(
            -1, self.edges, self.vertexes, self.vertexes
        )
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropoout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits = self.nodes_layer(output)
        nodes_logits = self.dropoout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(
        self,
        conv_dim,
        atom_dim,
        bond_dim,
        with_features=False,
        feat_dim=0,
        dropout_rate=0.0,
    ):
        super(discriminator, self).__init__()
        self.activation_f = nn.Tanh()
        graph_conv_dim, aux_dim, linear_dim = conv_dim
        # discriminator
        self.gcn_layer = graph_conv(
            atom_dim, graph_conv_dim, bond_dim, with_features, feat_dim, dropout_rate
        )
        self.agg_layer = graph_aggr(
            graph_conv_dim[-1] + atom_dim,
            aux_dim,
            self.activation_f,
            with_features,
            feat_dim,
            dropout_rate,
        )
        self.multi_dense_layer = multi_dense_layer(
            aux_dim, linear_dim, self.activation_f, dropout_rate=dropout_rate
        )

        self.output_layer = nn.Linear(linear_dim[-1], 1)

    def forward(self, adj, hidden, node, activation=None):
        adj = adj[:, :, :, 1:].permute(0, 3, 1, 2)
        h = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(h, node, hidden)
        h = self.multi_dense_layer(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h
