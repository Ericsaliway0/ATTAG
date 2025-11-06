import copy
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

import dgl
from dgl import DGLGraph
import dgl.function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax

from dgl.nn import (
    SAGEConv,
    GATConv,
    GINConv,
    GraphConv,
    ChebConv,
    TAGConv
)



class BaseGNN(nn.Module):
    """A wrapper class to ensure all models have a unified interface:
       - forward(graph) returns logits if training, embeddings if not
       - get_node_embeddings(graph) always returns embeddings
    """
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__()
        self.dim_latent = dim_latent
        self.num_layers = num_layers
        self.do_train = do_train


class TAGCNModel(BaseGNN):
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__(dim_latent, num_layers, do_train)
        self.linear = nn.Linear(1, dim_latent)
        self.conv_0 = TAGConv(in_feats=dim_latent, out_feats=dim_latent, k=2)

        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList([
            TAGConv(in_feats=dim_latent, out_feats=dim_latent, k=2)
            for _ in range(num_layers - 1)
        ])
        self.predict = nn.Linear(dim_latent, 1)

    def get_node_embeddings(self, graph):
        w = graph.ndata["weight"].unsqueeze(-1)
        x = self.linear(w)

        g = dgl.add_self_loop(graph)
        h = self.conv_0(g, x)

        for conv in self.layers:
            h = self.relu(h)
            h = conv(g, h)

        return h

    def forward(self, graph):
        emb = self.get_node_embeddings(graph)
        if not self.do_train:
            return emb.detach()
        return self.predict(emb)



class GraphSAGEModel(BaseGNN):
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__(dim_latent, num_layers, do_train)
        self.linear = nn.Linear(1, dim_latent)

        self.layers = nn.ModuleList([
            SAGEConv(dim_latent, dim_latent, aggregator_type='mean')
            for _ in range(num_layers)
        ])

        self.relu = nn.LeakyReLU()
        self.predict = nn.Linear(dim_latent, 1)

    def get_node_embeddings(self, graph):
        w = graph.ndata["weight"].unsqueeze(-1)
        h = self.linear(w)
        g = dgl.add_self_loop(graph)

        for conv in self.layers:
            h = self.relu(h)
            h = conv(g, h)

        return h

    def forward(self, graph):
        emb = self.get_node_embeddings(graph)
        if not self.do_train:
            return emb.detach()
        return self.predict(emb)


class GATModel(BaseGNN):
    def __init__(self, dim_latent: int, num_layers: int, heads=4, do_train=False):
        super().__init__(dim_latent, num_layers, do_train)
        self.linear = nn.Linear(1, dim_latent)

        self.layers = nn.ModuleList([
            GATConv(dim_latent, dim_latent // heads, num_heads=heads)
            for _ in range(num_layers)
        ])

        self.relu = nn.LeakyReLU()
        self.predict = nn.Linear(dim_latent, 1)

    def get_node_embeddings(self, graph):
        w = graph.ndata["weight"].unsqueeze(-1)
        h = self.linear(w)
        g = dgl.add_self_loop(graph)

        for conv in self.layers:
            h = self.relu(h)
            h = conv(g, h).flatten(1)

        return h

    def forward(self, graph):
        emb = self.get_node_embeddings(graph)
        if not self.do_train:
            return emb.detach()
        return self.predict(emb)


class GCNModel(BaseGNN):
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__(dim_latent, num_layers, do_train)
        self.linear = nn.Linear(1, dim_latent)

        self.layers = nn.ModuleList([
            GraphConv(dim_latent, dim_latent)
            for _ in range(num_layers)
        ])

        self.relu = nn.LeakyReLU()
        self.predict = nn.Linear(dim_latent, 1)

    def get_node_embeddings(self, graph):
        w = graph.ndata["weight"].unsqueeze(-1)
        h = self.linear(w)
        g = dgl.add_self_loop(graph)

        for conv in self.layers:
            h = self.relu(h)
            h = conv(g, h)

        return h

    def forward(self, graph):
        emb = self.get_node_embeddings(graph)
        if not self.do_train:
            return emb.detach()
        return self.predict(emb)


class GINModel(BaseGNN):
    def __init__(self, dim_latent: int, num_layers: int, do_train=False):
        super().__init__(dim_latent, num_layers, do_train)
        self.linear = nn.Linear(1, dim_latent)

        mlp = lambda: nn.Sequential(
            nn.Linear(dim_latent, dim_latent),
            nn.ReLU(),
            nn.Linear(dim_latent, dim_latent)
        )

        self.layers = nn.ModuleList([GINConv(mlp()) for _ in range(num_layers)])

        self.relu = nn.LeakyReLU()
        self.predict = nn.Linear(dim_latent, 1)

    def get_node_embeddings(self, graph):
        w = graph.ndata["weight"].unsqueeze(-1)
        h = self.linear(w)
        g = dgl.add_self_loop(graph)

        for conv in self.layers:
            h = self.relu(h)
            h = conv(g, h)

        return h

    def forward(self, graph):
        emb = self.get_node_embeddings(graph)
        if not self.do_train:
            return emb.detach()
        return self.predict(emb)


class ChebNetModel(BaseGNN):
    def __init__(self, dim_latent: int, num_layers: int, k=3, do_train=False):
        super().__init__(dim_latent, num_layers, do_train)
        self.linear = nn.Linear(1, dim_latent)

        self.layers = nn.ModuleList([
            ChebConv(dim_latent, dim_latent, k)
            for _ in range(num_layers)
        ])

        self.relu = nn.LeakyReLU()
        self.predict = nn.Linear(dim_latent, 1)

    def get_node_embeddings(self, graph):
        w = graph.ndata["weight"].unsqueeze(-1)
        h = self.linear(w)
        g = dgl.add_self_loop(graph)

        for conv in self.layers:
            h = self.relu(h)
            h = conv(g, h)

        return h

    def forward(self, graph):
        emb = self.get_node_embeddings(graph)
        if not self.do_train:
            return emb.detach()
        return self.predict(emb)


def choose_model(model_type, **kwargs):
    model_type = model_type.lower()

    if model_type == "graphsage":
        return model.SAGEModel(**kwargs)

    elif model_type == "gat":
        return model.GATModel(**kwargs)

    elif model_type == "gcn":
        return model.GCNModel(**kwargs)

    elif model_type == "gin":
        return model.GINModel(**kwargs)

    elif model_type == "chebnet":
        return model.ChebNetModel(**kwargs)

    elif model_type == "tagcn":
        return model.TAGCNModel(**kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

