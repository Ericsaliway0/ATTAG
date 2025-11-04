import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from dgl.nn import SAGEConv, GATConv, GraphConv, GINConv, ChebConv, TAGConv
from torch_geometric.utils import dropout_edge, negative_sampling, remove_self_loops, add_self_loops


class AttentionLayer(nn.Module):
    def __init__(self, hidden_feats):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(2 * hidden_feats, 1)  # Learnable attention weights

    def forward(self, g, features):
        with g.local_scope():
            # Expand node features for attention computation
            g.ndata['h'] = features
            g.apply_edges(self._compute_attention)
            g.edata['a'] = F.leaky_relu(g.edata['a'], negative_slope=0.2)
            g.edata['a'] = torch.softmax(g.edata['a'], dim=1)  # Normalize attention scores
            g.update_all(self._message_func, self._reduce_func)
            return g.ndata['h']

    def _compute_attention(self, edges):
        # Concatenate source and destination node features for edge attention
        z = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)
        a = self.attn(z)
        return {'a': a}

    def _message_func(self, edges):
        return {'m': edges.src['h'] * edges.data['a']}

    def _reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


class ATTAG(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.5):
        super(ATTAG, self).__init__()
        self.tag1 = TAGConv(in_feats, hidden_feats, k, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        self.attn = AttentionLayer(hidden_feats)
        self.tag2 = TAGConv(hidden_feats, hidden_feats, k, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_feats)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        # Assume g and features are already on GPU
        x = self.tag1(g, features)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.attn(g, x)
        x = self.tag2(g, x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return self.mlp(x)

class ATTAG_(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.5, activation=F.relu):
        super(ATTAG, self).__init__()
        self.activation = activation
        
        # First TAGConv block
        self.tag1 = TAGConv(in_feats, hidden_feats, k)
        self.bn1 = nn.BatchNorm1d(hidden_feats)

        # Attention layer
        self.attn = AttentionLayer(hidden_feats)

        # Second TAGConv block
        self.tag2 = TAGConv(hidden_feats, hidden_feats, k)
        self.bn2 = nn.BatchNorm1d(hidden_feats)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats)
        )

        # Projection layers for residuals if dimensions differ
        self.res_proj1 = nn.Linear(in_feats, hidden_feats) if in_feats != hidden_feats else None
        self.res_proj2 = None  # hidden_feats → hidden_feats (no projection needed)

    def forward(self, g, features):
        device = features.device
        g = g.to(device)

        # ===== First TAGConv + Attention with Residual =====
        res1 = features
        x = self.tag1(g, features)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.attn(g, x)  # Apply attention

        # Project residual if needed
        if self.res_proj1 is not None:
            res1 = self.res_proj1(res1)

        x = x + res1  # Residual connection

        # ===== Second TAGConv with Residual =====
        res2 = x
        x = self.tag2(g, x)
        x = self.bn2(x)
        x = self.activation(x)

        if self.res_proj2 is not None:
            res2 = self.res_proj2(res2)

        x = x + res2  # Residual connection

        # ===== Final MLP =====
        x = self.dropout(x)
        return self.mlp(x)

class MOGAT_no_learn(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, heads=4, dropout=0.6):
        super(MOGAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads=heads, feat_drop=dropout, attn_drop=dropout)
        self.gat2 = GATConv(hidden_feats * heads, hidden_feats, num_heads=1, feat_drop=dropout, attn_drop=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        """
        Forward pass for MOGAT.

        Parameters:
        - g: DGLGraph on the same device as the model.
        - features: Node features tensor.

        Returns:
        - logits: Raw output scores [num_nodes, out_feats]
        - embeddings: Node embeddings from the last GAT layer.
        """
        # First GAT layer (multi-head)
        x = self.gat1(g, features)   # Shape: [N, heads, hidden_feats]
        x = x.flatten(1)             # Merge heads → shape: [N, hidden_feats * heads]
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        # Second GAT layer (single head)
        x = self.gat2(g, x)          # Shape: [N, 1, hidden_feats]
        x = x.squeeze(1)             # → shape: [N, hidden_feats]
        x = F.elu(x)

        embeddings = x               # Store embeddings before classification
        logits = self.mlp(x)         # Final classification

        # return logits, embeddings
        return logits

class MOGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, heads=1, dropout=0.2):
        super(MOGAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads=heads,
                            feat_drop=dropout, attn_drop=dropout, activation=F.elu)
        self.gat2 = GATConv(hidden_feats * heads, hidden_feats, num_heads=1,
                            feat_drop=dropout, attn_drop=dropout, activation=F.elu)
        self.classifier = nn.Linear(hidden_feats, out_feats)

    def forward(self, g, features):
        x = self.gat1(g, features).flatten(1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(g, x).squeeze(1)
        x = F.dropout(x, p=0.2, training=self.training)
        return self.classifier(x)

class FeatureAttention(nn.Module):
    def __init__(self, feat_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or max(16, feat_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim)
        )

    def forward(self, x):
        gates = torch.sigmoid(self.net(x))
        return x * gates

class MomentAggregator:
    @staticmethod
    def compute_moments(g, features, eps=1e-6):
        with g.local_scope():
            g.ndata['h'] = features
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh_mean'))
            neigh_mean = g.ndata.get('neigh_mean', torch.zeros_like(features))

            g.ndata['h2'] = features * features
            g.update_all(fn.copy_u('h2', 'm2'), fn.mean('m2', 'neigh_m2'))
            neigh_m2 = g.ndata.get('neigh_m2', torch.zeros_like(features))
            neigh_var = torch.clamp(neigh_m2 - neigh_mean * neigh_mean, min=0.0)

            g.ndata['h3'] = features * features * features
            g.update_all(fn.copy_u('h3', 'm3'), fn.mean('m3', 'neigh_m3'))
            neigh_m3 = g.ndata.get('neigh_m3', torch.zeros_like(features))
            neigh_skew = neigh_m3 - 3 * neigh_mean * neigh_m2 + 2 * neigh_mean.pow(3)
            denom = (neigh_var + eps).pow(1.5)
            neigh_skew = neigh_skew / (denom + eps)

            return neigh_mean, neigh_var, neigh_skew

class DMGNN(nn.Module):
    def __init__(
        self,
        in_feat_dim,
        hidden_dim,
        out_dim,
        heads=4,
        dropout=0.5,
        use_moments=('mean', 'var', 'skew'),
        use_feature_attn=True,
        remote_emb_dim=0
    ):
        super().__init__()
        self.use_moments = use_moments
        self.use_feature_attn = use_feature_attn
        self.remote_emb_dim = remote_emb_dim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.heads = heads

        if use_feature_attn:
            self.feat_attn = FeatureAttention(in_feat_dim)

        moment_channels = 0
        if 'mean' in use_moments: moment_channels += 1
        if 'var' in use_moments: moment_channels += 1
        if 'skew' in use_moments: moment_channels += 1

        total_input = in_feat_dim * (1 + moment_channels) + remote_emb_dim

        self.moment_proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, num_heads=heads,
                            feat_drop=dropout, attn_drop=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, num_heads=1,
                            feat_drop=dropout, attn_drop=dropout)

        self.res_proj = nn.Linear(hidden_dim, hidden_dim)
        self.agg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def mix_moment_embed(self, g, features):
        neigh_mean, neigh_var, neigh_skew = MomentAggregator.compute_moments(g, features)
        parts = [features]
        if 'mean' in self.use_moments: parts.append(neigh_mean)
        if 'var' in self.use_moments: parts.append(neigh_var)
        if 'skew' in self.use_moments: parts.append(neigh_skew)
        return torch.cat(parts, dim=1)

    def forward(self, g, features, remote_emb=None):
        if self.use_feature_attn:
            features = self.feat_attn(features)

        mixed = self.mix_moment_embed(g, features)

        if remote_emb is not None and self.remote_emb_dim > 0:
            mixed = torch.cat([mixed, remote_emb], dim=1)

        h = self.moment_proj(mixed)

        x = self.gat1(g, h)
        x = x.view(x.shape[0], -1)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x2 = self.gat2(g, x).squeeze(1)
        x2 = F.elu(x2)

        if x.shape[1] != x2.shape[1]:
            x = self.res_proj(x)
        agg = self.agg_mlp(torch.cat([x, x2], dim=1))

        logits = self.classifier(agg)
        return logits  # ready for BCEWithLogitsLoss

class HGDC(torch.nn.Module):
    def __init__(self, args, weights=[0.95, 0.90, 0.15, 0.10]):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels
        self.linear1 = Linear(in_channels, hidden_channels)

        # 3 convolutional layers for the original network
        self.conv_k1_1 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_1 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        
        # 3 convolutional layers for the auxiliary network
        self.conv_k1_2 = GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k2_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)
        self.conv_k3_2 = GCNConv(2 * hidden_channels, hidden_channels, add_self_loops=False)

        self.linear_r0 = Linear(hidden_channels, 1)
        self.linear_r1 = Linear(2 * hidden_channels, 1)
        self.linear_r2 = Linear(2 * hidden_channels, 1)
        self.linear_r3 = Linear(2 * hidden_channels, 1)

        # Attention weights on outputs of different convolutional layers
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([weights[3]]), requires_grad=True)

    def forward(self, data):
        x_input = data.x
        edge_index_1 = data.edge_index
        edge_index_2 = data.edge_index_aux

        edge_index_1, _ = dropout_edge(edge_index_1, p=0.5, 
                                       force_undirected=True, 
                                       training=self.training)
        edge_index_2, _ = dropout_edge(edge_index_2, p=0.5, 
                                       force_undirected=True, 
                                       training=self.training)

        x_input = F.dropout(x_input, p=0.5, training=self.training)

        R0 = torch.relu(self.linear1(x_input))

        R_k1_1 = self.conv_k1_1(R0, edge_index_1)
        R_k1_2 = self.conv_k1_2(R0, edge_index_2)
        R1 = torch.cat((R_k1_1, R_k1_2), 1)

        R_k2_1 = self.conv_k2_1(R1, edge_index_1)
        R_k2_2 = self.conv_k2_2(R1, edge_index_2)
        R2 = torch.cat((R_k2_1, R_k2_2), 1)

        R_k3_1 = self.conv_k3_1(R2, edge_index_1)
        R_k3_2 = self.conv_k3_2(R2, edge_index_2)
        R3 = torch.cat((R_k3_1, R_k3_2), 1)

        R0 = F.dropout(R0, p=0.5, training=self.training)
        res0 = self.linear_r0(R0)
        R1 = F.dropout(R1, p=0.5, training=self.training)
        res1 = self.linear_r1(R1)
        R2 = F.dropout(R2, p=0.5, training=self.training)
        res2 = self.linear_r2(R2)
        R3 = F.dropout(R3, p=0.5, training=self.training)
        res3 = self.linear_r3(R3)

        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out

class MTGCN(torch.nn.Module):
    def __init__(self, args):
        super(MTGCN, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, 1, K=2, normalization="sym")

        self.lin1 = Linear(58, 100)
        self.lin2 = Linear(58, 100)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, data):
        edge_index, _ = dropout_edge(data.edge_index, p=0.5,
                                     force_undirected=True,
                                     num_nodes=data.x.size()[0],
                                     training=self.training)
        E = data.edge_index
        pb, _ = remove_self_loops(data.edge_index)
        pb, _ = add_self_loops(pb)

        x0 = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()

        neg_edge_index = negative_sampling(pb, data.num_nodes, data.num_edges)

        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        r_loss = pos_loss + neg_loss

        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x, r_loss, self.c1, self.c2

class EMOGI(torch.nn.Module):
    def __init__(self,args):
        super(EMOGI, self).__init__()
        self.args = args
        self.conv1 = ChebConv(58, 300, K=2)
        self.conv2 = ChebConv(300, 100, K=2)
        self.conv3 = ChebConv(100, 1, K=2)

    def forward(self, data):
        edge_index = data.edge_index
        x = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x

class Chebnet(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3):
        """
        Chebnet implementation using DGL's ChebConv.
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - k: Chebyshev polynomial order.
        """
        super(Chebnet, self).__init__()
        self.cheb1 = ChebConv(in_feats, hidden_feats, k)
        self.cheb2 = ChebConv(hidden_feats, hidden_feats, k)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        """
        Forward pass for Chebnet.
        
        Parameters:
        - g: DGL graph.
        - features: Input features tensor.
        
        Returns:
        - Output tensor after passing through Chebnet layers.
        """
        x = F.relu(self.cheb1(g, features))
        x = F.relu(self.cheb2(g, x))
        return self.mlp(x)

class GIN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GIN, self).__init__()
        # Define the first GIN layer
        self.gin1 = GINConv(
            nn.Sequential(
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ),
            'mean'  # Aggregation method: 'mean', 'max', or 'sum'
        )
        # Define the second GIN layer
        self.gin2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_feats, hidden_feats),
                nn.ReLU(),
                nn.Linear(hidden_feats, hidden_feats)
            ),
            'mean'
        )
        # MLP for final predictions
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        # Apply the first GIN layer
        x = F.relu(self.gin1(g, features))
        # Apply the second GIN layer
        x = F.relu(self.gin2(g, x))
        # Apply the MLP
        return self.mlp(x)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.sage2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        x = F.relu(self.sage1(g, features))
        x = F.relu(self.sage2(g, x))
        return self.mlp(x)

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=3):
        """
        Graph Attention Network (GAT).
        
        Parameters:
        - in_feats: Number of input features.
        - hidden_feats: Number of hidden layer features.
        - out_feats: Number of output features.
        - num_heads: Number of attention heads.
        """
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, num_heads, activation=F.relu)
        self.gat2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads, activation=F.relu)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats * num_heads, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        """
        Forward pass for GAT.
        
        Parameters:
        - g: DGL graph.
        - features: Input features tensor.
        
        Returns:
        - Output tensor after passing through GAT layers.
        """
        x = self.gat1(g, features)
        x = x.flatten(1)  # Flatten the output of multi-head attention
        x = self.gat2(g, x)
        x = x.flatten(1)  # Flatten the output again
        return self.mlp(x)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.gcn1 = GraphConv(in_feats, hidden_feats)
        self.gcn2 = GraphConv(hidden_feats, hidden_feats)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

    def forward(self, g, features):
        x = F.relu(self.gcn1(g, features))
        x = F.relu(self.gcn2(g, x))
        return self.mlp(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ##bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Ensure targets are of type float
        targets = targets.float()

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probas = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probas, 1 - probas)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class FastAttentionLayer(nn.Module):
    """
    Efficient multi-head dot-product attention layer for DGL graphs.

    - in_feats: input feature dimension
    - out_feats: output feature dimension per head (final output dim = heads * out_feats if concat=True)
    - heads: number of attention heads
    - dropout: attention dropout
    - concat: whether to concat heads (True) or average them (False)
    """
    def __init__(self, in_feats, out_feats, heads=4, dropout=0.1, concat=True, bias=True):
        super(FastAttentionLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.heads = heads
        self.concat = concat
        self.scale = 1.0 / math.sqrt(out_feats)

        # Project input features to heads * out_feats in one matrix multiply
        self.fc = nn.Linear(in_feats, heads * out_feats, bias=False)

        # Optional bias on output
        self.bias = nn.Parameter(torch.zeros(heads * out_feats)) if bias and concat else \
                    nn.Parameter(torch.zeros(out_feats)) if bias and not concat else None

        self.attn_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, g, h):
        """
        g: DGLGraph (must have same device as h)
        h: (N, in_feats)
        returns: (N, heads*out_feats) if concat else (N, out_feats)
        """
        device = h.device
        g = g.to(device)

        # Linear projection and reshape -> (N, heads, out_feats)
        Wh = self.fc(h).view(-1, self.heads, self.out_feats)  # (N, H, D)

        # Save Wh for message passing
        g.ndata['Wh'] = Wh

        # compute edge score = sum(Wh_u * Wh_v, dim=-1) * scale
        # We'll use built-in message functions: copy_u/v then compute inside apply_edges
        def edge_attention(edges):
            # edges.src['Wh'], edges.dst['Wh'] are (E, H, D)
            score = (edges.src['Wh'] * edges.dst['Wh']).sum(dim=-1)  # (E, H)
            return {'e': score * self.scale}

        g.apply_edges(edge_attention)

        # edge softmax across incoming edges per head (returns shape (E, H))
        e = g.edata.pop('e')  # (E, H)
        # edge_softmax works per edge scalar; for multi-head we need to flatten heads as separate "edge types"
        # Efficient trick: treat heads as extra dimension and compute softmax per-head manually using segment_softmax pattern
        # But DGL provides edge_softmax that accepts (E, 1) scalars. We'll compute per-head softmax manually:

        # Compute per-head softmax using scatter (vectorized)
        # Get destination node ids for each edge
        dst = g.edges()[1]  # (E,)
        # e: (E, H) -> we want normalized alpha: (E, H)
        # Use exponent + scatter by dst for stable softmax
        e_exp = torch.exp(e - e.max(dim=0, keepdim=True)[0])  # (E, H)
        # sum exp per destination node per head
        # create index for scatter add: (E, H) -> we will perform scatter_add on (num_nodes, H)
        num_nodes = g.num_nodes()
        denom = torch.zeros(num_nodes, self.heads, device=device).index_add_(0, dst, e_exp)
        alpha = e_exp / (denom[dst] + 1e-12)  # (E, H)

        # Apply attention dropout
        alpha = self.attn_dropout(alpha)

        # store alpha as edge data for message passing
        g.edata['alpha'] = alpha  # (E, H)

        # Message: m = Wh_src * alpha.unsqueeze(-1)  -> (E, H, D)
        def message_func(edges):
            # edges.src['Wh']: (E, H, D), edges.data['alpha']: (E, H)
            a = edges.data['alpha'].unsqueeze(-1)  # (E, H, 1)
            m = edges.src['Wh'] * a  # (E, H, D)
            return {'m': m}

        # Reduce: sum messages per dst -> (N, H, D)
        def reduce_func(nodes):
            # nodes.mailbox['m'] is (N, E_in, H, D) possibly large; but DGL will handle memory streaming
            m_sum = nodes.mailbox['m'].sum(dim=1)  # (N, H, D)
            return {'h_new': m_sum}

        g.update_all(message_func, reduce_func)

        h_new = g.ndata.pop('h_new')  # (N, H, D)

        # Combine heads
        if self.concat:
            h_new = h_new.reshape(-1, self.heads * self.out_feats)  # (N, H*D)
            if self.bias is not None:
                h_new = h_new + self.bias
        else:
            h_new = h_new.mean(dim=1)  # (N, D)
            if self.bias is not None:
                h_new = h_new + self.bias

        return self.activation(h_new)

# Requires: torch, dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import TAGConv
from dgl.nn.functional import edge_softmax

# ---------------------------
# Attention Layer (DGL)
# ---------------------------
class AttentionLayer(nn.Module):
    def __init__(self, hidden_feats):
        super(AttentionLayer, self).__init__()
        # maps concat([h_src, h_dst]) -> scalar score
        self.attn = nn.Linear(2 * hidden_feats, 1, bias=True)
        self.leaky_slope = 0.2

    def forward(self, g, h):
        """
        g: DGLGraph (can be batched)
        h: node features tensor (N, D)
        returns: updated node features (N, D)
        """
        with g.local_scope():
            g.ndata['h'] = h
            # compute unnormalized attention per edge
            def compute_edge_attention(edges):
                z = torch.cat([edges.src['h'], edges.dst['h']], dim=-1)  # (E, 2D)
                a = self.attn(z).squeeze(-1)  # (E,)
                return {'a': a}
            g.apply_edges(compute_edge_attention)

            # leaky relu
            g.edata['a'] = F.leaky_relu(g.edata['a'], negative_slope=self.leaky_slope)
            # normalize across incoming edges for each destination node
            g.edata['a'] = edge_softmax(g, g.edata['a'])  # (E,)

            # weighted message: m_ij = a_ij * h_i
            g.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h_new'))
            h_new = g.ndata['h_new']  # (N, D)
            return h_new

# ---------------------------
# TAG + Attention model (ATTAGv2)
# ---------------------------

class ATTAG(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, k=3, dropout=0.5, activation=F.relu):
        super(ATTAGv2, self).__init__()
        self.activation = activation
        # First TAGConv block
        self.tag1 = TAGConv(in_feats, hidden_feats, k=k, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_feats)

        # Attention layer
        self.attn = AttentionLayer(hidden_feats)

        # Second TAGConv block
        self.tag2 = TAGConv(hidden_feats, hidden_feats, k=k, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_feats)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_feats, out_feats)
        )

        # Residual projection if input dims differ
        self.res_proj1 = nn.Linear(in_feats, hidden_feats) if in_feats != hidden_feats else None

    def forward(self, g, features):
        device = features.device
        g = g.to(device)
        h = features

        # First TAGConv -> BN -> Act
        res1 = h
        x = self.tag1(g, h)  # (N, hidden)
        x = self.bn1(x)
        x = self.activation(x)

        # Attention-based aggregation (edge weights)
        a_out = self.attn(g, x)  # (N, hidden)

        # project residual if dims mismatch
        if self.res_proj1 is not None:
            res1 = self.res_proj1(res1)

        x = a_out + res1  # residual addition

        # Second TAGConv -> BN -> Act
        res2 = x
        x = self.tag2(g, x)
        x = self.bn2(x)
        x = self.activation(x)

        # add residual again
        x = x + res2

        x = self.dropout(x)
        out = self.mlp(x)  # (N, out_feats)
        return out
