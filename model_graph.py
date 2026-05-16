import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv
import dgl
import torch
class GCNWithWeightEdge(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCNWithWeightEdge, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, out_feats, allow_zero_in_degree=True)

    def forward(self, graph, node_feats):
        # # Compute edge weights from edge features
        # edge_weights = self.edge_fc(edge_feats).squeeze(-1)
        # edge_weights = self.edge_norm(graph, edge_weights)
        edge_weights = graph.edata["weight"]
        
        # First GCN layer
        h = self.conv1(graph, node_feats, edge_weight=edge_weights)
        h = F.relu(h)

        # Second GCN layer
        h = self.conv2(graph, h, edge_weight=edge_weights)
        return h

class WeightedGraphGNN(nn.Module):
   def __init__(self,args,):
        super(WeightedGraphGNN, self).__init__()
        in_feats =args.emb_dim
        # print(f"input feature is  in gnn {in_feats}")
        self.args = args
        hidden_feats =args.emb_dim
        out_feats =args.emb_dim
        self.encoder = GCNWithWeightEdge(in_feats, hidden_feats, out_feats)
    
   def forward(self, graph, features):
        # Encode
        graph = graph.to(self.args.gpu)
        features = features.to(self.args.gpu)
        z = self.encoder(graph, features)
        # Decode
        return z
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl
import torch

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=4):
        super(GAT, self).__init__()
        
        self.conv1 = dglnn.GATConv(
            in_feats, 
            hidden_feats, 
            num_heads=num_heads,
            feat_drop=0.1,
            attn_drop=0.1,
            residual=False,
            activation=F.elu,
            allow_zero_in_degree=True
        )
        

        self.conv2 = dglnn.GATConv(
            hidden_feats * num_heads, 
            out_feats, 
            num_heads=1,
            feat_drop=0.1,
            attn_drop=0.1,
            residual=False,
            activation=None,
            allow_zero_in_degree=True
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_feats * num_heads)
        self.layer_norm2 = nn.LayerNorm(out_feats)
        self.dropout = nn.Dropout(0.1)

    def forward(self, graph, node_feats):

        h = self.conv1(graph, node_feats)
        h = h.view(h.size(0), -1)  # [N, num_heads * hidden_feats]
        h = self.layer_norm1(h)
        h = self.dropout(h)
        
      
        h = self.conv2(graph, h)
        h = h.squeeze(1)  # [N, out_feats]
        h = self.layer_norm2(h)
        
        return h

class WeightedGATGraphGNN(nn.Module):
    def __init__(self, args):
        super(WeightedGATGraphGNN, self).__init__()
        self.args = args
        in_feats = args.emb_dim
        hidden_feats = args.emb_dim
        out_feats = args.emb_dim
        
        self.encoder = GAT(in_feats, hidden_feats, out_feats, num_heads=4)
    
    def forward(self, graph, features):
        device = torch.device(f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu')
        graph = graph.to(device)
        features = features.to(device)
        z = self.encoder(graph, features)
        return z