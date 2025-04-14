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
