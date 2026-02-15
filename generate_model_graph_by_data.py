

import pickle
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
# from sklearn.cluster import KMeans
# from kmodes.kmodes import KModes

# from scipy.spatial.distance import pdist, squareform
# from scipy.cluster.hierarchy import linkage, fcluster
# from tools import get_g, generate_group_triples_v1, add_feature_to_model_graph_nodes, create__model_graph
from kg_utiles import KnowledgeGraphUtils as kgu
from typing import Dict, Tuple, List

def build_model_graph(args) -> None:
    """Builds and saves a model graph from the main graph data.
    
    Args:
        args: Object containing configuration parameters (data_path, num_rel, data_name)
    """
    print("Building model graph from main graph...")
    
    # Load data efficiently
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Construct training graph
    if args.new_data=="old":
        train_data = (data['train_graph']['train'] + 
                 data['train_graph']['valid'] + 
                 data['train_graph']['test'])
       
    else:
         train_data = (data['train_graph']['train'] + 
                 data['train_graph']['valid'] )

    # train_g = get_g(train_data)
    train_g = kgu.create_directed_graph(np.array(train_data))
    num_nodes = train_g.num_nodes()
    
    # Prepare triples efficiently
    triples = torch.stack([
        train_g.edges()[0],
        train_g.edata['rel'],
        train_g.edges()[1]
    ]).T.tolist()
    
    # Initialize and populate node features
    # features = _create_node_features(train_g, num_nodes, args.num_rel)
    # features_in_matrix_form = _create_node_complex_features(train_g, num_nodes, args.num_rel)
    # features_in_matrix_form =create_rel_to_rel_binary_features(train_g,args.num_rel)
    # features_in_matrix_form1 =create_node_rel_matrix(train_g,args.num_rel)
    features_in_matrix_form1 =create_node_structural_matrices(train_g,args.num_rel)
    # print(f"the features size is   : { features_in_matrix_form.shape} ")
    # print(f"the features size is   : { features_in_matrix_form1.shape} ")
    # print(f"the features is  : { max(features_in_matrix_form.sum(axis=(1, 2)))} ")
    # print(f" is it equal {features_in_matrix_form[:,:,250] }")
    # print(f" is it equal {features_in_matrix_form1[:,:,250] }")
    # print(f" is it equal {features_in_matrix_form[:,:,0] ==features_in_matrix_form1[:,:,0]}")



    
    # Calculate entity statistics
    nentity = len(np.unique(np.array(triples)[:, [0, 2]]))
    print(f"Number of relations: {args.num_rel}")
    print(f"Number of entities: {nentity}")
    
    # Group nodes by similarity
    # groups, unique_rows = partitionNodeBysimilarty(features)
    # groups1, unique_rows1 = partition_nodes_by_matrix_similarity(features_in_matrix_form1)
    groups1, unique_rows1 = partition_nodes_by_2r_matrix_similarity(features_in_matrix_form1)
    # print(f"the groups in matrix form is {groups1}")

    # print(f"Number of unique feature groups: {len(groups)}")
    print(f"Number of unique feature in matrix form groups: {len(groups1)}")
    
    # Create entity-type mapping
    ent_type = _create_entity_type_mapping(groups1)
    print(f"Entity type mapping size: {len(ent_type)}")
    
    # Save unique features
    _save_unique_features(unique_rows1, args)
    
    # Generate group triples and relations
    entity_type_triples, inner_rel, output_relations, input_relations = (
        # generate_group_triples_v1(triples, ent_type, args.num_rel)
        kgu.generate_group_triples(triples,ent_type,args.num_rel)
    )
    # Validate entity_type_triples
   
    
    # Build and enhance model graph
    # model_graph = get_g(list(entity_type_triples))
    if args.is_wieghted_model_graph :
        model_graph = kgu.create_directed_graph(entity_type_triples,edge_key="weight")
    else:
        model_graph = kgu.create_directed_graph(entity_type_triples,edge_key="weight", is_weighted=False)
    if not args.is_directed_model_graph:
        model_graph = kgu.undirected_graph(model_graph)


    # model_graph = add_feature_to_model_graph_nodes(
    #     model_graph, inner_rel, output_relations, input_relations, args.num_rel
    # )
    model_graph = kgu.add_node_features(
        model_graph, inner_rel, output_relations, input_relations, args.num_rel
    )
    
    print(f"the model graph is {model_graph}")
    # print(f"the model graph edges are  {model_graph.edata}")
   
    
    # Prepare and save final data
    model_features = model_graph.ndata["feat"]
    print(f"Model graph triples count: {len(entity_type_triples)}")
    
    save_data = {
        'model_graph': {
            'triples': entity_type_triples,
            'ent_type': ent_type,
            'proper_feature': model_features
        }
    }
    if args.new_data=='old':
        with open(f'./dataset/{args.data_name}_model_graph.pkl', 'wb') as f:
            pickle.dump(save_data, f) 
    else :
        with open(f'./dataset/new_data/{args.data_name}_model_graph.pkl', 'wb') as f:
            pickle.dump(save_data, f)

def _create_node_features(train_g, num_nodes: int, num_rel: int) -> torch.Tensor:
    """Create node features based on graph relations."""
    features = torch.zeros((num_nodes, 2 * num_rel))
    src, dst = train_g.edges()
    etypes = train_g.edata['rel']
    
    src = src.to(torch.long)
    dst = dst.to(torch.long)
    etypes = etypes.to(torch.long)
    
    features[src, etypes] = 1  # Outgoing relations
    features[dst, etypes + num_rel] = 1  # Incoming relations
    return features
import torch

def create_node_rel_matrix(g, num_rel: int) -> torch.Tensor:
    """
    Returns tensor of shape (num_rel, num_rel, num_nodes)
    
    features[r1, r2, v] == 1.0 
        if node v has at least one neighbor u such that:
            - there is an edge between v and u using relation r1
            - AND node u participates (in or out) in relation r2
    """
    src, dst = g.edges()
    etype = g.edata['rel'].long()          # (E,)
    device = g.device
    N = g.num_nodes()

    # 1. Which relations touch each node? (in + out)
    touch = torch.zeros(num_rel, N, dtype=torch.bool, device=device)
    touch.index_put_((etype, src), torch.tensor(True, device=device), accumulate=True)
    touch.index_put_((etype, dst), torch.tensor(True, device=device), accumulate=True)
    # touch: (num_rel, N)

    # 2. Undirected neighbor adjacency matrix
    adj = torch.zeros(N, N, dtype=torch.bool, device=device)
    adj[src, dst] = True
    adj[dst, src] = True                     # treat graph as undirected for neighbors

    # 3. Magic einsum: for every node v, touch[r1, v] * adj[v, u] * touch[r2, u] → sum over u
    #    → becomes features[r1, r2, v]
    features = torch.einsum('rv, vu, wu -> rwu', touch.float(), adj.float(), touch.float())
    
    # Make binary
    features = (features > 0).float()        # shape: (num_rel, num_rel, N)

    return features  # Final shape: (9, 9, 2746) in your case


def create_rel_to_rel_binary_features(g, num_rel: int) -> torch.Tensor:
    """
    Returns tensor of shape (num_rel, num_rel, num_nodes) where:
        features[r1, r2, v] == 1 
        if there is at least one neighbor u of v such that:
            - (v --r1--> u) or (u --r1--> v)
            - AND u has at least one edge (in or out) with relation r2
    
    Extremely powerful local structural feature!
    """
    device = g.device
    N = g.num_nodes()
    src, dst = g.edges()
    etype = g.edata['rel'].long()  # (E,)

    # Step 1: Compute which relations touch each node (in + out)
    # rel_touch[r, node] = 1 if node has at least one edge with relation r
    rel_touch = torch.zeros(num_rel, N, dtype=torch.bool, device=device)
    
    rel_touch.index_put_((etype, src), torch.ones_like(etype, dtype=torch.bool), accumulate=True)
    rel_touch.index_put_((etype, dst), torch.ones_like(etype, dtype=torch.bool), accumulate=True)
    # Now rel_touch[r, u] = True if node u uses relation r at all

    # Step 2: For each node v, get its neighbors (undirected view)
    # We'll build adjacency list for neighbors
    neigh_list = [set() for _ in range(N)]
    for s, d in zip(src.tolist(), dst.tolist()):
        neigh_list[s].add(d)
        neigh_list[d].add(s)  # treat as undirected for neighbors

    # Step 3: Build final (num_rel, num_rel, N) binary tensor
    features = torch.zeros(num_rel, num_rel, N, dtype=torch.float, device=device)

    for v in range(N):
        neighbors = neigh_list[v]
        if len(neighbors) == 0:
            continue

        # Get all relations used to reach neighbors (r1)
        r1_mask = torch.zeros(num_rel, N, dtype=torch.bool, device=device)
        for u in neighbors:
            # Edges between v and u
            edge_mask = ((src == v) & (dst == u)) | ((src == u) & (dst == v))
            if edge_mask.any():
                r1s = etype[edge_mask]
                r1_mask[r1s, v] = True

        # Get all relations used by neighbors (r2)
        r2_used = rel_touch[:, list(neighbors)].any(dim=1)  # (num_rel,)

        # Broadcast: all r1 that reach neighbors × all r2 used by those neighbors
        r1_indices = r1_mask[:, v].nonzero(as_tuple=True)[0]  # relations touching v via neighbors
        if len(r1_indices) > 0 and r2_used.any():
            features[r1_indices] += r2_used.float().unsqueeze(1)  # outer product

    # Clamp to 0/1
    features = (features > 0).to(torch.float)

    return features  # shape: (num_rel, num_rel, num_nodes)
def _create_node_complex_features(train_g, num_nodes: int, num_rel: int) -> torch.Tensor:
    """Create node features based on graph relations."""
    features = torch.zeros((num_nodes,num_rel,  num_rel+1 ))
    src, dst = train_g.edges()
    etypes = train_g.edata['rel']
    
    src = src.to(torch.long)
    dst = dst.to(torch.long)
    etypes = etypes.to(torch.long)
    
    features[src,0, etypes] = 1  # Outgoing relations
    features[dst,0, etypes] = 1  # Incoming relations
    return features

def _create_entity_type_mapping(groups: Dict) -> Dict:
    """Create a mapping of entities to their types."""
    return {ent: type_ for type_, val in groups.items() for ent in val}

def _save_unique_features(unique_rows, args) -> None:
    """Save unique features to a pickle file."""
    with open( args.unique_features_for_model_graph, "wb") as f:
        pickle.dump(unique_rows, f)

def partitionNodeBysimilarty(features):

    unique_rows, indices = np.unique(features, axis=0, return_inverse=True)

# Group similar rows
    groups = {i: np.where(indices == i)[0].tolist() for i in range(len(unique_rows))}
    # print(f"the groups of features matrixs is {groups}")
    return groups, unique_rows


def create_node_structural_matrices(g, num_rel: int) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (num_nodes, 2*num_rel, 2*num_rel)
    
    For each node v:
      - Rows 0 to R-1: incoming relations r_in to v
        If no incoming r_in, row is zero
        Else, for each neighbor u sending r_in to v:
          - Columns 0 to R-1: 1 if u has incoming relation w
          - Columns R to 2R-1: 1 if u has outgoing relation w
      - Rows R to 2R-1: outgoing relations r_out from v
        Similar, for each neighbor u reached by r_out
    """
    device = g.device
    N = g.num_nodes()
    src, dst = g.edges()
    etype = g.edata['rel'].long()
    R = num_rel

    # Profile: (2R, N) - relations each node participates in
    # profile[0:R, u]: incoming to u (count of times u receives each rel)
    # profile[R:2R, u]: outgoing from u (count of times u sends each rel)
    profile = torch.zeros(2 * R, N, device=device)

    one_hot = F.one_hot(etype, num_classes=R).float().t()  # (R, E)

    # Incoming: u receives rel (dst)
    profile[:R].index_add_(1, dst, one_hot)

    # Outgoing: u sends rel (src)
    profile[R:].index_add_(1, src, one_hot)

    # Features: (N, 2R, 2R)
    features = torch.zeros(N, 2 * R, 2 * R, device=device)

    # Fill with loop to avoid dimension error
    for r in range(R):
        mask = (etype == r)
        if not mask.any():
            continue

        # Incoming: u --r--> v → row r of v += profile of u
        u = src[mask]
        v = dst[mask]
        features[v, r] += profile[:, u].t()

        # Outgoing: v --r--> u → row (R + r) of v += profile of u
        u = dst[mask]
        v = src[mask]
        features[v, R + r] += profile[:, u].t()

    # Binary
    features = (features > 0).float()

    return features

def partition_nodes_by_matrix_similarity(features: torch.Tensor):
    """
    Groups nodes that have EXACTLY identical (num_rel × num_rel) feature matrices.
    
    Input:
        features: torch.Tensor of shape (num_rel, num_rel, num_nodes)
                  e.g., (9, 9, 2746)
    
    Returns:
        groups: dict {group_id: [list of node ids]}
        unique_matrices: np.array of shape (num_groups, num_rel, num_rel)
    """
    # Move to CPU + convert to numpy: (num_nodes, num_rel, num_rel)
    matrices = features.permute(2, 0, 1).cpu().numpy()  # shape: (N, R, R)

    # Reshape to treat each matrix as a 1D vector of length R*R
    flat_matrices = matrices.reshape(matrices.shape[0], -1)  # (N, R*R)

    # Find unique matrices (exact match)
    unique_flat, inverse_indices = np.unique(
        flat_matrices, axis=0, return_inverse=True
    )

    # Reconstruct unique matrices back to 2D
    num_groups = unique_flat.shape[0]
    R = features.shape[0]
    unique_matrices = unique_flat.reshape(num_groups, 2*R, 2*R)

    # Build groups: {group_id: [node_ids]}
    groups = {}
    for group_id in range(num_groups):
        node_ids = np.where(inverse_indices == group_id)[0].tolist()
        groups[group_id] = node_ids

    return groups, unique_matrices


def partition_nodes_by_2r_matrix_similarity(features: torch.Tensor):
    """
    Groups nodes that have EXACTLY identical (2R × 2R) structural matrices.
    
    Input:
        features: torch.Tensor of shape (num_nodes, 2*num_rel, 2*num_rel)
                  e.g., (2746, 18, 18) if num_rel=9
    
    Returns:
        groups: dict {group_id: [list of node ids]}  → nodes with identical matrix
        unique_matrices: torch.Tensor of shape (num_groups, 2R, 2R)
    """
    # Move to CPU for np.unique (much faster and more reliable than torch.unique for 2D)
    matrices_np = features.cpu().numpy()                    # (N, 2R, 2R)
    N, height, width = matrices_np.shape
    assert height == width, "Matrix must be square"

    # Flatten each matrix into a 1D vector: (N, 2R*2R)
    flat_matrices = matrices_np.reshape(N, -1)              # (N, 4*R²)

    # Find unique rows (i.e. unique matrices)
    unique_flat, inverse_indices = np.unique(
        flat_matrices, axis=0, return_inverse=True
    )

    num_groups = unique_flat.shape[0]
    # print(f"Found {num_groups} unique structural roles (out of {N} nodes)")

    # Reconstruct unique matrices back to (num_groups, 2R, 2R)
    unique_matrices = unique_flat.reshape(num_groups, height, width)
    unique_matrices = torch.tensor(unique_matrices, device=features.device)

    # Build groups
    groups = {}
    for group_id in range(num_groups):
        node_ids = np.where(inverse_indices == group_id)[0].tolist()
        # if len(node_ids) > 1:  # optional: only show non-singletons
        #     print(f"  Role {group_id}: {len(node_ids)} nodes")
        groups[group_id] = node_ids

    return groups, unique_matrices