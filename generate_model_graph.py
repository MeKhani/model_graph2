

import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
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
    train_data = (data['train_graph']['train'] + 
                 data['train_graph']['valid'] + 
                 data['train_graph']['test'])
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
    features = _create_node_features(train_g, num_nodes, args.num_rel)
    
    # Calculate entity statistics
    nentity = len(np.unique(np.array(triples)[:, [0, 2]]))
    print(f"Number of relations: {args.num_rel}")
    print(f"Number of entities: {nentity}")
    
    # Group nodes by similarity
    groups, unique_rows = partitionNodeBysimilarty(features)
    print(f"Number of unique feature groups: {len(groups)}")
    
    # Create entity-type mapping
    ent_type = _create_entity_type_mapping(groups)
    print(f"Entity type mapping size: {len(ent_type)}")
    
    # Save unique features
    _save_unique_features(unique_rows, args)
    
    # Generate group triples and relations
    entity_type_triples, inner_rel, output_relations, input_relations = (
        # generate_group_triples_v1(triples, ent_type, args.num_rel)
        kgu.generate_group_triples(triples,ent_type,args.num_rel)
    )
    # Validate entity_type_triples
   
    
    # Build and enhance model graph
    # model_graph = get_g(list(entity_type_triples))
    model_graph = kgu.create_directed_graph(entity_type_triples,edge_key="weight")
    # model_graph = add_feature_to_model_graph_nodes(
    #     model_graph, inner_rel, output_relations, input_relations, args.num_rel
    # )
    model_graph = kgu.add_node_features(
        model_graph, inner_rel, output_relations, input_relations, args.num_rel
    )
    
    print(f"the model graph is {model_graph}")
   
    
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
    
    with open(f'./dataset/{args.data_name}_model_graph.pkl', 'wb') as f:
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