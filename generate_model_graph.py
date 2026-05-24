import pickle
import numpy as np
import torch
import torch.nn.functional as F
import dgl

from kg_utiles import KnowledgeGraphUtils as kgu
from typing import Dict, Tuple, Any


# ────────────────────────────────────────────────────────────
# Main: Unified model graph builder
# ────────────────────────────────────────────────────────────

def build_model_graph(args: Any) -> dgl.DGLGraph:
    """Build and save model graph. Dispatches to relation_base or entity_base logic.
    
    Args:
        args: Configuration object (data_path, num_rel, data_name, 
              model_graph_type, is_relation_model_graph, is_weighted_model_graph,
              is_directed_model_graph, benchmark, unique_features_for_model_graph)
    
    Returns:
        model_graph: DGL graph of entity types
    """
    print("Building model graph from main graph...")

    # ── 1. Load data ──────────────────────────────────────────
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    # ── 2. Construct training graph ───────────────────────────
    train_data = _get_train_data(data, args.benchmark)
    train_g = kgu.create_directed_graph(np.array(train_data))

    # ── 3. Extract triples ────────────────────────────────────
    triples = torch.stack([
        train_g.edges()[0],
        train_g.edata['rel'],
        train_g.edges()[1]
    ]).T.tolist()

    # Statistics
    nentity = len(np.unique(np.array(triples)[:, [0, 2]]))
    print(f"Number of relations: {args.num_rel}")
    print(f"Number of entities: {nentity}")

    # ── 4. Get entity type mapping ────────────────────────────
    if args.model_graph_type =="relation_base":
        ent_type = _get_ent_type_from_features(train_g, args)
        generate_fn = kgu.generate_group_triples
    else:
        ent_type = data['train_graph']['ent_type']
        generate_fn = kgu.generate_group_triples_by_for_hito

    print(f"Entity type mapping size: {len(ent_type)}")
    print(f"Unique entity types: {len(set(ent_type.values()))}")

    # ── 5. Generate entity-type triples ───────────────────────
    entity_type_triples, inner_rel, output_relations, input_relations = generate_fn(
        triples, ent_type, args.num_rel
    )

    # ── 6. Build model graph ──────────────────────────────────
    model_graph = _create_model_graph(entity_type_triples, args)

   

    print(f"Model graph: {model_graph}")
    print(f"Model graph triples count: {len(entity_type_triples)}")

    # ── 8. Save ───────────────────────────────────────────────
    _save_model_graph(model_graph, entity_type_triples, ent_type, args)

    return model_graph


# ────────────────────────────────────────────────────────────
# Helper: Get training data based on benchmark
# ────────────────────────────────────────────────────────────

def _get_train_data(data: dict, benchmark: str) -> list:
    """Extract training triples based on benchmark type."""
    tg = data['train_graph']
    if benchmark == "dataset":
        return tg['train'] + tg['valid'] + tg['test']
    else:
        return tg['train'] + tg['valid']


# ────────────────────────────────────────────────────────────
# Helper: Entity type from features (relation_base)
# ────────────────────────────────────────────────────────────

def _get_ent_type_from_features(train_g: dgl.DGLGraph, args: Any) -> dict:
    """Compute entity types by grouping nodes with similar relation features."""
    num_nodes = train_g.num_nodes()
    features = _create_node_features(train_g, num_nodes, args.num_rel)
    groups, unique_rows = _partition_nodes_by_similarity(features)

    print(f"Number of unique feature groups: {len(groups)}")

    # Save unique features
    with open(args.unique_features_for_model_graph, "wb") as f:
        pickle.dump(unique_rows, f)

    return _create_entity_type_mapping(groups)


# ────────────────────────────────────────────────────────────
# Helper: Create model graph from entity-type triples
# ────────────────────────────────────────────────────────────

def _create_model_graph(entity_type_triples: np.ndarray, args: Any) -> dgl.DGLGraph:
    """Build DGL graph from entity-type triples with appropriate settings."""
   
    model_graph = kgu.create_directed_graph(
            entity_type_triples,
            edge_key="weight",
            is_weighted=args.is_weighted_model_graph
        )
    if not args.is_directed_model_graph:
            model_graph = kgu.undirected_graph(model_graph)

    return model_graph


# ────────────────────────────────────────────────────────────
# Helper: Save model graph
# ────────────────────────────────────────────────────────────

def _save_model_graph(
    model_graph: dgl.DGLGraph,
    entity_type_triples: np.ndarray,
    ent_type: dict,
    args: Any
) -> None:
    """Save model graph data to pickle file."""
    save_data = {
        'model_graph': {
            'triples': entity_type_triples,
            'ent_type': ent_type,
        }
    }

    

    # Determine save path
    if args.benchmark == "grail":
        save_path = f'./dataset/{args.data_name}_model_graph.pkl'
    else:
        save_path = f'./dataset/new_data/{args.data_name}_model_graph.pkl'

    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Model graph saved to {save_path}")


# ────────────────────────────────────────────────────────────
# Node feature creation
# ────────────────────────────────────────────────────────────

def _create_node_features(train_g: dgl.DGLGraph, num_nodes: int, num_rel: int) -> torch.Tensor:
    """Create node features based on incoming/outgoing relations.
    
    Returns:
        features: (num_nodes, 2 * num_rel) tensor
            - Columns 0 to num_rel-1: outgoing relations
            - Columns num_rel to 2*num_rel-1: incoming relations
    """
    features = torch.zeros((num_nodes, 2 * num_rel), device=train_g.device)
    src, dst = train_g.edges()
    etypes = train_g.edata['rel'].long()

    features[src, etypes] = 1                    # Outgoing
    features[dst, etypes + num_rel] = 1          # Incoming
    return features


# ────────────────────────────────────────────────────────────
# Node partitioning by feature similarity
# ────────────────────────────────────────────────────────────

def _partition_nodes_by_similarity(features: torch.Tensor) -> Tuple[dict, np.ndarray]:
    """Group nodes with identical feature vectors.
    
    Args:
        features: (N, D) tensor of node features
    
    Returns:
        groups: {group_id: [node_ids]}
        unique_rows: (num_groups, D) array of unique feature vectors
    """
    features_np = features.cpu().numpy()
    unique_rows, indices = np.unique(features_np, axis=0, return_inverse=True)
    groups = {i: np.where(indices == i)[0].tolist() for i in range(len(unique_rows))}
    return groups, unique_rows


def _create_entity_type_mapping(groups: dict) -> dict:
    """Convert {group_id: [nodes]} to {node: group_id}."""
    return {ent: type_ for type_, val in groups.items() for ent in val}