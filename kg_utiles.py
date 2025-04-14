import pickle
import os
import random
import dgl
import torch
import json
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Any, Optional
from pathlib import Path

class KnowledgeGraphUtils:
    """Utilities for creating and manipulating knowledge graphs with DGL."""

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seeds for reproducibility."""
        dgl.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def serialize(data: Any) -> bytes:
        """Serialize data using pickle."""
        return pickle.dumps(data)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize data using pickle."""
        try:
            return pickle.loads(data)
        except pickle.UnpicklingError as e:
            raise ValueError(f"Failed to deserialize data: {e}")

    @staticmethod
    def create_directed_graph(triples: np.ndarray, edge_key: str = "rel") -> dgl.DGLGraph:
        """Create a directed DGL graph from triples.

        Args:
            triples: Array of shape (N, 3) with [head, relation, tail].
            edge_key: Name for edge data storing relations.

        Returns:
            DGL graph with edges head→tail and relations in edata[edge_key].
        """
        if triples.shape[1] != 3:
            raise ValueError("Triples must have shape (N, 3)")
        g = dgl.graph((triples[:, 0], triples[:, 2]))
        g.edata[edge_key] = torch.tensor(triples[:, 1], dtype=torch.float32)
        return g

    @staticmethod
    def create_bidirectional_graph(triples: torch.Tensor, num_rel: int) -> dgl.DGLGraph:
        """Create a bidirectional DGL graph from triples.

        Args:
            triples: Tensor of shape (N, 3) with [head, relation, tail].
            num_rel: Number of relation types.

        Returns:
            DGL graph with edges head↔tail and relation types in edata['type'].
        """
        src = torch.cat([triples[:, 0], triples[:, 2]])
        dst = torch.cat([triples[:, 2], triples[:, 0]])
        rels = torch.cat([triples[:, 1], triples[:, 1] + num_rel])
        g = dgl.graph((src, dst))
        g.edata['type'] = rels.to(torch.long)
        return g

    @staticmethod
    def map_head_relation_to_tail(triples: List[Tuple[int, int, int]]) -> Tuple[Dict, Dict]:
        """Map head-relation to tails and relation-tail to heads.

        Args:
            triples: List of (head, relation, tail) tuples.

        Returns:
            Tuple of (hr2t, rt2h) dictionaries.
        """
        hr2t, rt2h = defaultdict(list), defaultdict(list)
        for h, r, t in triples:
            hr2t[(h, r)].append(t)
            rt2h[(r, t)].append(h)
        return hr2t, rt2h

    @staticmethod
    def map_support_query_triples(
        sup_triples: List[Tuple[int, int, int]],
        que_triples: List[Tuple[int, int, int]]
    ) -> Tuple[Dict, Dict]:
        """Map head-relation to tails and relation-tail to heads for support and query triples.

        Args:
            sup_triples: Support triples (head, relation, tail).
            que_triples: Query triples (head, relation, tail).

        Returns:
            Tuple of (que_hr2t, que_rt2h) dictionaries for query triples.
        """
        hr2t, rt2h = defaultdict(list), defaultdict(list)
        for h, r, t in sup_triples + que_triples:
            hr2t[(h, r)].append(t)
            rt2h[(r, t)].append(h)
        que_hr2t = {(h, r): hr2t[(h, r)] for h, r, t in que_triples}
        que_rt2h = {(r, t): rt2h[(r, t)] for h, r, t in que_triples}
        return que_hr2t, que_rt2h

    @staticmethod
    def write_results(results: Dict, save_path: str,args:Any) -> None:
        """Write evaluation results to a JSON file.

        Args:
            results: Dictionary containing evaluation metrics.
            save_path: Path to save the JSON file.
        """
        print("write result on disk ")
        path1 = f"C:/Users/MEKhani/Documents/results/{args.data_name}/result.json"
        os.makedirs(os.path.dirname(path1), exist_ok=True)  # Ensure the directory exists
        try:
            with open(path1, "a", encoding="utf-8") as f:  # Open file in write mode
                json.dump(results, f, indent=4)  # Save dictionary as JSON
        except PermissionError as e:
            raise PermissionError(f"Cannot write to {save_path}: {e}")

        

    @staticmethod
    def get_num_relations(data_path: str) -> int:
        """Compute the number of unique relations from data.

        Args:
            data_path: Path to pickled data file.

        Returns:
            Number of unique relations.
        """
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            return len(np.unique(np.array(data['train_graph']['train'])[:, 1]))
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Failed to load relations from {data_path}: {e}")

    @staticmethod
    def generate_group_triples(
        triples: np.ndarray,
        entity_types: Dict[int, int],
        num_rel: int
    ) -> Tuple[np.ndarray, Dict[int, Set[int]], Dict[int, Set[int]], Dict[int, Set[int]]]:
        """Generate group-level triples and relation mappings.

        Args:
            triples: Array of shape (N, 3) with [e1, rel, e2].
            entity_types: Mapping of entities to group IDs.
            num_rel: Total number of relations.

        Returns:
            Tuple of (group_triples, inner_relations, output_relations, input_relations).
        """
        group_rels = defaultdict(list)
        inner_rels = defaultdict(set)
        out_rels = defaultdict(set)
        in_rels = defaultdict(set)

        for e1, rel, e2 in triples:
            k1 = entity_types.get(e1)
            k2 = entity_types.get(e2)
            if k1 is not None and k2 is not None:
                if k1 != k2:
                    group_rels[(k1, k2)].append(rel)
                    out_rels[k1].add(rel)
                    in_rels[k2].add(rel)
                else:
                    inner_rels[k1].add(rel)

        group_triples = {(k1, len(rels) / num_rel, k2) for (k1, k2), rels in group_rels.items()}
        triples_list = [[k1, score, k2] for k1, score, k2 in group_triples]
        ndarray_grouptriples=  np.array(triples_list, dtype=np.float64)
        return ndarray_grouptriples, inner_rels, out_rels, in_rels

    @staticmethod
    def add_node_features(
        graph: dgl.DGLGraph,
        inner_rels: Dict[int, Set[int]],
        out_rels: Dict[int, Set[int]],
        in_rels: Dict[int, Set[int]],
        num_rel: int
    ) -> dgl.DGLGraph:
        """Add features to graph nodes based on relations.

        Args:
            graph: Input DGL graph.
            inner_rels: Internal relations per node.
            out_rels: Outgoing relations per node.
            in_rels: Incoming relations per node.
            num_rel: Number of relations.

        Returns:
            Graph with updated node features.
        """
        num_nodes = graph.num_nodes()
        features = torch.zeros(num_nodes, num_rel * 3, dtype=torch.float)

        for node, rels in inner_rels.items():
            features[node, list(rels)] = 1
        for node, rels in out_rels.items():
            features[node, [num_rel + r for r in rels]] = 1
        for node, rels in in_rels.items():
            features[node, [2 * num_rel + r for r in rels]] = 1

        graph.ndata['feat'] = features
        return graph

    @staticmethod
    def load_inductive_test_data(args: Any) -> Tuple[Any, dgl.DGLGraph, Dict[int, int]]:
        """Load inductive test dataset and create training graph.

        Args:
            args: Object with data_path, num_rel, and other attributes.

        Returns:
            Tuple of (test_dataset, training_graph, entity_types).
        """
        from my_dataset import KGEEvalDataset  # Import here to avoid circular imports

        try:
            with open(args.data_path, 'rb') as f:
                data = pickle.load(f)['ind_test_graph']
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Failed to load inductive test data from {args.data_path}: {e}")

        entity_types = KnowledgeGraphUtils.get_entity_types(data, args)
        train_triples = np.array(data['train'])
        num_entities = len(np.unique(train_triples[:, [0, 2]]))

        hr2t, rt2h = KnowledgeGraphUtils.map_head_relation_to_tail(train_triples.tolist())
        test_dataset = KGEEvalDataset(args, data['test'], num_entities, hr2t, rt2h)
        training_graph = KnowledgeGraphUtils.create_bidirectional_graph(
            torch.tensor(train_triples, dtype=torch.long), args.num_rel
        )

        return test_dataset, training_graph, entity_types

    @staticmethod
    def get_entity_types(data: Dict, args: Any) -> Dict[int, int]:
        """Assign type indices to entities based on relation patterns.

        Args:
            data: Dictionary with 'train', 'valid', 'test' triples.
            args: Object with num_rel and unique_features_for_model_graph attributes.

        Returns:
            Dictionary mapping entities to type indices.
        """
        all_triples = np.array(data['train'] + data['valid'] + data['test'])
        graph = KnowledgeGraphUtils.create_directed_graph(all_triples)
        num_nodes = graph.num_nodes()
        num_rel = len(np.unique(all_triples[:, 1]))

        features = torch.zeros(num_nodes, 2 * args.num_rel, dtype=torch.float)
        src, dst = graph.edges()
        etypes = graph.edata['rel'].to(torch.long)
        features[src, etypes] = 1
        features[dst, etypes + args.num_rel] = 1

        try:
            with open(args.unique_features_for_model_graph, 'rb') as f:
                unique_rows = pickle.load(f)
        except FileNotFoundError as e:
            raise ValueError(f"Unique features file not found: {e}")

        unique_rows = torch.tensor(unique_rows, dtype=torch.float)
        _, feature_types = torch.unique(features, dim=0, return_inverse=True)
        return {i: t.item() for i, t in enumerate(feature_types)}

    @staticmethod
    def create_model_graph(triples: List[Tuple[int, float, int]]) -> dgl.DGLGraph:
        """Create a heterogeneous DGL graph from group-level triples.

        Args:
            triples: List of (src, weight, dst) tuples.

        Returns:
            Heterogeneous DGL graph with edge weights.
        """
        if not triples:
            return dgl.heterograph({('node', 'weight', 'node'): ([], [])})

        src_nodes, weights, dst_nodes = zip(*triples)
        graph = dgl.heterograph({
            ('node', 'weight', 'node'): (src_nodes, dst_nodes)
        })
        graph.edata['weight'] = torch.tensor(weights, dtype=torch.float)
        return graph