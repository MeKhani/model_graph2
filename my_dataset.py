from typing import Tuple, List, Dict, Any
import torch
import numpy as np
import lmdb
from torch.utils.data import Dataset, DataLoader
# from tools import deserialize
import kg_utiles


class SubgraphDataset(Dataset):
    """Base dataset class for subgraphs stored in LMDB."""
    
    def __init__(self, args: Any, db_name: str) -> None:
        """Initialize the dataset with LMDB environment.
        
        Args:
            args: Configuration object with db_path, num_train_subgraph, etc.
            db_name: Name of the LMDB database (e.g., 'train_subgraphs', 'valid_subgraphs')
        """
        self.args = args
        self.env = lmdb.open(args.db_path, readonly=True, max_dbs=5, lock=False)
        self.subgraphs_db = self.env.open_db(db_name.encode())
        self.db_name = db_name
        
        # Determine dataset size
        with self.env.begin(db=self.subgraphs_db) as txn:
            self.num_entries = (args.num_train_subgraph if db_name == "train_subgraphs" 
                              else txn.stat()['entries'])

    def __len__(self) -> int:
        """Return the number of subgraphs in the dataset."""
        return self.num_entries

    def _fetch_subgraph(self, idx: int) -> Tuple[List, List, Dict, Dict, Dict]:
        """Fetch a single subgraph from LMDB."""
        with self.env.begin(db=self.subgraphs_db) as txn:
            str_id = f"{idx:08}".encode('ascii')
            return kg_utiles.KnowledgeGraphUtils.deserialize(txn.get(str_id))

    @staticmethod
    def _sample_negatives(triples: np.ndarray, mapping: Dict[Tuple[int, int], List[int]], 
                         nentity: int, num_neg: int, is_head: bool = False) -> np.ndarray:
        """Sample negative entities for triples."""
        all_entities = np.arange(nentity)
        neg_samples = np.zeros((len(triples), num_neg), dtype=int)
        
        for i, (h, r, t) in enumerate(triples):
            key = (r, t) if is_head else (h, r)
            exclude = mapping.get(key, [])
            valid_choices = np.delete(all_entities, exclude)
            if len(valid_choices)>=num_neg :
                neg_samples[i] = np.random.choice(valid_choices, num_neg, replace=False)
            else :
                neg_samples[i] = np.random.choice(valid_choices, num_neg, replace=True)
        
        return neg_samples

class TrainSubgraphDataset(SubgraphDataset):
    """Dataset for training subgraphs."""
    
    def __init__(self, args: Any) -> None:
        super().__init__(args, "train_subgraphs")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve and process a training subgraph."""
        sup_tri, que_tri, hr2t, rt2h, ent_type = self._fetch_subgraph(idx)
        
        sup_tri = np.array(sup_tri)
        que_tri = np.array(que_tri)
        nentity = len(np.unique(sup_tri[:, [0, 2]]))
        
        neg_tail_ent = self._sample_negatives(que_tri, hr2t, nentity, self.args.num_neg)
        neg_head_ent = self._sample_negatives(que_tri, rt2h, nentity, self.args.num_neg, is_head=True)
        
        ent_type_tensor = torch.tensor([[k, v] for k, v in ent_type.items()], dtype=torch.int64)
        
        return (torch.tensor(sup_tri, dtype=torch.int64),
                ent_type_tensor,
                torch.tensor(que_tri, dtype=torch.int64),
                torch.tensor(neg_tail_ent, dtype=torch.int64),
                torch.tensor(neg_head_ent, dtype=torch.int64))

    @staticmethod
    def collate_fn(data):
        return data

class ValidSubgraphDataset(SubgraphDataset):
    """Dataset for validation subgraphs."""
    
    def __init__(self, args: Any) -> None:
        super().__init__(args, "valid_subgraphs")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve and process a validation subgraph, returning all outputs as tensors.

        Args:
            idx (int): Index of the subgraph to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Support triples tensor (shape: [N, 3], dtype: int64).
                - Entity types tensor (shape: [M, 2], dtype: int64, [[entity, type]]).
                - Query triples tensor (shape: [K, 3], dtype: int64).
                - hr2t tensor (shape: [P, 3], dtype: int64, [[head, relation, tail]]).
                - rt2h tensor (shape: [Q, 3], dtype: int64, [[relation, tail, head]]).

        Raises:
            ValueError: If the subgraph is empty or invalid.
        """
        sup_tri, que_tri, hr2t, rt2h, ent_type = self._fetch_subgraph(idx)

        if not sup_tri or not que_tri:
            raise ValueError(f"Empty subgraph at index {idx}")

        # Convert support triples to tensor
        sup_tri = np.array(sup_tri)
        sup_tri_tensor = torch.tensor(sup_tri, dtype=torch.int64)

        # Convert query triples to tensor
        que_tri = np.array(que_tri)
        que_tri_tensor = torch.tensor(que_tri, dtype=torch.int64)

        # Convert entity types to tensor (key-value pairs as in original)
        ent_type_tensor = torch.tensor([[k, v] for k, v in ent_type.items()], dtype=torch.int64)

        # Convert hr2t to tensor: [[head, relation, tail], ...]
         # Convert hr2t and rt2h dictionaries to tensors
        hr2t_tensor = torch.tensor(
            [[k[0], k[1], v] for k, v_list in hr2t.items() for v in v_list],
            dtype=torch.int64
        )
        
        rt2h_tensor = torch.tensor(
            [[k[0], k[1], v] for k, v_list in rt2h.items() for v in v_list],
            dtype=torch.int64)

        return sup_tri_tensor, ent_type_tensor, que_tri_tensor, hr2t_tensor, rt2h_tensor
    @staticmethod
    def collate_fn(data):
        return data

class KGETrainDataset(Dataset):
    """Dataset for training KGE models with negative sampling."""
    
    def __init__(self, args: Any, train_triples: List, num_ent: int, num_neg: int, 
                hr2t: Dict, rt2h: Dict) -> None:
        self.args = args
        self.triples = np.array(train_triples)
        self.num_ent = num_ent
        self.num_neg = num_neg
        self.hr2t = hr2t
        self.rt2h = rt2h

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a triple with negative samples."""
        h, r, t = self.triples[idx]
        
        neg_tail_ent = np.random.choice(np.delete(np.arange(self.num_ent), self.hr2t.get((h, r), [])),
                                      self.num_neg, replace=False)
        neg_head_ent = np.random.choice(np.delete(np.arange(self.num_ent), self.rt2h.get((r, t), [])),
                                      self.num_neg, replace=False)
        
        return (torch.tensor([h, r, t], dtype=torch.long),
                torch.tensor(neg_tail_ent, dtype=torch.long),
                torch.tensor(neg_head_ent, dtype=torch.long))

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate function for batching KGE training data."""
        pos_triple, neg_tail, neg_head = zip(*batch)
        return (torch.stack(pos_triple, dim=0),
                torch.stack(neg_tail, dim=0),
                torch.stack(neg_head, dim=0))

class KGEEvalDataset(Dataset):
    """Dataset for evaluating KGE models."""
    
    def __init__(self, args: Any, eval_triples: List, num_ent: int, 
                hr2t: Dict, rt2h: Dict) -> None:
        self.args = args
        self.triples = np.array(eval_triples)
        self.num_ent = num_ent
        self.hr2t = hr2t
        self.rt2h = rt2h
        self.num_cand = getattr(args, 'num_cand', 'all')  # Default to 'all' if not specified

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a triple with labels or candidates."""
        h, r, t = self.triples[idx]
        pos_triple = torch.tensor([h, r, t], dtype=torch.long)
        
        if self.num_cand == 'all':
            tail_label, head_label = self._get_label(self.hr2t.get((h, r), []), 
                                                   self.rt2h.get((r, t), []))
            return pos_triple, tail_label, head_label
        else:
            neg_tail_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.hr2t.get((h, r), [])),
                                           self.num_cand, replace=False)
            neg_head_cand = np.random.choice(np.delete(np.arange(self.num_ent), self.rt2h.get((r, t), [])),
                                           self.num_cand, replace=False)
            tail_cand = torch.tensor(np.concatenate(([t], neg_tail_cand)), dtype=torch.long)
            head_cand = torch.tensor(np.concatenate(([h], neg_head_cand)), dtype=torch.long)
            return pos_triple, tail_cand, head_cand

    def _get_label(self, true_tail: List[int], true_head: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate binary label tensors for true entities."""
        y_tail = torch.zeros(self.num_ent, dtype=torch.float32)
        y_head = torch.zeros(self.num_ent, dtype=torch.float32)
        y_tail[true_tail] = 1.0
        y_head[true_head] = 1.0
        return y_tail, y_head

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate function for batching KGE evaluation data."""
        pos_triple, tail_data, head_data = zip(*batch)
        return (torch.stack(pos_triple, dim=0),
                torch.stack(tail_data, dim=0),
                torch.stack(head_data, dim=0))