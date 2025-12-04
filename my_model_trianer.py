from typing import Any, Tuple, Dict,List
from torch.utils.data import DataLoader,Dataset
from torch import optim
import os
import torch
import torch.nn.functional as F
from my_dataset import TrainSubgraphDataset, ValidSubgraphDataset
from rgcn_model import RGCN
from model_graph import WeightedGraphGNN
from kge_model import KGEModel
# from tools import get_g_bidir , write_evaluation_result,get_indtest_test_dataset_and_train_g
from kg_utiles import KnowledgeGraphUtils as kgu
from tqdm import tqdm
from my_dataset import KGEEvalDataset
import dgl
import numpy as np
import shutil
from collections import defaultdict as ddict

class ModelTrainer:
    """Trainer class for managing model training with subgraph datasets."""
    
    def __init__(self, args: Any, model_graph: Any) -> None:
        """
        Initialize the ModelTrainer with arguments and model graph.
        
        Args:
            args: Configuration object containing training parameters (e.g., gpu, state_dir, batch_size
            ).
            model_graph: Graph structure for the model.
        """
        """
        ModelTrainer adopts 'CWA' for practical' reasons—negative' sampling and 'filtered' evaluation rely on treating absences as false
        """
        self.args = args
        self.name = args.name
        self.model_graph = model_graph

        # Initialize datasets
        self.train_subgraph_dataloader = self._create_dataloader(
            TrainSubgraphDataset(args), args.batch_size
            , shuffle=True, 
            collate_fn=TrainSubgraphDataset.collate_fn
        )
        self.valid_subgraph_dataloader = self._create_dataloader(
            ValidSubgraphDataset(args), args.batch_size
            , shuffle=False, 
            collate_fn=ValidSubgraphDataset.collate_fn
        )

        # Inductive test datasets
        # indtest_test_dataset, indtest_train_g, self.ind_ent_type = get_indtest_test_dataset_and_train_g(args)
        if args.task =="inductve":
            indtest_test_dataset, indtest_train_g, self.ind_ent_type = kgu.load_inductive_test_data(args)
            self.indtest_train_g = indtest_train_g.to(self.args.gpu)
            self.indtest_test_dataloader = self._create_dataloader(
                indtest_test_dataset, args.indtest_eval_bs, shuffle=False, 
                collate_fn=KGEEvalDataset.collate_fn
            )

        # State directory setup
        self.state_path = os.path.join(args.state_dir, self.name)
        os.makedirs(self.state_path, exist_ok=True)  # More efficient than checking existence first

        # Build and initialize models
        self.model_g, self.rgcn, self.kge_model = self.build_model()
        self.optimizer = optim.Adam(
            list(self.model_g.parameters()) + list(self.rgcn.parameters()) + list(self.kge_model.parameters()), 
            lr=args.lr
        )

    def _create_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool, 
                          collate_fn: Any) -> DataLoader:
        """Helper method to create a DataLoader with consistent settings."""
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                         collate_fn=collate_fn, num_workers=0, pin_memory=self.args.gpu == "cuda")

    def build_model(self) -> Tuple[WeightedGraphGNN, RGCN, KGEModel]:
        """
        Build and initialize the models for training.
        
        Returns:
            Tuple containing the graph autoencoder, RGCN, and KGE model.
        """
        print(f"Using device: {self.args.gpu}")
        model_g = WeightedGraphGNN(self.args).to(self.args.gpu)
        rgcn = RGCN(self.args).to(self.args.gpu)
        kge_model = KGEModel(self.args).to(self.args.gpu)
        return model_g, rgcn, kge_model
    def train(self) -> Dict[str, float]:
        """Train the model and evaluate on validation subgraphs, saving the best checkpoint."""
        # kgu.write_results("-" * 50 + "\n", self.args)
        
        best_step = 0
        best_eval_rst = {'mrr': 0.0, 'hits@1': 0.0, 'hits@3': 0.0, 'hits@5': 0.0, 'hits@10': 0.0}
        bad_count = 0
        global_step = 0
        
        for epoch in tqdm(range(self.args.train_num_epoch), desc="Training Epochs"):
            self.model_g.train()
            self.rgcn.train()
            self.kge_model.train()
            
            for batch in self.train_subgraph_dataloader:
                embedding = self.get_embedding_from_model_graph()
                batch_sup_g = dgl.batch([
                    self.add_model_graph_embedding_to_sub_graphs(d, embedding) 
                    for d in batch
                ]).to(self.args.gpu)
                
                # Forward pass
                ent_emb = self.rgcn(batch_sup_g)
                sup_g_list = dgl.unbatch(batch_sup_g)
                
                # Compute batch loss
                batch_loss = self._compute_batch_loss(batch, sup_g_list)
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                global_step += 1
                print(f"Epoch {epoch}, Step {global_step} | Loss: {batch_loss.item():.4f}")
                
                # Evaluate periodically
                if global_step % self.args.metatrain_check_per_step == 0:
                    eval_res = self.evaluate_valid_subgraphs()
                    print(f"Validation results: {eval_res}")
                    
                    is_better_result = eval_res['mrr'] > best_eval_rst['mrr']
                    if is_better_result:
                        best_eval_rst = eval_res
                        best_step = global_step
                        self.save_checkpoint(global_step)
                        bad_count = 0
                        print("find best result")
                    else:
                        bad_count += 1
                    
                    result_dict = (
                        {"step": global_step, "result": eval_res, "best_step": best_step, "best_result": best_eval_rst}
                        if is_better_result else
                        {"step": global_step, "result": eval_res, "bad_count": bad_count}
                    )
                    kgu.write_results(result_dict, self.args.save_result,self.args)
        
        self.save_model(best_step)
        self.before_test_load()
        self.evaluate_indtest_test_triples(num_cand=50)
        return best_eval_rst

    def _compute_batch_loss(self, batch: List[Tuple], sup_g_list: List[dgl.DGLGraph]) -> torch.Tensor:
        """Compute the total loss for a batch of subgraphs."""
        batch_loss = 0.0
        for batch_i, data in enumerate(batch):
            ent_type, que_tri, que_neg_tail_ent, que_neg_head_ent = [d.to(self.args.gpu) for d in data[1:]]
            ent_emb = sup_g_list[batch_i].ndata['h']
            loss = self.get_loss(que_tri.to(torch.int64), que_neg_tail_ent, que_neg_head_ent, ent_emb)
            batch_loss += loss
        return batch_loss / len(batch)

    def save_checkpoint(self, step: int) -> None:
        """Save the current model state as a checkpoint, removing previous ones."""
        state = {
            'model_g': self.model_g.state_dict(),
            'rgcn': self.rgcn.state_dict(),
            'kge_model': self.kge_model.state_dict()
        }
        print(f"the step to save {step}")
        checkpoint_path = os.path.join(self.state_path, f"{self.name}.{step}.ckpt")
        
        # Remove previous checkpoints efficiently
        for filename in os.listdir(self.state_path):
            if self.name in filename and filename.endswith('.ckpt'):
                os.remove(os.path.join(self.state_path, filename))
        
        torch.save(state, checkpoint_path)

    def save_model(self, step):
        old_path = f'./state/{self.name}/{self.name}.{step}.ckpt'
        new_path = f'./state/{self.name}/{self.name}.best'
        
        try:
            # Remove if exists, then rename
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(old_path, new_path)
            print(f"Model saved as best: {new_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            # Fallback: copy instead of move
            shutil.copy2(old_path, new_path)

    def get_embedding_from_model_graph(self) -> torch.Tensor:
        """Generate embeddings from the model graph."""
        node_embeddings= self.model_g(self.model_graph, self.model_graph.ndata["feat"])
        return node_embeddings

    def add_model_graph_embedding_to_sub_graphs(self, data: Tuple, embeddings: torch.Tensor) -> dgl.DGLGraph:
        """Add model graph embeddings to a subgraph."""
        # sub_g = get_g_bidir(data[0], self.args)
        sub_g = kgu.create_bidirectional_graph( data[0], self.args.num_rel)
        ent_type = data[1]
        
        num_nodes = sub_g.num_nodes()
        if len(ent_type) != num_nodes:
            raise ValueError(f"Node count mismatch: ent_type ({len(ent_type)}) vs graph ({num_nodes})")
        
        type_indices = ent_type[:, 1]  # Assuming ent_type is (n_nodes, 2) with [id, type]
        sub_g.ndata['feat'] = embeddings[type_indices]
        return sub_g

    def add_model_graph_embedding_to_graphs(self, graph: dgl.DGLGraph, embeddings: torch.Tensor, 
                                           ent_type: Dict[int, int]) -> dgl.DGLGraph:
        """Add embeddings to a graph using entity type mapping."""
        num_nodes = graph.num_nodes()
        if len(ent_type) != num_nodes:
            raise ValueError(f"Node count mismatch: ent_type ({len(ent_type)}) vs graph ({num_nodes})")
        
        type_indices = torch.tensor([ent_type[i] for i in range(num_nodes)], dtype=torch.long, device=self.args.gpu)
        graph.ndata['feat'] = embeddings[type_indices]
        return graph

    def get_loss(self, tri: torch.Tensor, neg_tail_ent: torch.Tensor, neg_head_ent: torch.Tensor, 
                ent_emb: torch.Tensor) -> torch.Tensor:
        """Compute the KGE loss for positive and negative samples."""
        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, mode='head-batch')
        neg_score = torch.cat([neg_tail_score, neg_head_score])
        #self-adversarial negative sampling technique
        """
            Positive neg_score (bad) → large negative logsigmoid(-neg_score) → high loss.
            Negative neg_score (good) → small negative logsigmoid(-neg_score) → low loss.
            Penalizing high scores for negatives (logsigmoid).
        """
        neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach() * F.logsigmoid(-neg_score)).sum(dim=1)

        pos_score = self.kge_model(tri, ent_emb)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)

        return (pos_score.mean() + neg_score.mean()) / -2  # Negative mean for minimization

    def evaluate(self, ent_emb: torch.Tensor, eval_dataloader: DataLoader, num_cand: str = 'all') -> Dict[str, float]:
        """Evaluate the model on a given dataset."""
        results = ddict(float)
        count = 0
        eval_dataloader.dataset.num_cand = num_cand

        for batch in eval_dataloader:
            batch = [b.to(self.args.gpu) for b in batch]
            if num_cand == 'all':
                pos_triple, tail_label, head_label = batch
                ranks = self._compute_ranks(pos_triple, ent_emb, tail_label, head_label)
            else:
                pos_triple, tail_cand, head_cand = batch
                ranks = self._compute_candidate_ranks(pos_triple, ent_emb, tail_cand, head_cand)

            ranks = ranks.float()
            count += ranks.numel()
            results['mr'] += ranks.sum().item()
            results['mrr'] += (1.0 / ranks).sum().item()
            for k in [1,3, 5, 10]:
                results[f'hits@{k}'] += (ranks <= k).sum().item()

        return {k: v / count for k, v in results.items()}

    def _compute_ranks(self, pos_triple: torch.Tensor, ent_emb: torch.Tensor, 
                      tail_label: torch.Tensor, head_label: torch.Tensor) -> torch.Tensor:
        """Compute ranks for all-candidate evaluation.
        
        The ranking process involves:

             1-Scoring all entities using the KGE model.
             2-Sorting these scores in descending order (higher score = better match).
             3-Finding the position (rank) of the true entity in this sorted list.
             A realistic evaluation fairly reflects the model’s ability to rank the true entity against incorrect ones, 
             aligning with the intended task of link prediction in KGE.

        Why It’s Realistic:
            Filtered Setting: This approach follows the "filtered" evaluation 
            protocol standard in KGE literature (e.g., TransE, Bordes et al., 2013). In this setting:
            All known true triples (from the knowledge graph) are excluded from ranking competition, 
            except the specific true entity being tested.
            This mimics the real-world task: given (h, r, ?), 
            rank the correct t against incorrect entities, not against other valid answers.
        """
        b_range = torch.arange(pos_triple.size(0), device=self.args.gpu)
        head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

        # Tail prediction
        pred = self.kge_model((pos_triple, None), ent_emb, mode='tail-batch')
        #Extract True Tail Scores
        target_pred = pred[b_range, tail_idx]
        #Mask True Tails
        pred = torch.where(tail_label.bool(), -torch.ones_like(pred) * 1e6, pred)
        #Restores the original score of the true tail
        pred[b_range, tail_idx] = target_pred
        #Compute Tail Ranks:
        tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), 
                                      dim=1, descending=False)[b_range, tail_idx]

        # Head prediction
        pred = self.kge_model((pos_triple, None), ent_emb, mode='head-batch')
        #Extract True head Scores
        target_pred = pred[b_range, head_idx]
        #Mask True heads
        pred = torch.where(head_label.bool(), -torch.ones_like(pred) * 1e6, pred)
        #Restores the original score of the true heads
        pred[b_range, head_idx] = target_pred
        #compute head rank
        head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), 
                                      dim=1, descending=False)[b_range, head_idx]

        return torch.cat([tail_ranks, head_ranks])

    def _compute_candidate_ranks(self, pos_triple: torch.Tensor, ent_emb: torch.Tensor, 
                                tail_cand: torch.Tensor, head_cand: torch.Tensor) -> torch.Tensor:
        """Compute ranks for candidate-based evaluation."""
        b_range = torch.arange(pos_triple.size(0), device=self.args.gpu)
        target_idx = torch.zeros(pos_triple.size(0), dtype=torch.int64, device=self.args.gpu)

        pred = self.kge_model((pos_triple, tail_cand), ent_emb, mode='tail-batch')
        tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), 
                                      dim=1, descending=False)[b_range, target_idx]

        pred = self.kge_model((pos_triple, head_cand), ent_emb, mode='head-batch')
        head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), 
                                      dim=1, descending=False)[b_range, target_idx]

        return torch.cat([tail_ranks, head_ranks])

    def evaluate_valid_subgraphs(self) -> Dict[str, float]:
        """Evaluate the model on validation subgraphs."""
        self.model_g.eval()
        self.rgcn.eval()
        self.kge_model.eval()
        with torch.no_grad():
            all_results = ddict(float)
            for batch in self.valid_subgraph_dataloader:
                embedding = self.get_embedding_from_model_graph()
                batch_sup_g = dgl.batch([
                    self.add_model_graph_embedding_to_sub_graphs(d, embedding) 
                    for d in batch
                ]).to(self.args.gpu)
                ent_emb = self.rgcn(batch_sup_g)
                sup_g_list = dgl.unbatch(batch_sup_g)

                for batch_i, data in enumerate(batch):
                    sup_tri_tensor, ent_type_tensor, que_tri_tensor, hr2t_tensor, rt2h_tensor =data

                    # Move tensors to GPU
                    sup_tri_tensor = sup_tri_tensor.to(self.args.gpu)
                    ent_type_tensor = ent_type_tensor.to(self.args.gpu)
                    que_tri_tensor = que_tri_tensor.to(self.args.gpu)
                    hr2t_tensor = hr2t_tensor.to(self.args.gpu)  # Optional
                    rt2h_tensor = rt2h_tensor.to(self.args.gpu)  # Optional

                    # Reconstruct dictionaries
                    hr2t = self.hr2t_tensor_to_dict(hr2t_tensor)
                    rt2h = self.rt2h_tensor_to_dict(rt2h_tensor)
                    # Compute nentity
                    nentity = len(torch.unique(sup_tri_tensor[:, [0, 2]]))

                    # Create que_dataset
                    que_dataset = KGEEvalDataset(self.args, que_tri_tensor, nentity, hr2t, rt2h)

                    # Create que_dataloader
                    que_dataloader = DataLoader(
                        que_dataset,
                        batch_size=que_tri_tensor.shape[0],
                        collate_fn=KGEEvalDataset.collate_fn,
                        shuffle=False
                    )
                    # que_dataloader = data[2]
                    ent_emb = sup_g_list[batch_i].ndata['h']
                    results = self.evaluate(ent_emb, que_dataloader)
                    for k, v in results.items():
                        all_results[k] += v

            return {k: v / self.args.num_valid_subgraph for k, v in all_results.items()}

    def before_test_load(self) -> None:
        """Load the best model state before testing."""
        state = torch.load(os.path.join(self.state_path, f"{self.name}.best"), map_location=self.args.gpu)
        self.model_g.load_state_dict(state['model_g'])
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])

    def evaluate_indtest_test_triples(self, num_cand: str = 'all') -> Dict[str, float]:
        """Evaluate the model on inductive test triples."""
        self.model_g.eval()
        self.rgcn.eval()
        self.kge_model.eval()
        with torch.no_grad():
            embedding = self.get_embedding_from_model_graph()
            self.indtest_train_g = self.add_model_graph_embedding_to_graphs(
                self.indtest_train_g, embedding, self.ind_ent_type
            )
            ent_emb = self.rgcn(self.indtest_train_g)
            results = self.evaluate(ent_emb, self.indtest_test_dataloader, num_cand=num_cand)

            result_str = (
                f"Test on ind-test-graph, num_cand: {num_cand}, "
                f"mrr: {results['mrr']:.4f}, hits@1: {results['hits@1']:.4f}, "
                f" hits@3: {results['hits@3']:.4f}, "
                f"hits@5: {results['hits@5']:.4f}, hits@10: {results['hits@10']:.4f}"
            )
            result_dict = {
                "test_on": "ind-test-graph", "num_cand": num_cand, "mrr": results['mrr'],
                "hits@1": results['hits@1'],"hits@3": results['hits@3'], "hits@5": results['hits@5'], "hits@10": results['hits@10']
            }
            # kgu.luation_result("-" * 50 + "\n", self.args, type="test")
            kgu.write_results(result_dict, self.args.save_result,self.args)
            print(result_str)
            return results

    def hr2t_tensor_to_dict(self,hr2t_tensor: torch.Tensor) -> dict:
        """Convert flattened (h,r,t) tensor back to hr2t dictionary.
        Args:
            hr2t_tensor: Shape [N, 3] where each row is (h, r, t)
        Returns:
            {(h,r): [t1, t2, ...]}
        """
        # Group by (h,r) pairs and collect tails
        hr2t_dict = ddict(list)
        if hr2t_tensor.numel() > 0:  # Handle empty tensor case
            # Convert to numpy for easier grouping (or use torch_scatter if on GPU)
            h_r = hr2t_tensor[:, :2].numpy()
            t = hr2t_tensor[:, 2].numpy()
            
            # Create dictionary using vectorized operations
            for (h, r), t_val in zip(map(tuple, h_r), t):
                hr2t_dict[(h, r)].append(t_val)
        return hr2t_dict

    def rt2h_tensor_to_dict(self,rt2h_tensor: torch.Tensor) -> dict:
        """Convert flattened (r,t,h) tensor back to rt2h dictionary.
        Args:
            rt2h_tensor: Shape [N, 3] where each row is (r, t, h)
        Returns:
            {(r,t): [h1, h2, ...]}
        """
        rt2h_dict = ddict(list)
        if rt2h_tensor.numel() > 0:
            r_t = rt2h_tensor[:, :2].numpy()
            h = rt2h_tensor[:, 2].numpy()
            
            for (r, t), h_val in zip(map(tuple, r_t), h):
                rt2h_dict[(r, t)].append(h_val)
        return rt2h_dict