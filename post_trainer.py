from typing import Any, Tuple, Dict,List
import torch
from torch import optim
import numpy as np
from kg_utiles import KnowledgeGraphUtils as kgu
from torch.utils.data import DataLoader
from my_dataset import KGETrainDataset, KGEEvalDataset
from rgcn_model import RGCN
from model_graph import WeightedGraphGNN
from kge_model import KGEModel
import torch.nn.functional as F
from collections import defaultdict as ddict

import dgl
import os


class PostTrainer:
    def __init__(self, args, model_graph):
        # super(PostTrainer, self).__i
        self.args = args
        self.model_graph = model_graph
        indtest_test_dataset, indtest_train_g, self.ind_ent_type = kgu.load_inductive_test_data(args)
        self.indtest_train_g = indtest_train_g.to(self.args.gpu)
        self.model_g, self.rgcn, self.kge_model = self.build_model()
        self.state_path = os.path.join(args.state_dir, self.args.name)
        self.load_metatrain()
        train_dataset, valid_dataset = kgu.get_posttrain_train_valid_dataset(args)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size,
                                      collate_fn=KGETrainDataset.collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                      collate_fn=KGEEvalDataset.collate_fn)
        self.indtest_test_dataloader = DataLoader(indtest_test_dataset, batch_size=self.args.batch_size, shuffle=True, 
                         collate_fn=KGEEvalDataset.collate_fn, num_workers=0, pin_memory=self.args.gpu == "cuda")

        self.optimizer = optim.Adam(list(self.model_g.parameters()) + list(self.rgcn.parameters())
                                    + list(self.kge_model.parameters()), lr=self.args.lr)
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

        # dataloader
      

    def load_metatrain(self):
        best_model_path = os.path.join(self.state_path, f"{self.args.name}.best")
        # state = torch.load(self.args.metatrain_state, map_location=self.args.gpu)
        state = torch.load(best_model_path, map_location=self.args.gpu)
        self.model_g.load_state_dict(state['model_g'])
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])

    

     

    def train(self):



        
        print('start fine-tuning')

        best_step = 0
        best_eval_rst = {'mrr': 0.0, 'hits@1': 0.0, 'hits@3': 0.0, 'hits@5': 0.0, 'hits@10': 0.0}
        bad_count = 0
        global_step = 0

        # print epoch test rst
        self.evaluate_indtest_test_triples(num_cand=50)

        for i in range(1, self.args.posttrain_num_epoch + 1):
            self.model_g.train()
            self.rgcn.train()
            self.kge_model.train()
            losses = []
            for batch in self.train_dataloader:
                embedding = self.get_embedding_from_model_graph()
                self.indtest_train_g = self.add_model_graph_embedding_to_graphs(
                self.indtest_train_g, embedding, self.ind_ent_type
            )
                ent_emb = self.rgcn(self.indtest_train_g)
                pos_triple, neg_tail_ent, neg_head_ent = [b.to(self.args.gpu) for b in batch]
                batch_loss = self.get_loss(
                    pos_triple, neg_tail_ent, neg_head_ent, ent_emb)
                # print(f"the batch loss is {batch_loss}")
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                # ent_emb = self.get_ent_emb(self.indtest_train_g)
                # loss = self.get_loss(pos_triple, neg_tail_ent, neg_head_ent, ent_emb)

                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()

                losses.append(batch_loss.item())

            print('epoch: {} | loss: {:.4f}'.format(i, np.mean(losses)))
            self.evaluate_indtest_test_triples(num_cand=50)

            # if i % self.args.posttrain_check_per_epoch == 0:
    def add_model_graph_embedding_to_graphs(self, graph: dgl.DGLGraph, embeddings: torch.Tensor, 
                                           ent_type: Dict[int, int]) -> dgl.DGLGraph:
        """Add embeddings to a graph using entity type mapping."""
        num_nodes = graph.num_nodes()
        if len(ent_type) != num_nodes:
            raise ValueError(f"Node count mismatch: ent_type ({len(ent_type)}) vs graph ({num_nodes})")
        
        type_indices = torch.tensor([ent_type[i] for i in range(num_nodes)], dtype=torch.long, device=self.args.gpu)
        graph.ndata['feat'] = embeddings[type_indices]
        return graph

    def evaluate_indtest_valid_triples(self, num_cand='all'):
        ent_emb = self.get_ent_emb(self.indtest_train_g)

        results = self.evaluate(ent_emb, self.valid_dataloader, num_cand)

        self.logger.info('valid on ind-test-graph')
        self.logger.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10']))

        return results
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
    def _compute_batch_loss(self, batch: List[Tuple], ent_emb) -> torch.Tensor:
        """Compute the total loss for a batch of subgraphs."""
        batch_loss = 0.0
        for batch_i, data in enumerate(batch):
            enque_tri, que_neg_tail_ent, que_neg_head_ent = [d.to(self.args.gpu) for d in data[1:]]
            # ent_emb = sup_g_list[batch_i].ndata['h']
            loss = self.get_loss(que_tri.to(torch.int64), que_neg_tail_ent, que_neg_head_ent, ent_emb)
            batch_loss += loss
        return batch_loss / len(batch)