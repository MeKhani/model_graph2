
from typing import Any, Tuple
from torch.utils.data import DataLoader,Dataset
from torch import optim
import os
import torch
import torch.nn.functional as F
from my_dataset import TrainSubgraphDataset, ValidSubgraphDataset
from rgcn_model import RGCN
from model import WeightedGraphAutoEncoder
from ent_init_model import EntInit
from kge_model import KGEModel
from tools import get_g_bidir , write_evaluation_result,get_indtest_test_dataset_and_train_g
from tqdm import tqdm
from my_dataset import KGEEvalDataset
import dgl
from collections import defaultdict as ddict

class ModelTrainer:
    """Trainer class for managing model training with subgraph datasets."""
    
    def __init__(self, args: Any, model_graph: Any) -> None:
        """
        Initialize the ModelTrainer with arguments and model graph.
        
        Args:
            args: Configuration object containing training parameters (e.g., gpu, state_dir, metatrain_bs).
            model_graph: Graph structure for the model.
        """
        self.args = args
        self.name = args.name
        self.model_graph = model_graph

        # Initialize datasets
        self.train_subgraph_dataloader = self._create_dataloader(
            TrainSubgraphDataset(args), args.metatrain_bs, shuffle=True, 
            collate_fn=TrainSubgraphDataset.collate_fn
        )
        self.valid_subgraph_dataloader = self._create_dataloader(
            ValidSubgraphDataset(args), args.metatrain_bs, shuffle=False, 
            collate_fn=ValidSubgraphDataset.collate_fn
        )

        # Inductive test datasets
        indtest_test_dataset, indtest_train_g, self.ind_ent_type = get_indtest_test_dataset_and_train_g(args)
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
            lr=args.metatrain_lr
        )

    def _create_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool, 
                          collate_fn: Any) -> DataLoader:
        """Helper method to create a DataLoader with consistent settings."""
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                         collate_fn=collate_fn, num_workers=0, pin_memory=self.args.gpu == "cuda")

    def build_model(self) -> Tuple[WeightedGraphAutoEncoder, RGCN, KGEModel]:
        """
        Build and initialize the models for training.
        
        Returns:
            Tuple containing the graph autoencoder, RGCN, and KGE model.
        """
        print(f"Using device: {self.args.gpu}")
        model_g = WeightedGraphAutoEncoder(self.args).to(self.args.gpu)
        rgcn = RGCN(self.args).to(self.args.gpu)
        kge_model = KGEModel(self.args).to(self.args.gpu)
        return model_g, rgcn, kge_model
   
    def train(self):
         write_evaluation_result("-" * 50+"\n" ,self.args)
         best_step = 0
         best_eval_rst = {'mrr': 0, 'hits@1': 0, 'hits@5': 0, 'hits@10': 0}
         bad_count = 0
         pbar = tqdm(range(self.args.train_num_epoch))
         step1= 0
         for step in pbar :
              
              for batch in self.train_subgraph_dataloader:
                    batch_loss = 0
                    embedding = self.get_embedding_from_model_graph()
                    batch_sup_g = dgl.batch([self.add_model_graph_embedding_to_sub_graphs(d, embedding) for d in batch]).to(self.args.gpu)
                    
                    #forward data 
                    ent_emb = self.rgcn(batch_sup_g)
                    is_better_result =True
                    sup_g_list = dgl.unbatch(batch_sup_g)
                    for batch_i, data in enumerate(batch):
                        ent_type,que_tri, que_neg_tail_ent, que_neg_head_ent = [d.to(self.args.gpu) for d in data[1:]]
                        ent_emb = sup_g_list[batch_i].ndata['h']
                        # kge loss
                        loss = self.get_loss(que_tri.to(torch.int64), que_neg_tail_ent, que_neg_head_ent, ent_emb)

                        batch_loss += loss

                    batch_loss /= len(batch)
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    self.optimizer.step()

                    step1 += 1
                    print('step : {} in batch size {} | loss: {:.4f}'.format(step,step1, batch_loss.item()))

                    if step1 % self.args.metatrain_check_per_step  == 0  :
                        eval_res = self.evaluate_valid_subgraphs()
                        print(f"the eval result  is {eval_res}")
                        

                        if eval_res['mrr'] > best_eval_rst['mrr']:
                            best_eval_rst = eval_res
                            best_step = step
                            is_better_result= True
                            self.save_checkpoint(step)
                            bad_count = 0
                        else:
                            bad_count += 1
                            is_better_result= False
                            
              result_best ={f"result in : {step}":eval_res,f"best result in {best_step}":best_eval_rst}if is_better_result else {f"result in {step}":eval_res,f"bad count is {step}":bad_count} 
              write_evaluation_result(result_best,self.args)
            
         self.save_model(best_step)

        

         self.before_test_load()
         self.evaluate_indtest_test_triples(num_cand=50)
    def save_checkpoint(self, step):
        # state = {'ent_init': self.ent_init.state_dict(),
        state = {'model_g': self.model_g.state_dict(),
                 'rgcn': self.rgcn.state_dict(),
                 'kge_model': self.kge_model.state_dict()}
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.name,
                                       self.name + '.' + str(step) + '.ckpt'))
    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.name + '.best'))

    def get_embedding_from_model_graph(self):
        
        node_embeddings, _ = self.model_g(self.model_graph,self.model_graph.ndata["feat"]) 
        return node_embeddings
    def add_model_graph_embedding_to_sub_graphs(self,b,embedddings ):
        # Step 1: Validate input
        sub_g = get_g_bidir(b[0],self.args)
        ent_type= b[1]

        num_nodes = sub_g.num_nodes()
        if len(ent_type) != num_nodes:
            raise ValueError(f"Number of nodes in enttype ({len(ent_type)}) does not match main_graph ({num_nodes})")
        # Step 2: Create a mapping tensor of type indices
        type_indices = ent_type[:,1]
        # Step 3: Assign features using the type indices
        # sub_g.ndata['feat'] = torch.cat((embedddings[type_indices],main_graph.ndata['feat']) ,dim=1)
        sub_g.ndata['feat'] = embedddings[type_indices]#+ sub_g.ndata['feat']

        # Debug: Print summary of assigned features
        # print(f"Assigned features to {num_nodes} nodes. Feature shape: {main_graph.ndata['feat'].shape}")
        
        return sub_g
    def add_model_graph_embedding_to_graphs(self,graph,embedddings, ent_type ):
        # Step 1: Validate input
       

        num_nodes = graph.num_nodes()
        if len(ent_type) != num_nodes:
            raise ValueError(f"Number of nodes in enttype ({len(ent_type)}) does not match main_graph ({num_nodes})")
        # Step 2: Create a mapping tensor of type indices
         # Step 2: Create a mapping tensor of type indices
        type_indices = torch.tensor(
        [ent_type[node_id] for node_id in range(num_nodes)],
        dtype=torch.long
    )
        # Step 3: Assign features using the type indices
        # sub_g.ndata['feat'] = torch.cat((embedddings[type_indices],main_graph.ndata['feat']) ,dim=1)
        graph.ndata['feat'] = embedddings[type_indices]#+ sub_g.ndata['feat']

        # Debug: Print summary of assigned features
        # print(f"Assigned features to {num_nodes} nodes. Feature shape: {main_graph.ndata['feat'].shape}")
        
        return graph
        
    def get_loss(self, tri, neg_tail_ent, neg_head_ent, ent_emb):

        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, mode='head-batch')
        neg_score = torch.cat([neg_tail_score, neg_head_score])
        neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
                     * F.logsigmoid(-neg_score)).sum(dim=1)

        pos_score = self.kge_model(tri, ent_emb)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)

        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2 

        return loss 
    def evaluate(self, ent_emb, eval_dataloader, num_cand='all'):
        
        results = ddict(float)
        count = 0

        eval_dataloader.dataset.num_cand = num_cand

        if num_cand == 'all':
            for batch in eval_dataloader:
                pos_triple, tail_label, head_label = [b.to(self.args.gpu) for b in batch]
                head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

                # tail prediction
                pred = self.kge_model((pos_triple, None), ent_emb, mode='tail-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, tail_idx]
                pred = torch.where(tail_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, tail_idx] = target_pred

                tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, tail_idx]

                # head prediction
                pred = self.kge_model((pos_triple, None), ent_emb, mode='head-batch')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, head_idx]
                pred = torch.where(head_label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, head_idx] = target_pred

                head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                               dim=1, descending=False)[b_range, head_idx]

                ranks = torch.cat([tail_ranks, head_ranks])
                ranks = ranks.float()
                count += torch.numel(ranks)
                results['mr'] += torch.sum(ranks).item()
                results['mrr'] += torch.sum(1.0 / ranks).item()

                for k in [1, 5, 10]:
                    results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

            for k, v in results.items():
                results[k] = v / count

        else:
            for i in range(self.args.num_sample_cand):
                for batch in eval_dataloader:
                    pos_triple, tail_cand, head_cand = [b.to(self.args.gpu) for b in batch]

                    b_range = torch.arange(pos_triple.size()[0], device=self.args.gpu)
                    target_idx = torch.zeros(pos_triple.size()[0], device=self.args.gpu, dtype=torch.int64)
                    # tail prediction
                    pred = self.kge_model((pos_triple, tail_cand), ent_emb, mode='tail-batch')
                    tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]
                    # head prediction
                    pred = self.kge_model((pos_triple, head_cand), ent_emb, mode='head-batch')
                    head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
                                                   dim=1, descending=False)[b_range, target_idx]

                    ranks = torch.cat([tail_ranks, head_ranks])
                    ranks = ranks.float()
                    count += torch.numel(ranks)
                    results['mr'] += torch.sum(ranks).item()
                    results['mrr'] += torch.sum(1.0 / ranks).item()

                    for k in [1, 5, 10]:
                        results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

            for k, v in results.items():
                results[k] = v / count

        return results
    def evaluate_valid_subgraphs(self):
                
                all_results = ddict(int)
                for batch in self.valid_subgraph_dataloader:
                   
                    embedding = self.get_embedding_from_model_graph()
                    batch_sup_g = dgl.batch([self.add_model_graph_embedding_to_sub_graphs(d, embedding) for d in batch]).to(self.args.gpu)
                    ent_emb = self.rgcn(batch_sup_g)

                    sup_g_list = dgl.unbatch(batch_sup_g)

                    for batch_i, data in enumerate(batch):
                        que_dataloader = data[2]
                        ent_emb = sup_g_list[batch_i].ndata['h']

                        results = self.evaluate(ent_emb, que_dataloader)

                        for k, v in results.items():
                            all_results[k] += v

                for k, v in all_results.items():
                    all_results[k] = v / self.args.num_valid_subgraph


                return all_results
    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        # self.ent_init.load_state_dict(state['ent_init'])
        self.model_g.load_state_dict(state['model_g'])
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])

    def evaluate_indtest_test_triples(self, num_cand='all'):
        """do evaluation on test triples of ind-test-graph"""
        # ent_emb = self.get_ent_emb(self.indtest_train_g)
        # self.ent_init(self.indtest_train_g)
        # self.model_g(self.indtest_train_g)
        embedding = self.get_embedding_from_model_graph()
        self.indtest_train_g = self.add_model_graph_embedding_to_graphs(self.indtest_train_g,embedding,self.ind_ent_type)
        ent_emb = self.rgcn(self.indtest_train_g)

        results = self.evaluate(ent_emb, self.indtest_test_dataloader, num_cand=num_cand)

        test_result = 'test on ind-test-graph, sample {:.4f} , mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(num_cand,
            results['mrr'], results['hits@1'],
            results['hits@5'], results['hits@10'])
        resultss = {f"test on ind-test-graph  num_cand":num_cand,f"bad count is mrr ": results['mrr'],f"hits@1":results['hits@1']
                       ,f" hits@5": results['hits@5'],f"hits@10":results['hits@10']} 
        
        write_evaluation_result("-"*50+"\n", self.args, type ="test")
        write_evaluation_result(resultss, self.args, type ="test")
        return results
