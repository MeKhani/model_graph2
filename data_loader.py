

import os 
import torch
from generate_model_graph import build_model_graph
from data_prosessing import data2pkl 
from subgraph_genrator import gen_subgraph_datasets 
from tools import set_seed , get_num_rel, get_g
import pickle


def load_and_pre_processing_data(args):
     args.ent_dim = args.emb_dim
     args.rel_dim = args.emb_dim
     args.gpu = "cuda" if  torch.cuda.is_available() else "cpu"
     if args.kge in ['ComplEx', 'RotatE']:
        args.ent_dim = args.emb_dim * 2
     if args.kge in ['ComplEx']:
        args.rel_dim = args.emb_dim * 2
    # specify the paths for original data and subgraph db
     args.data_path = f'dataset/{args.data_name}.pkl'
     args.data_model_graph = f'./dataset/{args.data_name}_model_graph.pkl'
     args.db_path = f'dataset/{args.data_name}_subgraph'
     args.save_result= f"dataset/result/{args.data_name}/result.json"
     if not os.path.exists(args.save_result):
        os.makedirs(args.save_result)

     set_seed(args.seed)

    # load original data and make index
     if not os.path.exists(args.data_path):
        data2pkl(args.data_name)
     
     args.num_rel = get_num_rel(args)

     if not os.path.exists(args.data_model_graph):
         build_model_graph(args)
      

     if not os.path.exists(args.db_path):
        gen_subgraph_datasets(args,)
     model_data = pickle.load(open(args.data_model_graph, 'rb'))
     model_triples, _ = model_data['model_graph']['triples'],model_data['model_graph']['ent_type']
     model_graph = get_g(list(model_triples), name_edge="weight")
     num_nodes = model_graph.num_nodes() 
     model_graph.ndata["feat"] = torch.randn(num_nodes,args.emb_dim)
     return model_graph

        