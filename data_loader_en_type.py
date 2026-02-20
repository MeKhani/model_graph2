import os 
import torch
from generate_model_graph_by_data import build_model_graph
from data_prosessing_entity_type import data2pkl 
from subgraph_genrator import gen_subgraph_datasets 
# from tools import set_seed , get_num_rel, get_g
from kg_utiles import KnowledgeGraphUtils as kgu
import pickle
import numpy as np
import re


def load_and_pre_processing_data(args):
     if args.task in ['transductive']:
        args.data_name ="/kgc/"+re.sub(r"_v.*","",args.data_name)
     if args.new_data=='new':
        root_data = "dataset/new_data"
     elif(args.new_data=='old'):
        root_data = "dataset"
      
        
     print(f"the data set is {args.name }")
     args.ent_dim = args.emb_dim
     args.rel_dim = args.emb_dim
     args.gpu = "cuda" if  torch.cuda.is_available() else "cpu"
     if args.kge in ['ComplEx', 'RotatE']:
        args.ent_dim = args.emb_dim * 2
     if args.kge in ['ComplEx']:
        args.rel_dim = args.emb_dim * 2
    # specify the paths for original data and subgraph db
     args.data_path = f'{root_data}/{args.data_name}.pkl'
     args.data_model_graph = f'./{root_data}/{args.data_name}_model_graph.pkl'
     args.unique_features_for_model_graph = f"{root_data}/unique_features/unique_features_{args.data_name}.pkl"
     args.db_path = f'{root_data}/{args.data_name}_subgraph'
     args.save_result= f"../../results/{args.data_name}/"
     if not os.path.exists(args.save_result):
        os.makedirs(args.save_result)
     if not os.path.exists(args.unique_features_for_model_graph.replace(args.unique_features_for_model_graph.split("/")[-1],"")):
        os.makedirs(args.unique_features_for_model_graph.replace(args.unique_features_for_model_graph.split("/")[-1],""))

     kgu.set_seed(args.seed)

     if not os.path.exists(args.data_path):
        data2pkl(args.data_name,args)
    # load original data and make index
     args.num_rel = kgu.get_num_relations(args.data_path)
   #   if not os.path.exists(args.data_model_graph):
   #   build_model_graph(args)
     model_graph = build_model_graph(args)
     

     
      

     if not os.path.exists(args.db_path):
        gen_subgraph_datasets(args,)
   #   model_data = pickle.load(open(args.data_model_graph, 'rb'))
   #   model_triples, _ = model_data['model_graph']['triples'],model_data['model_graph']['ent_type']
   # #   model_graph = get_g(list(model_triples), name_edge="weight")
   #   model_graph = kgu.create_directed_graph(np.array(model_triples),edge_key="weight")
     num_nodes = model_graph.num_nodes() 
     model_graph.ndata["feat"] = torch.randn(num_nodes,args.emb_dim).to(args.gpu)
     # If using DGL, ensure graph is on the correct device (assuming get_g returns a DGLGraph)
     if hasattr(model_graph, 'to'):
        model_graph = model_graph.to(args.gpu)
     return model_graph

        