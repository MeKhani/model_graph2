import pickle
import torch
import numpy as np
from collections import defaultdict as ddict
import lmdb
from tqdm import tqdm
import random
# from tools import get_g, get_hr2t_rt2h_sup_que , serialize
from kg_utiles import KnowledgeGraphUtils as kgu
import dgl
from typing import Tuple, List, Dict, Any


def gen_subgraph_datasets(args):
    print(f'-----There is no sub-graphs for {args.data_name}, so start generating sub-graphs before meta-training!-----')
    data = pickle.load(open(args.data_path, 'rb'))
    model_data = pickle.load(open(args.data_model_graph, 'rb'))
    _, ent_type = model_data['model_graph']['triples'],model_data['model_graph']['ent_type']
    # train_g = get_g(data['train_graph']['train'] + data['train_graph']['valid']
    #                 + data['train_graph']['test'])
    train_g = kgu.create_directed_graph(np.array(data['train_graph']['train'] + data['train_graph']['valid']
                    + data['train_graph']['test']))

    BYTES_PER_DATUM = get_average_subgraph_size(args, args.num_sample_for_estimate_size, train_g,ent_type) * 2
    map_size = (args.num_train_subgraph + args.num_valid_subgraph) * BYTES_PER_DATUM
    env = lmdb.open(args.db_path, map_size=map_size, max_dbs=2)
    train_subgraphs_db = env.open_db("train_subgraphs".encode())
    valid_subgraphs_db = env.open_db("valid_subgraphs".encode())

    for idx in tqdm(range(args.num_train_subgraph)):
        str_id = '{:08}'.format(idx).encode('ascii')
        datum = sample_one_subgraph(args, train_g,ent_type)
        with env.begin(write=True, db=train_subgraphs_db) as txn:
            txn.put(str_id, kgu.serialize(datum))

    for idx in tqdm(range(args.num_valid_subgraph)):
        str_id = '{:08}'.format(idx).encode('ascii')
        datum = sample_one_subgraph(args, train_g,ent_type)
        with env.begin(write=True, db=valid_subgraphs_db) as txn:
            txn.put(str_id, kgu.serialize(datum))


def sample_one_subgraph(args, bg_train_g: dgl.DGLGraph,
                       ent_type: Dict) -> Tuple[List, List, Dict, Dict, Dict]:
    """Sample a single subgraph with support and query triples from the training graph.
    
    Args:
        args: Configuration object with rw_0, rw_1, rw_2 parameters
        bg_train_g: Input training graph
        model_triples: Model triples to be returned unchanged
        ent_type: Entity type mapping
    
    Returns:
        Tuple containing support triples, query triples, hr2t mapping, rt2h mapping,
        entity type subset, and model triples
    """
    # Create undirected graph efficiently
    edges_src, edges_dst = bg_train_g.edges()
    undirected_edges = (torch.cat([edges_src, edges_dst]), 
                       torch.cat([edges_dst, edges_src]))
    bg_train_g_undir = dgl.graph(undirected_edges)

    # Sample nodes and induce subgraph
    while True:
        sel_nodes = _sample_nodes(bg_train_g_undir, bg_train_g.num_nodes(), 
                                args.rw_0, args.rw_1, args.rw_2)
        sub_g = dgl.node_subgraph(bg_train_g, sel_nodes)
        
        if sub_g.num_nodes() >= 50:
            # Process subgraph triples
            sub_triples = torch.stack([sub_g.edges()[0], 
                                     sub_g.edata['rel'], 
                                     sub_g.edges()[1]]).T.tolist()
            random.shuffle(sub_triples)
            
            # Reindex entities and compute frequencies
            ent_freq, rel_freq, ent_type_sub, triples_reidx, _ = _process_triples(sub_triples, ent_type)
            
            # Split into support and query triples
            sup_tris, que_tris = _split_triples(triples_reidx, ent_freq, rel_freq)
            
            if len(que_tris) >= int(len(triples_reidx) * 0.05):
                break

    # Generate mapping dictionaries
    # hr2t, rt2h = get_hr2t_rt2h_sup_que(sup_tris, que_tris)
    hr2t, rt2h = kgu.map_support_query_triples(sup_tris, que_tris)
    
    return sup_tris, que_tris, hr2t, rt2h, ent_type_sub

def _sample_nodes(graph: dgl.DGLGraph, num_nodes: int, rw_0: int, rw_1: int, 
                 rw_2: int) -> List[int]:
    """Sample nodes using random walks on the graph."""
    sel_nodes = set()  # Use set for faster uniqueness checking
    cand_nodes = np.arange(num_nodes)
    
    for i in range(rw_0):
        if i > 0 and not sel_nodes:
            break
        cand_nodes = cand_nodes if i == 0 else list(sel_nodes)
        rw_starts = np.random.choice(cand_nodes, 1, replace=False).repeat(rw_1)
        rw, _ = dgl.sampling.random_walk(graph, rw_starts, length=rw_2)
        sel_nodes.update(n for n in rw.reshape(-1) if n != -1)
    
    return list(sel_nodes)

def _process_triples(triples: List[List[int]], ent_type: Dict) -> Tuple[Dict, Dict, Dict, List, Dict]:
    """Process triples: reindex entities and compute frequencies."""
    ent_freq = ddict(int)
    rel_freq = ddict(int)
    ent_type_sub = ddict(int)
    triples_reidx = []
    ent_reidx = {}
    entidx = 0

    for h, r, t in triples:
        h_idx = ent_reidx.setdefault(h, entidx)
        if h_idx == entidx:
            ent_type_sub[entidx] = ent_type[h]
            entidx += 1
        
        t_idx = ent_reidx.setdefault(t, entidx)
        if t_idx == entidx:
            ent_type_sub[entidx] = ent_type[t]
            entidx += 1
        
        ent_freq[h_idx] += 1
        ent_freq[t_idx] += 1
        rel_freq[r] += 1
        triples_reidx.append([h_idx, r, t_idx])
    
    return ent_freq, rel_freq, ent_type_sub, triples_reidx, ent_reidx

def _split_triples(triples: List[List[int]], ent_freq: Dict, rel_freq: Dict) -> Tuple[List, List]:
    """Split triples into support and query sets based on frequency criteria."""
    que_tris = []
    sup_tris = []
    max_que_count = int(len(triples) * 0.1)

    for idx, (h, r, t) in enumerate(triples):
        if (len(que_tris) < max_que_count and 
            ent_freq[h] > 2 and ent_freq[t] > 2 and rel_freq[r] > 2):
            que_tris.append([h, r, t])
            ent_freq[h] -= 1
            ent_freq[t] -= 1
            rel_freq[r] -= 1
        else:
            sup_tris.append([h, r, t])
    
    return sup_tris, que_tris


def get_average_subgraph_size(args, sample_size, bg_train_g,ent_type):
    total_size = 0
    for i in range(sample_size):
        datum = sample_one_subgraph(args, bg_train_g,ent_type)
        total_size += len(kgu.serialize(datum))
    return total_size / sample_size