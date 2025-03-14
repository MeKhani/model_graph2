
import pickle
import os
import random 
import dgl 
import torch
import json
import pandas as pd 
import numpy as np
from collections import defaultdict as ddict


def serialize(data):
    return pickle.dumps(data)


def deserialize(data):
    data_tuple = pickle.loads(data)
    return data_tuple



def get_g(tri_list, name_edge = "rel"):
    triples = np.array(tri_list)
    g = dgl.graph((triples[:, 0].T, triples[:, 2].T))
    g.edata[name_edge] = torch.tensor(triples[:, 1].T, dtype=torch.float32)
    return g


def get_g_bidir(triples, args):
    g = dgl.graph((
    torch.cat([triples[:, 0].T, triples[:, 2].T]).to(torch.int64),
    torch.cat([triples[:, 2].T, triples[:, 0].T]).to(torch.int64)
))
    g.edata['type'] = torch.cat([triples[:, 1].T, triples[:, 1].T + args.num_rel]).to(torch.int64)
    return g


def get_hr2t_rt2h(tris):
    hr2t = ddict(list)
    rt2h = ddict(list)
    for tri in tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    return hr2t, rt2h

def get_hr2t_rt2h_sup_que(sup_tris, que_tris):
    hr2t = ddict(list)
    rt2h = ddict(list)
    for tri in sup_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    for tri in que_tris:
        h, r, t = tri
        hr2t[(h, r)].append(t)
        rt2h[(r, t)].append(h)

    que_hr2t = dict()
    que_rt2h = dict()
    for tri in que_tris:
        h, r, t = tri
        que_hr2t[(h, r)] = hr2t[(h, r)]
        que_rt2h[(r, t)] = rt2h[(r, t)]

    return que_hr2t, que_rt2h

def set_seed(seed):
    dgl.seed(seed)
    dgl.random.seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def write_evaluation_result(result_best, args, type ="main"):
    path1 = f"E:/phd/semester-5/results/{args.data_name}/{type}result.json"
    print("write result on disk ")
    os.makedirs(os.path.dirname(path1), exist_ok=True)  # Ensure the directory exists

    try:
        with open(path1, "a", encoding="utf-8") as f:  # Open file in write mode
            json.dump(result_best, f, indent=4)  # Save dictionary as JSON
    except PermissionError:
        print(f"Permission denied: {path1}. Make sure the file is not open elsewhere.")


def get_num_rel(args):
    data = pickle.load(open(args.data_path, 'rb'))
    num_rel = len(np.unique(np.array(data['train_graph']['train'])[:, 1]))

    return num_rel
def generate_group_triples_v1(triples, type_ent, num_rel):
    """
    Generate group-level triples and intra-group/outgoing/incoming relations.

    Args:
        triples (list): A list of triples of the form (e1, rel, e2).
        type_ent (dict): A dictionary mapping entities (e1, e2, ...) to their groups (k1, k2, ...).
        num_rel (int): Total number of relations.

    Returns:
        tuple: 
            - group_triples (set): A set of triples of the form (k1, rel_score, k2).
            - inner_relations (defaultdict): Relations within the same group (k1 -> relations).
            - output_relations (defaultdict): Relations leaving each group (k1 -> relations).
            - input_relations (defaultdict): Relations entering each group (k2 -> relations).
    """
    # Initialize relation tracking dictionaries
    group_relations = ddict(list)  # (k1, k2) -> list of rel
    inner_relations = ddict(set)   # k1 -> set of relations within the group
    output_relations = ddict(set)  # k1 -> set of outgoing relations
    input_relations = ddict(set)   # k2 -> set of incoming relations

    # Iterate through triples
    for e1, rel, e2 in triples:
        k1 = type_ent.get(e1)  # Group of e1
        k2 = type_ent.get(e2)  # Group of e2

        if k1 is not None and k2 is not None:  # Both entities are mapped to groups
            if k1 != k2:
                group_relations[(k1, k2)].append(rel)
                output_relations[k1].add(rel)
                input_relations[k2].add(rel)
            else:
                inner_relations[k1].add(rel)
        # print(f"the group triplest is {group_relations}")
    # Generate group triples with relation scores
    group_triples = {
        (k1, len(rels) / num_rel, k2)  # Score is the proportion of relations over total
        for (k1, k2), rels in group_relations.items()
    }
    # print(f"the group triplest  weighted is {group_triples}")


    return group_triples, inner_relations, output_relations, input_relations
def add_feature_to_model_graph_nodes(graph, i_r, output_relations, input_relations, num_rel):
    """
    Add features to graph nodes based on input, output, and relation-based features.

    Args:
        graph (DGLGraph): Input graph.
        i_r (dict): Dictionary mapping nodes to their internal relations.
        output_relations (dict): Dictionary mapping nodes to their output relations.
        input_relations (dict): Dictionary mapping nodes to their input relations.
        num_rel (int): Number of relations.

    Returns:
        DGLGraph: Graph with updated node features.
    """
    # Initialize node features with zeros
    num_nodes = graph.num_nodes()
    graph.ndata["feat"] = torch.zeros(num_nodes, num_rel * 3)

    # Batch update internal relation features
    if i_r:
        for node, rels in i_r.items():
            graph.ndata["feat"][node, list(rels)] = 1

    # Batch update output relation features
    if output_relations:
        for node, rels in output_relations.items():
            graph.ndata["feat"][node, [num_rel + r for r in rels]] = 1

    # Batch update input relation features
    if input_relations:
        for node, rels in input_relations.items():
            graph.ndata["feat"][node, [2 * num_rel + r for r in rels]] = 1

    return graph


def get_indtest_test_dataset_and_train_g(args):

    data = pickle.load(open(args.data_path, 'rb'))['ind_test_graph']
    ent_type= get_ent_types(data,args)
    num_ent = len(np.unique(np.array(data['train'])[:, [0, 2]]))

    hr2t, rt2h = get_hr2t_rt2h(data['train'])

    from datasets import KGEEvalDataset
    test_dataset = KGEEvalDataset(args, data['test'], num_ent, hr2t, rt2h)

    g = get_g_bidir(torch.LongTensor(data['train']), args)

    return test_dataset, g ,ent_type

def get_ent_types(data, args):
    train_g = get_g(data['train'] + data['valid']
                    + data['test'])
    num_rel = len(np.unique(np.array(data['train'])[:, 1]))
    print(f"number relation is {num_rel}")
    num_nodes = train_g.num_nodes()
    triples = torch.stack([train_g.edges()[0],
                               train_g.edata['rel'],
                               train_g.edges()[1]])
    triples = triples.T.tolist()
     # Initialize node features with zeros
    features = torch.zeros((num_nodes, 2 * args.num_rel))
    
    # Get edges and their types
    src, dst = train_g.edges()  # Get edge endpoints
    etypes = train_g.edata['rel']  # Get edge relation types
    src = src.to(torch.long)
    etypes = etypes.to(torch.long)

    # Assign outgoing relation features
    features[src, etypes] = 1  # Outgoing relations
    features[dst, etypes + args.num_rel] = 1  # Incoming relations
    
    # Find unique rows and their indices
    # Step 3: Load the trained model
    with open(f"unique_features_{args.data_name}.pkl", "rb") as f:
        unique_rows = pickle.load(f)

    unique_rows = torch.tensor(unique_rows, dtype=torch.float32)

    # Find index of each row in `features` within `unique_rows`
    _, feature_types = torch.unique(features, dim=0, return_inverse=True)

    # `feature_types` now contains an integer type index for each row in `features`
    print("Assigned Type Indices:", feature_types)
    print(f"the size is of features  {features.shape}")
    print(f"the size is of features  {len(feature_types)}")
    # return
    # groups = {i: np.where(indices == i)[0].tolist() for i in range(len(unique_rows))}
    ent_type = {ent : type.item() for ent, type in enumerate(feature_types)}
    print(f"the entity type is {ent_type}")
    return ent_type 
def create__model_graph(triples):
     # Extract node and edge information
    src_nodes = [t[0] for t in triples]  # subjects
    dst_nodes = [t[2] for t in triples]  # objects
    weight =torch.tensor([t[1] for t in triples] ) # relations

    # Create a DGL graph
    g = dgl.heterograph({
        ('node', 'weight', 'node'): (src_nodes, dst_nodes)
    })
    g.edata["weight"] = weight
    
    return g