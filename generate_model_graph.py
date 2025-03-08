

import pickle
import torch
import numpy as np
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from tools import get_g, generate_group_triples_v1, add_feature_to_model_graph_nodes, create__model_graph
def build_model_graph(args):
    print("building model graph from main graph ....")
    data = pickle.load(open(args.data_path, 'rb'))
    train_g = get_g(data['train_graph']['train'] + data['train_graph']['valid']
                    + data['train_graph']['test'])
   
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

    # Assign the feature matrix to the graph
    train_g.ndata['feat'] = features
    numcluster = (int)( np.round( num_nodes*0.05) )
    groupsOfnodes= clusterEntitiesKg(features, numcluster, args)
    ent_type = {ent : type for ent, type in enumerate(groupsOfnodes)}
    # print(ent_type)
    entity_type_triples ,inner_rel ,output_relations,input_relations = generate_group_triples_v1(triples,ent_type,args.num_rel)
    model_graph = get_g(list(entity_type_triples))
    model_graph = add_feature_to_model_graph_nodes(model_graph, inner_rel, output_relations, input_relations, args.num_rel)
    model_features = model_graph.ndata["feat"]
    print(f"the size of model_graph triples id {len(entity_type_triples)}")
    save_data = {'model_graph': {'triples': entity_type_triples, 'ent_type': ent_type, 'proper_feature': model_features}}

    pickle.dump(save_data, open(f'./dataset/{args.data_name}_model_graph.pkl', 'wb'))
    return 

def partitionNodeBysimilarty(features):

    unique_rows, indices = np.unique(features, axis=0, return_inverse=True)

# Group similar rows
    groups = {i: np.where(indices == i)[0].tolist() for i in range(len(unique_rows))}
    print(f"the groups of features matrixs is {groups}")
    return groups

def partitionNodeKmeans(features, best_k):
    
   # Perform final clustering with the best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # Organize rows into clusters
    clusters = {i: np.where(labels == i)[0].tolist() for i in range(best_k)}

    print("Clustered Rows:", clusters)
    return clusters

def clusterEntitiesKg(binary_matrix,num_clusters, args):
   # Number of clusters (adjust as needed)
        

        # K-Modes clustering (Hamming similarity)
        km = KModes(n_clusters=num_clusters, init='Huang', n_init=5, verbose=1)
        clusters = km.fit_predict(binary_matrix)
        with open(f"similarty_model_of_{args.data_name}.pkl", "wb") as f:
            pickle.dump(km, f)

        return(clusters)

        
def clusterEntitiesKg_percentage(binary_matrix):
    

    

    # Compute pairwise Hamming distances (normalized)
    hamming_distances = pdist(binary_matrix, metric='hamming')

    # Perform hierarchical clustering
    linkage_matrix = linkage(hamming_distances, method='average')

    # Define similarity threshold (e.g., 80% similarity → 20% Hamming distance)
    similarity_threshold = 0.1  # 20% Hamming distance (80% similarity)

    # Convert distance to clusters
    clusters = fcluster(linkage_matrix, t=similarity_threshold, criterion='distance')

    # Print cluster assignments
    for i, cluster in enumerate(clusters):
        print(f"Row {i} assigned to Cluster {cluster}")


        



