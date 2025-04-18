
import pickle
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
def data2pkl(data_name):

    
    if "primekg" not in data_name:

        train_tri = []
        file = open('dataset/{}/train.txt'.format(data_name))
        train_tri.extend([l.strip().split() for l in file.readlines()])
        file.close()

        valid_tri = []
        file = open('dataset/{}/valid.txt'.format(data_name))
        valid_tri.extend([l.strip().split() for l in file.readlines()])
        file.close()

        test_tri = []
        file = open('dataset/{}/test.txt'.format(data_name))
        test_tri.extend([l.strip().split() for l in file.readlines()])
        file.close()

        train_tri, fix_rel_reidx, ent_reidx = reidx(train_tri)
        valid_tri = reidx_withr_ande(valid_tri, fix_rel_reidx, ent_reidx)
        test_tri = reidx_withr_ande(test_tri, fix_rel_reidx, ent_reidx)


        file = open('dataset/{}_ind/train.txt'.format(data_name))
        ind_train_tri = ([l.strip().split() for l in file.readlines()])
        file.close()

        file = open('dataset/{}_ind/valid.txt'.format(data_name))
        ind_valid_tri = ([l.strip().split() for l in file.readlines()])
        file.close()

        file = open('dataset/{}_ind/test.txt'.format(data_name))
        ind_test_tri = ([l.strip().split() for l in file.readlines()])
        file.close()

        test_train_tri, ent_reidx_ind = reidx_withr(ind_train_tri, fix_rel_reidx)
        test_valid_tri = reidx_withr_ande(ind_valid_tri, fix_rel_reidx, ent_reidx_ind)
        test_test_tri = reidx_withr_ande(ind_test_tri, fix_rel_reidx, ent_reidx_ind)

        save_data = {'train_graph': {'train': train_tri, 'valid': valid_tri, 'test': test_tri},
                    'ind_test_graph': {'train': test_train_tri, 'valid': test_valid_tri, 'test': test_test_tri}}

        pickle.dump(save_data, open(f'./dataset/{data_name}.pkl', 'wb'))
    else:
        
        print(f"the data set name is {data_name}/{data_name}")
        path = 'dataset/{}/{}'.format(data_name, "kg.csv")
        all_prime_kg = pd.read_csv(path)
        train_data, ind_data= train_test_split(all_prime_kg, test_size=0.2, random_state=42)
        #splite data for train graph 
        train_tr,test_val_tr = train_test_split(train_data, test_size=0.1, random_state=42)
        valid_tr, test_tr =  train_test_split(test_val_tr, test_size=0.4, random_state=42)
        train_tri = get_triples(train_tr)
        valid_tri = get_triples(valid_tr)
        test_tri = get_triples(test_tr)
        #splite data for inductive data 
        train_ind,test_val_ind = train_test_split(ind_data, test_size=0.2, random_state=42)
        valid_ind, test_ind =  train_test_split(test_val_ind, test_size=0.5, random_state=42)
        test_train_tri = get_triples(train_ind)
        test_valid_tri = get_triples(valid_ind)
        test_test_tri =  get_triples(test_ind)


        


        save_data = {'train_graph': {'train': train_tri, 'valid': valid_tri, 'test': test_tri},
                    'ind_test_graph': {'train': test_train_tri, 'valid': test_valid_tri, 'test': test_test_tri}}

        pickle.dump(save_data, open(f'./dataset/{data_name}.pkl', 'wb'))

def reidx(tri):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0
    rel_reidx = dict()
    relidx = 0
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        if r not in rel_reidx.keys():
            rel_reidx[r] = relidx
            relidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx, dict(rel_reidx), dict(ent_reidx)


def reidx_withr(tri, rel_reidx):
    tri_reidx = []
    ent_reidx = dict()
    entidx = 0
    for h, r, t in tri:
        if h not in ent_reidx.keys():
            ent_reidx[h] = entidx
            entidx += 1
        if t not in ent_reidx.keys():
            ent_reidx[t] = entidx
            entidx += 1
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx, dict(ent_reidx)


def reidx_withr_ande(tri, rel_reidx, ent_reidx):
    tri_reidx = []
    for h, r, t in tri:
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx
def get_triples(data_df):
    ent_type_id = {etype: idx for idx, etype in enumerate(data_df["x_type"].unique())}
    ent_x_id = {en: idx for idx, en in enumerate(data_df["x_name"].unique())}
    ent_y_id = {en: idx for idx, en in enumerate(data_df["y_name"].unique())}
    relations_id = {rel: idx for idx, rel in enumerate(data_df["relation"].unique())}
    num_rel = len(relations_id)  # Count unique relations
    all_en = set(ent_x_id.keys()).union(set(ent_y_id.keys()))
    ent_id = {en: idx for idx, en in enumerate(all_en)}


    # ✅ Extract triples with numerical relation IDs (FAST)
    triples = data_df[['x_name', 'relation', 'y_name']].to_numpy()
    triples = [(ent_id[x], relations_id[r],ent_id[y]) for x, r, y in triples]
    return triples 