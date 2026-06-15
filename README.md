## Model Graph Inductive Learning (MGIL) for Knowledge Graph Completion

This repository contains the official implementation of the paper:

---

###  Overview

Link prediction in knowledge graphs relies heavily on high-quality embeddings. However, most existing approaches focus only on local neighborhood aggregation and ignore the global structure of the graph. To address this limitation, we propose **MGIL (Model Graph Inductive Learning)**, a novel framework that:

- Constructs a **model graph** from the original knowledge graph
- Captures **global structural patterns**
- Generates **high-quality initial embeddings** for entities

---
###  Key Idea

MGIL builds **two type model graphs** using two strategies:

#### 1. Relation-based Clustering
Entities are grouped based on the similarity of their:
- Incoming relations
- Outgoing relations


#### 2. Type-based Clustering
Entities are grouped based on their semantic types:
- Example: drugs, proteins, diseases
### Edge in Model Graph

In the model graph, nodes represent groups of entities that share identical relational feature vectors. An undirected edge between two nodes \(U_j\) and \(U_k\) is created if there exists at least one triple \((v_p, r, v_q)\) in the knowledge graph such that \(v_p \in U_j\) and \(v_q \in U_k\), where \(r \in \mathcal{R}\).

This means that if any entity in group \(U_j\) is connected to any entity in group \(U_k\) through any observed relation in the original knowledge graph, an edge is added between the corresponding model graph nodes. The edge captures aggregated interactions between entity groups rather than individual entity-level connections.

A **Graph Neural Network (GNN)** is then applied to the model graph to learn embeddings, which are transferred to the original graph.

---
### Framework Pipeline

1. Construct model graph (relation-based or type-based)
2. Apply GNN on the model graph
3. Generate global-aware embeddings
4. Initialize original KG embeddings
5. Perform link prediction
### Framework Overview

![Framework](figure/mgil.svg)
###  Model Graph Construction

#### Relation-based Model Graph
![Relation](figure/relation_based.svg)

#### Entity-type Model Graph
![Type](figure/type_based.svg)

###  Inductive Datasets

We evaluate MGIL on several widely-used and recently proposed inductive knowledge graph completion benchmarks:

- **FB15k-237**  
- **WN18RR**  
- **NELL-995**  

- **Shomer  Benchmarks:**  
  - **CoDEx-M_E**  
  - **WN18RR_E**  
  - **HetioNet_E**
  
---


## Benchmark Summary

| Benchmark | Directory | Datasets | `model_graph_type` |
|-----------|-----------|----------|---------------------|
| Grail | `dataset` | `nell_v1` ~ `nell_v4` | `relation_base` |
| Grail | `dataset` | `fb237_v1` ~ `fb237_v4` | `relation_base` |
| Grail | `dataset` | `wn18rr_v1` ~ `wn18rr_v4` | `relation_base` |
| Shomer | `dataset/new_data` | `codex_m_E` | `relation_base` |
| Shomer | `dataset/new_data` | `wn18rr_E` | `relation_base` |
| Shomer | `dataset/new_data` | `hetionet_E` | `relation_base` or `entity_base` |
> **Note:** Shomer datasets support two inductive inference settings:
> - `inference_1` — evaluates on test split 1 (`test_0_graph.txt` / `test_0_samples.txt`)
> - `inference_2` — evaluates on test split 2 (`test_1_graph.txt` / `test_1_samples.txt`)

---
## 🚀 How to Run

We provide a simple command-line interface to train and evaluate the MGIL framework.


### Quick Reference

**Grail Benchmark** (`--benchmark dataset`)

| Dataset | Command |
|---------|---------|
| nell_v1 | `python main.py --step meta_train --data_name nell_v1 --benchmark dataset --model_graph_type relation_base` |
| nell_v2 | `python main.py --step meta_train --data_name nell_v2 --benchmark dataset --model_graph_type relation_base` |
| nell_v3 | `python main.py --step meta_train --data_name nell_v3 --benchmark dataset --model_graph_type relation_base` |
| nell_v4 | `python main.py --step meta_train --data_name nell_v4 --benchmark dataset --model_graph_type relation_base` |
| fb237_v1 | `python main.py --step meta_train --data_name fb237_v1 --benchmark dataset --model_graph_type relation_base` |
| fb237_v2 | `python main.py --step meta_train --data_name fb237_v2 --benchmark dataset --model_graph_type relation_base` |
| fb237_v3 | `python main.py --step meta_train --data_name fb237_v3 --benchmark dataset --model_graph_type relation_base` |
| fb237_v4 | `python main.py --step meta_train --data_name fb237_v4 --benchmark dataset --model_graph_type relation_base` |
| wn18rr_v1 | `python main.py --step meta_train --data_name wn18rr_v1 --benchmark dataset --model_graph_type relation_base` |
| wn18rr_v2 | `python main.py --step meta_train --data_name wn18rr_v2 --benchmark dataset --model_graph_type relation_base` |
| wn18rr_v3 | `python main.py --step meta_train --data_name wn18rr_v3 --benchmark dataset --model_graph_type relation_base` |
| wn18rr_v4 | `python main.py --step meta_train --data_name wn18rr_v4 --benchmark dataset --model_graph_type relation_base` |

**Shomer Benchmark** (`--benchmark dataset/new_data`)

| Dataset | `model_graph_type` | Command |
|---------|---------------------|---------|
| codex_m_E | `relation_base` | `python main.py --step meta_train --data_name codex_m_E --benchmark dataset/new_data --model_graph_type relation_base` |
| wn18rr_E | `relation_base` | `python main.py --step meta_train --data_name wn18rr_E --benchmark dataset/new_data --model_graph_type relation_base` |
| hetionet_E | `relation_base` | `python main.py --step meta_train --data_name hetionet_E --benchmark dataset/new_data --model_graph_type relation_base` |
| hetionet_E | `entity_base` | `python main.py --step meta_train --data_name hetionet_E --benchmark dataset/new_data --model_graph_type entity_base` |



### Fine-tuning

```bash
python main.py \
  --step fine_tune \
  --data_name nell_v1 \
  --benchmark dataset \
  --model_graph_type relation_base \
  --metatrain_state ./state/nell_v1/nell_v1.best

### 🔹 1. Meta-Training

Train the model on base datasets:

```bash
python main.py \
  --step meta_train \
  --data_name codex_m_E \
  --model_graph_type relation_base \
  --kge TransE \
  --num_layers 3 \
  --emb_dim 32
```



---
### Key Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--data_name` | `str` | `nell_v1` | — | Name of the dataset to use |
| `--benchmark` | `str` | `dataset` | `dataset`, `dataset/new_data` | Benchmark type: `dataset` for Grail, `dataset/new_data` for Shomer |
| `--model_graph_type` | `str` | `relation_base` | `relation_base`, `entity_base` | Model graph construction strategy |
| `--step` | `str` | `meta_train` | `meta_train`, `fine_tune` | Training mode |
| `--test_type` | `str` | `inference_1` | `inference_1`, `inference_2` | Inductive test split (**Shomer datasets only**) |
| `--kge` | `str` | `TransE` | `TransE`, `DistMult`, `ComplEx`, `RotatE` | Knowledge graph embedding model |
| `--emb_dim` | `int` | `32` | — | Embedding dimension |
| `--num_layers` | `int` | `3` | — | Number of R-GCN layers |
| `--num_bases` | `int` | `4` | — | Number of bases for R-GCN weight decomposition |
| `--batch_size` | `int` | `64` | — | Batch size for training |
| `--lr` | `float` | `0.01` | — | Learning rate |
| `--gamma` | `float` | `10.0` | — | Margin parameter for KGE loss |
| `--adv_temp` | `float` | `1.0` | — | Temperature for adversarial negative sampling |
| `--num_neg` | `int` | `32` | — | Number of negative samples per positive |
| `--train_num_epoch` | `int` | `3` | — | Number of meta-training epochs |
| `--posttrain_num_epoch` | `int` | `50` | — | Number of post-training epochs |
| `--seed` | `int` | `1234` | — | Random seed for reproducibility |
| `--gpu` | `str` | `cuda:0` | — | GPU device identifier |
| `--metatrain_state` | `str` | `./state/fb237_v1_transe/fb237_v1_transe.best` | — | Path to pre-trained state file (required for `--step fine_tune`) |

---

#### Subgraph Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_train_subgraph` | `int` | `10000` | Number of training subgraphs |
| `--num_valid_subgraph` | `int` | `200` | Number of validation subgraphs |
| `--num_sample_for_estimate_size` | `int` | `50` | Number of samples for size estimation |
| `--rw_0` | `int` | `10` | Random walk parameter 0 |
| `--rw_1` | `int` | `10` | Random walk parameter 1 |
| `--rw_2` | `int` | `5` | Random walk parameter 2 |
| `--num_sample_cand` | `int` | `5` | Number of sample candidates |

---

#### Model Graph Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--is_weighted_model_graph` | `bool` | `False` | Use weighted edges in model graph |
| `--is_directed_model_graph` | `bool` | `False` | Use directed edges in model graph |
| `--indtest_eval_bs` | `int` | `512` | Batch size for inductive test evaluation |
| `--metatrain_check_per_step` | `int` | `625` | Checkpoint interval during meta-training |
| `--posttrain_check_per_epoch` | `int` | `625` | Checkpoint interval during post-training |

##  Citation

- Mohommad Esmaeil Khani,  Mahdieh Hasheminejad, Ali Taherkhani, and Hossein Hajiabolhassan, 
Model Graph Inductive Learning for Knowledge Graph Completion, 
