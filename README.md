# Model Graph Inductive Learning (MGIL) for Knowledge Graph Completion

This repository contains the official implementation of the paper:

**Model Graph Inductive Learning for Knowledge Graph Completion**
---

##  Overview

Link prediction in knowledge graphs relies heavily on high-quality embeddings. However, most existing approaches focus only on local neighborhood aggregation and ignore the global structure of the graph.

To address this limitation, we propose **MGIL (Model Graph Inductive Learning)**, a novel framework that:

- Constructs a **model graph** from the original knowledge graph
- Captures **global structural patterns**
- Generates **high-quality initial embeddings** for entities

---
##  Key Idea

MGIL builds a *model graph* using two strategies:

### 1. Relation-based Clustering
Entities are grouped based on the similarity of their:
- Incoming relations
- Outgoing relations

### 2. Type-based Clustering
Entities are grouped based on their semantic types:
- Example: drugs, proteins, diseases

A **Graph Neural Network (GNN)** is then applied to the model graph to learn embeddings, which are transferred to the original graph.

---
## Framework Pipeline

1. Construct model graph (relation-based or type-based)
2. Apply GNN on the model graph
3. Generate global-aware embeddings
4. Initialize original KG embeddings
5. Perform link prediction
## Framework Overview

![Framework](figure/mgil.svg)
##  Model Graph Construction

### Relation-based Model Graph
![Relation](figure/relation_based.svg)

### Entity-type Model Graph
![Type](figure/type_based.svg)

##  Datasets

We evaluate MGIL on several widely-used and recently proposed inductive knowledge graph completion benchmarks:

- **FB15k-237 (Inductive)**  
- **WN18RR (Inductive)**  
- **NELL-995**  

- **Shomer Inductive Benchmarks:**  
  - **CoDEx-M_E**  
  - **WN18RR_E**  
  - **HetioNet_E**
## 🚀 How to Run

We provide a simple command-line interface to train and evaluate the MGIL framework.

---

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

### 🔹 2. Fine-Tuning (Inductive Setting)

Adapt the pre-trained model to a new dataset:

```bash
python main.py \
  --step fine_tune \
  --data_name WN18RR_E \
  --metatrain_state ./state/pretrained_model.best
```

---

### 🔹 3. Evaluation

Evaluate the model on inductive test sets:

```bash
python main.py \
  --step meta_train \
  --data_name HetioNet_E \
  --test_type inference_1
```

---

### 🔹 4. Key Arguments

* `--data_name`: Dataset name (e.g., CoDEx-M_E, WN18RR_E, HetioNet_E)
* `--step`: Training mode (`meta_train` or `fine_tune`)
* `--model_graph_type`: Graph construction type
* `--kge`: KGE model (TransE, RotatE, etc.)
* `--num_layers`: Number of GNN layers
* `--emb_dim`: Embedding dimension

For a full list of parameters, see the **Command Line Arguments** section.



##  Citation

If you find this work useful, please cite:

```bibtex
@article{khani2025mgil,
  title={Model Graph Inductive Learning for Knowledge Graph Completion},
  author={Khani, Mohommad Esmaeil and Hasheminejad, Mahdieh and Taherkhani, Ali and Hajiabolhassan, Hosein},
  journal={},
  year={2025}
}
