import os
import torch
from pathlib import Path

from generate_model_graph import build_model_graph
import generate_model_graph_by_data 
from data_prosessing import data_to_pickle
from subgraph_genrator import gen_subgraph_datasets
from kg_utiles import KnowledgeGraphUtils as kgu
def load_and_preprocess_data(args):
    """
    Load / prepare dataset, model graph, and subgraph DB.
    Creates necessary directories and files if missing.
    """
    # ── 1. Determine root directory ─────────────────────────────────────
    root = Path(args.benchmark)

    print(f"Dataset: {args.data_name} (mode: {args.benchmark})")

    # ── 2. Embedding dimensions ─────────────────────────────────────────
    args.ent_dim = args.emb_dim
    args.rel_dim = args.emb_dim

    if args.kge in {'ComplEx', 'RotatE'}:
        args.ent_dim = args.emb_dim * 2
    if args.kge == 'ComplEx':
        args.rel_dim = args.emb_dim * 2

    # ── 3. Device ───────────────────────────────────────────────────────
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.gpu = str(args.device)

    # ── 4. Define all important paths as Path objects ───────────────────
    paths = {
        'pkl':          root / f"{args.data_name}.pkl",
        'model_graph':  root / f"{args.data_name}_model_graph.pkl",
        'unique_feat':  root / "unique_features" / f"unique_features_{args.data_name}.pkl",
        'subgraph_db':  root / f"{args.data_name}_subgraph",
    }

    # ── 5. Determine graph type directory ───────────────────────────────
    if args.is_directed_model_graph and args.is_weighted_model_graph:
        graph_type = "directed_and_weighted"
    elif not args.is_directed_model_graph and args.is_weighted_model_graph:
        graph_type = "undirected_and_weighted"
    elif not args.is_directed_model_graph and not args.is_weighted_model_graph:
        graph_type = "undirected_and_unweighted"
    elif args.is_directed_model_graph and not args.is_weighted_model_graph:
        graph_type = "directed_and_unweighted"

    # ── 6. Build results path ──────────────────────────────────────────
    if args.benchmark == "dataset/new_data":
        results_path = Path("../random_seed") / f"seed_{args.seed}" / args.data_name / args.test_type / graph_type
    else:
        results_path = Path("../results") / f"seed_{args.seed}" / args.data_name / graph_type

    # Remove spaces and normalize path
    args.save_result = str(results_path).strip()
    
    # Create directories
    results_path.mkdir(parents=True, exist_ok=True)
    paths['unique_feat'].parent.mkdir(parents=True, exist_ok=True)

    # Attach remaining paths to args
    args.data_path                     = str(paths['pkl'])
    args.data_model_graph              = str(paths['model_graph'])
    args.unique_features_for_model_graph = str(paths['unique_feat'])
    args.db_path                       = str(paths['subgraph_db'])

    
    # ── 5. Seed ─────────────────────────────────────────────────────────
    kgu.set_seed(args.seed)
    

    # ── 6. Generate pickled data if missing ─────────────────────────────
    if not paths['pkl'].exists():
        print(f"Pickle file missing → running data_to_pickle for {args.data_name}")
        data_to_pickle(args)   # ← note: most implementations expect (data_name, args)
  

    # ── 7. Get number of relations ──────────────────────────────────────
    args.num_rel = kgu.get_num_relations(args.data_path)
    # return 

    # ── 8. Build / load model graph ─────────────────────────────────────
    # if not paths['model_graph'].exists():
    #     print("Model graph missing → building...")
    #     build_model_graph(args)

    print("Building/loading model graph...")
    
    model_graph = build_model_graph(args)
   


    print(f"Model graph created: {model_graph}")

    # ── 9. Generate subgraph datasets if missing ────────────────────────
    if not paths['subgraph_db'].exists():
        print("Subgraph DB missing → generating...")
        gen_subgraph_datasets(args)

    # ── 10. Prepare node features & move to device ──────────────────────
    num_nodes = model_graph.num_nodes()

    # Random features (you may want to replace this later with learned / precomputed ones)
    features = torch.randn(num_nodes, args.emb_dim, device=args.device)

    model_graph.ndata["feat"] = features

    # Move graph to GPU if it's a DGL graph and has .to() method
    if hasattr(model_graph, 'to'):
        model_graph = model_graph.to(args.device)

    return model_graph