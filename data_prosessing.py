from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# ────────────────────────────────────────────────
# Type aliases
# ────────────────────────────────────────────────

Triple = List[str]
IndexedTriple = List[int]
EntityMap = Dict[str, int]
RelationMap = Dict[str, int]
EntityTypeGroups = Dict[int, List[int]]
FlatEntityTypes = Dict[int, int]


# ────────────────────────────────────────────────
# Dataset classification
# ────────────────────────────────────────────────

# Grail benchmark: standard datasets (relation_base only)
GRAIL_DATASETS = {
    "nell_v1", "nell_v2", "nell_v3", "nell_v4",
    "fb237_v1", "fb237_v2", "fb237_v3", "fb237_v4",
    "wn18rr_v1", "wn18rr_v2", "wn18rr_v3", "wn18rr_v4"
}

# Shomer benchmark: new datasets (all relation_base, hetionet also entity_base)
SHOMER_DATASETS = {"codex_m_e", "hetionet_e", "wn18rr_e"}

# Only hetionet supports entity_base mode
ENTITY_BASE_DATASETS = {"hetionet_e"}


# ────────────────────────────────────────────────
# File reading helpers
# ────────────────────────────────────────────────

def read_triples(path: str | Path) -> List[Triple]:
    """Read space-separated triples from a text file, one per line."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    triples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                triples.append(parts)
    return triples


# ────────────────────────────────────────────────
# Re-indexing functions
# ────────────────────────────────────────────────

def create_full_index(triples: List[Triple]) -> Tuple[List[IndexedTriple], RelationMap, EntityMap]:
    """Build fresh entity & relation indices."""
    ent2idx: EntityMap = {}
    rel2idx: RelationMap = {}
    e_cnt = r_cnt = 0
    result = []

    for h, r, t in triples:
        if h not in ent2idx:
            ent2idx[h] = e_cnt
            e_cnt += 1
        if t not in ent2idx:
            ent2idx[t] = e_cnt
            e_cnt += 1
        if r not in rel2idx:
            rel2idx[r] = r_cnt
            r_cnt += 1
        result.append([ent2idx[h], rel2idx[r], ent2idx[t]])

    return result, dict(rel2idx), dict(ent2idx)


def index_new_entities_fixed_rels(
    triples: List[Triple],
    fixed_rels: RelationMap
) -> Tuple[List[IndexedTriple], EntityMap]:
    """New entity ids, reuse fixed relation ids, skip unknown relations."""
    ent2idx: EntityMap = {}
    e_cnt = 0
    result = []

    for h, r, t in triples:
        if r not in fixed_rels:
            continue
        if h not in ent2idx:
            ent2idx[h] = e_cnt
            e_cnt += 1
        if t not in ent2idx:
            ent2idx[t] = e_cnt
            e_cnt += 1
        result.append([ent2idx[h], fixed_rels[r], ent2idx[t]])

    return result, dict(ent2idx)


def remap_to_known_ids(
    triples: List[Triple],
    ent2idx: EntityMap,
    rel2idx: RelationMap
) -> List[IndexedTriple]:
    """Keep only triples where both entities and relation are known."""
    result = []
    skipped = 0

    for h, r, t in triples:
        if h in ent2idx and t in ent2idx and r in rel2idx:
            result.append([ent2idx[h], rel2idx[r], ent2idx[t]])
        else:
            skipped += 1

    if skipped:
        print(f"  skipped {skipped} triples with unknown entities/relations")
    return result


def extract_type_groups(
    triples: List[Triple],
    ent2idx: EntityMap,
    type2id: Dict[str, int] | None = None
) -> Tuple[EntityTypeGroups, EntityMap, Dict[str, int]]:
    """Extract type -> [ent_idx] mapping (for :: notation)."""
    if type2id is None:
        type2id = {}
    
    type_counter = len(type2id)
    groups: EntityTypeGroups = defaultdict(list)

    for h, _, t in triples:
        h_type = h.split("::", 1)[0]
        t_type = t.split("::", 1)[0]

        if h_type not in type2id:
            type2id[h_type] = type_counter
            type_counter += 1
        if t_type not in type2id:
            type2id[t_type] = type_counter
            type_counter += 1

        groups[type2id[h_type]].append(ent2idx[h])
        groups[type2id[t_type]].append(ent2idx[t])

    return dict(groups), ent2idx, type2id


def flatten_type_map(groups: EntityTypeGroups) -> FlatEntityTypes:
    """Convert group mapping to flat entity->type mapping."""
    return {ent: typ for typ, ents in groups.items() for ent in ents}


# ────────────────────────────────────────────────
# Helper: normalize dataset name
# ────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Convert dataset name to lowercase for case-insensitive matching."""
    return name.lower().strip()


# ────────────────────────────────────────────────
# Helper: check dataset membership
# ────────────────────────────────────────────────

def _in_dataset_set(name: str, dataset_set: set) -> bool:
    """Check if dataset name is in set (case-insensitive)."""
    return _normalize_name(name) in dataset_set


# ────────────────────────────────────────────────
# Helper: process inference split
# ────────────────────────────────────────────────

def _process_inference_split(
    data_dir: Path,
    test_type: str,
    rel_map: RelationMap,
    ent_map: EntityMap | None = None,
    type2id: Dict[str, int] | None = None,
    with_types: bool = False
) -> dict:
    """Load and process a single inference split (inference_1 or inference_2)."""
    suffix = "0" if test_type == "inference_1" else "1"
    
    ind_train_raw = read_triples(data_dir / f"test_{suffix}_graph.txt")
    ind_valid_raw = read_triples(data_dir / f"test_{suffix}_samples.txt")
    
    ind_train_idx, ind_ent_map = index_new_entities_fixed_rels(ind_train_raw, rel_map)
    ind_valid_idx = remap_to_known_ids(ind_valid_raw, ind_ent_map, rel_map)
    
    result = {
        "train": ind_train_idx,
        "valid": ind_valid_idx,
    }
    
    if with_types and ent_map is not None and type2id is not None:
        type_groups_ind, _ , type2id = extract_type_groups(
            ind_train_raw, ind_ent_map, type2id
        )
        result["ent_type"] = flatten_type_map(type_groups_ind)
    
    return result


# ────────────────────────────────────────────────
# Helper: process Shomer benchmark
# ────────────────────────────────────────────────

def _process_shomer_benchmark(data_dir: Path, data_name: str, args: Any) -> dict:
    """Process Shomer benchmark (codex_m_E, hetionet_E, wn18rr_E)."""
    train_raw = read_triples(data_dir / "train_graph.txt")
    valid_raw = read_triples(data_dir / "valid_samples.txt")
    
    train_idx, rel_map, ent_map = create_full_index(train_raw)
    
    # Check if entity_base is requested and dataset supports it
    use_entity_base = (
        args.model_graph_type == "entity_base" 
        and _in_dataset_set(data_name, ENTITY_BASE_DATASETS)
    )
    
    if use_entity_base:
        # Entity base mode (only for hetionet_E)
        type_groups_tr, ent_map, type2id = extract_type_groups(train_raw, ent_map)
        valid_idx = remap_to_known_ids(valid_raw, ent_map, rel_map)
        
        ind_result = _process_inference_split(
            data_dir, args.test_type, rel_map, ent_map, type2id, with_types=True
        )
        
        save_data = {
            "train_graph": {
                "train": train_idx,
                "valid": valid_idx,
                "ent_type": flatten_type_map(type_groups_tr)
            },
            "ind_test_graph": ind_result,
        }
        
        print(f"  Entity base mode: {len(flatten_type_map(type_groups_tr))} train types")
        print(f"  Entity base mode: {len(ind_result.get('ent_type', {}))} ind types")
    else:
        # Relation base mode (all Shomer datasets)
        valid_idx = remap_to_known_ids(valid_raw, ent_map, rel_map)
        ind_result = _process_inference_split(data_dir, args.test_type, rel_map)
        
        save_data = {
            "train_graph": {"train": train_idx, "valid": valid_idx},
            "ind_test_graph": ind_result,
        }
    
    return save_data


# ────────────────────────────────────────────────
# Helper: process Grail benchmark
# ────────────────────────────────────────────────

def _process_grail_benchmark(data_dir: Path, data_name: str) -> dict:
    """Process Grail benchmark (nell_v1-4, fb237_v1-4, wn18rr_v1-4)."""
    train_raw = read_triples(data_dir / "train.txt")
    valid_raw = read_triples(data_dir / "valid.txt")
    test_raw = read_triples(data_dir / "test.txt")

    train_idx, rel_map, ent_map = create_full_index(train_raw)
    valid_idx = remap_to_known_ids(valid_raw, ent_map, rel_map)
    test_idx = remap_to_known_ids(test_raw, ent_map, rel_map)

    # Inductive split
    ind_path = Path("dataset") / f"{data_name}_ind"
    ind_tr = read_triples(ind_path / "train.txt")
    ind_va = read_triples(ind_path / "valid.txt")
    ind_te = read_triples(ind_path / "test.txt")

    ind_tr_idx, ind_ent = index_new_entities_fixed_rels(ind_tr, rel_map)
    ind_va_idx = remap_to_known_ids(ind_va, ind_ent, rel_map)
    ind_te_idx = remap_to_known_ids(ind_te, ind_ent, rel_map)

    return {
        "train_graph": {
            "train": train_idx,
            "valid": valid_idx,
            "test": test_idx
        },
        "ind_test_graph": {
            "train": ind_tr_idx,
            "valid": ind_va_idx,
            "test": ind_te_idx
        }
    }


# ────────────────────────────────────────────────
# Main logic
# ────────────────────────────────────────────────

def data_to_pickle(args: Any) -> None:
    """Load raw data, process, and save as pickle."""
    data_name = args.data_name
    normalized_name = _normalize_name(data_name)
    data_dir = Path(args.benchmark) / data_name
    out_path = Path(args.benchmark) / f"{data_name}.pkl"

    # Dispatch based on benchmark type
    if args.benchmark == "dataset/new_data":
        if not _in_dataset_set(normalized_name, SHOMER_DATASETS):
            print(f"Warning: '{data_name}' not in known Shomer datasets")
            return
        save_data = _process_shomer_benchmark(data_dir, data_name, args)

    elif args.benchmark == "dataset":
        if not _in_dataset_set(normalized_name, GRAIL_DATASETS):
            print(f"Warning: '{data_name}' not in known Grail datasets")
            return
        save_data = _process_grail_benchmark(data_dir, data_name)

    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}. Use 'shomer' or 'grail'.")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved -> {out_path}")
    print(f"   Train: {len(save_data['train_graph']['train'])} triples")
    print(f"   Ind:   {len(save_data['ind_test_graph']['train'])} triples")