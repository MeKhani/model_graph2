from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────
#  Type aliases
# ────────────────────────────────────────────────

Triple = List[str]                      # raw: ["head", "rel", "tail"]
IndexedTriple = List[int]               # [hid, rid, tid]
EntityMap = Dict[str, int]
RelationMap = Dict[str, int]
EntityTypeGroups = Dict[int, List[int]]  # type_id → [ent ids]
FlatEntityTypes = Dict[int, int]         # ent_id → type_id


# ────────────────────────────────────────────────
#  File reading helpers
# ────────────────────────────────────────────────

def read_triples(path: str | Path) -> List[Triple]:
    """Read space-separated triples from a text file, one per line"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    triples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:          # fast path — most lines are good
                triples.append(parts)
            elif (clean := line.strip()) and len(clean.split()) == 3:
                triples.append(clean.split())
    return triples


# ────────────────────────────────────────────────
#  Re-indexing functions
# ────────────────────────────────────────────────

def create_full_index(triples: List[Triple]) -> Tuple[List[IndexedTriple], RelationMap, EntityMap]:
    """Build fresh entity & relation indices"""
    ent2idx: EntityMap = {}
    rel2idx: RelationMap = {}
    e_cnt = r_cnt = 0
    result = []

    for h, r, t in triples:
        if h not in ent2idx: ent2idx[h] = e_cnt; e_cnt += 1
        if t not in ent2idx: ent2idx[t] = e_cnt; e_cnt += 1
        if r not in rel2idx: rel2idx[r] = r_cnt; r_cnt += 1
        result.append([ent2idx[h], rel2idx[r], ent2idx[t]])

    return result, dict(rel2idx), dict(ent2idx)


def index_new_entities_fixed_rels(
    triples: List[Triple],
    fixed_rels: RelationMap
) -> Tuple[List[IndexedTriple], EntityMap]:
    """New entity ids, reuse fixed relation ids, skip unknown relations"""
    ent2idx: EntityMap = {}
    e_cnt = 0
    result = []

    for h, r, t in triples:
        if r not in fixed_rels:
            continue
        if h not in ent2idx: ent2idx[h] = e_cnt; e_cnt += 1
        if t not in ent2idx: ent2idx[t] = e_cnt; e_cnt += 1
        result.append([ent2idx[h], fixed_rels[r], ent2idx[t]])

    return result, dict(ent2idx)


def remap_to_known_ids(
    triples: List[Triple],
    ent2idx: EntityMap,
    rel2idx: RelationMap
) -> List[IndexedTriple]:
    """Keep only triples where both entities and relation are known"""
    result = []
    skipped = 0

    for h, r, t in triples:
        if h not in ent2idx or t not in ent2idx or r not in rel2idx:
            skipped += 1
            continue
        result.append([ent2idx[h], rel2idx[r], ent2idx[t]])

    if skipped:
        print(f"  skipped {skipped} triples with unknown entities/relations")
    return result


def extract_type_groups(
    triples: List[Triple],
    ent2idx: EntityMap,
    type2id :Dict[str, int]
) -> Tuple[EntityTypeGroups, Dict[str, int]]:
    """Extract type → [ent_idx] mapping (for :: notation)"""
    if len(type2id)==0:
        type2id = {}
    else:
        type2id = type2id 
    type_counter = len(type2id)
        
    groups: EntityTypeGroups = {}

    for h, _, t in triples:
        h_type = h.split("::", 1)[0]
        t_type = t.split("::", 1)[0]

        if h_type not in type2id:
            type2id[h_type] = type_counter
            type_counter += 1
        if t_type not in type2id:
            type2id[t_type] = type_counter
            type_counter += 1

        groups.setdefault(type2id[h_type], []).append(ent2idx[h])
        groups.setdefault(type2id[t_type], []).append(ent2idx[t])

    return groups, type2id
def extract_type_groups_for_nell(
    triples: List[Triple],
    ent2idx: EntityMap,
     type2id: Dict[str, int]
) -> Tuple[EntityTypeGroups, Dict[str, int]]:
    """Extract type → [ent_idx] mapping (for :: notation)"""
    if len(type2id)==0:
        type2id = {}
    else:
        type2id = type2id 
    type_counter = len(type2id)
    
    groups: EntityTypeGroups = {}

    for h, _, t in triples:
        h_type = h.split(":")[1]
        t_type = t.split(":")[1]

        if h_type not in type2id:
            type2id[h_type] = type_counter
            type_counter += 1
        if t_type not in type2id:
            type2id[t_type] = type_counter
            type_counter += 1

        groups.setdefault(type2id[h_type], []).append(ent2idx[h])
        groups.setdefault(type2id[t_type], []).append(ent2idx[t])

    return groups, type2id


def flatten_type_map(groups: EntityTypeGroups) -> FlatEntityTypes:
    return {ent: typ for typ, ents in groups.items() for ent in ents}


# ────────────────────────────────────────────────
#  PrimeKG specific helper
# ────────────────────────────────────────────────

def triples_from_primekg_df(df: pd.DataFrame) -> List[IndexedTriple]:
    all_entities = set(df["x_name"]) | set(df["y_name"])
    ent2idx = {e: i for i, e in enumerate(sorted(all_entities))}
    rel2idx = {r: i for i, r in enumerate(sorted(df["relation"].unique()))}

    return [
        [ent2idx[row["x_name"]], rel2idx[row["relation"]], ent2idx[row["y_name"]]]
        for _, row in df.iterrows()
    ]


# ────────────────────────────────────────────────
#  Main logic
# ────────────────────────────────────────────────

def data_to_pickle(args: Any) -> None:
    data_name = args.data_name
    base = Path("dataset")
    out_path = base / f"{data_name}.pkl"

    if args.new_data == "new":
        data_dir = base / "new_data" / data_name

        train_raw = read_triples(data_dir / "train_graph.txt")
        valid_raw = read_triples(data_dir / "valid_samples.txt")

        if args.is_relation_model_graph:
            # ── relation model graph (no types) ────────────────────────
            train_idx, rel_map, ent_map = create_full_index(train_raw)
            valid_idx = remap_to_known_ids(valid_raw, ent_map, rel_map)

            ind_train_raw = read_triples(data_dir / "test_0_graph.txt")
            ind_valid_raw = read_triples(data_dir / "test_0_sample.txt")   # ← fixed filename

            ind_train_idx, ind_ent_map = index_new_entities_fixed_rels(ind_train_raw, rel_map)
            ind_valid_idx = remap_to_known_ids(ind_valid_raw, ind_ent_map, rel_map)

            save_data = {
                "train_graph": {"train": train_idx, "valid": valid_idx},
                "ind_test_graph": {"train": ind_train_idx, "valid": ind_valid_idx},
            }

        elif "hito" in data_name.lower():
            # ── hito-style (with entity types) ─────────────────────────
            train_idx, rel_map, ent_map = create_full_index(train_raw)
            type_groups_tr, type2id = extract_type_groups(train_raw, ent_map,typ2id={})
            valid_idx = remap_to_known_ids(valid_raw, ent_map, rel_map)

            ind_train_raw = read_triples(data_dir / "test_0_graph.txt")
            # Note: original code read same file twice → most likely bug
            # Here we assume test_0_sample.txt is the validation set (as in relation_model_graph branch)
            ind_valid_raw = read_triples(data_dir / "test_0_sample.txt")

            ind_train_idx, ind_ent_map, type_groups_ind = extract_type_groups(
                ind_train_raw,
                index_new_entities_fixed_rels(ind_train_raw, rel_map)[1]
                ,type2id
            )
            ind_valid_idx = remap_to_known_ids(ind_valid_raw, ind_ent_map, rel_map)

            save_data = {
                "train_graph": {
                    "train": train_idx,
                    "valid": valid_idx,
                    "ent_type": flatten_type_map(type_groups_tr)
                },
                "ind_test_graph": {
                    "train": ind_train_idx,
                    "test": ind_valid_idx,   # ← kept your key name
                    "ent_type": flatten_type_map(type_groups_ind)
                }
            }

            print(f"Train entity types count: {len(flatten_type_map(type_groups_tr))}")
            print(f"Ind  entity types count: {len(flatten_type_map(type_groups_ind))}")

        else:
            raise ValueError(f"Unknown new_data mode for dataset {data_name}")

    else:
        # ── Classic / PrimeKG style ───────────────────────────────────
        if "primekg" in data_name.lower():
            kg_path = base / data_name / "kg.csv"
            df = pd.read_csv(kg_path)

            train_df, ind_df = train_test_split(df, test_size=0.2, random_state=42)

            tr_tr, tr_val = train_test_split(train_df, test_size=0.1, random_state=42)
            va_tr, te_tr = train_test_split(tr_val, test_size=0.4, random_state=42)

            tr_in, in_val = train_test_split(ind_df, test_size=0.2, random_state=42)
            va_in, te_in = train_test_split(in_val, test_size=0.5, random_state=42)

            save_data = {
                "train_graph": {
                    "train": triples_from_primekg_df(tr_tr),
                    "valid": triples_from_primekg_df(va_tr),
                    "test": triples_from_primekg_df(te_tr),
                },
                "ind_test_graph": {
                    "train": triples_from_primekg_df(tr_in),
                    "valid": triples_from_primekg_df(va_in),
                    "test": triples_from_primekg_df(te_in),
                }
            }

        else:
            # Standard KG + optional inductive split
            if args.is_relation_model_graph:

                dpath = base / data_name

                train_raw = read_triples(dpath / "train.txt")
                valid_raw = read_triples(dpath / "valid.txt")
                test_raw  = read_triples(dpath / "test.txt")

                train_idx, rel_map, ent_map = create_full_index(train_raw)
                valid_idx = remap_to_known_ids(valid_raw, ent_map, rel_map)
                test_idx  = remap_to_known_ids(test_raw,  ent_map, rel_map)

            
                ind_path = base / f"{data_name}_ind"
                ind_tr = read_triples(ind_path / "train.txt")
                ind_va = read_triples(ind_path / "valid.txt")
                ind_te = read_triples(ind_path / "test.txt")

                ind_tr_idx, ind_ent = index_new_entities_fixed_rels(ind_tr, rel_map)
                ind_va_idx = remap_to_known_ids(ind_va, ind_ent, rel_map)
                ind_te_idx = remap_to_known_ids(ind_te, ind_ent, rel_map)

                save_data = {
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
            if not args.is_relation_model_graph and "nell" in data_name.lower():
                dpath = base / data_name

                train_raw = read_triples(dpath / "train.txt")
                valid_raw = read_triples(dpath / "valid.txt")
                test_raw  = read_triples(dpath / "test.txt")
                train_idx, rel_map, ent_map = create_full_index(train_raw)
                type_groups_tr, type2id = extract_type_groups_for_nell(train_raw, ent_map,type2id={})
                valid_idx = remap_to_known_ids(valid_raw, ent_map, rel_map)
                test_idx  = remap_to_known_ids(test_raw,  ent_map, rel_map)

            
                ind_path = base / f"{data_name}_ind"
                ind_tr = read_triples(ind_path / "train.txt")
                ind_va = read_triples(ind_path / "valid.txt")
                ind_te = read_triples(ind_path / "test.txt")

                ind_tr_idx, ind_ent = index_new_entities_fixed_rels(ind_tr, rel_map)
                type_groups_id_tr, type2id = extract_type_groups_for_nell(ind_tr, ind_ent,type2id)
                ind_va_idx = remap_to_known_ids(ind_va, ind_ent, rel_map)
                ind_te_idx = remap_to_known_ids(ind_te, ind_ent, rel_map)
                # print(f"the type_group  is{len(type_groups_tr)}")
                # print(f"the type_group inductive is{len(type_groups_id_tr)}")
                # print(f"the type_group inductive is{(type_groups_id_tr)}")
                # return
                save_data = {
                    "train_graph": {
                        "train": train_idx,
                        "valid": valid_idx,
                        "test": test_idx ,
                        "ent_type": flatten_type_map(type_groups_tr)
    

                    },
                    "ind_test_graph": {
                        "train": ind_tr_idx,
                        "valid": ind_va_idx,
                        "test": ind_te_idx,
                        "ent_type": flatten_type_map(type_groups_id_tr)
                    }
                }
                
                


    # ── Save ────────────────────────────────────────────────────────
    with out_path.open("wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved processed dataset → {out_path}")
