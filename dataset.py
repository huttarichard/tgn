import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from kgclient import Client
from kgclient.kg import KGRelationship, KGEntity, stream_kg


def _parse_iso8601_to_epoch(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    # Accept both ...Z and ...+00:00
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _one_hot(index: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def _csv_quote(text: str) -> str:
    # Wrap in double quotes and escape contained quotes by doubling them
    return '"' + text.replace('"', '""') + '"'


def build_tgn_dataset_from_kgclient(
    dataset_name: str = "kgclient",
    limit: int = 50000,
    target_relations: Optional[Sequence[str]] = None,
    node_feat_dim: int = 64,
) -> Tuple[str, str, str]:
    """
    Streams relationships from kgclient and writes:
      - data/ml_{dataset}.csv          (columns: u, i, ts, label, idx)
      - data/ml_{dataset}.npy          (edge features; row 0 is padding)
      - data/ml_{dataset}_node.npy     (node features; row 0 is padding)
    Returns the three file paths.
    """
    client = Client()
    target_rel_set: Optional[Set[str]] = (
        set(target_relations) if target_relations else None
    )

    # Collect edges and entity metadata
    node_id_by_hash: Dict[str, int] = {}
    edges: List[Tuple[int, int, float, int, str]] = []
    entity_meta: Dict[str, Tuple[str, str]] = {}  # hash -> (name, kind)

    def _node_id(h: str) -> int:
        # Reserve 0 as padding, start at 1
        if h not in node_id_by_hash:
            node_id_by_hash[h] = len(node_id_by_hash) + 1
        return node_id_by_hash[h]

    for obj in stream_kg(client):
        if isinstance(obj, KGEntity):
            # Cache entity metadata for later human-readable mapping
            entity_meta[obj.hash] = (obj.name or "", obj.kind or "")
            continue
        if isinstance(obj, KGRelationship):
            rel_type = obj.relationship or ""
            if target_rel_set is not None and rel_type not in target_rel_set:
                continue
            ts = _parse_iso8601_to_epoch(obj.first_seen_at) or _parse_iso8601_to_epoch(
                obj.last_seen_at
            )
            if ts is None:
                continue
            u = _node_id(obj.head_hash)
            v = _node_id(obj.tail_hash)
            edges.append((u, v, float(ts), 1, rel_type))
            if len(edges) >= limit:
                break
        # Ignore any other streamed object types

    if not edges:
        raise RuntimeError(
            "No relationships collected from kgclient stream (after filtering)."
        )

    # Sort by timestamp for stable temporal ordering
    edges.sort(key=lambda x: x[2])

    # Build DataFrame (1-indexed edge idx; reserve idx=0 for padding in features)
    df = pd.DataFrame(
        [(u, v, ts, label, idx + 1) for idx, (u, v, ts, label, _) in enumerate(edges)],
        columns=["u", "i", "ts", "label", "idx"],
    )

    # Build edge features as simple one-hot over the observed relation types (or target set)
    if target_rel_set:
        rel_vocab = sorted(target_rel_set)
    else:
        rel_vocab = sorted({r for *_, r in edges})
    rel_to_ix = {r: i for i, r in enumerate(rel_vocab)}
    feat_dim = len(rel_vocab) if len(rel_vocab) > 0 else 1

    edge_features = np.zeros(
        (len(edges) + 1, feat_dim), dtype=np.float32
    )  # row 0 is padding
    for (u, v, ts, label, rel), row_idx in zip(edges, range(1, len(edges) + 1)):
        if feat_dim == 1:
            edge_features[row_idx, 0] = 1.0
        else:
            edge_features[row_idx] = _one_hot(rel_to_ix.get(rel, -1), feat_dim)

    # Build node features (random small values); row 0 is padding
    max_node_id = max(node_id_by_hash.values()) if node_id_by_hash else 0
    node_features = np.zeros((max_node_id + 1, node_feat_dim), dtype=np.float32)
    if max_node_id >= 1:
        node_features[1:] = np.random.uniform(
            low=-0.01, high=0.01, size=(max_node_id, node_feat_dim)
        ).astype(np.float32)

    # Write files
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"ml_{dataset_name}.csv"
    edge_feat_path = data_dir / f"ml_{dataset_name}.npy"
    node_feat_path = data_dir / f"ml_{dataset_name}_node.npy"
    node_meta_path = data_dir / f"ml_{dataset_name}_nodes.csv"

    df.to_csv(csv_path, index=False)
    np.save(edge_feat_path, edge_features)
    np.save(node_feat_path, node_features)
    # Save node id -> metadata (hash, name, kind) for human-readable evaluation
    with node_meta_path.open("w", encoding="utf-8") as f:
        f.write("id,hash,name,kind\n")
        # node_id_by_hash: hash -> id ; write sorted by id
        for h, i in sorted(node_id_by_hash.items(), key=lambda kv: kv[1]):
            name, kind = entity_meta.get(h, ("", ""))
            # CSV-quote name/kind so commas are safe
            name_escaped = _csv_quote(name)
            kind_escaped = _csv_quote(kind)
            f.write(f"{i},{h},{name_escaped},{kind_escaped}\n")

    return str(csv_path), str(edge_feat_path), str(node_feat_path)


def main():
    parser = argparse.ArgumentParser("Build a minimal TGN dataset from kgclient stream")
    parser.add_argument(
        "--data", type=str, default="kgclient", help="Dataset name (used in file names)"
    )
    parser.add_argument(
        "--limit", type=int, default=50000, help="Max number of relationships to stream"
    )
    parser.add_argument(
        "--node-dim", type=int, default=64, help="Node feature dimension"
    )
    parser.add_argument(
        "--relations",
        type=str,
        # default="INVESTS_IN,IMPACT,POSITIVE_IMPACT_ON,NEGATIVE_IMPACT_ON,CONTROL",
        default=None,
        help="Comma-separated list of relation types to include; leave empty for all",
    )
    args = parser.parse_args()

    rels = (
        [r.strip() for r in args.relations.split(",") if r.strip()]
        if args.relations
        else None
    )
    csv_path, edge_path, node_path = build_tgn_dataset_from_kgclient(
        dataset_name=args.data,
        limit=args.limit,
        target_relations=rels,
        node_feat_dim=args.node_dim,
    )
    print("Wrote:")
    print(f"  CSV:  {csv_path}")
    print(f"  EFEAT:{edge_path}")
    print(f"  NFEAT:{node_path}")
    print()


if __name__ == "__main__":
    main()
