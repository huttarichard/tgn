import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import pandas as pd

from utils.data_processing import get_data, compute_time_statistics
from utils.utils import RandEdgeSampler, get_neighbor_finder
from model.tgn import TGN


def load_node_meta(data_name: str) -> Dict[int, Tuple[str, str, str]]:
    """
    Load node metadata written by the dataset builder.
    Returns: {node_id: (hash, name, kind)}
    """
    meta_path = Path(f"data/ml_{data_name}_nodes.csv")
    mapping: Dict[int, Tuple[str, str, str]] = {}
    if not meta_path.exists():
        return mapping
    df = pd.read_csv(meta_path)
    for _, row in df.iterrows():
        try:
            mapping[int(row["id"])] = (
                str(row["hash"]),
                str(row["name"]),
                str(row["kind"]),
            )
        except Exception:
            continue
    return mapping


def main():
    parser = argparse.ArgumentParser("Predict likely new edges (human-readable)")
    parser.add_argument("--data", type=str, default="kgclient")
    parser.add_argument(
        "--prefix", type=str, default="kg", help="Prefix used during training"
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Top-K predictions to keep per source"
    )
    parser.add_argument(
        "--candidates", type=int, default=200, help="Random candidates per source"
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="How many test interactions to sample"
    )
    parser.add_argument("--gpu", type=int, default=0)
    # Model hyperparams (must match training)
    parser.add_argument("--n_degree", type=int, default=10)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--drop_out", type=float, default=0.1)
    parser.add_argument("--use_memory", action="store_true")
    parser.add_argument(
        "--embedding_module",
        type=str,
        default="graph_attention",
        choices=["graph_attention", "graph_sum", "identity", "time"],
    )
    parser.add_argument(
        "--message_function", type=str, default="identity", choices=["mlp", "identity"]
    )
    parser.add_argument(
        "--memory_updater", type=str, default="gru", choices=["gru", "rnn"]
    )
    parser.add_argument("--aggregator", type=str, default="last")
    parser.add_argument("--message_dim", type=int, default=100)
    parser.add_argument("--memory_dim", type=int, default=172)
    args = parser.parse_args()

    # Load data
    (
        node_features,
        edge_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    ) = get_data(
        args.data,
        different_new_nodes_between_val_and_test=False,
        randomize_features=False,
    )

    # Neighbor finders
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False)

    # Device
    device_string = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)

    # Time stats
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = (
        compute_time_statistics(
            full_data.sources, full_data.destinations, full_data.timestamps
        )
    )

    # Build model and load checkpoint
    tgn = TGN(
        neighbor_finder=full_ngh_finder,
        node_features=node_features,
        edge_features=edge_features,
        device=device,
        n_layers=args.n_layer,
        n_heads=args.n_head,
        dropout=args.drop_out,
        use_memory=args.use_memory,
        message_dimension=args.message_dim,
        memory_dimension=args.memory_dim,
        memory_update_at_start=True,
        embedding_module_type=args.embedding_module,
        message_function=args.message_function,
        aggregator_type=args.aggregator,
        memory_updater_type=args.memory_updater,
        n_neighbors=args.n_degree,
        mean_time_shift_src=mean_time_shift_src,
        std_time_shift_src=std_time_shift_src,
        mean_time_shift_dst=mean_time_shift_dst,
        std_time_shift_dst=std_time_shift_dst,
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
        dyrep=False,
    )
    tgn = tgn.to(device)
    model_path = Path(f"./saved_models/{args.prefix}-{args.data}.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")
    state = torch.load(model_path, map_location=device)
    tgn.load_state_dict(state)
    tgn.eval()

    # Candidate sampler across all nodes
    rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=123)

    # Optional: backup memory to avoid mutating between batches
    memory_backup = None
    if args.use_memory:
        memory_backup = tgn.memory.backup_memory()

    # Prepare output
    out_dir = Path("results") / f"{args.prefix}-{args.data}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "predicted_edges.csv"

    node_meta = load_node_meta(args.data)  # {id: (hash, name, kind)}

    records: List[Dict] = []
    # Sample first N test interactions (sorted by time already via split)
    n = min(args.limit, len(test_data.sources))
    for idx in range(n):
        s = int(test_data.sources[idx])
        ts = float(test_data.timestamps[idx])
        # Sample candidate tails
        _, cand = rand_sampler.sample(args.candidates)
        # Remove self and duplicates
        cand = np.array([c for c in cand if c != s], dtype=np.int64)
        if cand.size > 1:
            cand = np.unique(cand)
        if len(cand) == 0:
            continue
        # Build inputs
        sources_batch = np.full(shape=(len(cand),), fill_value=s, dtype=np.int64)
        destinations_batch = cand
        negatives_batch = cand  # placeholder to satisfy API; we ignore neg outputs
        timestamps_batch = np.full(shape=(len(cand),), fill_value=ts, dtype=np.float64)
        edge_idxs_batch = np.zeros(
            shape=(len(cand),), dtype=np.int64
        )  # 0 = padding feature
        with torch.no_grad():
            pos_prob, _ = tgn.compute_edge_probabilities(
                sources_batch,
                destinations_batch,
                negatives_batch,
                timestamps_batch,
                edge_idxs_batch,
                args.n_degree,
            )
            scores = pos_prob.cpu().numpy()
        # Top-K unique destinations
        order = np.argsort(-scores)
        seen = set()
        taken = 0
        for j in order:
            dst = int(cand[j])
            if dst in seen:
                continue
            seen.add(dst)
            score = float(scores[j])
            s_hash, s_name, s_kind = node_meta.get(s, ("", "", ""))
            d_hash, d_name, d_kind = node_meta.get(dst, ("", "", ""))
            records.append(
                {
                    "ts": ts,
                    "source_id": s,
                    "source_hash": s_hash,
                    "source_name": s_name,
                    "source_kind": s_kind,
                    "pred_id": dst,
                    "pred_hash": d_hash,
                    "pred_name": d_name,
                    "pred_kind": d_kind,
                    "score": score,
                }
            )
            taken += 1
            if taken >= args.topk:
                break

    # Restore memory
    if memory_backup is not None:
        tgn.memory.restore_memory(memory_backup)

    # Save CSV
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    print(f"Wrote predictions: {out_csv}")


if __name__ == "__main__":
    main()
