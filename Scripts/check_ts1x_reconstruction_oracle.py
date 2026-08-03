import argparse
import csv
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_FILE = ROOT / "Data" / "reactions_test.pickle"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct Transition1x TS coordinates from exact TS distance "
            "matrices to check the coordinate-reconstruction error floor"
        )
    )
    parser.add_argument("--hdf5", required=True, help="Transition1x HDF5 file")
    parser.add_argument(
        "--split-file",
        default=str(DEFAULT_SPLIT_FILE),
        help="Reaction split pickle (default: Data/reactions_test.pickle)",
    )
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=100,
        help="Maximum L-BFGS iterations for coordinate reconstruction",
    )
    parser.add_argument(
        "--lbfgs-lr",
        type=float,
        default=0.1,
        help="L-BFGS learning rate for coordinate reconstruction",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Optional number of split records to check",
    )
    parser.add_argument("--output-csv", help="Optional per-reaction CSV output")
    return parser


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_check(args):
    import pickle

    import h5py

    from Data.Transition1x import generator
    from Scripts.evaluate_ts1x_xyz import calc_dmae, calc_rmsd
    from Utils import Kabsch_alignment, generate_fully_connected
    from Utils.distance_geometry import pairwise_dist_to_coord_linear_interp

    hdf5_path = Path(args.hdf5)
    split_path = Path(args.split_file)
    for label, path in (("HDF5", hdf5_path), ("split", split_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    if args.lbfgs_max_iter <= 0:
        raise ValueError("--lbfgs-max-iter must be greater than zero")
    if args.lbfgs_lr <= 0:
        raise ValueError("--lbfgs-lr must be greater than zero")

    device = torch.device(args.device)
    with split_path.open("rb") as source:
        reactions = pickle.load(source)
    if args.max_items is not None:
        reactions = reactions[: args.max_items]

    rows = []
    with h5py.File(hdf5_path, "r") as h5_file:
        h5_data = h5_file["data"]
        for index, (formula, reaction) in enumerate(reactions):
            group = h5_data[formula][reaction]
            reactant = next(generator(formula, reaction, group["reactant"]))
            product = next(generator(formula, reaction, group["product"]))
            ts = next(generator(formula, reaction, group["transition_state"]))

            x = torch.tensor(reactant["atomic_numbers"], dtype=torch.long, device=device)
            batch = torch.zeros_like(x)
            reactant_pos = torch.as_tensor(
                reactant["positions"], dtype=torch.float32, device=device
            )
            product_pos = torch.as_tensor(
                product["positions"], dtype=torch.float32, device=device
            )
            ts_pos = torch.as_tensor(ts["positions"], dtype=torch.float32, device=device)
            product_pos = Kabsch_alignment(product_pos, reactant_pos, batch)
            ts_pos = Kabsch_alignment(ts_pos, reactant_pos, batch)

            src, dst = generate_fully_connected(batch)
            exact_ts_dist = torch.linalg.vector_norm(ts_pos[src] - ts_pos[dst], dim=-1)
            pred_pos, reconstruction_loss = pairwise_dist_to_coord_linear_interp(
                reactant_pos,
                product_pos,
                exact_ts_dist,
                max_iter=args.lbfgs_max_iter,
                lr=args.lbfgs_lr,
            )

            predicted = pred_pos.detach().cpu().numpy()
            target = ts_pos.detach().cpu().numpy()
            rows.append(
                {
                    "index": index,
                    "formula": formula,
                    "reaction": reaction,
                    "rmsd": calc_rmsd(predicted, target),
                    "rmsd_reflection": calc_rmsd(
                        predicted, target, allow_reflection=True
                    ),
                    "dmae": calc_dmae(predicted, target),
                    "reconstruction_loss": float(reconstruction_loss.cpu().item()),
                }
            )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "index",
                    "formula",
                    "reaction",
                    "rmsd",
                    "rmsd_reflection",
                    "dmae",
                    "reconstruction_loss",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"Checked {len(rows)} exact-distance reconstructions")
    for key, label in (
        ("rmsd", "RMSD"),
        ("rmsd_reflection", "RMSD(reflection)"),
        ("dmae", "DMAE"),
    ):
        summary = summarize([row[key] for row in rows])
        print(
            f"{label} mean={summary['mean']:.6f} median={summary['median']:.6f} "
            f"min={summary['min']:.6f} max={summary['max']:.6f}"
        )


def main():
    run_check(build_parser().parse_args())


if __name__ == "__main__":
    main()
