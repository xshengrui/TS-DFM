import argparse
import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_FILE = ROOT / "Data" / "reactions_test.pickle"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted Transition1x TS XYZ files with paper RMSD/DMAE metrics"
    )
    parser.add_argument("--hdf5", required=True, help="Transition1x HDF5 file")
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory containing predicted XYZ files and manifest.csv",
    )
    parser.add_argument(
        "--manifest",
        help="Prediction manifest CSV (default: <pred-dir>/manifest.csv)",
    )
    parser.add_argument(
        "--split-file",
        default=str(DEFAULT_SPLIT_FILE),
        help="Reaction split pickle (default: Data/reactions_test.pickle)",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path for per-reaction metric CSV output",
    )
    return parser


def kabsch_align(predicted, target):
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target coordinates must have matching [N, D] shapes")
    pred_center = predicted.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    pred = predicted - pred_center
    ref = target - target_center
    u, _, vh = np.linalg.svd(pred.T @ ref)
    correction = np.eye(predicted.shape[1])
    correction[-1, -1] = np.linalg.det(u @ vh)
    rotation = u @ correction @ vh
    return pred @ rotation + target_center


def calc_rmsd(predicted, target):
    aligned = kabsch_align(predicted, target)
    return float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))


def calc_dmae(predicted, target):
    pred_dist = np.linalg.norm(predicted[:, None, :] - predicted[None, :, :], axis=-1)
    target_dist = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=-1)
    mask = ~np.eye(predicted.shape[0], dtype=bool)
    return float(np.mean(np.abs(pred_dist[mask] - target_dist[mask])))


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def load_manifest(manifest_path):
    with manifest_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    required = {"index", "formula", "reaction", "predicted_xyz"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return rows


def run_evaluation(args):
    import pickle

    import h5py
    from ase.io import read

    from Data.Transition1x import generator

    hdf5_path = Path(args.hdf5)
    pred_dir = Path(args.pred_dir)
    manifest_path = Path(args.manifest) if args.manifest else pred_dir / "manifest.csv"
    split_path = Path(args.split_file)

    for label, path in (
        ("HDF5", hdf5_path),
        ("prediction directory", pred_dir),
        ("manifest", manifest_path),
        ("split", split_path),
    ):
        if label == "prediction directory":
            if not path.is_dir():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        elif not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")

    with split_path.open("rb") as source:
        split_records = [tuple(item) for item in pickle.load(source)]
    manifest_rows = load_manifest(manifest_path)
    if len(manifest_rows) != len(split_records):
        raise ValueError(
            f"Manifest has {len(manifest_rows)} rows, split has {len(split_records)} records"
        )

    results = []
    with h5py.File(hdf5_path, "r") as h5_file:
        h5_data = h5_file["data"]
        for row in manifest_rows:
            index = int(row["index"])
            formula = row["formula"]
            reaction = row["reaction"]
            if (formula, reaction) != split_records[index]:
                raise ValueError(
                    f"Manifest row {index} does not match split: {formula}/{reaction}"
                )

            pred_atoms = read(pred_dir / row["predicted_xyz"])
            true_ts = next(
                generator(
                    formula,
                    reaction,
                    h5_data[formula][reaction]["transition_state"],
                )
            )
            true_numbers = np.asarray(true_ts["atomic_numbers"], dtype=int)
            pred_numbers = np.asarray(pred_atoms.numbers, dtype=int)
            if not np.array_equal(pred_numbers, true_numbers):
                raise ValueError(f"Atomic numbers differ for {formula}/{reaction}")

            predicted = np.asarray(pred_atoms.positions, dtype=float)
            target = np.asarray(true_ts["positions"], dtype=float)
            rmsd = calc_rmsd(predicted, target)
            dmae = calc_dmae(predicted, target)
            if not math.isfinite(rmsd) or not math.isfinite(dmae):
                raise RuntimeError(f"Non-finite metrics for {formula}/{reaction}")
            results.append(
                {
                    "index": index,
                    "formula": formula,
                    "reaction": reaction,
                    "rmsd": rmsd,
                    "dmae": dmae,
                }
            )

    rmsd_summary = summarize([row["rmsd"] for row in results])
    dmae_summary = summarize([row["dmae"] for row in results])

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=("index", "formula", "reaction", "rmsd", "dmae"),
            )
            writer.writeheader()
            writer.writerows(results)

    print(f"Evaluated {len(results)} predicted TS structures")
    print(
        "RMSD mean={mean:.6f} median={median:.6f} min={min:.6f} max={max:.6f}".format(
            **rmsd_summary
        )
    )
    print(
        "DMAE mean={mean:.6f} median={median:.6f} min={min:.6f} max={max:.6f}".format(
            **dmae_summary
        )
    )


def main():
    run_evaluation(build_parser().parse_args())


if __name__ == "__main__":
    main()

"""
python -m Scripts.evaluate_ts1x_xyz --hdf5 Data/Transition1x.h5 --pred-dir logs/dynamics_flow/tsdfm_ts1x_2026_07_08__10_07_48/test_xyz_fixed --output-csv logs/dynamics_flow/tsdfm_ts1x_2026_07_08__10_07_48/test_xyz_fixed/metrics.csv
"""