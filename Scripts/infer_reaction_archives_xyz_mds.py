import argparse
import csv
import math
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Scripts.infer_reaction_archives_xyz import archive_label, iter_reaction_texts
from Scripts.infer_ts1x_xyz import make_output_filename


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate TS XYZ files from reaction archives using Classical MDS "
            "initialization and LBFGS refinement"
        )
    )
    parser.add_argument("--config", required=True, help="Dynamics YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint_best.pth")
    parser.add_argument(
        "--archives",
        nargs="+",
        required=True,
        help="One or more .tar.gz archives containing raw/<reaction>/{R,P}.xyz",
    )
    parser.add_argument("--output", required=True, help="Root output directory")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.05,
        help="ODE midpoint integration step size (default: 0.05)",
    )
    parser.add_argument(
        "--lbfgs-max-iter",
        type=int,
        default=100,
        help="Maximum LBFGS iterations after MDS initialization (default: 100)",
    )
    parser.add_argument(
        "--lbfgs-lr",
        type=float,
        default=0.1,
        help="LBFGS learning rate after MDS initialization (default: 0.1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many reactions per archive (for smoke tests)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing predicted XYZ files",
    )
    return parser


def run_inference(args):
    import torch
    import yaml
    from ase import Atoms
    from ase.io import write
    from easydict import EasyDict

    from Data.rgd1_preprocess import parse_xyz
    from Model.model import DistFlowMatchingNetwork, ODEWrapper2
    from Utils import Kabsch_alignment, generate_fully_connected, seed_all
    from Utils.distance_geometry import pairwise_dist_to_coord_mds

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    archive_paths = [Path(path) for path in args.archives]
    output_root = Path(args.output)

    for label, path in (
        ("config", config_path),
        ("checkpoint", checkpoint_path),
        *(("archive", path) for path in archive_paths),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    if args.step_size <= 0:
        raise ValueError("--step-size must be greater than zero")
    if args.lbfgs_max_iter <= 0:
        raise ValueError("--lbfgs-max-iter must be greater than zero")
    if args.lbfgs_lr <= 0:
        raise ValueError("--lbfgs-lr must be greater than zero")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}) but is not available")

    with config_path.open("r", encoding="utf-8") as handle:
        config = EasyDict(yaml.safe_load(handle))
    seed_all(config.train.seed)
    device = torch.device(args.device)
    model = DistFlowMatchingNetwork(**config.dynamic_model.parameters).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ode = ODEWrapper2(model).to(device)
    ode.eval()
    output_root.mkdir(parents=True, exist_ok=True)

    for archive_path in archive_paths:
        dataset = archive_label(archive_path)
        output_dir = output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        existing = list(output_dir.glob("*_pred.xyz"))
        if existing and not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already contains {len(existing)} predicted XYZ files; "
                "use --overwrite to replace them"
            )

        manifest_path = output_dir / "manifest.csv"
        fields = (
            "index",
            "dataset",
            "reaction",
            "atom_count",
            "predicted_xyz",
            "reconstruction_method",
            "reconstruction_loss",
            "lbfgs_max_iter",
            "lbfgs_lr",
        )
        processed = 0

        with (
            tarfile.open(archive_path, mode="r:gz") as archive,
            manifest_path.open("w", newline="", encoding="utf-8") as manifest_file,
        ):
            writer = csv.DictWriter(manifest_file, fieldnames=fields)
            writer.writeheader()

            for index, (reaction, reactant_text, product_text) in enumerate(
                iter_reaction_texts(archive)
            ):
                if args.limit is not None and index >= args.limit:
                    break
                try:
                    reactant_atoms, reactant_pos = parse_xyz(reactant_text)
                    product_atoms, product_pos = parse_xyz(product_text)
                    if not torch.equal(reactant_atoms, product_atoms):
                        raise ValueError("R.xyz and P.xyz have different atom ordering")

                    x = reactant_atoms.to(device)
                    reactant_pos = reactant_pos.to(device)
                    product_pos = product_pos.to(device)
                    single_batch = torch.zeros_like(x)
                    product_pos = Kabsch_alignment(product_pos, reactant_pos, single_batch)
                    src, dst = generate_fully_connected(single_batch)
                    edge_index = torch.stack((src, dst), dim=0)
                    dist_reactant = torch.linalg.vector_norm(
                        reactant_pos[src] - reactant_pos[dst], dim=-1
                    )
                    dist_product = torch.linalg.vector_norm(
                        product_pos[src] - product_pos[dst], dim=-1
                    )
                    dist_initial = 0.5 * (dist_reactant + dist_product)

                    with torch.no_grad():
                        dist_pred = ode(
                            x,
                            edge_index,
                            dist_reactant,
                            dist_product,
                            dist_initial,
                            single_batch,
                            step_size=args.step_size,
                        ).detach()

                    pred_pos, reconstruction_loss = pairwise_dist_to_coord_mds(
                        reactant_pos,
                        product_pos,
                        dist_pred,
                        max_iter=args.lbfgs_max_iter,
                        lr=args.lbfgs_lr,
                    )
                    reconstruction_loss = float(reconstruction_loss.cpu().item())
                    if not torch.isfinite(pred_pos).all() or not math.isfinite(
                        reconstruction_loss
                    ):
                        raise RuntimeError(
                            "coordinate reconstruction produced non-finite values"
                        )

                    filename = make_output_filename(index, dataset, reaction)
                    atoms = Atoms(
                        numbers=x.detach().cpu().numpy(),
                        positions=pred_pos.cpu().numpy(),
                    )
                    atoms.info.update(
                        dataset=dataset,
                        reaction=reaction,
                        reconstruction_method="classical_mds_lbfgs",
                        reconstruction_loss=reconstruction_loss,
                    )
                    write(output_dir / filename, atoms, format="extxyz")
                    writer.writerow(
                        {
                            "index": index,
                            "dataset": dataset,
                            "reaction": reaction,
                            "atom_count": len(x),
                            "predicted_xyz": filename,
                            "reconstruction_method": "classical_mds_lbfgs",
                            "reconstruction_loss": reconstruction_loss,
                            "lbfgs_max_iter": args.lbfgs_max_iter,
                            "lbfgs_lr": args.lbfgs_lr,
                        }
                    )
                    manifest_file.flush()
                    processed += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"Inference failed for {dataset} reaction {index}: {reaction}"
                    ) from exc

                print(f"[{dataset} {processed}] {reaction} -> {filename}", flush=True)

        print(f"Generated {processed} MDS+LBFGS TS structures in {output_dir}")
        print(f"Manifest: {manifest_path}")


def main():
    run_inference(build_parser().parse_args())


if __name__ == "__main__":
    main()
