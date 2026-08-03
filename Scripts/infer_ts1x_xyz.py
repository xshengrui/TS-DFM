import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_FILE = ROOT / "Data" / "reactions_test.pickle"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate predicted Transition1x test-set TS structures as XYZ files"
    )
    parser.add_argument("--config", required=True, help="Dynamics YAML configuration")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint_best.pth")
    parser.add_argument("--hdf5", required=True, help="Transition1x HDF5 file")
    parser.add_argument("--output", required=True, help="Directory for XYZ files and manifest.csv")
    parser.add_argument(
        "--split-file",
        default=str(DEFAULT_SPLIT_FILE),
        help="Reaction split pickle (default: Data/reactions_test.pickle)",
    )
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
        help="Maximum L-BFGS iterations for coordinate reconstruction (default: 100)",
    )
    parser.add_argument(
        "--lbfgs-lr",
        type=float,
        default=0.1,
        help="L-BFGS learning rate for coordinate reconstruction (default: 0.1)",
    )
    parser.add_argument(
        "--reconstruction-restarts",
        type=int,
        default=1,
        help=(
            "Number of coordinate reconstruction initializations. 1 preserves "
            "the paper linear-interpolation reconstruction; >1 enables robust "
            "multi-start reconstruction."
        ),
    )
    parser.add_argument(
        "--reconstruction-noise-scale",
        type=float,
        default=0.05,
        help="Noise scale for robust reconstruction random restarts (default: 0.05)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing inference output directory",
    )
    return parser


def make_output_filename(index, formula, reaction):
    def clean(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")

    return f"{index:04d}_{clean(formula)}_{clean(reaction)}_pred.xyz"


def _prepare_output_directory(output_dir, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*_pred.xyz"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} predicted XYZ files; "
            "use --overwrite to replace them"
        )


def run_inference(args):
    import pickle

    import h5py
    import torch
    import yaml
    from ase import Atoms
    from ase.io import write
    from easydict import EasyDict

    from Data.Transition1x import generator
    from Model.model import DistFlowMatchingNetwork, ODEWrapper2
    from Utils import (
        Kabsch_alignment,
        generate_fully_connected,
        seed_all,
    )
    from Utils.distance_geometry import (
        pairwise_dist_to_coord_linear_interp,
        pairwise_dist_to_coord_multi_start,
    )

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    hdf5_path = Path(args.hdf5)
    split_path = Path(args.split_file)
    output_dir = Path(args.output)

    for label, path in (
        ("config", config_path),
        ("checkpoint", checkpoint_path),
        ("HDF5", hdf5_path),
        ("split", split_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")

    if args.step_size <= 0:
        raise ValueError("--step-size must be greater than zero")
    if args.lbfgs_max_iter <= 0:
        raise ValueError("--lbfgs-max-iter must be greater than zero")
    if args.lbfgs_lr <= 0:
        raise ValueError("--lbfgs-lr must be greater than zero")
    if args.reconstruction_restarts <= 0:
        raise ValueError("--reconstruction-restarts must be greater than zero")
    if args.reconstruction_noise_scale < 0:
        raise ValueError("--reconstruction-noise-scale cannot be negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}) but is not available")

    _prepare_output_directory(output_dir, args.overwrite)

    with config_path.open("r", encoding="utf-8") as handle:
        config = EasyDict(yaml.safe_load(handle))
    with split_path.open("rb") as handle:
        reactions = pickle.load(handle)

    if not isinstance(reactions, list) or not reactions:
        raise ValueError(f"Reaction split must be a non-empty list: {split_path}")

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

    manifest_path = output_dir / "manifest.csv"
    fields = (
        "index",
        "formula",
        "reaction",
        "atom_count",
        "predicted_xyz",
        "reconstruction_method",
        "reconstruction_loss",
        "lbfgs_max_iter",
        "lbfgs_lr",
        "reconstruction_restarts",
        "reconstruction_selected_start",
    )

    with (
        h5py.File(hdf5_path, "r") as h5_file,
        manifest_path.open("w", newline="", encoding="utf-8") as manifest_file,
    ):
        if "data" not in h5_file:
            raise KeyError(f"HDF5 file has no 'data' group: {hdf5_path}")
        h5_data = h5_file["data"]
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()

        for index, record in enumerate(reactions):
            if not isinstance(record, (tuple, list)) or len(record) != 2:
                raise ValueError(f"Invalid reaction record at index {index}: {record!r}")
            formula, reaction = record

            try:
                group = h5_data[formula][reaction]
                reactant = next(generator(formula, reaction, group["reactant"]))
                product = next(generator(formula, reaction, group["product"]))

                x = torch.tensor(reactant["atomic_numbers"], dtype=torch.long, device=device)
                reactant_pos = torch.as_tensor(
                    reactant["positions"], dtype=torch.float32, device=device
                )
                product_pos = torch.as_tensor(
                    product["positions"], dtype=torch.float32, device=device
                )
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

                # Coordinate reconstruction uses LBFGS and therefore must run with gradients.
                reconstruction_selected_start = 0
                if args.reconstruction_restarts == 1:
                    pred_pos, reconstruction_loss = pairwise_dist_to_coord_linear_interp(
                        reactant_pos,
                        product_pos,
                        dist_pred,
                        max_iter=args.lbfgs_max_iter,
                        lr=args.lbfgs_lr,
                    )
                else:
                    (
                        pred_pos,
                        reconstruction_loss,
                        reconstruction_selected_start,
                    ) = pairwise_dist_to_coord_multi_start(
                        reactant_pos,
                        product_pos,
                        dist_pred,
                        max_iter=args.lbfgs_max_iter,
                        lr=args.lbfgs_lr,
                        restarts=args.reconstruction_restarts,
                        noise_scale=args.reconstruction_noise_scale,
                        seed=config.train.seed + index,
                    )
                pred_pos = pred_pos.detach()
                reconstruction_loss = float(reconstruction_loss.detach().cpu().item())
                if not torch.isfinite(pred_pos).all() or not torch.isfinite(
                    torch.tensor(reconstruction_loss)
                ):
                    raise RuntimeError("coordinate reconstruction produced non-finite values")

                filename = make_output_filename(index, formula, reaction)
                atoms = Atoms(
                    numbers=x.detach().cpu().numpy(),
                    positions=pred_pos.cpu().numpy(),
                )
                atoms.info.update(
                    formula=str(formula),
                    reaction=str(reaction),
                    reconstruction_method=(
                        "linear_interp_lbfgs"
                        if args.reconstruction_restarts == 1
                        else "multi_start_lbfgs"
                    ),
                    reconstruction_loss=reconstruction_loss,
                    reconstruction_restarts=args.reconstruction_restarts,
                    reconstruction_selected_start=reconstruction_selected_start,
                )
                write(output_dir / filename, atoms, format="extxyz")

                writer.writerow(
                    {
                        "index": index,
                        "formula": formula,
                        "reaction": reaction,
                        "atom_count": len(x),
                        "predicted_xyz": filename,
                        "reconstruction_method": (
                            "linear_interp_lbfgs"
                            if args.reconstruction_restarts == 1
                            else "multi_start_lbfgs"
                        ),
                        "reconstruction_loss": reconstruction_loss,
                        "lbfgs_max_iter": args.lbfgs_max_iter,
                        "lbfgs_lr": args.lbfgs_lr,
                        "reconstruction_restarts": args.reconstruction_restarts,
                        "reconstruction_selected_start": reconstruction_selected_start,
                    }
                )
                manifest_file.flush()
            except Exception as exc:
                raise RuntimeError(
                    f"Inference failed for test reaction {index}: {formula}/{reaction}"
                ) from exc

            print(
                f"[{index + 1}/{len(reactions)}] {formula}/{reaction} -> {filename}",
                flush=True,
            )

    print(f"Generated {len(reactions)} predicted TS structures in {output_dir}")
    print(f"Manifest: {manifest_path}")


def main():
    run_inference(build_parser().parse_args())


if __name__ == "__main__":
    main()

"""

python -m Scripts.infer_ts1x_xyz   --config Configs/Dynamics.yml   --checkpoint logs/dynamics_flow/tsdfm_ts1x_2026_07_08__10_07_48/checkpoints/checkpoint_best.pth   --hdf5 Data/Transition1x.h5   --output logs/dynamics_flow/tsdfm_ts1x_2026_07_08__10_07_48/test_xyz   --device cuda
修复1x的代码
python -m Scripts.infer_ts1x_xyz   --config Configs/Dynamics.yml   --checkpoint logs/dynamics_flow/tsdfm_ts1x_2026_07_08__10_07_48/checkpoints/checkpoint_best.pth   --hdf5 Data/Transition1x.h5   --output logs/dynamics_flow/tsdfm_ts1x_2026_07_08__10_07_48/test_xyz_fixed  --device cuda
修复1x的代码+retrain
2025
python -m Scripts.infer_ts1x_xyz   --config Configs/Dynamics.yml   --checkpoint logs/dynamics_flow/tsdfm_ts1x_paper_seed2025_2026_07_31__06_55_55/checkpoints/checkpoint_best.pth   --hdf5 Data/Transition1x.h5   --output logs/dynamics_flow/tsdfm_ts1x_paper_seed2025_2026_07_31__06_55_55/test_xyz_fixed  --device cuda
2026
python -m Scripts.infer_ts1x_xyz   --config Configs/Dynamics.yml   --checkpoint logs/dynamics_flow/tsdfm_ts1x_paper_seed2026_2026_07_31__06_56_15/checkpoints/checkpoint_best.pth   --hdf5 Data/Transition1x.h5   --output logs/dynamics_flow/tsdfm_ts1x_paper_seed2026_2026_07_31__06_56_15/test_xyz_fixed  --device cuda


"""
