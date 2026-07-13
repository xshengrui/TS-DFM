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
        pairwise_dist_to_coord,
        seed_all,
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
        "reconstruction_loss",
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
                pred_pos, reconstruction_loss = pairwise_dist_to_coord(
                    x, reactant_pos, product_pos, dist_pred
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
                    reconstruction_loss=reconstruction_loss,
                )
                write(output_dir / filename, atoms, format="extxyz")

                writer.writerow(
                    {
                        "index": index,
                        "formula": formula,
                        "reaction": reaction,
                        "atom_count": len(x),
                        "predicted_xyz": filename,
                        "reconstruction_loss": reconstruction_loss,
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
