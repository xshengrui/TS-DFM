import argparse
import csv
import sys
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate predicted TS XYZ files from R.xyz/P.xyz reaction archives"
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


def archive_label(path):
    name = Path(path).name
    if name.lower().endswith(".tar.gz"):
        return name[:-7]
    return Path(name).stem


def iter_reaction_texts(archive):
    pending = {}
    completed = set()

    for member in archive:
        if not member.isfile():
            continue
        path = PurePosixPath(member.name)
        if len(path.parts) != 3 or path.parts[0] != "raw":
            continue
        filename = path.name
        if filename not in {"R.xyz", "P.xyz"}:
            continue

        reaction = path.parts[1]
        if reaction in completed:
            raise ValueError(f"Duplicate endpoint files for reaction {reaction}")
        files = pending.setdefault(reaction, {})
        if filename in files:
            raise ValueError(f"Duplicate {filename} for reaction {reaction}")

        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"Unable to read archive member {member.name}")
        files[filename] = source.read().decode("utf-8")

        if set(files) == {"R.xyz", "P.xyz"}:
            yield reaction, files["R.xyz"], files["P.xyz"]
            completed.add(reaction)
            del pending[reaction]

    if pending:
        examples = ", ".join(list(pending)[:5])
        raise ValueError(
            f"Archive reactions missing R.xyz or P.xyz ({len(pending)}): {examples}"
        )


def run_inference(args):
    import math

    import torch
    import yaml
    from ase import Atoms
    from ase.io import write
    from easydict import EasyDict

    sys.path.insert(0, str(ROOT))
    from Data.rgd1_preprocess import parse_xyz
    from Model.model import DistFlowMatchingNetwork, ODEWrapper2
    from Scripts.infer_ts1x_xyz import make_output_filename
    from Utils import (
        Kabsch_alignment,
        generate_fully_connected,
        pairwise_dist_to_coord,
        seed_all,
    )

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
            "reconstruction_loss",
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

                    pred_pos, reconstruction_loss = pairwise_dist_to_coord(
                        x, reactant_pos, product_pos, dist_pred
                    )
                    pred_pos = pred_pos.detach()
                    reconstruction_loss = float(
                        reconstruction_loss.detach().cpu().item()
                    )
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
                            "reconstruction_loss": reconstruction_loss,
                        }
                    )
                    manifest_file.flush()
                    processed += 1
                except Exception as exc:
                    raise RuntimeError(
                        f"Inference failed for {dataset} reaction {index}: {reaction}"
                    ) from exc

                print(f"[{dataset} {processed}] {reaction} -> {filename}", flush=True)

        print(f"Generated {processed} predicted TS structures in {output_dir}")
        print(f"Manifest: {manifest_path}")


def main():
    run_inference(build_parser().parse_args())


if __name__ == "__main__":
    main()
