import csv
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import torch


FORMAT_VERSION = 1
DEFAULT_CSV_NAME = "RGD1CHNO_AMsmiles_with_split.csv"
DEFAULT_ARCHIVE_NAME = "RGD1_CHNO_xyz.tar.gz"
REQUIRED_FILES = ("energy", "RG.xyz", "PG.xyz", "TSG.xyz")
ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8}


def parse_xyz(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("XYZ content is missing its header")
    try:
        atom_count = int(lines[0])
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count: {lines[0]!r}") from exc
    atom_lines = lines[2:]
    if len(atom_lines) != atom_count:
        raise ValueError(
            f"XYZ declares {atom_count} atoms but contains {len(atom_lines)} atom rows"
        )

    atomic_numbers = []
    positions = []
    for line_number, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"Malformed XYZ row {line_number}: {line!r}")
        symbol = fields[0]
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"Unsupported RGD1 element: {symbol!r}")
        try:
            coordinate = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Malformed XYZ coordinates on row {line_number}") from exc
        atomic_numbers.append(ATOMIC_NUMBERS[symbol])
        positions.append(coordinate)

    return (
        torch.tensor(atomic_numbers, dtype=torch.long),
        torch.tensor(positions, dtype=torch.float32),
    )


def parse_energy(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Energy content is missing its header or values")
    header = lines[0].split()
    values = lines[1].split()
    if len(header) != len(values):
        raise ValueError("Energy header/value column counts do not match")
    columns = dict(zip(header, values))
    missing = [name for name in ("R_E", "P_E", "TS_E") if name not in columns]
    if missing:
        raise ValueError(f"Energy content is missing columns: {', '.join(missing)}")
    try:
        return torch.tensor(
            [float(columns["R_E"]), float(columns["P_E"]), float(columns["TS_E"])],
            dtype=torch.float64,
        )
    except ValueError as exc:
        raise ValueError("Energy content has malformed numeric values") from exc


def _read_reaction_ids(bundle, csv_name):
    try:
        csv_file = bundle.open(csv_name)
    except KeyError as exc:
        raise ValueError(f"RGD1 ZIP is missing {csv_name}") from exc
    with csv_file, io.TextIOWrapper(csv_file, encoding="utf-8-sig", newline="") as text:
        reader = csv.DictReader(text)
        if not reader.fieldnames or "reaction" not in reader.fieldnames:
            raise ValueError(f"{csv_name} does not contain a reaction column")
        reaction_ids = [row["reaction"].strip() for row in reader]
    if any(not reaction_id for reaction_id in reaction_ids):
        raise ValueError(f"{csv_name} contains an empty reaction ID")
    if len(set(reaction_ids)) != len(reaction_ids):
        raise ValueError(f"{csv_name} contains duplicate reaction IDs")
    return reaction_ids


def _decode_member(archive, member):
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Unable to read archive member {member.name}")
    return source.read().decode("utf-8")


def _build_record(reaction_id, files):
    r_atoms, reactant_pos = parse_xyz(files["RG.xyz"])
    p_atoms, product_pos = parse_xyz(files["PG.xyz"])
    ts_atoms, transition_state_pos = parse_xyz(files["TSG.xyz"])
    if not torch.equal(r_atoms, p_atoms) or not torch.equal(r_atoms, ts_atoms):
        raise ValueError(f"{reaction_id} has inconsistent atoms across geometries")
    return {
        "reaction_id": reaction_id,
        "atomic_numbers": r_atoms,
        "reactant_pos": reactant_pos,
        "product_pos": product_pos,
        "transition_state_pos": transition_state_pos,
        "energies": parse_energy(files["energy"]),
    }


def _write_shard(staging_dir, shard_index, records):
    filename = f"shard-{shard_index:05d}.pt"
    temporary_path = staging_dir / f"{filename}.tmp"
    final_path = staging_dir / filename
    torch.save(records, temporary_path)
    temporary_path.replace(final_path)
    return {"file": filename, "count": len(records)}


def preprocess_rgd1_zip(
    source_zip,
    output_dir,
    shard_size=2048,
    expected_count=176898,
    csv_name=DEFAULT_CSV_NAME,
    archive_name=DEFAULT_ARCHIVE_NAME,
):
    source_zip = Path(source_zip).resolve()
    output_dir = Path(output_dir).resolve()
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if output_dir.exists():
        raise FileExistsError(f"RGD1 cache output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )

    try:
        with zipfile.ZipFile(source_zip) as bundle:
            reaction_ids = _read_reaction_ids(bundle, csv_name)
            if len(reaction_ids) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} RGD1 CSV reactions, found {len(reaction_ids)}"
                )
            whitelist = set(reaction_ids)
            try:
                nested_archive = bundle.open(archive_name)
            except KeyError as exc:
                raise ValueError(f"RGD1 ZIP is missing {archive_name}") from exc

            pending = {}
            seen = set()
            shard_records = []
            shards = []
            with nested_archive, tarfile.open(fileobj=nested_archive, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    path = PurePosixPath(member.name)
                    if len(path.parts) < 3:
                        continue
                    reaction_id = path.parts[-2]
                    filename = path.parts[-1]
                    if reaction_id not in whitelist or filename not in REQUIRED_FILES:
                        continue
                    files = pending.setdefault(reaction_id, {})
                    if filename in files:
                        raise ValueError(f"{reaction_id} contains duplicate {filename}")
                    files[filename] = _decode_member(archive, member)
                    if all(required in files for required in REQUIRED_FILES):
                        shard_records.append(_build_record(reaction_id, files))
                        seen.add(reaction_id)
                        del pending[reaction_id]
                        if len(shard_records) == shard_size:
                            shards.append(
                                _write_shard(staging_dir, len(shards), shard_records)
                            )
                            shard_records = []

            if shard_records:
                shards.append(_write_shard(staging_dir, len(shards), shard_records))

        missing_ids = [reaction_id for reaction_id in reaction_ids if reaction_id not in seen]
        if missing_ids:
            reaction_id = missing_ids[0]
            available = pending.get(reaction_id, {})
            missing_files = [name for name in REQUIRED_FILES if name not in available]
            raise ValueError(
                f"{reaction_id} is missing required files: {', '.join(missing_files)}"
            )

        record_count = sum(shard["count"] for shard in shards)
        if record_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} processed RGD1 reactions, found {record_count}"
            )

        source_stat = source_zip.stat()
        reaction_hash = hashlib.sha256(
            "\n".join(reaction_ids).encode("utf-8")
        ).hexdigest()
        manifest = {
            "format_version": FORMAT_VERSION,
            "source_zip": {
                "name": source_zip.name,
                "size": source_stat.st_size,
                "mtime_ns": source_stat.st_mtime_ns,
            },
            "csv_name": csv_name,
            "archive_name": archive_name,
            "expected_count": expected_count,
            "record_count": record_count,
            "reaction_ids_sha256": reaction_hash,
            "shard_size": shard_size,
            "shards": shards,
        }
        manifest_tmp = staging_dir / "manifest.json.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        manifest_tmp.replace(staging_dir / "manifest.json")
        staging_dir.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
