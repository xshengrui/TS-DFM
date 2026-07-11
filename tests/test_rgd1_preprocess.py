import csv
import importlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch


def _subject():
    assert importlib.util.find_spec("Data.rgd1_preprocess") is not None, (
        "Data.rgd1_preprocess has not been implemented"
    )
    return importlib.import_module("Data.rgd1_preprocess")


def _xyz(reaction_id, label, symbols=("C", "H"), offset=0.0):
    lines = [str(len(symbols)), f"{reaction_id} {label}"]
    for index, symbol in enumerate(symbols):
        value = index + offset
        lines.append(f"{symbol} {value:.3f} {value + 1:.3f} {value + 2:.3f}")
    return "\n".join(lines) + "\n"


def _energy(values=(-1.0, -2.0, -0.5)):
    return (
        "R_E P_E TS_E DE_F DE_B\n"
        f"{values[0]} {values[1]} {values[2]} 0.5 1.5\n"
    )


def _write_source_zip(path, csv_reactions, archive_reactions, missing=None):
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(
        ["reaction", "reactant", "product", "DE_F", "DE_B", "DG_F", "DG_B", "DH", "dataset"]
    )
    for reaction_id in csv_reactions:
        writer.writerow([reaction_id, "R", "P", 1, 2, 3, 4, 5, "test"])

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as archive:
        for reaction_id in archive_reactions:
            files = {
                "energy": _energy(),
                "RG.xyz": _xyz(reaction_id, "RG", offset=0.0),
                "PG.xyz": _xyz(reaction_id, "PG", offset=0.1),
                "TSG.xyz": _xyz(reaction_id, "TSG", offset=0.2),
            }
            for filename, content in files.items():
                if missing == (reaction_id, filename):
                    continue
                payload = content.encode("utf-8")
                info = tarfile.TarInfo(f"RGD1_CHNO/{reaction_id}/{filename}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("RGD1CHNO_AMsmiles_with_split.csv", csv_buffer.getvalue())
        bundle.writestr("RGD1_CHNO_xyz.tar.gz", tar_buffer.getvalue())


class RGD1PreprocessTests(unittest.TestCase):
    def test_parse_xyz_and_energy_use_model_atomic_numbers(self):
        module = _subject()

        atomic_numbers, positions = module.parse_xyz(
            _xyz("rxn-1", "RG", symbols=("H", "C", "N", "O"))
        )
        energies = module.parse_energy(_energy((-10.0, -11.0, -9.5)))

        self.assertEqual(atomic_numbers.tolist(), [1, 6, 7, 8])
        self.assertEqual(positions.shape, (4, 3))
        self.assertEqual(positions.dtype, torch.float32)
        torch.testing.assert_close(
            energies, torch.tensor([-10.0, -11.0, -9.5], dtype=torch.float64)
        )

    def test_preprocess_uses_csv_as_whitelist_and_writes_valid_shards(self):
        module = _subject()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            source_zip = root / "RGD1.zip"
            output_dir = root / "cache"
            _write_source_zip(
                source_zip,
                csv_reactions=["keep-1", "keep-2"],
                archive_reactions=["keep-1", "extra", "keep-2"],
            )

            manifest = module.preprocess_rgd1_zip(
                source_zip,
                output_dir,
                shard_size=1,
                expected_count=2,
            )

            self.assertEqual(manifest["record_count"], 2)
            self.assertTrue(manifest["reaction_ids_sha256"])
            self.assertEqual([item["count"] for item in manifest["shards"]], [1, 1])
            saved_manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(saved_manifest["record_count"], 2)
            records = []
            for shard in manifest["shards"]:
                records.extend(torch.load(output_dir / shard["file"], weights_only=True))
            self.assertEqual(
                [record["reaction_id"] for record in records], ["keep-1", "keep-2"]
            )

    def test_preprocess_rejects_missing_required_geometry(self):
        module = _subject()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            source_zip = root / "RGD1.zip"
            _write_source_zip(
                source_zip,
                csv_reactions=["broken"],
                archive_reactions=["broken"],
                missing=("broken", "TSG.xyz"),
            )

            with self.assertRaisesRegex(ValueError, r"broken.*TSG\.xyz"):
                module.preprocess_rgd1_zip(
                    source_zip,
                    root / "cache",
                    shard_size=1,
                    expected_count=1,
                )


if __name__ == "__main__":
    unittest.main()
