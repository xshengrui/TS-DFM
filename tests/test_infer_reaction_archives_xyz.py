import importlib.util
import io
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Scripts" / "infer_reaction_archives_xyz.py"
MIX_RUN_SCRIPT = ROOT / "run_scripts" / "infer_mix_gdb10_17.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("infer_reaction_archives_xyz", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive_with(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files:
            payload = content.encode("utf-8")
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    buffer.seek(0)
    return tarfile.open(fileobj=buffer, mode="r:gz")


class ReactionArchiveXyzInferenceTests(unittest.TestCase):
    def test_mixed_gdb_runner_uses_mixed_model_and_both_archives(self):
        self.assertTrue(MIX_RUN_SCRIPT.is_file(), "Mixed GDB runner is missing")

        content = MIX_RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Configs/Dynamics_mixed.yml", content)
        self.assertIn("logs/dynamics_flow_mixed", content)
        self.assertIn("Data/GDB-10-rxn_raw.tar.gz", content)
        self.assertIn("Data/GDB-17-rxn_raw.tar.gz", content)
        self.assertIn("python -m Scripts.infer_reaction_archives_xyz", content)
        self.assertIn('CHECKPOINT="${CHECKPOINT:-', content)
        self.assertIn('RUN_DIR="${RUN_DIR:-', content)
        self.assertIn("--skip-existing", content)

    def test_cli_accepts_multiple_archives(self):
        self.assertTrue(SCRIPT.is_file(), "Reaction archive inference CLI is missing")

        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--config", "--checkpoint", "--archives", "--output"):
            self.assertIn(option, result.stdout)

    def test_archive_label_removes_both_suffixes(self):
        module = _load_module()

        self.assertEqual(
            module.archive_label(Path("Data/GDB-10-rxn_raw.tar.gz")),
            "GDB-10-rxn_raw",
        )

    def test_reaction_reader_uses_only_r_and_p_files(self):
        module = _load_module()
        xyz = "1\ncomment\nH 0 0 0\n"
        archive = _archive_with(
            [
                ("raw/reaction2/R.xyz", xyz),
                ("raw/reaction2/P.xyz", xyz.replace("0 0 0", "0 0 1")),
                ("raw/reaction2/TS.xyz", "not read"),
                ("raw/reaction2/IRC.xyz", "not read"),
            ]
        )

        with archive:
            records = list(module.iter_reaction_texts(archive))

        self.assertEqual(len(records), 1)
        reaction, reactant, product = records[0]
        self.assertEqual(reaction, "reaction2")
        self.assertIn("H 0 0 0", reactant)
        self.assertIn("H 0 0 1", product)

    def test_reaction_reader_rejects_missing_endpoint(self):
        module = _load_module()
        archive = _archive_with(
            [("raw/reaction3/R.xyz", "1\ncomment\nH 0 0 0\n")]
        )

        with archive, self.assertRaisesRegex(ValueError, "missing R.xyz or P.xyz"):
            list(module.iter_reaction_texts(archive))


if __name__ == "__main__":
    unittest.main()
