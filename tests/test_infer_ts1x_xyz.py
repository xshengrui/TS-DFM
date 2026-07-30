import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Scripts" / "infer_ts1x_xyz.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("infer_ts1x_xyz", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Transition1xXyzInferenceTests(unittest.TestCase):
    def test_cli_exposes_required_inference_paths(self):
        self.assertTrue(SCRIPT.is_file(), "Transition1x XYZ inference CLI is missing")

        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--config",
            "--checkpoint",
            "--hdf5",
            "--output",
            "--lbfgs-max-iter",
            "--lbfgs-lr",
        ):
            self.assertIn(option, result.stdout)

    def test_output_filename_is_stable_and_filesystem_safe(self):
        module = _load_module()

        filename = module.make_output_filename(3, "C2H2N2O2", "rxn/12")

        self.assertEqual(filename, "0003_C2H2N2O2_rxn_12_pred.xyz")


if __name__ == "__main__":
    unittest.main()
