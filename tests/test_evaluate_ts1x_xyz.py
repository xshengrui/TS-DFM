import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Scripts" / "evaluate_ts1x_xyz.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_ts1x_xyz", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Transition1xXyzEvaluationTests(unittest.TestCase):
    def test_cli_exposes_required_metric_inputs(self):
        self.assertTrue(SCRIPT.is_file(), "Transition1x XYZ evaluation CLI is missing")

        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--hdf5", "--pred-dir", "--manifest", "--output-csv"):
            self.assertIn(option, result.stdout)

    def test_rmsd_is_kabsch_aligned_and_dmae_uses_pair_distances(self):
        module = _load_module()
        target = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=float,
        )
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        predicted = target @ rotation + np.array([3.0, -2.0, 0.5])

        self.assertLess(module.calc_rmsd(predicted, target), 1e-12)
        self.assertLess(module.calc_dmae(predicted, target), 1e-12)


if __name__ == "__main__":
    unittest.main()
