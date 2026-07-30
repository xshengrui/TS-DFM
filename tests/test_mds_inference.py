import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Utils" / "distance_geometry.py"
T1X_SCRIPT = ROOT / "Scripts" / "infer_ts1x_xyz_mds.py"
ARCHIVE_SCRIPT = ROOT / "Scripts" / "infer_reaction_archives_xyz_mds.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("distance_geometry", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MdsReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        self.distances = torch.cdist(self.coords, self.coords)

    def test_classical_mds_recovers_pairwise_distances_in_three_dimensions(self):
        module = _load_module()

        reconstructed = module.classical_mds(self.distances, dimensions=3)

        torch.testing.assert_close(
            torch.cdist(reconstructed, reconstructed),
            self.distances,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_mds_lbfgs_reconstruction_does_not_use_random_noise(self):
        module = _load_module()
        src, dst = torch.where(~torch.eye(4, dtype=torch.bool))
        edge_distances = self.distances[src, dst]

        with patch("torch.randn_like", side_effect=AssertionError("random noise used")):
            reconstructed, loss = module.pairwise_dist_to_coord_mds(
                self.coords,
                self.coords,
                edge_distances,
                max_iter=20,
                lr=0.1,
            )

        self.assertLess(float(loss), 1e-8)
        torch.testing.assert_close(
            torch.cdist(reconstructed, reconstructed),
            self.distances,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_paper_reconstruction_uses_linear_interpolation_without_random_noise(self):
        module = _load_module()
        src, dst = torch.where(~torch.eye(4, dtype=torch.bool))
        edge_distances = self.distances[src, dst]
        reactant = self.coords + torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64)
        product = self.coords + torch.tensor([-0.2, 0.1, -0.05], dtype=torch.float64)

        with patch("torch.randn_like", side_effect=AssertionError("random noise used")):
            reconstructed, loss = module.pairwise_dist_to_coord_linear_interp(
                reactant,
                product,
                edge_distances,
                max_iter=20,
                lr=0.1,
            )

        self.assertLess(float(loss), 1e-8)
        torch.testing.assert_close(
            torch.cdist(reconstructed, reconstructed),
            self.distances,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_additional_mds_clis_preserve_required_dataset_inputs(self):
        for script, dataset_option in (
            (T1X_SCRIPT, "--hdf5"),
            (ARCHIVE_SCRIPT, "--archives"),
        ):
            self.assertTrue(script.is_file(), f"MDS inference CLI is missing: {script.name}")
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for option in (
                "--config",
                "--checkpoint",
                dataset_option,
                "--output",
                "--lbfgs-max-iter",
                "--lbfgs-lr",
            ):
                self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
