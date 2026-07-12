import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _factory_module():
    assert importlib.util.find_spec("Data.dataloader_factory") is not None, (
        "Data.dataloader_factory has not been implemented"
    )
    return importlib.import_module("Data.dataloader_factory")


class TrainingDataEntrypointTests(unittest.TestCase):
    def test_mix_server_script_uses_mixed_config(self):
        script = ROOT / "run_scripts" / "run_mix_paper.sh"
        self.assertTrue(script.is_file(), "Mixed server training script is missing")

        content = script.read_text(encoding="utf-8")
        self.assertIn("conda activate reactot", content)
        self.assertIn(
            "cd /inspire/qb-ilm/project/chemicalreaction/czxs25220150/projects/TS-DFM",
            content,
        )
        self.assertIn("--config_file Configs/Dynamics_mixed.yml", content)
        self.assertIn("--log_prefix tsdfm_mix", content)
        self.assertIn("--device cuda", content)

    def test_preprocess_cli_exposes_source_output_and_shard_options(self):
        script = ROOT / "Scripts" / "preprocess_rgd1_full.py"
        self.assertTrue(script.is_file(), "RGD1 preprocessing CLI is missing")

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source-zip", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--shard-size", result.stdout)

    def test_mixed_config_declares_both_data_sources(self):
        config_path = ROOT / "Configs" / "Dynamics_mixed.yml"
        self.assertTrue(config_path.is_file(), "Mixed dynamics config is missing")

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertIn("transition1x_path", config["data"])
        self.assertEqual(config["data"]["rgd1_cache_path"], "Data/processed/rgd1_full")
        self.assertEqual(config["data"]["batch_size"], 32)

    def test_factory_preserves_transition1x_only_route(self):
        module = _factory_module()
        config = SimpleNamespace(
            train=SimpleNamespace(seed=5),
            data=SimpleNamespace(path="transition1x.h5", batch_size=8),
        )
        sentinel = object()

        with patch.object(module, "generate_transition1x_dataloaders", return_value=sentinel) as loader:
            result = module.build_dynamics_dataloaders(config)

        self.assertIs(result, sentinel)
        loader.assert_called_once_with("transition1x.h5", 8)

    def test_factory_selects_mixed_route_when_cache_is_configured(self):
        module = _factory_module()
        config = SimpleNamespace(
            train=SimpleNamespace(seed=2025),
            data=SimpleNamespace(
                transition1x_path="transition1x.h5",
                rgd1_cache_path="Data/processed/rgd1_full",
                batch_size=32,
                num_workers=2,
            ),
        )
        sentinel = object()

        with patch.object(module, "generate_mixed_dataloaders", return_value=sentinel) as loader:
            result = module.build_dynamics_dataloaders(config)

        self.assertIs(result, sentinel)
        loader.assert_called_once_with(
            transition1x_hdf5="transition1x.h5",
            rgd1_cache_dir="Data/processed/rgd1_full",
            batch_size=32,
            seed=2025,
            num_workers=2,
        )


if __name__ == "__main__":
    unittest.main()
