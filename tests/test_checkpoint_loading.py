import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Utils" / "checkpoint_loading.py"


def _load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError("Utils/checkpoint_loading.py has not been implemented")
    spec = importlib.util.spec_from_file_location("checkpoint_loading", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTorch:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.calls = []

    def load(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return self.checkpoint


class _FakeModel:
    def __init__(self):
        self.calls = []

    def load_state_dict(self, state_dict, **kwargs):
        self.calls.append((state_dict, kwargs))


class CheckpointLoadingTests(unittest.TestCase):
    def test_load_model_only_uses_cpu_and_does_not_restore_training_state(self):
        load_model_only = _load_module().load_model_only

        checkpoint = {
            "model": {"layer.weight": "weights"},
            "optimizer": {"lr": 1e-6},
            "scheduler": {"best": 0.1},
            "epoch": 705,
        }
        fake_torch = _FakeTorch(checkpoint)
        model = _FakeModel()

        metadata = load_model_only(
            model,
            "mix/checkpoint_best.pth",
            torch_module=fake_torch,
        )

        self.assertEqual(
            fake_torch.calls,
            [
                (
                    "mix/checkpoint_best.pth",
                    {"map_location": "cpu", "weights_only": True},
                )
            ],
        )
        self.assertEqual(
            model.calls,
            [({"layer.weight": "weights"}, {"strict": True})],
        )
        self.assertEqual(
            metadata,
            {"checkpoint_path": "mix/checkpoint_best.pth", "source_epoch": 705},
        )

    def test_load_model_only_rejects_checkpoint_without_model_state(self):
        load_model_only = _load_module().load_model_only

        fake_torch = _FakeTorch({"optimizer": {}})

        with self.assertRaisesRegex(ValueError, "does not contain a 'model' state_dict"):
            load_model_only(
                _FakeModel(),
                "bad.pth",
                torch_module=fake_torch,
            )


if __name__ == "__main__":
    unittest.main()
