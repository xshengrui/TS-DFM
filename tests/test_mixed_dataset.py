import importlib
import json
import tempfile
import unittest
from pathlib import Path

import torch


def _subject():
    assert importlib.util.find_spec("Data.MixedRGD1Transition1x") is not None, (
        "Data.MixedRGD1Transition1x has not been implemented"
    )
    return importlib.import_module("Data.MixedRGD1Transition1x")


def _record(reaction_id, offset):
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    return {
        "reaction_id": reaction_id,
        "atomic_numbers": torch.tensor([6, 1, 8], dtype=torch.long),
        "reactant_pos": base + offset,
        "product_pos": base + offset + 0.1,
        "transition_state_pos": base + offset + 0.2,
        "energies": torch.tensor([-10.0, -11.0, -9.5], dtype=torch.float64),
    }


def _write_cache(root, count=4):
    root.mkdir()
    shards = []
    for shard_index, start in enumerate(range(0, count, 2)):
        records = [
            _record(f"rgd-{index}", float(index))
            for index in range(start, min(start + 2, count))
        ]
        filename = f"shard-{shard_index:05d}.pt"
        torch.save(records, root / filename)
        shards.append({"file": filename, "count": len(records)})
    manifest = {
        "format_version": 1,
        "expected_count": count,
        "record_count": count,
        "shards": shards,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class MixedDatasetTests(unittest.TestCase):
    def test_manifest_validation_rejects_missing_shard(self):
        module = _subject()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            cache = Path(temp_dir) / "cache"
            _write_cache(cache)
            (cache / "shard-00001.pt").unlink()

            with self.assertRaisesRegex(ValueError, "missing shard"):
                module.load_cache_manifest(cache, expected_count=4)

    def test_rgd1_dataset_converts_every_record_once(self):
        module = _subject()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            cache = Path(temp_dir) / "cache"
            _write_cache(cache)

            dataset = module.RGD1ShardDataset(cache, seed=17, expected_count=4)
            records = list(dataset)

            self.assertEqual(len(dataset), 4)
            self.assertEqual({data.reaction_id for data in records}, {"rgd-0", "rgd-1", "rgd-2", "rgd-3"})
            self.assertTrue(all(data.source == "rgd1" for data in records))
            self.assertTrue(all(data.x.dtype == torch.long for data in records))
            self.assertTrue(all(data.reactant_pos.shape == (3, 3) for data in records))
            self.assertTrue(all(data.energies.dtype == torch.float32 for data in records))

    def test_rgd1_dataset_order_is_deterministic_for_fixed_seed(self):
        module = _subject()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            cache = Path(temp_dir) / "cache"
            _write_cache(cache)

            first = module.RGD1ShardDataset(cache, seed=99, expected_count=4)
            second = module.RGD1ShardDataset(cache, seed=99, expected_count=4)

            self.assertEqual(
                [data.reaction_id for data in first],
                [data.reaction_id for data in second],
            )

    def test_worker_partition_is_disjoint_and_complete(self):
        module = _subject()
        items = list(range(11))

        partitions = [module.partition_for_worker(items, worker, 3) for worker in range(3)]

        flattened = [item for partition in partitions for item in partition]
        self.assertEqual(sorted(flattened), items)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_transition1x_manifests_have_required_split_counts(self):
        module = _subject()
        self.assertTrue(hasattr(module, "validate_transition1x_split_manifests"))

        counts = module.validate_transition1x_split_manifests(Path("Data"))

        self.assertEqual(counts, {"train": 9561, "valid": 225, "test": 287})
        self.assertEqual(
            module.mixed_split_lengths(176898, counts),
            {"train": 186459, "val": 225, "test": 287},
        )

    def test_proportional_mix_is_deterministic_and_exactly_once(self):
        module = _subject()
        self.assertTrue(hasattr(module, "proportional_mix"))

        first = list(
            module.proportional_mix(
                iter(["r0", "r1", "r2", "r3"]),
                iter(["t0", "t1"]),
                4,
                2,
                seed=23,
            )
        )
        second = list(
            module.proportional_mix(
                iter(["r0", "r1", "r2", "r3"]),
                iter(["t0", "t1"]),
                4,
                2,
                seed=23,
            )
        )

        self.assertEqual(first, second)
        self.assertEqual(set(first), {"r0", "r1", "r2", "r3", "t0", "t1"})
        self.assertEqual(len(first), 6)

    def test_loader_factory_reports_required_memberships(self):
        module = _subject()
        self.assertTrue(hasattr(module, "generate_mixed_dataloader_dynamics"))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            cache = Path(temp_dir) / "cache"
            _write_cache(cache, count=4)

            loaders = module.generate_mixed_dataloader_dynamics(
                transition1x_hdf5="not-opened-until-iteration.h5",
                rgd1_cache_dir=cache,
                batch_size=32,
                seed=2025,
                rgd1_expected_count=4,
                transition1x_split_dir=Path("Data"),
            )

            self.assertEqual(len(loaders["train"].dataset), 4 + 9561)
            self.assertEqual(len(loaders["val"].dataset), 225)
            self.assertEqual(len(loaders["test"].dataset), 287)


if __name__ == "__main__":
    unittest.main()
