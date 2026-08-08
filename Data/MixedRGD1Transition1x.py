import json
import pickle
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from Utils.alignment import Kabsch_alignment


RGD1_EXPECTED_COUNT = 176898
TRANSITION1X_SPLIT_COUNTS = {"train": 9561, "valid": 225, "test": 287}
CACHE_FORMAT_VERSION = 1


def partition_for_worker(items, worker_id, num_workers):
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id is outside the worker range")
    return list(items)[worker_id::num_workers]


def load_cache_manifest(cache_dir, expected_count=RGD1_EXPECTED_COUNT):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"RGD1 cache manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read RGD1 cache manifest: {manifest_path}") from exc

    if manifest.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported RGD1 cache format version: {manifest.get('format_version')!r}"
        )
    if manifest.get("record_count") != expected_count:
        raise ValueError(
            f"RGD1 cache contains {manifest.get('record_count')} records; "
            f"expected {expected_count}"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("RGD1 cache manifest does not contain shards")

    shard_total = 0
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("count"), int):
            raise ValueError("RGD1 cache manifest contains an invalid shard entry")
        shard_path = cache_dir / str(shard.get("file", ""))
        if not shard_path.is_file():
            raise ValueError(f"RGD1 cache is missing shard: {shard_path}")
        shard_total += shard["count"]
    if shard_total != expected_count:
        raise ValueError(
            f"RGD1 shard counts sum to {shard_total}; expected {expected_count}"
        )
    return manifest


def validate_transition1x_split_manifests(split_dir):
    split_dir = Path(split_dir)
    counts = {}
    memberships = {}
    for split, expected_count in TRANSITION1X_SPLIT_COUNTS.items():
        path = split_dir / f"reactions_{split}.pickle"
        if not path.is_file():
            raise ValueError(f"Transition1x split manifest is missing: {path}")
        with path.open("rb") as source:
            reactions = pickle.load(source)
        if not isinstance(reactions, list):
            raise ValueError(f"Transition1x {split} manifest must contain a list")
        normalized = [tuple(item) for item in reactions]
        if len(normalized) != expected_count:
            raise ValueError(
                f"Transition1x {split} contains {len(normalized)} reactions; "
                f"expected {expected_count}"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Transition1x {split} contains duplicate reactions")
        counts[split] = len(normalized)
        memberships[split] = set(normalized)

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = memberships[left] & memberships[right]
        if overlap:
            raise ValueError(
                f"Transition1x {left}/{right} manifests overlap by {len(overlap)} reactions"
            )
    return counts


def mixed_split_lengths(rgd1_count, transition1x_counts):
    return {
        "train": int(rgd1_count) + int(transition1x_counts["train"]),
        "val": int(transition1x_counts["valid"]),
        "test": int(transition1x_counts["test"]),
    }


def proportional_mix(left, right, left_count, right_count, seed):
    left_remaining = int(left_count)
    right_remaining = int(right_count)
    if left_remaining < 0 or right_remaining < 0:
        raise ValueError("source counts cannot be negative")
    rng = random.Random(seed)
    while left_remaining or right_remaining:
        choose_left = rng.randrange(left_remaining + right_remaining) < left_remaining
        if choose_left:
            try:
                item = next(left)
            except StopIteration as exc:
                raise RuntimeError("left source ended before its declared count") from exc
            left_remaining -= 1
        else:
            try:
                item = next(right)
            except StopIteration as exc:
                raise RuntimeError("right source ended before its declared count") from exc
            right_remaining -= 1
        yield item

    try:
        next(left)
    except StopIteration:
        pass
    else:
        raise RuntimeError("left source yielded more than its declared count")
    try:
        next(right)
    except StopIteration:
        pass
    else:
        raise RuntimeError("right source yielded more than its declared count")


def rgd1_record_to_data(record):
    x = torch.as_tensor(record["atomic_numbers"], dtype=torch.long).clone()
    reactant_pos = torch.as_tensor(
        record["reactant_pos"], dtype=torch.float32
    ).clone()
    product_pos = torch.as_tensor(record["product_pos"], dtype=torch.float32).clone()
    transition_state_pos = torch.as_tensor(
        record["transition_state_pos"], dtype=torch.float32
    ).clone()
    if (
        reactant_pos.shape != product_pos.shape
        or reactant_pos.shape != transition_state_pos.shape
        or reactant_pos.shape != (x.numel(), 3)
    ):
        raise ValueError(f"{record.get('reaction_id')} has inconsistent tensor shapes")
    batch = torch.zeros_like(x)
    product_pos = Kabsch_alignment(product_pos, reactant_pos, batch)
    transition_state_pos = Kabsch_alignment(
        transition_state_pos, reactant_pos, batch
    )
    return Data(
        x=x,
        reactant_pos=reactant_pos,
        product_pos=product_pos,
        transition_state_pos=transition_state_pos,
        energies=torch.as_tensor(record["energies"], dtype=torch.float32).clone(),
        reaction_id=record["reaction_id"],
        source="rgd1",
    )


class RGD1ShardDataset(IterableDataset):
    def __init__(self, cache_dir, seed, expected_count=RGD1_EXPECTED_COUNT):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.seed = int(seed)
        self.manifest = load_cache_manifest(self.cache_dir, expected_count)
        self.record_count = expected_count
        self._iteration = 0

    def __len__(self):
        return self.record_count

    def worker_record_count(self, worker_id, num_workers):
        shards = partition_for_worker(
            self.manifest["shards"], worker_id, num_workers
        )
        return sum(shard["count"] for shard in shards)

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
            iteration_seed = self.seed + self._iteration
            self._iteration += 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers
            iteration_seed = worker.seed

        rng = random.Random(iteration_seed)
        shards = partition_for_worker(
            self.manifest["shards"], worker_id, num_workers
        )
        rng.shuffle(shards)
        for shard in shards:
            records = torch.load(
                self.cache_dir / shard["file"], map_location="cpu", weights_only=True
            )
            if len(records) != shard["count"]:
                raise ValueError(
                    f"RGD1 shard {shard['file']} contains {len(records)} records; "
                    f"expected {shard['count']}"
                )
            rng.shuffle(records)
            for record in records:
                yield rgd1_record_to_data(record)


class Transition1xSplitDataset(IterableDataset):
    def __init__(self, hdf5_file, split, split_dir, seed):
        super().__init__()
        if split not in TRANSITION1X_SPLIT_COUNTS:
            raise ValueError(f"Unsupported Transition1x split: {split}")
        self.hdf5_file = hdf5_file
        self.split = split
        self.seed = int(seed)
        self._iteration = 0
        manifest_path = Path(split_dir) / f"reactions_{split}.pickle"
        with manifest_path.open("rb") as source:
            self.datalist = pickle.load(source)
        expected = TRANSITION1X_SPLIT_COUNTS[split]
        if len(self.datalist) != expected:
            raise ValueError(
                f"Transition1x {split} contains {len(self.datalist)} reactions; "
                f"expected {expected}"
            )

    def __len__(self):
        return len(self.datalist)

    def worker_record_count(self, worker_id, num_workers):
        return len(partition_for_worker(self.datalist, worker_id, num_workers))

    def __iter__(self):
        import h5py

        from Data.Transition1x import get_dynamics_data

        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
            iteration_seed = self.seed + self._iteration
            self._iteration += 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers
            iteration_seed = worker.seed
        reactions = partition_for_worker(self.datalist, worker_id, num_workers)
        random.Random(iteration_seed).shuffle(reactions)
        with h5py.File(self.hdf5_file, "r") as source:
            data = source["data"]
            for formula, reaction_id in reactions:
                item = get_dynamics_data(formula, reaction_id, data)
                item.reaction_id = reaction_id
                item.source = "transition1x"
                yield item


class MixedTrainingDataset(IterableDataset):
    def __init__(self, rgd1_dataset, transition1x_dataset, seed):
        super().__init__()
        self.rgd1_dataset = rgd1_dataset
        self.transition1x_dataset = transition1x_dataset
        self.seed = int(seed)
        self._iteration = 0

    def __len__(self):
        return len(self.rgd1_dataset) + len(self.transition1x_dataset)

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
            iteration_seed = self.seed + self._iteration
            self._iteration += 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers
            iteration_seed = worker.seed + self.seed
        rgd1_count = self.rgd1_dataset.worker_record_count(worker_id, num_workers)
        transition_count = self.transition1x_dataset.worker_record_count(
            worker_id, num_workers
        )
        yield from proportional_mix(
            iter(self.rgd1_dataset),
            iter(self.transition1x_dataset),
            rgd1_count,
            transition_count,
            seed=iteration_seed,
        )


def generate_mixed_dataloader_dynamics(
    transition1x_hdf5,
    rgd1_cache_dir,
    batch_size,
    seed,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None,
    rgd1_expected_count=RGD1_EXPECTED_COUNT,
    transition1x_split_dir="Data",
):
    validate_transition1x_split_manifests(transition1x_split_dir)
    rgd1 = RGD1ShardDataset(
        rgd1_cache_dir, seed=seed, expected_count=rgd1_expected_count
    )
    transition_train = Transition1xSplitDataset(
        transition1x_hdf5,
        "train",
        transition1x_split_dir,
        seed=seed + 1,
    )
    train = MixedTrainingDataset(rgd1, transition_train, seed=seed + 2)
    val = Transition1xSplitDataset(
        transition1x_hdf5,
        "valid",
        transition1x_split_dir,
        seed=seed + 3,
    )
    test = Transition1xSplitDataset(
        transition1x_hdf5,
        "test",
        transition1x_split_dir,
        seed=seed + 4,
    )
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_options["prefetch_factor"] = prefetch_factor
    return {
        "train": DataLoader(train, **loader_options),
        "val": DataLoader(val, **loader_options),
        "test": DataLoader(test, **loader_options),
    }
