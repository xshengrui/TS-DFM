# RGD1 + Transition1x Mixed Training Data Design

## Goal

Add a training-data path with these fixed memberships:

- `train`: all 176,898 reactions listed in `RGD1CHNO_AMsmiles_with_split.csv`, plus the existing 9,561 Transition1x training reactions (186,459 total).
- `val`: the existing 225 Transition1x validation reactions only.
- `test`: the existing 287 Transition1x test reactions only.

The `dataset` labels already present in the RGD1 CSV are ignored. The CSV reaction IDs are the canonical whitelist because the nested geometry archive contains 94 additional reaction directories.

## Architecture

### RGD1 preprocessing

Add `Scripts/preprocess_rgd1_full.py`. It reads `RGD1.zip` without extracting the complete archive, streams the nested `RGD1_CHNO_xyz.tar.gz`, and converts only reaction IDs present in `RGD1CHNO_AMsmiles_with_split.csv`.

For every reaction, the preprocessor validates and stores:

- matching atom symbols and atom counts across `RG.xyz`, `PG.xyz`, and `TSG.xyz`;
- atomic numbers using `H=1`, `C=6`, `N=7`, and `O=8`;
- reactant, product, and transition-state coordinates;
- `R_E`, `P_E`, and `TS_E` from the reaction's `energy` file;
- the reaction ID for integrity checks.

Records are written as numbered PyTorch shard files under a configurable cache directory. A JSON manifest records the format version, source ZIP identity, canonical CSV name, expected and actual counts, shard names, and per-shard counts. Output is written through temporary files and renamed only after validation, so an interrupted preprocessing run cannot look complete.

### Runtime datasets

Add `Data/MixedRGD1Transition1x.py` with three responsibilities:

1. Load and validate the RGD1 cache manifest.
2. Iterate RGD1 shards in randomized shard order and randomized record order while keeping memory bounded to one shard per worker.
3. Mix the RGD1 iterator with the existing Transition1x training iterator using source selection proportional to each source's remaining sample count. This produces every RGD1 and Transition1x training reaction exactly once per epoch while avoiding a source-blocked order.

RGD1 records are converted to the same PyG contract already used by Transition1x:

```text
Data(
  x,
  reactant_pos,
  product_pos,
  transition_state_pos,
  energies,
)
```

Product and transition-state coordinates are Kabsch-aligned to the reactant, matching the existing RGD1 and Transition1x behavior. Energies remain in their source Hartree units.

Worker-aware partitioning assigns disjoint RGD1 shards and disjoint Transition1x reaction subsets to DataLoader workers. The default zero-worker configuration remains supported. Dataset length is reported explicitly so the mixed training loader has a stable length.

### Configuration and training entry point

Add `Configs/Dynamics_mixed.yml` with separate paths for:

- the Transition1x HDF5 file;
- the preprocessed RGD1 cache directory;
- batch size and seed.

Update the existing flow-matching training entry point to choose the mixed loader only when the mixed-data configuration is present. Existing Transition1x-only configuration and behavior remain unchanged.

The preprocessing cache directory is ignored by Git; source code, tests, configuration, and the cache manifest schema remain tracked.

## Data Flow

```text
RGD1.zip -> CSV whitelist + nested XYZ/energy stream -> validated shard cache
                                                           |
Transition1x train manifest + HDF5 -------------------------+-> proportional mixed train iterator
Transition1x valid manifest + HDF5 ----------------------------> validation iterator
Transition1x test manifest + HDF5 -----------------------------> test iterator
```

## Failure Handling

Preprocessing fails with a clear error if:

- the expected CSV or nested archive is missing;
- a CSV reaction ID is duplicated;
- a whitelisted reaction lacks any required geometry or energy file;
- atom symbols/counts disagree across the three geometries;
- an unsupported element or malformed numeric value is encountered;
- the completed record count is not exactly 176,898.

Runtime loading fails before training if the manifest is missing, its format version is unsupported, a shard is absent, shard counts do not sum to 176,898, or the Transition1x split manifests do not have the expected 9,561/225/287 sizes.

## Testing

Tests are written before production changes and cover:

- XYZ and energy parsing;
- CSV-whitelist behavior, including exclusion of extra archive directories;
- manifest and shard-count validation;
- RGD1-to-PyG field shapes and atom mapping;
- exact mixed split lengths: 186,459 train, 225 validation, 287 test;
- exactly-once source mixing on small synthetic datasets;
- deterministic ordering for a fixed seed;
- disjoint worker partitioning;
- preservation of the existing Transition1x-only loader path.

After unit tests pass, the full RGD1 preprocessing command is run against the provided ZIP. The generated manifest and shard totals are then audited, followed by a loader smoke test that counts a complete mixed epoch without training the model.

## Non-goals

- Reusing the original RGD1 train/validation/test labels.
- Adding RGD1 samples to validation or test.
- Changing the model, loss, optimizer, or Transition1x split membership.
- Deduplicating chemically similar but non-identical reactions across RGD1 and Transition1x; only exact source membership and record integrity are enforced.
