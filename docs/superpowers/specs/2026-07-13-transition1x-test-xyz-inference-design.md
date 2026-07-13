# Transition1x Test XYZ Inference Design

## Goal

Add a standalone command-line script that loads a trained TS-DFM best checkpoint, runs inference over the unchanged Transition1x test split, reconstructs predicted transition-state coordinates from the predicted pairwise distances, and writes one XYZ file per test reaction.

## Chosen approach

Add `Scripts/infer_ts1x_xyz.py` instead of changing the training entry point or relying on a notebook. The command will be repeatable, will not create an optimizer or resume training, and will preserve the test manifest order and reaction identifiers.

The alternatives were:

- Extend `train_flow_matching_dist_ts1x.py` with an inference mode. This would couple training and export concerns and could accidentally create a new training log or optimizer state.
- Adapt `Notebooks/TestFlowmatchingDist.ipynb`. This already demonstrates the numerical path, but it is not a stable batch command and does not preserve a reproducible file manifest.

## Command-line interface

The script will require:

- `--config`: the Dynamics YAML used to construct the trained model.
- `--checkpoint`: a checkpoint containing the `model` state dictionary.
- `--hdf5`: the Transition1x HDF5 file.
- `--output`: the output directory.

It will accept `--device` (default `cuda`) and `--step-size` (default `0.05`).

## Data flow

1. Load the model architecture from the supplied Dynamics configuration.
2. Load `checkpoint["model"]` with `map_location` set to the requested device, then switch the model and `ODEWrapper2` to evaluation mode.
3. Load `Data/reactions_test.pickle` directly and iterate it in its stored order. Do not use `Dataset_dynamics.__iter__`, because that iterator shuffles its reaction list and drops `formula/rxn` metadata from yielded PyG objects.
4. For each `(formula, rxn)`, call the existing `get_dynamics_data` against the HDF5 `data` group.
5. Build a single-reaction fully connected graph, compute reactant/product distances and their midpoint, and run `ODEWrapper2` under `torch.no_grad()`.
6. Run the existing `pairwise_dist_to_coord` outside `torch.no_grad()` because its LBFGS coordinate reconstruction requires gradients.
7. Write the predicted coordinates as `{index}_{formula}_{rxn}_pred.xyz` and append a row to `manifest.csv` containing the index, formula, reaction ID, relative XYZ path, and coordinate-reconstruction loss.

The true Transition1x transition-state coordinates will not be used for Kabsch alignment or any other post-processing of the saved prediction. This prevents test-label leakage. They are present in the existing data object but are not consumed by the inference calculation.

## Validation and error handling

The script will fail early with a clear error when the configuration, checkpoint, HDF5 file, or test manifest is missing; when the checkpoint lacks a `model` key; or when the requested device is unavailable. Output files will be written only under the requested directory.

On successful completion, the command must produce exactly one manifest row and one predicted XYZ per unique test reaction. With the current manifest, the expected count is 287.

## Testing

Unit tests will cover import safety, argument/path validation, checkpoint validation, deterministic output naming, and manifest writing without requiring CUDA or the full Transition1x HDF5 file. A lightweight injected inference/export path will use small synthetic tensors or fakes where the full model and ODE would be unnecessarily expensive.

Repository verification will run the new focused tests followed by the existing test suite. A full 287-reaction GPU inference run cannot be executed in the local workspace because the configured HDF5 file and checkpoint live on the remote server; the final handoff will include the exact server command.
