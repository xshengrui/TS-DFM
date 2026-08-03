# TS-DFM Transition1x Reproduction Audit

This note tracks the settings that must match the Nature Communications paper
result for Transition1x direct prediction.

## Paper-Aligned Settings

| Item | Paper / official setting | Local status |
| --- | --- | --- |
| Dataset | Official Transition1x split | Uses `Data/reactions_train.pickle`, `Data/reactions_valid.pickle`, `Data/reactions_test.pickle` |
| Train split | 9561 reactions | Preserved by split manifests |
| Validation split | 225 reactions | Preserved by split manifests |
| Test split | 287 reactions | Preserved by split manifests |
| Batch size | 32 | `Configs/Dynamics.yml` uses `data.batch_size: 32` |
| Epochs | 2000 | `Configs/Dynamics.yml` uses `train.epochs: 2000` |
| Optimizer | Adam | `Scripts/train_flow_matching_dist_ts1x.py` uses Adam |
| Learning rate | 5e-4 | `Configs/Dynamics.yml` uses `optimizer.lr: 0.0005` |
| LR scheduler | Reduce on plateau, factor 0.8, patience 40 | Configured |
| TSDVNet blocks | 6 | `dynamic_model.parameters.num_layers: 6` |
| Hidden dimension | 128 | `dynamic_model.parameters.hidden_dim: 128` |
| Cutoff | 20 Angstrom | `dynamic_model.parameters.cutoff: 20.0` |
| Training noise | sigma 0.1 | `train.random_noise: true`, `train.noise_scale: 0.1` |
| ODE step size | 0.05 | Inference default is `--step-size 0.05` |
| Coordinate reconstruction | Linear interpolation from reactant/product, weighted L-BFGS | `Scripts/infer_ts1x_xyz.py` uses `linear_interp_lbfgs` |
| RMSD metric | Kabsch-aligned coordinate RMSD | `Scripts/evaluate_ts1x_xyz.py`; it also reports a reflection-allowed RMSD matching the repository Kabsch convention |
| DMAE metric | Mean absolute pairwise-distance error | `Scripts/evaluate_ts1x_xyz.py` |

## Important Findings

The old run `tsdfm_ts1x_2026_07_08__10_07_48` was not paper-aligned:

- It used `batch_size: 256`, not 32.
- It did not log `random_noise` or `noise_scale`, so it did not enable the
  paper noise scale of 0.1.

The old checkpoint evaluated with the corrected inference/evaluation pipeline:

- RMSD mean: 0.368561
- DMAE mean: 0.146142

Since DMAE is also high, the gap is not only a coordinate reconstruction issue.
The learned distance matrix is worse than the paper result.

For new runs where DMAE is close to the paper but RMSD mean remains high, run:

```bash
python -m Scripts.check_ts1x_reconstruction_oracle \
  --hdf5 Data/Transition1x.h5 \
  --output-csv logs/oracle_reconstruction_metrics.csv
```

If the oracle reconstruction has near-zero RMSD, remaining high-RMSD cases are
mainly model-distance outliers. If the oracle reconstruction has high RMSD for
the same reactions, the coordinate reconstruction procedure is the bottleneck.

## GPU Utilization

Do not increase `data.batch_size` when reproducing Table 1; that changes the
optimization regime. To improve GPU utilization while preserving batch size 32,
use dataloader throughput options:

```yaml
data:
  batch_size: 32
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

Safe ranges for server trials:

- RTX 4090: start with `num_workers: 4`, then try 6 or 8 if GPU utilization is
  below 50%.
- H200: start with `num_workers: 8`. If utilization is still below 50%, prefer
  running on 4090 for this small 5.2M-parameter model rather than changing
  batch size.

Transition1x `IterableDataset` now partitions data across workers. Before this
fix, setting `num_workers > 0` would risk duplicated split traversal.
