# Introduction1

This repo contains the code for "Generative Flow Model on Distance Geometry for Predicting Transition States of Chemical Reactions" published at *Nature Communications*.

# Dataset and pretrained checkpoints

Before reproducing the results on Transition1x dataset, please download the `.h5` file in [Item - Transition1x - figshare - Figshare](https://figshare.com/articles/dataset/Transition1x/19614657/4?file=36035789).

The preprocessed RGD1 subset used in our work can be downloaded from [https://doi.org/10.5281/zenodo.20179684](https://doi.org/10.5281/zenodo.20179684).

We have provided the trained checkpoints of baseline methods and our proposed TS-DFM. They can be downloaded from [https://doi.org/10.5281/zenodo.20179684](https://doi.org/10.5281/zenodo.20179684)

# Reproduction

For reproducing the results of TS-DFM, please run the Python scripts in the `Scripts` folder. The files containing hyperparameters are in the `Configs` folder.

## Transition1x paper setting

The main-paper Transition1x result in Table 1 uses `Configs/Dynamics.yml`:

- batch size: 32
- TSDVNet hidden dimension: 128
- update blocks: 6
- cutoff: 20 Angstrom
- learning rate: 5e-4
- training noise scale: 0.1
- ODE inference step size: 0.05

After setting `data.path` in `Configs/Dynamics.yml` to the downloaded
Transition1x HDF5 file, train with:

```bash
python Scripts/train_flow_matching_dist_ts1x.py \
  --config_file Configs/Dynamics.yml \
  --log_prefix tsdfm_ts1x \
  --device cuda
```

The training script reports validation/test loss on the predicted TS distance
matrix. To compare with the paper's RMSD/DMAE values, generate XYZ structures
and evaluate them with the paper metrics:

```bash
python -m Scripts.infer_ts1x_xyz \
  --config Configs/Dynamics.yml \
  --checkpoint logs/dynamics_flow/<run>/checkpoints/checkpoint_best.pth \
  --hdf5 Data/Transition1x.h5 \
  --output logs/dynamics_flow/<run>/test_xyz \
  --device cuda

python -m Scripts.evaluate_ts1x_xyz \
  --hdf5 Data/Transition1x.h5 \
  --pred-dir logs/dynamics_flow/<run>/test_xyz \
  --output-csv logs/dynamics_flow/<run>/test_xyz/metrics.csv
```

For server training, keep `data.batch_size: 32` for paper reproduction. Improve
GPU utilization with `data.num_workers`, `pin_memory`, `persistent_workers`,
and `prefetch_factor` instead of increasing the batch size. See
`docs/reproduction_audit.md` for the checked settings.

## Mixed RGD1 + Transition1x training

This is a custom mixed-data experiment and should not be directly compared with
the Transition1x-only result reported in the main paper Table 1.

The mixed-data configuration trains on all 176,898 reactions listed in the
RGD1 CSV together with the 9,561-reaction Transition1x training split. Its
validation and test sets remain the 225- and 287-reaction Transition1x splits.

First create the sharded RGD1 cache:

```bash
python Scripts/preprocess_rgd1_full.py \
  --source-zip "D:/path/to/RGD1.zip" \
  --output-dir Data/processed/rgd1_full
```

Set `data.transition1x_path` in `Configs/Dynamics_mixed.yml` to the downloaded
Transition1x HDF5 file, then start training with:

```bash
python Scripts/train_flow_matching_dist_ts1x.py \
  --config_file Configs/Dynamics_mixed.yml
```

The RGD1 CSV `dataset` column is intentionally ignored. CSV reaction IDs are
used as the full-data whitelist, while validation and testing use only the
unchanged Transition1x manifests in `Data/`.

The baseline methods are provided in the following folders:

````bash
React-OT on ts1x:react-ot-main/
React-OT on rgd1:react-ot-rgd1/
PSI-based:learnts_main/
OA-ReactDiff: OAReactDiff-main/
NeuralNEB: NeuralNEB/
````

# Citation

If you find this repo useful, please cite our article as follows
````latex
@article{luoGenerativeFlowModel2026a,
  title = {Generative Flow Model on Distance Geometry for Predicting Transition States of Chemical Reactions},
  author = {Luo, Yufei and Gu, Xiang and Sun, Jian},
  year = 2026,
  month = jun,
  journal = {Nature Communications},
  issn = {2041-1723},
  doi = {10.1038/s41467-026-74101-0},
````
