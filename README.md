# Early Prediction of Future Behavioral Strategy from Process Traces

Code for the NeurIPS 2026 paper "Early Prediction of Future Behavioral Strategy from Process
Traces". The paper introduces Process-Level Latent Variable Models (PLVMs). A PLVM predicts how a
person will approach a *future* task from the process traces of tasks they have already done:
the moment-to-moment record of what they did, not just their final score.

The paper studies two domains:

- **PowerWash**: cleaning traces from players of a commercial game, used to predict each
  player's spatial strategy (navigation style) on a level they have not yet played.
- **Simulations**: a paired-task gridworld simulation, used as a controlled test of the same idea.

## Setup

```bash
conda env create -f environment.yml
conda activate plvm
```

Training the encoders and PLVMs is much faster on a GPU. Regenerating the figures and tables from
the artifact bundle runs on a CPU.

## Repository layout

```text
paper/         Generators for every figure, table and in-text number in the paper
plvm/          Shared model code, training utilities and the artifact-path resolver
powerwash/     PowerWash pipeline: data/ (raw -> features, labels), train/, analysis/
simulation/    MiniGrid simulator and its training code (see simulation/README.md)
tests/         Fast regression tests
```

Two folders are created locally and are not in git:

```text
artifacts/            Datasets, trained models and training logs (download, see below)
manuscript_figures/   Output of paper/make_paper.py
```

## Artifacts

The datasets, trained models and training logs are distributed separately from the code. They are provided as supplementary material on openreview.
Download the artifact bundle and unpack it as `artifacts/` in the repository root, or point the
code at another location:

```bash
export PLVM_ARTIFACT_ROOT=/path/to/artifacts
```

All relative paths in the commands below are resolved against this artifact root.

## Regenerating the paper's figures and tables

Everything in the paper is built from the artifacts by one command:

```bash
python paper/make_paper.py
```

This runs every generator, in the order the items appear in the paper, and writes:

- figures to `manuscript_figures/figures/Figure_*.pdf`
- tables to `manuscript_figures/tables/`
- in-text numbers to `manuscript_figures/tables/in_text_numbers.{json,md}`

Useful flags: `--list` shows the jobs, `--dry_run` previews the commands, `--only fig02,tab09`
rebuilds a subset, and `--keep_going` continues past a failure.

Each job is one row of `paper/paper_manifest.tsv`, which maps a generator script to the paper
items it produces. To rebuild a single item, run its script directly, for example:

```bash
python paper/figures/fig02_example_process_traces.py
```

## Reproducing the PowerWash pipeline

This is optional, because the figures and tables only need the artifact bundle. Every script
documents its inputs, outputs and options in its module docstring and `--help`. The defaults are
the settings used for the runs in the artifact bundle.

**1. Data.** The raw PowerWash dataset is on OSF: <https://osf.io/wpeh6/overview>. Download and
unzip it, then run the scripts in `powerwash/data/` in order:

```bash
python powerwash/data/generate_simplified_data.py /path/to/osf/data   # raw CSVs -> per-task files
python powerwash/data/extract_player_sequences.py                                # per-attempt traces
python powerwash/data/derive_features_for_transformer.py                         # encoder inputs and targets
python powerwash/data/create_firestation_zone_planning_labels.py                 # Fire Station labels and folds
```

`extract_player_sequences.py` writes two folders in one pass: the sequences with the implicit-gap
correction, and an uncorrected copy that `derive_features_for_transformer.py` falls back on.
`create_firestation_zone_planning_labels.py --task storybook_house` builds the Storybook House
labels.

**2. Task encoders.** Train one frozen transformer encoder per observed level:

```bash
python powerwash/train/train_task_specific_behavior_model_powerwash.py --task_name back_garden
python powerwash/train/train_task_specific_behavior_model_powerwash.py --task_name bungalow
```

The encoders the PLVM uses are listed in `models/powerwash/pw_task_specific_next_subtask.json`.

**3. PLVM and baselines.** Train one run per fold (`fold_00` ... `fold_09`) and label set:

```bash
LABELS=powerwash/outputs/firestation_zone_labels/fold_00
python powerwash/train/train_plvm_powerwash.py \
    --player_label_csv $LABELS/firestation_zone_labels_2_class_clean.csv \
    --player_split_manifest $LABELS/player_split_manifest.csv
```

This trains PLVM-2 on the 2-class label. For the 3-class label, use
`firestation_zone_labels_3_class.csv` with `--target_mode player_label_3_class_zonehopping`. The
same script trains the single-task, linear-probe and two-task MLP baselines through
`--model_type` and `--prediction_head_type`; see its docstring. The outcome-only random-forest
baseline is `powerwash/train/train_rf_zone_planning_baseline.py`.

Training runs take GPU-hours in total. The artifact bundle lets you skip them.

## Reproducing the Simulation results

`simulation/README.md` describes the simulator and gives the commands to generate the three
simulated conditions and to train the encoders, PLVMs and baselines.

## Tests

```bash
pytest tests
```

The tests are small, fast checks and do not need the artifacts.

## License

The code is released under the MIT License (see `LICENSE`). The PowerWash Simulator dataset was
collected and released by Vuorre et al. (2023) on OSF under CC0 1.0; the derived data in the
artifact bundle is released under the same CC0 1.0 license. If you use the PowerWash data, please
also cite the dataset paper:

```bibtex
@article{vuorre2023intensive,
  title   = {An intensive longitudinal dataset of in-game player behaviour and well-being in
             PowerWash Simulator},
  author  = {Vuorre, Matti and Magnusson, Kristoffer and Johannes, Niklas and Butlin, James and
             Przybylski, Andrew K},
  journal = {Scientific Data},
  year    = {2023}
}
```

## Citation

If you use this code or the artifact bundle, please cite:

```bibtex
@inproceedings{kasumba2026early,
  title     = {Early Prediction of Future Behavioral Strategy from Process Traces},
  author    = {Kasumba, Robert and Barbour, Dennis and Ho, Chien-Ju},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
