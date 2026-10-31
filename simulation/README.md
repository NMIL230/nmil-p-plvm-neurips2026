# Simulation

This folder holds the paired-task simulation study: synthetic players with latent drives `P`
(patch persistence) and `O` (opportunistic reward seeking) complete a `foraging` and a
`cleaning` grid-world task. Their label (`Planner` or `Hopper`) is set by whether the two
drives agree. The agent's choice rule and all simulator settings are in `agents.py` and
`config.py`.

Script defaults are the settings of the reported runs, so the commands below only pass what
changes between runs. Run them from the repository root. Outputs go under the artifact root
(`artifacts/`, or `$PLVM_ARTIFACT_ROOT`).

```bash
D=artifacts/simulation/outputs/simulations
M=artifacts/models/simulations/minigrid_foraging_cleaning
```

## 1. Simulate

```bash
python -m simulation.simulate                                                # complementary
python -m simulation.simulate --relationship_condition redundant
python -m simulation.simulate --relationship_condition weak_shared_structure
```

Each run writes 10,000 players to `$D/foraging_cleaning[_redundant|_weak_structure]` (CPU,
about 7 minutes and 4 GB). The `--relationship_condition` presets set how strongly each task
reveals `P` and `O`; see `config.py`.

## 2. Task encoders

```bash
for task in foraging cleaning; do
  python -m simulation.train.train_task_specific_transformer --task_name $task
done
```

There is one next-action transformer per task. Both are trained on the complementary data
and shared by all three conditions. The PLVM trainer reads the encoders named in
`$M/pw_task_specific_models_best_256_new.json`, which lists the encoders included in the
artifact bundle. To use your own, point `--task_model_spec` at a file of the same form.
Retrained encoders are close to, but not bit-identical with, the ones in the bundle.

## 3. PLVM and baselines

All models are trained by `simulation.train.train_plvm_label_minigrid`, except the outcome
baseline (step 3d). Each run reads one condition's episodes and writes to that condition's
output folder:

| Condition | `--data_dir` | `--output_dir` |
|---|---|---|
| Complementary (the script defaults) | `$D/foraging_cleaning` | `$M/plvm_label` |
| Redundant | `$D/foraging_cleaning_redundant` | `$M/plvm_label_redundant` |
| Weak shared structure | `$D/foraging_cleaning_weak_structure` | `$M/plvm_label_weak_structure` |

Pass `--cache_dir <output_dir>/embedding_cache` with the same folder so each condition keeps its
own embedding cache. The commands below set these once per condition:

```bash
TRAIN="python -m simulation.train.train_plvm_label_minigrid"
use_condition() {   # usage: use_condition <data folder> <output folder>
  ARGS="--data_dir $D/$1 --output_dir $M/$2 --cache_dir $M/$2/embedding_cache"
}
```

**3a. PLVM and single-task models (all three conditions).** For the complementary condition:

```bash
use_condition foraging_cleaning plvm_label
$TRAIN $ARGS --model_type plvm_both_partial                                        # PLVM-2
$TRAIN $ARGS --model_type single_task --partial_task_name foraging --latent_dim 1  # single task
$TRAIN $ARGS --model_type single_task --partial_task_name cleaning --latent_dim 1  # single task
```

Then run the same three commands after each of:

```bash
use_condition foraging_cleaning_redundant plvm_label_redundant
use_condition foraging_cleaning_weak_structure plvm_label_weak_structure
```

**3b. Linear heads (complementary condition only).**

```bash
use_condition foraging_cleaning plvm_label
$TRAIN $ARGS --model_type plvm_both_partial --prediction_head_type logistic
$TRAIN $ARGS --model_type single_task --partial_task_name foraging --latent_dim 1 --prediction_head_type logistic
$TRAIN $ARGS --model_type single_task --partial_task_name cleaning --latent_dim 1 --prediction_head_type logistic
```

**3c. Two-task MLP on the concatenated embeddings (all three conditions).**

```bash
use_condition foraging_cleaning plvm_label
$TRAIN $ARGS --model_type plvm_both_partial --prediction_head_type mlp

use_condition foraging_cleaning_redundant plvm_label_redundant
$TRAIN $ARGS --model_type plvm_both_partial --prediction_head_type mlp

use_condition foraging_cleaning_weak_structure plvm_label_weak_structure
$TRAIN $ARGS --model_type plvm_both_partial --prediction_head_type mlp
```

**3d. Outcome-only random-forest baseline (complementary condition, CPU).**

```bash
BASELINE="python -m simulation.train.train_outcome_observation_baseline_minigrid"
$BASELINE --model_type plvm_both_partial
$BASELINE --model_type single_task --partial_task_name foraging
$BASELINE --model_type single_task --partial_task_name cleaning
```

Runs with a non-default head are filed in a folder named after the head (for example 3b's
two-task run goes to `$M/plvm_label_concat_logistic_probe`). Every folder records its runs in
`training_logs/model_performance.csv`, which the table and figure scripts in `paper/` read.

The simulation figures and tables are rebuilt with
`python paper/make_paper.py --only fig04a,fig04b,fig04c,fig05,fig07,tab08,tab09,tab09_matched_head,tab12`.
