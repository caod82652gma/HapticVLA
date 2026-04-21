# Training

Training code for the HapticVLA paper. All scripts expect to be run from
the **repo root** so that relative paths (`training/configs/...`,
`../datasets_labeled`, `outputs/...`) resolve the same way they did on the
paper-era training boxes.

## Entry points

| Script | Purpose | Paper reference |
|--------|---------|-----------------|
| `train.py`                | Joint trainer for SmolVLA IL baselines and the SA-RWFM teacher. Reward weighting / anchor regularization are toggled through the YAML config. | Stages A, baselines |
| `distill_precompute_v2.py`| Roll the SA-RWFM teacher over the student's training set and save soft action-chunk targets. | Stage B |
| `train_distill_v2.py`     | Train the tactile-free student on blended (GT + teacher) targets. | Stage C |

## Modules

| File | Role |
|------|------|
| `crab_smolvla_wrapper.py` | SmolVLA wrapper that adds the tactile encoder, RWFM loss reduction, and anchor regularization. Used by every entry point above. |
| `crab_dataset.py`         | LeRobot-format dataset with per-sample reward fields (step_reward, chunk_return, episode_reward, success, damage). |
| `crab_dataloader.py`      | Builds train/val loaders, including the sim/real mixture sampler used by the sim+real cotrain config. |
| `tactile_encoder.py`      | Dual-pad (L/R) 10×10 taxel encoder that emits the 128-dim tactile embedding consumed by the teacher. |
| `crab_xvla_wrapper.py`    | X-VLA baseline wrapper for the comparison in Fig. 4. |

## Configs

Every released checkpoint has a one-to-one config in `configs/`:

| Config | Trains | HF checkpoint |
|--------|--------|----------------|
| `train_6dof_left_arm_multitask_12v_v4.yaml`  | Left-arm IL baseline (no tactile)       | `armteam/crab-smolvla-left-arm` |
| `train_6dof_right_arm_multitask_12v_v4.yaml` | Right-arm IL baseline (no tactile)      | `armteam/crab-smolvla-right-arm` |
| `train_6dof_right_arm_rwfm_v3.yaml`          | SA-RWFM teacher (tactile encoder on)    | `armteam/crab-smolvla-rwfm` |
| `train_distill_rwfm_v3.yaml`                 | Tactile-distilled student (HapticVLA)   | `armteam/crab-smolvla-hapticsvla` |
| `train_6dof_right_arm_sim_real_cotrain.yaml` | Sim + real mixture baseline (α = 0.85)  | *(not published — used for ablation rows)* |
| `train_crab_xvla.yaml`                       | X-VLA baseline                           | *(weights from the X-VLA authors)* |

## Auxiliary scripts

`scripts/`:

- `build_dataset_manifest.py` — (re)build `docs/dataset_manifest.csv`
  from a datasets directory.
- `compute_reward_weighting_stats.py` — precompute per-group robust stats
  consumed by `crab_smolvla_wrapper.py` to produce RWFM sample weights.

## Documentation

- `docs/reward_weighting.md` — full SA-RWFM weighting spec (mirror of the
  paper's Eq. 11–18).
- `docs/dataset_manifest.csv` — episode-level inventory used by the
  paper's runs.
- `docs/reward_weighting_stats_recomputed_20260220.json` — paper-era
  reward weighting statistics.

For the end-to-end walkthrough (data → train → distill → deploy) see
[`docs/paper/REPRODUCING.md`](../docs/paper/REPRODUCING.md).
