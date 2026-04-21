# Reproducing HapticVLA

End-to-end walkthrough for reproducing the paper's three contact-rich
manipulation results (egg, waffles, marmalade jar). If you only want to
**deploy** HapticVLA, skip to "Inference" — everything before it is the
training pipeline.

## 0. Prerequisites

- Hardware matching [`HARDWARE.md`](HARDWARE.md) (the tactile right arm is
  required for teacher training; the student runs on vision-only setups).
- Python 3.10+, CUDA-capable GPU (RTX 4090 for teacher, H100 for student
  distillation; a single 24 GB GPU is enough for either with some batch
  reduction).
- `pip install -e .` from the repo root to pull in the LeRobot stack
  (PyTorch, smolvla, pyarrow, av, torchvision, etc.).

## 1. Dataset

The paper uses an in-house dataset of 310 real-world episodes (70
successful + 20 faulty per single-arm task; 100 successful + 30 faulty for
the bimanual egg task). Faulty episodes are demonstrations where excessive
grip force damaged the object — they are kept so SA-RWFM can assign
negative rewards and sharpen the high-reward region of the target
distribution.

Expected layout after teleoperation + labeling:

```
datasets_labeled/
├── egg_carton_to_tray/
│   ├── pick_place_soft_20260217_1847/
│   ├── pick_place_soft_20260226_1142/
│   ├── ...
├── marmalade_jar/
├── waffles/
└── <each episode dir contains the LeRobot parquet/video layout>
```

The exact episode mix for the four paper-era runs is hard-wired into each
YAML under `training/configs/`. See `training/docs/dataset_manifest.csv`
for the full list.

### Recording your own data

`examples/crab/record.py` produces new LeRobot-format episodes with all
three cameras, joint state, and tactile arrays. Label each episode with
`success` / `damage` columns so the reward computation can pick them up.

### Reward computation

Before training the SA-RWFM teacher, precompute the reward-weighting
statistics for your dataset mixture:

```bash
python training/scripts/compute_reward_weighting_stats.py \
  --root <path-to-datasets_labeled> \
  --group-key task --horizon 50 --gamma 0.99 \
  --mix-beta 0.7 --alpha 0.25 --w-min 0.25 --w-max 4.0 \
  --batch-size 16 --sim-batches 4000
```

The paper uses the bundled
`training/docs/reward_weighting_stats_recomputed_20260220.json`. See
`training/docs/reward_weighting.md` for the full spec.

## 2. Stage A — SA-RWFM teacher

Fine-tune SmolVLA-450M with the tactile encoder on and reward weighting
active. Paper schedule: AdamW, cosine LR with 8 k step linear warmup to
2e-4, 160 000 steps on a single RTX 4090.

```bash
python training/train.py \
  --config training/configs/train_6dof_right_arm_rwfm_v3.yaml
```

Published checkpoint:
[`armteam/crab-smolvla-rwfm`](https://huggingface.co/armteam/crab-smolvla-rwfm).

IL baselines (for comparison rows in the paper) use the same script with
`train_6dof_{left,right}_arm_multitask_12v_v4.yaml`. The sim+real mixture
ablation (α = 0.85) uses `train_6dof_right_arm_sim_real_cotrain.yaml`.

## 3. Stage B — Precompute teacher targets

Once the teacher converges, roll out its predicted action chunks on the
student's training set so distillation can be fully offline:

```bash
python training/distill_precompute_v2.py \
  --teacher-config     training/configs/train_6dof_right_arm_rwfm_v3.yaml \
  --teacher-checkpoint <path>/rwfm_v3/best/model.pt \
  --student-config     training/configs/train_distill_rwfm_v3.yaml \
  --output             outputs/teacher_targets_v2.pt
```

This generates the ~51 k soft targets referenced in the paper.

## 4. Stage C — Tactile distillation

Train a tactile-free student. The student is initialized from the teacher
(tactile encoder dropped, state projection sliced to proprioceptive dims),
then optimized against the 50/50 blend of ground-truth and teacher
predictions. Paper schedule: AdamW, LR 3e-4, batch 32, 30 000 steps on an
H100.

```bash
python training/train_distill_v2.py \
  --config          training/configs/train_distill_rwfm_v3.yaml \
  --teacher-targets outputs/teacher_targets_v2.pt
```

Published checkpoint:
[`armteam/crab-smolvla-hapticsvla`](https://huggingface.co/armteam/crab-smolvla-hapticsvla).

## 5. Inference

On the Jetson Orin, the one-command launcher wires up both arms and the
three cameras via tmux. Drop the published checkpoints (or your training
outputs) into `~/crab_smolvla_6dof_*` first — see
[`WEIGHTS.md`](WEIGHTS.md) for the expected paths.

```bash
./examples/crab/launch_crab.sh distill                 # HapticVLA (default)
./examples/crab/launch_crab.sh rwfm_v3 -t "Pick waffle"  # SA-RWFM teacher
./examples/crab/launch_crab.sh v4 --sync               # IL baseline, sync inference
./examples/crab/launch_crab.sh --list                   # list all wired-in models
./examples/crab/launch_crab.sh --kill                   # stop everything
```

Sync vs async: the paper's ablation (Tab. I) shows synchronous chunking
often beats async at the cost of higher wall-clock latency. Async is the
default for `launch_crab.sh`; add `--sync` to match the "sync" rows of the
ablation. `--rtc` enables the RealTime Chunking server (see
`examples/crab/run_inference_async_server.py`).

For the X-VLA baseline (Fig. 4), use
`examples/crab/inference_xvla_standalone.py` + the weights described in
the X-VLA paper.

## 6. Evaluation

The paper reports 20 trials per task per model. There is no headless
benchmark harness in this repo — evaluation is physical. Track
success/damage manually using the acceptance criteria stated in
Section IV-B of the paper.

## Troubleshooting

- **Teacher OOM on 4090**: drop `batch_size` and raise
  `gradient_accumulation_steps` proportionally (the v3 config already
  uses an effective batch of 32 via `batch=8, grad_accum=4`).
- **Distillation student diverges**: verify the blending coefficient
  stays at `alpha = 0.5` at train time and is switched to `alpha = 0`
  during validation (the distillation trainer handles this automatically).
- **Tactile stream desync**: the 120 Hz tactile feed must be resampled to
  15 Hz to line up with the camera streams. `crab_dataset.py` handles
  this; check the per-episode reward JSON is non-empty before training.
