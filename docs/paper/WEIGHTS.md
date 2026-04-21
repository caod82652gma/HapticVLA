# Pretrained Weights

All four checkpoints from the paper are released on the Hugging Face Hub
under [`armteam/`](https://huggingface.co/armteam).

| HF model | Paper name | Tactile at inference | Use case |
|----------|------------|----------------------|----------|
| [`armteam/crab-smolvla-hapticsvla`](https://huggingface.co/armteam/crab-smolvla-hapticsvla) | Distilled RWFM v3 student — **HapticVLA (ours)** | **No** | Deploy this for contact-rich manipulation without tactile hardware. |
| [`armteam/crab-smolvla-rwfm`](https://huggingface.co/armteam/crab-smolvla-rwfm) | SA-RWFM v3 teacher | Yes | Teacher used to generate distillation targets; also the `w/ TD` ablation row. |
| [`armteam/crab-smolvla-right-arm`](https://huggingface.co/armteam/crab-smolvla-right-arm) | Right-arm multitask 12 V v4 | No | IL baseline for the right arm (`SmolVLA` row in Fig. 4). |
| [`armteam/crab-smolvla-left-arm`](https://huggingface.co/armteam/crab-smolvla-left-arm) | Left-arm multitask 12 V v4 | No | Left arm IL baseline, paired with any right-arm checkpoint for the bimanual egg task. |

Each model repository contains:

- `config.yaml` — the exact YAML used to train the checkpoint (mirrors one
  of the files under `training/configs/`).
- `best/model.pt` — model weights.
- `best/metadata.json` — training step and val loss at save time (teacher
  and student only).

## Downloading

```bash
huggingface-cli download armteam/crab-smolvla-hapticsvla \
  --local-dir ~/crab_smolvla_6dof_right_arm_distill_rwfm_v3
```

Paths matching `~/crab_smolvla_6dof_<slug>/best/model.pt` are the layout
expected by `examples/crab/launch_crab.sh` on the Jetson, so this one
command sets up deployment.

## Bimanual inference

The bimanual tasks (Egg) use the **left arm baseline** paired with the
**right arm HapticVLA student**:

| Arm | Default checkpoint |
|-----|--------------------|
| Left | `armteam/crab-smolvla-left-arm` |
| Right | `armteam/crab-smolvla-hapticsvla` |

`launch_crab.sh distill` wires both of these automatically.

## Licensing

Weights inherit the Apache-2.0 license of the training code.
