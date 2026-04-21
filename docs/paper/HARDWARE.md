# Hardware

The Crab platform used in the paper is a two-arm setup built on LeRobot's
SO-101 manipulators, instrumented with a custom tactile gripper and three
RGB cameras. Inference runs on a Jetson Orin NX; training runs on a single
workstation / server GPU.

## Manipulators

| Role | Arm | Servos | Gripper |
|------|-----|--------|---------|
| Left | SO-101 (stock) | 7.4 V Feetech STS3215 | Stock parallel |
| Right | SO-101 (12 V) | 12 V Feetech STS3215 | Custom parallel gripper with tactile array |

The 12 V right arm provides the torque headroom needed to manipulate the
heavier objects (marmalade jar, egg carton) in the paper's task set.

## Tactile sensor

- Two electrodes, one per fingertip of the right gripper.
- 10×10 taxels per electrode → **200 taxels total**.
- Per-taxel force range: 1–9 N.
- Sampling rate: 120 Hz.
- Driver: `src/lerobot/robots/crab/tactile_sensor.py`.

The sensor is only required at **training** time for the SA-RWFM teacher.
Once distillation is complete, the student (`armteam/crab-smolvla-hapticsvla`)
runs with vision and proprioception only.

## Cameras

| Stream | Sensor | Resolution | Role |
|--------|--------|------------|------|
| `observation.images.main_camera` | Intel RealSense D435 (RGB) | 640×480 | Overhead scene view |
| `observation.images.left_arm_camera` | IMX335 5 MP | 640×480 | Wrist-mounted, left arm |
| `observation.images.right_arm_camera` | IMX335 5 MP | 640×480 | Wrist-mounted, right arm |

All streams are captured at the same 15 FPS used by the training data.
Images are resized to 256×256 before entering SmolVLA.

## Compute

- **Inference**: NVIDIA Jetson Orin NX 16 GB. Runs the full
  bimanual SmolVLA stack at 10 FPS in async mode via
  `examples/crab/launch_crab.sh`.
- **SA-RWFM teacher training**: single RTX 4090 (24 GB VRAM). 160 000
  steps, AdamW, cosine LR to 2e-4 with 8 000-step linear warmup.
- **Tactile distillation**: single H100. 30 000 steps, AdamW, LR 3e-4,
  batch size 32.

## Tasks in the paper

| Task | Instruction | Focus |
|------|-------------|-------|
| Jar pick-and-place | "Pick a jar of marmalade and place it into the box" | Deformable container |
| Waffles pick-and-place | "Pick waffles and place them into the box" | Fragile, crumbly contents |
| Egg pick-and-place | "Open the carton. Pick an egg and place it on the tray." | Fragile + bimanual coordination |
