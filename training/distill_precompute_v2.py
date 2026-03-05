#!/usr/bin/env python3
"""Pre-compute RWFM teacher action predictions for distillation.

Runs the tactile-enabled RWFM teacher on the student's training datasets
(27 clean episodes) and saves predicted action chunks per sample.
These serve as soft targets for distillation training.

Usage:
    python3 -u distill_precompute_v2.py \
        --teacher-config configs/train_6dof_right_arm_rwfm_v3.yaml \
        --teacher-checkpoint ../outputs/rwfm_v3_teacher.pt \
        --student-config configs/train_distill_rwfm_v3.yaml \
        --output ../outputs/teacher_targets_v2.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from crab_smolvla_wrapper import build_model
from crab_dataset import CrabEpisodeDataset

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Minimal collation for teacher inference."""
    images = {}
    for cam_key in batch[0]["images"]:
        images[cam_key] = torch.stack([b["images"][cam_key] for b in batch])

    return {
        "images": images,
        "state": torch.stack([b["state"] for b in batch]),
        "tactile_left": torch.stack([b["tactile_left"] for b in batch]),
        "tactile_right": torch.stack([b["tactile_right"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "task": [b["task"] for b in batch],
        "task_name": [b["task_name"] for b in batch],
        "dataset_name": [b["dataset_name"] for b in batch],
        "domain": [b["domain"] for b in batch],
        "step_reward": torch.stack([b["step_reward"] for b in batch]),
        "chunk_return": torch.stack([b["chunk_return"] for b in batch]),
        "episode_reward": torch.stack([b["episode_reward"] for b in batch]),
        "episode_success": torch.stack([b["episode_success"] for b in batch]),
        "episode_damage": torch.stack([b["episode_damage"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-compute teacher targets for distillation")
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-config", required=True)
    parser.add_argument("--output", default="../outputs/teacher_targets_v2.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda")

    # Load teacher config and build model
    with open(args.teacher_config) as f:
        teacher_cfg = yaml.safe_load(f)

    logger.info("Building teacher model (with tactile encoder)...")
    teacher = build_model(teacher_cfg)

    max_action_dim = teacher_cfg["model"].get("max_action_dim", 32)
    max_state_dim = teacher_cfg["model"].get("max_state_dim", 32)
    teacher.smolvla.config.output_features["action"].shape = (max_action_dim,)
    teacher.smolvla.config.input_features["observation.state"].shape = (max_state_dim,)

    logger.info(f"Loading teacher checkpoint from {args.teacher_checkpoint}")
    state_dict = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    teacher.load_state_dict(state_dict)
    teacher.to(device)
    teacher.eval()
    del state_dict
    torch.cuda.empty_cache()
    logger.info("Teacher model ready")

    # Load student config to get the 27 clean datasets
    with open(args.student_config) as f:
        student_cfg = yaml.safe_load(f)

    ds_cfg = student_cfg["dataset"]
    _config_dir = Path(args.student_config).resolve().parent.parent
    episodes_dir = _config_dir / ds_cfg["episodes_dir"]

    # Build train dataset (no augmentation for precompute)
    train_ds = CrabEpisodeDataset(
        episodes_dir=episodes_dir,
        episode_names=ds_cfg["episode_names"],
        chunk_size=teacher_cfg["model"]["chunk_size"],
        image_size=tuple(ds_cfg["image_size"]),
        fps=ds_cfg["fps"],
        task_description=ds_cfg.get("task_description", "Pick and place object"),
        val_ratio=ds_cfg.get("val_ratio", 0.1),
        image_keys=ds_cfg["image_keys"],
        action_indices=ds_cfg.get("action_indices"),
        state_indices=ds_cfg.get("state_indices"),
        split="train",
        video_cache_dir=ds_cfg.get("video_cache_dir"),
    )

    logger.info(f"Train dataset: {len(train_ds)} samples from {len(ds_cfg['episode_names'])} episodes")

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Run teacher inference
    all_teacher_actions = []
    t_start = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            smolvla_batch = teacher.prepare_batch_for_smolvla(batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                actions = teacher.smolvla.predict_action_chunk(smolvla_batch)  # [B, chunk_size, action_dim]
            # Crop to actual action_dim
            if actions.shape[-1] > teacher.action_dim:
                actions = actions[..., :teacher.action_dim]

            all_teacher_actions.append(actions.cpu().float())

            if (i + 1) % 50 == 0:
                n_done = min((i + 1) * args.batch_size, len(train_ds))
                elapsed = time.time() - t_start
                rate = n_done / elapsed
                eta = (len(train_ds) - n_done) / max(rate, 0.01) / 60
                logger.info(f"Processed {n_done}/{len(train_ds)} samples ({rate:.1f} samples/s, ETA: {eta:.1f}min)")

    all_teacher_actions = torch.cat(all_teacher_actions, dim=0)
    logger.info(f"Teacher targets shape: {all_teacher_actions.shape}")
    logger.info(f"Total time: {(time.time() - t_start) / 60:.1f} min")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_teacher_actions, output_path)
    logger.info(f"Saved to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
