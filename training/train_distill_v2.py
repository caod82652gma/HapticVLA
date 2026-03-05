#!/usr/bin/env python3
"""
Distillation training: RWFM teacher (tactile) -> SmolVLA student (no tactile).

Approach: Offline Action-Level Knowledge Distillation
  1. Pre-computed teacher action predictions loaded from .pt file
  2. Student initialized from teacher backbone (minus tactile encoder)
  3. Training targets blended: action = (1-alpha)*GT + alpha*teacher_pred
  4. Standard flow matching loss on blended targets
  5. At inference: standard SmolVLA, no modifications needed

This is NOT FD-VLA (which adds a force prediction module at inference).
Our approach transfers tactile knowledge purely through soft action targets
during training. The student architecture is identical to baseline SmolVLA
at inference time — no extra modules, no force prediction head.

The teacher (RWFM v3) was trained with:
  - Reward-Weighted Flow Matching on 34 labeled datasets (27 clean + 7 crack)
  - Dual tactile encoders (left+right MLP: 100->128->64, total 128-dim)
  - Anchor regularization to prevent mode collapse

The student learns force-aware manipulation behavior implicitly through
the teacher's action predictions, which encode tactile-conditioned policies.

Usage:
    # Step 1: Pre-compute teacher targets (run distill_precompute_v2.py first)
    # Step 2: Train student
    python3 -u train_distill_v2.py -c configs/train_distill_rwfm_v3.yaml
"""

import argparse
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = str(Path(config_path).resolve().parent.parent)
    return cfg


def init_student_from_teacher(student_model, teacher_checkpoint_path, device):
    """Initialize student from teacher weights, handling state_augment_proj mismatch.

    Teacher has state_augment_proj: Linear(134, 32) — 6 state + 128 tactile dims.
    Student has state_augment_proj: Linear(6, 32) — 6 state dims only.

    Strategy: Copy all shared weights. For state_augment_proj, copy the first 6
    columns of the teacher's weight matrix (the state-corresponding dimensions).
    Tactile encoder keys are skipped entirely (student has tactile_encoder.enabled=false).
    """
    logger.info(f"Initializing student from teacher: {teacher_checkpoint_path}")
    teacher_sd = torch.load(teacher_checkpoint_path, map_location=device, weights_only=False)

    student_sd = student_model.state_dict()
    loaded = 0
    skipped_keys = []
    adapted = 0

    for key in teacher_sd:
        if key not in student_sd:
            skipped_keys.append(key)
            continue

        teacher_tensor = teacher_sd[key]
        student_tensor = student_sd[key]

        if teacher_tensor.shape == student_tensor.shape:
            student_sd[key] = teacher_tensor
            loaded += 1
        elif "state_augment_proj.weight" in key:
            # Teacher: [32, 134] (state + tactile), Student: [32, 6] (state only)
            # Copy the first state_dim columns (state dimensions come first in concatenation)
            n_cols = student_tensor.shape[1]
            student_sd[key] = teacher_tensor[:, :n_cols].clone()
            adapted += 1
            logger.info(
                f"  Adapted {key}: teacher {list(teacher_tensor.shape)} -> "
                f"student {list(student_tensor.shape)} (copied first {n_cols} columns)"
            )
        else:
            skipped_keys.append(key)
            logger.warning(
                f"  Shape mismatch {key}: teacher={list(teacher_tensor.shape)}, "
                f"student={list(student_tensor.shape)}, skipping"
            )

    student_model.load_state_dict(student_sd)

    # Log skipped keys summary
    tactile_keys = [k for k in skipped_keys if "tactile" in k.lower()]
    other_skipped = [k for k in skipped_keys if "tactile" not in k.lower()]
    logger.info(
        f"Teacher -> Student init: {loaded} loaded, {adapted} adapted, "
        f"{len(tactile_keys)} tactile keys skipped, {len(other_skipped)} other keys skipped"
    )
    if other_skipped:
        logger.info(f"  Other skipped keys: {other_skipped}")

    del teacher_sd
    torch.cuda.empty_cache()
    return student_model


def build_optimizer(model, cfg):
    train_cfg = cfg["training"]
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {n_trainable:,} / {n_total:,} "
        f"({100 * n_trainable / n_total:.1f}%)"
    )
    return torch.optim.AdamW(
        params,
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        eps=train_cfg["eps"],
        weight_decay=train_cfg["weight_decay"],
    )


def build_scheduler(optimizer, cfg):
    train_cfg = cfg["training"]
    warmup_steps = train_cfg["warmup_steps"]
    total_steps = train_cfg["steps"]
    min_lr = train_cfg["min_lr"]
    base_lr = train_cfg["learning_rate"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model, optimizer, scheduler, step, train_loss, val_loss, output_dir, is_best=False
):
    try:
        ckpt_dir = output_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                f,
                indent=2,
            )
        if is_best:
            best_dir = output_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)
            logger.info(f"  >> New best checkpoint at step {step} (val_loss={val_loss:.4f})")
        logger.info(f"Saved checkpoint to {ckpt_dir}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint at step {step}: {e}")
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        logger.info("Continuing training despite save failure...")


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate on GT actions only (no blending during eval)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
        loss = output["loss"] if isinstance(output, dict) else output
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


class DistillDataset(torch.utils.data.Dataset):
    """Wraps CrabEpisodeDataset to include pre-computed teacher action predictions."""

    def __init__(self, base_dataset, teacher_actions):
        """
        Args:
            base_dataset: CrabEpisodeDataset (train split, 27 clean datasets)
            teacher_actions: Tensor [N, chunk_size, action_dim] from teacher inference
        """
        self.base = base_dataset
        self.teacher_actions = teacher_actions
        if len(self.base) != len(self.teacher_actions):
            raise ValueError(
                f"Dataset size {len(self.base)} != teacher targets {len(self.teacher_actions)}. "
                f"Re-run distill_precompute_v2.py with matching config."
            )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        sample["teacher_action"] = self.teacher_actions[idx]
        return sample


def collate_distill_batch(batch):
    """Collate that includes teacher_action field alongside standard CrabBatch fields."""
    images = {}
    for cam_key in batch[0]["images"]:
        images[cam_key] = torch.stack([b["images"][cam_key] for b in batch])

    return {
        "images": images,
        "state": torch.stack([b["state"] for b in batch]),
        "tactile_left": torch.stack([b["tactile_left"] for b in batch]),
        "tactile_right": torch.stack([b["tactile_right"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "teacher_action": torch.stack([b["teacher_action"] for b in batch]),
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
    parser = argparse.ArgumentParser(
        description="Distillation: RWFM teacher (tactile) -> SmolVLA student (no tactile)"
    )
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    cfg = load_config(args.config)
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.steps:
        cfg["training"]["steps"] = args.steps

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    distill_cfg = cfg["distillation"]

    alpha = distill_cfg["alpha"]
    _config_dir = Path(cfg.get("_config_dir", "."))

    # Resolve relative paths against _config_dir
    teacher_ckpt_path = Path(distill_cfg["teacher_checkpoint"])
    if not teacher_ckpt_path.is_absolute():
        teacher_ckpt_path = _config_dir / teacher_ckpt_path

    teacher_targets_path = Path(distill_cfg["teacher_targets_path"])
    if not teacher_targets_path.is_absolute():
        teacher_targets_path = _config_dir / teacher_targets_path

    logger.info(f"Distillation config:")
    logger.info(f"  alpha = {alpha} (blend: {1-alpha:.0%} GT + {alpha:.0%} teacher)")
    logger.info(f"  teacher checkpoint: {teacher_ckpt_path}")
    logger.info(f"  teacher targets: {teacher_targets_path}")

    # ── Build student model (no tactile encoder) ──
    from crab_smolvla_wrapper import build_model

    logger.info("Building student model (no tactile)...")
    model = build_model(cfg)

    max_action_dim = model_cfg.get("max_action_dim", 32)
    max_state_dim = model_cfg.get("max_state_dim", 32)
    model.smolvla.config.output_features["action"].shape = (max_action_dim,)
    model.smolvla.config.input_features["observation.state"].shape = (max_state_dim,)

    model.to(device)

    # Initialize from teacher backbone (unless resuming from checkpoint)
    if not args.resume:
        model = init_student_from_teacher(model, teacher_ckpt_path, device)

    model.train()

    # ── Load pre-computed teacher targets ──
    logger.info(f"Loading teacher targets from {teacher_targets_path}")
    teacher_actions = torch.load(teacher_targets_path, map_location="cpu", weights_only=False)
    logger.info(f"Teacher targets shape: {list(teacher_actions.shape)}")

    # ── Build datasets ──
    from crab_dataset import CrabEpisodeDataset

    ds_cfg = cfg["dataset"]
    episodes_dir = _config_dir / ds_cfg["episodes_dir"]

    train_ds = CrabEpisodeDataset(
        episodes_dir=episodes_dir,
        episode_names=ds_cfg["episode_names"],
        chunk_size=model_cfg["chunk_size"],
        image_size=tuple(ds_cfg["image_size"]),
        fps=ds_cfg["fps"],
        task_description=ds_cfg.get("task_description", "Pick and place object"),
        val_ratio=ds_cfg.get("val_ratio", 0.1),
        image_keys=ds_cfg["image_keys"],
        action_indices=ds_cfg.get("action_indices"),
        state_indices=ds_cfg.get("state_indices"),
        split="train",
        video_cache_dir=ds_cfg.get("video_cache_dir"),
        transform=ds_cfg.get("augmentation"),
    )

    val_ds = CrabEpisodeDataset(
        episodes_dir=episodes_dir,
        episode_names=ds_cfg["episode_names"],
        chunk_size=model_cfg["chunk_size"],
        image_size=tuple(ds_cfg["image_size"]),
        fps=ds_cfg["fps"],
        task_description=ds_cfg.get("task_description", "Pick and place object"),
        val_ratio=ds_cfg.get("val_ratio", 0.1),
        image_keys=ds_cfg["image_keys"],
        action_indices=ds_cfg.get("action_indices"),
        state_indices=ds_cfg.get("state_indices"),
        split="val",
        video_cache_dir=ds_cfg.get("video_cache_dir"),
    )

    # Verify teacher targets match train split size
    if abs(len(train_ds) - teacher_actions.shape[0]) > 1:
        raise ValueError(
            f"Train dataset has {len(train_ds)} samples but teacher targets have "
            f"{teacher_actions.shape[0]}. Re-run distill_precompute_v2.py."
        )
    # Handle off-by-one from rounding in train/val split
    if len(train_ds) != teacher_actions.shape[0]:
        n = min(len(train_ds), teacher_actions.shape[0])
        teacher_actions = teacher_actions[:n]
        logger.warning(f"Trimmed teacher targets to {n} samples (±1 rounding)")

    distill_train_ds = DistillDataset(train_ds, teacher_actions)
    logger.info(f"Train: {len(distill_train_ds)} samples, Val: {len(val_ds)} samples")

    train_loader = torch.utils.data.DataLoader(
        distill_train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_distill_batch,
        drop_last=True,
    )

    from crab_dataloader import collate_crab_batch

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_crab_batch,
        drop_last=False,
    )

    # ── Optimizer & scheduler ──
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── Output directory ──
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ──
    start_step = 0
    if args.resume:
        resume_dir = Path(args.resume)
        logger.info(f"Resuming from {resume_dir}")
        model.load_state_dict(
            torch.load(resume_dir / "model.pt", map_location=device, weights_only=False)
        )
        optimizer.load_state_dict(
            torch.load(resume_dir / "optimizer.pt", map_location=device, weights_only=False)
        )
        scheduler.load_state_dict(
            torch.load(resume_dir / "scheduler.pt", map_location=device, weights_only=False)
        )
        with open(resume_dir / "metadata.json") as f:
            start_step = json.load(f)["step"]
        logger.info(f"Resumed from step {start_step}")

    # ── Training loop ──
    total_steps = train_cfg["steps"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    log_every = train_cfg["log_every_steps"]
    eval_every = train_cfg["eval_every_steps"]
    save_every = train_cfg["checkpoint"]["save_every_steps"]
    best_val_loss = float("inf")

    logger.info(
        f"Training for {total_steps} steps "
        f"(batch={train_cfg['batch_size']}, grad_accum={grad_accum}, "
        f"effective={train_cfg['batch_size'] * grad_accum})"
    )
    logger.info(
        f"Distillation blend: {1-alpha:.0%} ground-truth + {alpha:.0%} teacher targets"
    )

    step = start_step
    running_loss = 0.0
    loss_count = 0
    t_start = time.perf_counter()

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break

            # ── Blend GT and teacher action targets ──
            gt_action = batch["action"]           # [B, chunk_size, action_dim]
            teacher_action = batch["teacher_action"]  # [B, chunk_size, action_dim]
            blended_action = (1 - alpha) * gt_action + alpha * teacher_action
            batch["action"] = blended_action

            # Remove teacher_action before model forward (model doesn't expect it)
            del batch["teacher_action"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch)
            loss = output["loss"] if isinstance(output, dict) else output

            (loss / grad_accum).backward()
            running_loss += loss.item()
            loss_count += 1

            if loss_count % grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    train_cfg["grad_clip_norm"],
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step % log_every == 0:
                    avg_loss = running_loss / loss_count
                    lr = optimizer.param_groups[0]["lr"]
                    elapsed = time.perf_counter() - t_start
                    sps = (step - start_step) / max(1, elapsed)
                    eta = (total_steps - step) / max(0.01, sps)
                    logger.info(
                        f"step {step}/{total_steps} | loss={avg_loss:.4f} | "
                        f"lr={lr:.2e} | {sps:.2f} steps/s | ETA: {eta / 60:.0f}min"
                    )
                    running_loss = 0.0
                    loss_count = 0

                if step % eval_every == 0:
                    val_loss = evaluate(model, val_loader, device)
                    logger.info(f"step {step} | val_loss={val_loss:.4f}")
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                    if step % save_every == 0 or is_best:
                        save_checkpoint(
                            model, optimizer, scheduler, step,
                            running_loss / max(1, loss_count), val_loss,
                            output_dir, is_best,
                        )

                elif step % save_every == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, step,
                        running_loss / max(1, loss_count), best_val_loss,
                        output_dir,
                    )

    # ── Final evaluation & save ──
    val_loss = evaluate(model, val_loader, device)
    logger.info(f"Final val_loss={val_loss:.4f}")
    save_checkpoint(
        model, optimizer, scheduler, step, 0.0, val_loss,
        output_dir, is_best=(val_loss < best_val_loss),
    )
    logger.info("Distillation training complete!")


if __name__ == "__main__":
    main()
