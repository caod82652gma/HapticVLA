"""
Dataloader builder for Crab robot co-training.

Handles: collation, weighted domain sampling (sim/real), train/val split.
Extracted from crab_dataset.py to keep modules focused.
"""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from crab_dataset import CrabEpisodeDataset

logger = logging.getLogger(__name__)


def collate_crab_batch(batch: list[dict]) -> dict:
    images = {}
    for cam_key in batch[0]["images"]:
        images[cam_key] = torch.stack([b["images"][cam_key] for b in batch])

    return {
        "images": images,
        "state": torch.stack([b["state"] for b in batch]),
        "tactile_left": torch.stack([b["tactile_left"] for b in batch]),
        "tactile_right": torch.stack([b["tactile_right"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "step_reward": torch.stack([b["step_reward"] for b in batch]),
        "chunk_return": torch.stack([b["chunk_return"] for b in batch]),
        "episode_reward": torch.stack([b["episode_reward"] for b in batch]),
        "episode_success": torch.stack([b["episode_success"] for b in batch]),
        "episode_damage": torch.stack([b["episode_damage"] for b in batch]),
        "task": [b["task"] for b in batch],
        "task_name": [b["task_name"] for b in batch],
        "dataset_name": [b["dataset_name"] for b in batch],
        "domain": [b["domain"] for b in batch],
        "metadata": [b["metadata"] for b in batch],
    }


def _build_domain_sampler(
    dataset: CrabEpisodeDataset, alpha: float
) -> WeightedRandomSampler | None:
    """Build weighted sampler for sim/real co-training.

    Each batch slot independently samples sim with probability alpha,
    real with probability (1 - alpha).
    """
    domains = [s["domain"] for s in dataset.samples]
    n_sim = sum(1 for d in domains if d == "sim")
    n_real = len(domains) - n_sim

    if n_sim == 0 or n_real == 0:
        logger.warning(
            f"Co-training requested but only one domain present (sim={n_sim}, real={n_real})"
        )
        return None

    weights = [alpha / n_sim if d == "sim" else (1.0 - alpha) / n_real for d in domains]
    logger.info(
        f"Co-training sampler: α={alpha}, sim={n_sim} (w={alpha / n_sim:.2e}), "
        f"real={n_real} (w={(1 - alpha) / n_real:.2e})"
    )
    return WeightedRandomSampler(
        weights=weights, num_samples=len(domains), replacement=True
    )


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders from config."""
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["training"]

    base_dir = Path(cfg.get("_config_dir", "."))
    episodes_dir = base_dir / ds_cfg["episodes_dir"]

    # Co-training config
    co_cfg = ds_cfg.get("co_training")
    sim_episodes = co_cfg.get("sim_prefixes", []) if co_cfg else []

    common_kwargs = dict(
        episodes_dir=episodes_dir,
        episode_names=ds_cfg["episode_names"],
        chunk_size=cfg["model"]["chunk_size"],
        image_size=tuple(ds_cfg["image_size"]),
        fps=ds_cfg["fps"],
        task_description=ds_cfg.get("task_description", "Pick and place object"),
        val_ratio=ds_cfg["val_ratio"],
        image_keys=ds_cfg["image_keys"],
        action_indices=ds_cfg.get("action_indices"),
        state_indices=ds_cfg.get("state_indices"),
        chunk_return_gamma=ds_cfg.get("chunk_return_gamma", 0.99),
        video_cache_dir=ds_cfg.get("video_cache_dir"),
        sim_episodes=sim_episodes,
        transform=ds_cfg.get("augmentation", {}),
    )

    train_ds = CrabEpisodeDataset(split="train", **common_kwargs)
    val_ds = CrabEpisodeDataset(
        split="val", shared_video_cache=train_ds._video_cache, **common_kwargs
    )

    num_workers = ds_cfg.get("num_workers", 4)

    # Weighted domain sampler for co-training (train only)
    sampler = None
    use_shuffle = True
    if co_cfg and co_cfg.get("enabled", False):
        alpha = co_cfg["alpha"]
        sampler = _build_domain_sampler(train_ds, alpha)
        if sampler is not None:
            use_shuffle = False  # sampler and shuffle are mutually exclusive

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=use_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_crab_batch,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_crab_batch,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
