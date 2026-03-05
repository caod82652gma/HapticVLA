"""
Dataset loader for Crab robot LeRobot v3.0 format.
Handles: parquet data, video frames, tactile matrices, multi-episode, multi-chunk videos.

v6: Sim-real co-training support.
    Handles missing tactile columns (returns zeros for sim data).
    Multi-chunk video support (discovers all chunks per camera automatically).
    Per-sample domain tracking ("sim"/"real") for weighted batch sampling.

v5: Reward-aware batch fields for RWFM.
v4: Disk-backed video cache using numpy memmap.
v3: Per-dataset task descriptions, action/state indices, global video indexing.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

VIDEO_CACHE_DIR = Path("/tmp/crab_video_cache")


class CrabEpisodeDataset(Dataset):
    """Dataset for Crab bimanual robot in LeRobot v3.0 format."""

    def __init__(
        self,
        episodes_dir: str | Path,
        episode_names: list[str],
        chunk_size: int = 50,
        image_size: tuple[int, int] = (256, 256),
        fps: int = 15,
        task_description: str = "Pick and place object",
        split: str = "train",
        val_ratio: float = 0.1,
        image_keys: list[str] | None = None,
        action_indices: list[int] | None = None,
        state_indices: list[int] | None = None,
        chunk_return_gamma: float | None = 0.99,
        transform=None,
        shared_video_cache: dict | None = None,
        video_cache_dir: str | Path | None = None,
        sim_episodes: list[str] | None = None,
    ):
        self.episodes_dir = Path(episodes_dir)
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.fps = fps
        self.task_description = task_description
        self.split = split
        self.transform = transform

        # Data augmentation config
        if isinstance(transform, dict):
            self.augmentation_cfg = transform
        else:
            self.augmentation_cfg = {}
        self._build_augmentation()

        if chunk_return_gamma is None:
            self.chunk_return_gamma = None
        else:
            self.chunk_return_gamma = float(chunk_return_gamma)
            if not (0.0 <= self.chunk_return_gamma <= 1.0):
                raise ValueError(
                    f"chunk_return_gamma must be in [0,1], got {self.chunk_return_gamma}"
                )
        self._discount_cache: dict[int, np.ndarray] = {}

        self._cache_dir = Path(video_cache_dir) if video_cache_dir else VIDEO_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self.action_indices = action_indices
        self.state_indices = state_indices
        if action_indices is not None:
            logger.info(
                f"Action indices: {action_indices} ({len(action_indices)} dims)"
            )
        if state_indices is not None:
            logger.info(f"State indices: {state_indices} ({len(state_indices)} dims)")

        self.image_keys = image_keys or [
            "observation.images.main_camera",
            "observation.images.left_arm_camera",
            "observation.images.right_arm_camera",
        ]

        # Sim/real domain tracking for co-training
        self._sim_episodes = set(sim_episodes or [])
        self._domain_counts: dict[str, int] = {"sim": 0, "real": 0}

        self.samples: list[dict] = []
        self.episode_boundaries: list[tuple[int, int]] = []
        self._video_cache: dict[str, np.ndarray] = (
            shared_video_cache if shared_video_cache is not None else {}
        )
        self._task_counts: dict[str, int] = {}

        for ep_name in episode_names:
            self._load_episode(self.episodes_dir / ep_name, val_ratio)

        logger.info(
            f"CrabEpisodeDataset [{split}]: {len(self.samples)} samples from {len(episode_names)} episode dirs"
        )
        for task, count in self._task_counts.items():
            logger.info(f'  task: "{task}" — {count} samples')
        if self._sim_episodes:
            logger.info(
                f"  domains: sim={self._domain_counts['sim']}, real={self._domain_counts['real']}"
            )

        self._preload_videos()

    def _build_augmentation(self):
        """Build torchvision augmentation transforms from config."""
        from torchvision import transforms as T

        aug_cfg = self.augmentation_cfg
        if not aug_cfg.get("enabled", False):
            self._augment_fn = None
            return

        transform_list = []

        cj = aug_cfg.get("color_jitter", {})
        if cj:
            transform_list.append(T.ColorJitter(
                brightness=cj.get("brightness", 0.0),
                contrast=cj.get("contrast", 0.0),
                saturation=cj.get("saturation", 0.0),
                hue=cj.get("hue", 0.0),
            ))

        rc = aug_cfg.get("random_crop", {})
        if rc.get("enabled", False):
            h, w = self.image_size
            transform_list.append(T.RandomResizedCrop(
                (h, w),
                scale=tuple(rc.get("scale", [0.9, 1.0])),
                ratio=(0.95, 1.05),
                antialias=True,
            ))

        if transform_list:
            self._augment_fn = T.Compose(transform_list)
            logger.info(f"Data augmentation enabled: {transform_list}")
        else:
            self._augment_fn = None

    def _apply_augmentations(self, images: dict) -> dict:
        """Apply same random augmentation to all camera views."""
        if self._augment_fn is None:
            return images

        import random as _random
        state = _random.getstate()
        torch_state = torch.random.get_rng_state()

        augmented = {}
        for cam_key, img_tensor in images.items():
            _random.setstate(state)
            torch.random.set_rng_state(torch_state.clone())
            augmented[cam_key] = self._augment_fn(img_tensor)

        return augmented

    def _read_task_from_parquet(self, ep_path: Path) -> str | None:
        tasks_path = ep_path / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return None
        try:
            tasks_df = pd.read_parquet(tasks_path)
            if "__index_level_0__" in tasks_df.columns:
                return str(tasks_df["__index_level_0__"].iloc[0])
            if tasks_df.index.name and len(tasks_df.index) > 0:
                return str(tasks_df.index[0])
        except Exception as e:
            logger.warning(f"Failed to read tasks.parquet from {ep_path}: {e}")
        return None

    def _safe_meta_value(self, value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value)
        return value

    def _read_episode_meta(self, ep_path: Path) -> dict:
        episodes_files = sorted((ep_path / "meta" / "episodes").rglob("*.parquet"))
        if not episodes_files:
            return {}
        try:
            meta_df = pd.concat(
                [pd.read_parquet(f) for f in episodes_files], ignore_index=True
            )
        except Exception as e:
            logger.warning(f"Failed to read meta/episodes from {ep_path}: {e}")
            return {}
        if "episode_index" not in meta_df.columns:
            return {}

        episode_meta = {}
        for _, row in meta_df.iterrows():
            ep_idx = int(row["episode_index"])
            episode_meta[ep_idx] = {
                "episode_reward": self._safe_meta_value(row["episode_reward"])
                if "episode_reward" in meta_df.columns
                else None,
                "episode_success": self._safe_meta_value(row["episode_success"])
                if "episode_success" in meta_df.columns
                else None,
                "episode_damage": self._safe_meta_value(row["episode_damage"])
                if "episode_damage" in meta_df.columns
                else None,
            }
        return episode_meta

    def _load_episode(self, ep_path: Path, val_ratio: float):
        with open(ep_path / "meta" / "info.json") as f:
            info = json.load(f)

        task = self._read_task_from_parquet(ep_path) or self.task_description
        if task == self.task_description:
            logger.warning(f'No tasks.parquet in {ep_path}, using fallback: "{task}"')
        else:
            logger.info(f'Dataset {ep_path.name}: task = "{task}"')

        data_files = sorted((ep_path / "data").rglob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
        if "reward" not in df.columns:
            df["reward"] = 0.0

        # Detect tactile columns
        has_tactile = "observation.tactile_left" in df.columns

        episode_meta_lookup = self._read_episode_meta(ep_path)

        try:
            dataset_name = ep_path.relative_to(self.episodes_dir).as_posix()
        except Exception:
            dataset_name = ep_path.name

        # Determine domain
        domain = (
            "sim"
            if any(dataset_name.startswith(s) for s in self._sim_episodes)
            else "real"
        )

        with open(ep_path / "meta" / "stats.json") as f:
            stats = json.load(f)

        episode_indices = sorted(df["episode_index"].unique())
        n_episodes = len(episode_indices)
        n_val = max(1, int(n_episodes * val_ratio))
        selected = (
            episode_indices[:-n_val]
            if self.split == "train"
            else episode_indices[-n_val:]
        )

        for ep_idx in selected:
            ep_df = (
                df[df["episode_index"] == ep_idx]
                .sort_values("frame_index")
                .reset_index(drop=True)
            )
            n_frames = len(ep_df)
            ep_rewards = ep_df["reward"].to_numpy(dtype=np.float32)
            episode_meta = episode_meta_lookup.get(int(ep_idx), {})

            self._task_counts[task] = self._task_counts.get(task, 0) + n_frames
            self._domain_counts[domain] += n_frames

            for start_frame in range(n_frames):
                self.samples.append(
                    {
                        "ep_path": ep_path,
                        "ep_df": ep_df,
                        "frame_idx": start_frame,
                        "n_frames": n_frames,
                        "info": info,
                        "stats": stats,
                        "task": task,
                        "task_name": task,
                        "dataset_name": dataset_name,
                        "domain": domain,
                        "has_tactile": has_tactile,
                        "ep_rewards": ep_rewards,
                        "episode_meta": episode_meta,
                        "episode_reward_fallback": float(ep_rewards.sum()),
                    }
                )

    def __len__(self):
        return len(self.samples)

    def _video_cache_path(self, cache_key: str) -> Path:
        h, w = self.image_size
        key = f"{cache_key}_{h}x{w}"
        return self._cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.npy"

    def _preload_videos(self):
        """Pre-decode all video files to disk-backed memmap. Handles multi-chunk videos."""
        import av
        from torchvision.transforms.functional import resize

        # Collect unique camera root directories (not individual chunk files)
        cam_roots = set()
        for sample in self.samples:
            for cam_key in self.image_keys:
                cam_roots.add(str(sample["ep_path"] / "videos" / cam_key))

        logger.info(
            f"Loading {len(cam_roots)} camera streams (disk-backed cache at {self._cache_dir})..."
        )
        n_cached = 0
        n_decoded = 0

        for cam_root_str in sorted(cam_roots):
            if cam_root_str in self._video_cache:
                continue

            cache_file = self._video_cache_path(cam_root_str)
            meta_file = cache_file.with_suffix(".json")

            if cache_file.exists() and meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                n_frames = meta["n_frames"]
                h, w = meta["height"], meta["width"]
                self._video_cache[cam_root_str] = np.memmap(
                    cache_file, dtype=np.uint8, mode="r", shape=(n_frames, 3, h, w)
                )
                n_cached += 1
            else:
                cam_root = Path(cam_root_str)
                video_files = sorted(cam_root.rglob("*.mp4"))
                if not video_files:
                    logger.warning(f"No video files under {cam_root_str}")
                    continue
                try:
                    # Count total frames first (quick probe, no full decode)
                    total_frames = 0
                    for vf in video_files:
                        container = av.open(str(vf))
                        total_frames += container.streams.video[0].frames
                        container.close()

                    h, w = self.image_size
                    mmap = np.memmap(
                        cache_file,
                        dtype=np.uint8,
                        mode="w+",
                        shape=(total_frames, 3, h, w),
                    )

                    # Decode directly into memmap (no RAM accumulation)
                    write_idx = 0
                    for vf in video_files:
                        container = av.open(str(vf))
                        for frame in container.decode(video=0):
                            img = frame.to_ndarray(format="rgb24")
                            t = torch.from_numpy(img).permute(2, 0, 1)
                            t = resize(t, list(self.image_size), antialias=True)
                            mmap[write_idx] = t.to(torch.uint8).numpy()
                            write_idx += 1
                        container.close()
                    mmap.flush()
                    del mmap

                    with open(meta_file, "w") as f:
                        json.dump(
                            {
                                "n_frames": write_idx,
                                "height": h,
                                "width": w,
                                "cam_root": cam_root_str,
                                "n_files": len(video_files),
                            },
                            f,
                        )

                    self._video_cache[cam_root_str] = np.memmap(
                        cache_file, dtype=np.uint8, mode="r", shape=(write_idx, 3, h, w)
                    )
                    n_decoded += 1
                    logger.info(
                        f"  Decoded {cam_root_str}: {len(video_files)} chunks, {write_idx} frames"
                    )
                except Exception as e:
                    logger.warning(f"Failed to decode videos from {cam_root_str}: {e}")

        logger.info(
            f"Video loading complete. {n_cached} cached, {n_decoded} decoded. {len(self._video_cache)} total."
        )

    def _get_video_frame(
        self, ep_path: Path, camera_key: str, global_index: int
    ) -> torch.Tensor:
        video_root = str(ep_path / "videos" / camera_key)
        if video_root not in self._video_cache:
            logger.warning(f"Video not in cache: {video_root}")
            return torch.zeros(3, *self.image_size, dtype=torch.float32)

        mmap = self._video_cache[video_root]
        if global_index < mmap.shape[0]:
            return torch.from_numpy(mmap[global_index].copy()).float() / 255.0
        logger.warning(f"global_index {global_index} >= video frames {mmap.shape[0]}")
        return torch.zeros(3, *self.image_size, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ep_df = sample["ep_df"]
        frame_idx = sample["frame_idx"]
        n_frames = sample["n_frames"]
        ep_path = sample["ep_path"]
        ep_rewards = sample["ep_rewards"]
        row = ep_df.iloc[frame_idx]

        # State
        state = torch.tensor(row["observation.state"], dtype=torch.float32)
        if self.state_indices is not None:
            state = state[self.state_indices]

        # Tactile — zeros for sim data (no tactile columns)
        if sample["has_tactile"]:
            tactile_left = torch.tensor(
                row["observation.tactile_left"], dtype=torch.float32
            )
            tactile_right = torch.tensor(
                row["observation.tactile_right"], dtype=torch.float32
            )
        else:
            tactile_left = torch.zeros(100, dtype=torch.float32)
            tactile_right = torch.zeros(100, dtype=torch.float32)

        # Action chunk
        action_frames = []
        for i in range(self.chunk_size):
            future_idx = min(frame_idx + i, n_frames - 1)
            action = torch.tensor(ep_df.iloc[future_idx]["action"], dtype=torch.float32)
            if self.action_indices is not None:
                action = action[self.action_indices]
            action_frames.append(action)
        actions = torch.stack(action_frames)

        # Images
        global_index = int(row["index"])
        images = {
            k: self._get_video_frame(ep_path, k, global_index) for k in self.image_keys
        }

        # Apply data augmentation (training only)
        if self.split == "train" and self._augment_fn is not None:
            images = self._apply_augmentations(images)

        # Reward fields for RWFM
        step_reward = (
            float(ep_rewards[frame_idx]) if frame_idx < len(ep_rewards) else 0.0
        )
        chunk_end = min(frame_idx + self.chunk_size, len(ep_rewards))
        future_rewards = ep_rewards[frame_idx:chunk_end]
        if future_rewards.size == 0:
            chunk_return = 0.0
        elif self.chunk_return_gamma is None or np.isclose(
            self.chunk_return_gamma, 1.0
        ):
            chunk_return = float(future_rewards.sum())
        else:
            h = future_rewards.shape[0]
            if h not in self._discount_cache:
                self._discount_cache[h] = np.power(
                    self.chunk_return_gamma, np.arange(h, dtype=np.float32)
                )
            chunk_return = float(np.sum(future_rewards * self._discount_cache[h]))

        episode_meta = sample.get("episode_meta", {})
        episode_reward = float(
            episode_meta.get("episode_reward")
            or sample.get("episode_reward_fallback", 0.0)
        )
        episode_damage = episode_meta.get("episode_damage")
        if episode_damage is None:
            episode_damage = 1.0 if "cracked" in str(ep_path).lower() else 0.0
        episode_damage = float(episode_damage)
        episode_success = episode_meta.get("episode_success")
        if episode_success is None:
            episode_success = (
                1.0 if (episode_reward > 0.0 and episode_damage < 0.5) else 0.0
            )
        episode_success = float(episode_success)

        return {
            "images": images,
            "state": state,
            "tactile_left": tactile_left,
            "tactile_right": tactile_right,
            "action": actions,
            "task": sample["task"],
            "task_name": sample["task_name"],
            "dataset_name": sample["dataset_name"],
            "domain": sample["domain"],
            "step_reward": torch.tensor(step_reward, dtype=torch.float32),
            "chunk_return": torch.tensor(chunk_return, dtype=torch.float32),
            "episode_reward": torch.tensor(episode_reward, dtype=torch.float32),
            "episode_success": torch.tensor(episode_success, dtype=torch.float32),
            "episode_damage": torch.tensor(episode_damage, dtype=torch.float32),
            "metadata": {
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "timestamp": float(row["timestamp"]),
                "index": global_index,
                "dataset_name": sample["dataset_name"],
            },
        }
