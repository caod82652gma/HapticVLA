import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from crab_dataset import CrabEpisodeDataset
episode = Path(sys.argv[1]).resolve()
root = episode.parent.parent; name = episode.relative_to(root).as_posix()
ds = CrabEpisodeDataset(root, [name], chunk_size=50, image_size=(256, 256), fps=15, split="val", val_ratio=1.0)
sample = ds[0]
print("vision", {k: tuple(v.shape) for k, v in sample["images"].items()}, "tactile", tuple(sample["tactile_left"].shape), tuple(sample["tactile_right"].shape), "action", tuple(sample["action"].shape))
print("trajectory_len", int(ds.samples[0]["n_frames"]))