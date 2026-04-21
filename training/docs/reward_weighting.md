# Safety-Aware Reward Weighting (SA-RWFM)

Specification for the tactile-based safety reward and the reward-weighting
formulas used by the SA-RWFM action expert. See the [paper](../../docs/paper/)
for the derivation; this doc is the implementation reference.

## Scope

- Data scope: `pick_place_soft` (egg carton to tray, marmalade jar, waffles),
  both clean and cracked-grasp variants.
- Training mode: offline.
- Goal: reward-weighted flow matching with stable, bounded sample weights that
  penalize excessive grasping force and suboptimal trajectories.

## Weighting signal

Per-sample mixed advantage:

```
A = clip( beta * z_episode + (1 - beta) * z_chunk , -c_A, c_A )
```

- `beta = 0.7`, `c_A = 6`.
- `z_episode`: robust-normalized `episode_reward`.
- `z_chunk`: robust-normalized finite-horizon discounted chunk return from
  step rewards.

## Chunk return

- Horizon `H = 50` (matches the action chunk).
- Discount `gamma = 0.99`.
- `R_chunk(t) = sum_{k=0..H-1} gamma^k * r_{t+k}` (truncated at episode end).

## Robust normalization

Per-group (group key = task string from `meta/tasks.parquet`):

- `median_g = median(x_g)`
- `MAD_g  = median(|x_g - median_g|)`
- `scale_g = max(1.4826 * MAD_g, eps)`, `eps = 1e-6`.
- Fallback: if `scale_g < eps`, use `std`; if still degenerate, use `1.0`.
- Robust z-score: `z = (x - median_g) / scale_g`.
- Clip each of `z_episode`, `z_chunk`, and the mixed advantage `A` to
  `[-c_A, c_A]`.

## Sample weight (RWFM)

- Raw weight: `w_raw = exp(alpha * A)`, `alpha = 0.25` (calibrated 2026-02-20).
- Hard clip: `w_clip = clip(w_raw, w_min, w_max)`, `w_min = 0.25`, `w_max = 4.0`.
- Batch-mean normalization: `w = w_clip / mean_batch(w_clip)`.

`w` multiplies the per-sample flow-matching loss before reduction. A quadratic
L2 anchor regularizer against the initial IL checkpoint is added on top; see
`crab_smolvla_wrapper.py`.

## Computing the stats

Stats used by the paper-era runs:
`training/docs/reward_weighting_stats_recomputed_20260220.json`.

To recompute for a different dataset manifest:

```bash
python training/scripts/compute_reward_weighting_stats.py \
  --root <dataset-root> \
  --group-key task --horizon 50 --gamma 0.99 \
  --mix-beta 0.7 --alpha 0.25 --w-min 0.25 --w-max 4.0 \
  --batch-size 16 --sim-batches 4000
```

## Acceptance criteria

A weighting configuration is accepted if:

- `clip_high_frac <= 0.01`
- `clip_low_frac  <= 0.25`
- `ESS_p10       >= 0.65` (at the target batch composition and size)

The paper configuration (`alpha = 0.25`, `w_min = 0.25`, `w_max = 4.0`) passes
all three on both clean and cracked-controlled mixtures.
