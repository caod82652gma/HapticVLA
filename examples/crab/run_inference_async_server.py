#!/usr/bin/env python3
"""Async inference server for Crab robot.

Subclasses PolicyServer to load our custom CrabSmolVLAWrapper
(build_model + torch.load) instead of from_pretrained().

Usage:
    python run_inference_async_server.py \
        -m ~/crab_smolvla_6dof_right_arm_sim_real_cotrain/best/model.pt \
        -c ~/crab_smolvla_6dof_right_arm_sim_real_cotrain/config.yaml

    python run_inference_async_server.py \
        -m ~/crab_smolvla_6dof_right_arm_multitask_12v_v3/best/model.pt \
        -c ~/crab_smolvla_6dof_right_arm_multitask_12v_v3/config.yaml
"""

import argparse
import base64
import logging
import pickle  # nosec
import sys
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import torch
from torchvision.transforms.functional import resize

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import (
    TimedAction,
    TimedObservation,
    get_logger,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)

logger = logging.getLogger(__name__)

# ---- Constants from run_inference.py ----
CAMERA_KEYS = ["main_camera", "left_arm_camera", "right_arm_camera"]
CAMERA_DS_KEYS = [f"observation.images.{k}" for k in CAMERA_KEYS]

STATE_KEYS = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
    "base_x.vel",
    "base_theta.vel",
]

# Add training package to path (same approach as run_inference.py)
TRAINING_PKG = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(TRAINING_PKG))


def load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, checkpoint_path: str, device: torch.device):
    """Build CrabSmolVLAWrapper and load trained weights (same as run_inference.py)."""
    from training.crab_smolvla_wrapper import build_model

    logger.info("Building model...")
    model = build_model(cfg)

    max_action_dim = cfg["model"].get("max_action_dim", 32)
    max_state_dim = cfg["model"].get("max_state_dim", 32)
    smolvla_cfg = model.smolvla.config
    smolvla_cfg.output_features["action"].shape = (max_action_dim,)
    smolvla_cfg.input_features["observation.state"].shape = (max_state_dim,)

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Warmup
    action_dim = cfg["model"].get("action_dim", 14)
    state_indices = cfg["dataset"].get("state_indices", None)
    state_dim = len(state_indices) if state_indices else 14
    image_size = tuple(cfg["dataset"]["image_size"])
    chunk_size = cfg["model"]["chunk_size"]

    logger.info("Warming up model...")
    dummy = {
        "images": {k: torch.randn(1, 3, *image_size, device=device) for k in CAMERA_DS_KEYS},
        "state": torch.randn(1, state_dim, device=device),
        "tactile_left": torch.randn(1, 100, device=device),
        "tactile_right": torch.randn(1, 100, device=device),
        "action": torch.zeros(1, chunk_size, action_dim, device=device),
        "task": ["warmup"],
    }
    for _ in range(3):
        with torch.no_grad(), torch.cuda.amp.autocast():
            model.predict_action(dummy)
    logger.info("Model ready")
    return model


def obs_to_batch(
    obs: dict,
    task: str,
    image_size: tuple,
    chunk_size: int,
    device: torch.device,
    state_indices: list | None = None,
    action_dim: int = 14,
):
    """Convert raw robot observation to model batch dict (from run_inference.py)."""
    if state_indices:
        state = torch.tensor([obs.get(STATE_KEYS[i], 0.0) for i in state_indices], dtype=torch.float32)
    else:
        state = torch.tensor([obs.get(k, 0.0) for k in STATE_KEYS], dtype=torch.float32)

    images = {}
    for cam_key, ds_key in zip(CAMERA_KEYS, CAMERA_DS_KEYS, strict=True):
        img = obs.get(cam_key)
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        if isinstance(img, np.ndarray):
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        else:
            img_t = img.permute(2, 0, 1).float() / 255.0
        img_t = resize(img_t, list(image_size), antialias=True)
        images[ds_key] = img_t.unsqueeze(0)

    tactile_left = obs.get("tactile_left")
    if tactile_left is None:
        tactile_left = np.zeros(100, dtype=np.float32)
    elif isinstance(tactile_left, str):
        raw_bytes = base64.b64decode(tactile_left)
        tactile_left = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.float32)
    else:
        tactile_left = np.asarray(tactile_left, dtype=np.float32).flatten()

    tactile_right = obs.get("tactile_right")
    if tactile_right is None:
        tactile_right = np.zeros(100, dtype=np.float32)
    elif isinstance(tactile_right, str):
        raw_bytes = base64.b64decode(tactile_right)
        tactile_right = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.float32)
    else:
        tactile_right = np.asarray(tactile_right, dtype=np.float32).flatten()

    dummy_action = torch.zeros(1, chunk_size, action_dim, dtype=torch.float32)

    return {
        "images": {k: v.to(device) for k, v in images.items()},
        "state": state.unsqueeze(0).to(device),
        "tactile_left": torch.from_numpy(tactile_left).unsqueeze(0).to(device),
        "tactile_right": torch.from_numpy(tactile_right).unsqueeze(0).to(device),
        "action": dummy_action.to(device),
        "task": [task],
    }


class CrabPolicyServer(PolicyServer):
    """PolicyServer that loads CrabSmolVLAWrapper instead of using from_pretrained."""

    def __init__(self, config: PolicyServerConfig, model, model_cfg: dict, rtc_enabled: bool = False):
        super().__init__(config)
        self.crab_model = model
        self.model_cfg = model_cfg
        self.rtc_enabled = rtc_enabled
        self._prev_chunk_left_over = None
        self._last_inference_time = 0.0
        self._last_chunk_timestamp = 0.0
        self.image_size = tuple(model_cfg["dataset"]["image_size"])
        self.chunk_size = model_cfg["model"]["chunk_size"]
        self.action_dim = model_cfg["model"].get("action_dim", 14)
        self.state_indices = model_cfg["dataset"].get("state_indices", None)
        self.action_indices = model_cfg["dataset"].get("action_indices", None)
        self.robot_action_dim = 14  # Crab robot: 12 arm + 2 base
        self.crab_logger = get_logger("crab_server")

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Accept client metadata but skip from_pretrained — model is pre-loaded."""
        if not self.running:
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.device = policy_specs.device

        self.crab_logger.info(
            f"Client {client_id} connected | "
            f"Actions/chunk: {self.actions_per_chunk} | "
            f"Device: {self.device} | "
            f"Model pre-loaded (skipping from_pretrained)"
        )
        return services_pb2.Empty()

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Run inference using CrabSmolVLAWrapper pipeline (same as run_inference.py).

        When RTC is enabled, passes inference_delay and prev_chunk_left_over
        to predict_action_chunk for prefix guidance.
        """
        raw_obs = observation_t.get_observation()
        task = raw_obs.get("task", "")

        start = time.perf_counter()
        batch = obs_to_batch(
            raw_obs,
            task,
            self.image_size,
            self.chunk_size,
            self.device,
            state_indices=self.state_indices,
            action_dim=self.action_dim,
        )
        prep_time = time.perf_counter() - start

        inf_start = time.perf_counter()
        smolvla_batch = self.crab_model.prepare_batch_for_smolvla(batch)

        if self.rtc_enabled:
            # RTC needs gradients for prefix guidance (torch.enable_grad inside denoise_step)
            # Compute inference delay from measured latency
            fps = self.config.fps or 15
            inference_delay = max(1, int(round(self._last_inference_time * fps)))

            with torch.cuda.amp.autocast():
                action = self.crab_model.smolvla.predict_action_chunk(
                    smolvla_batch,
                    inference_delay=inference_delay,
                    prev_chunk_left_over=self._prev_chunk_left_over,
                )
        else:
            with torch.no_grad(), torch.cuda.amp.autocast():
                action = self.crab_model.smolvla.predict_action_chunk(smolvla_batch)

        inf_time = time.perf_counter() - inf_start
        self._last_inference_time = inf_time

        # action: [B, chunk_size, max_action_dim] -> crop to action_dim, trim, squeeze
        if action.shape[-1] > self.action_dim:
            action = action[..., : self.action_dim]

        # For RTC: store full action chunk as leftover for next call's prefix guidance
        if self.rtc_enabled:
            now = time.perf_counter()
            if self._last_chunk_timestamp > 0:
                elapsed = now - self._last_chunk_timestamp
                fps = self.config.fps or 15
                consumed = min(int(elapsed * fps), self.chunk_size)
            else:
                consumed = 0
            self._last_chunk_timestamp = now

            # Store unconsumed portion of current chunk for next RTC guidance
            full_action = action.squeeze(0)  # [chunk_size, action_dim]
            if consumed < full_action.shape[0]:
                self._prev_chunk_left_over = full_action[consumed:].clone().detach()
            else:
                self._prev_chunk_left_over = None

        action = action[:, : self.actions_per_chunk, :].squeeze(0).float()

        # Pad partial actions (e.g. 6-DOF right arm) to full robot action space (14-DOF)
        # Non-predicted joints are NaN — client skips them (no command = arm stays put)
        if self.action_indices and action.shape[-1] < self.robot_action_dim:
            full = torch.full((action.shape[0], self.robot_action_dim), float("nan"), device=action.device)
            for model_idx, robot_idx in enumerate(self.action_indices):
                full[:, robot_idx] = action[:, model_idx]
            action = full

        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action), observation_t.get_timestep()
        )

        rtc_info = ""
        if self.rtc_enabled:
            fps = self.config.fps or 15
            delay = max(1, int(round(self._last_inference_time * fps)))
            has_prev = self._prev_chunk_left_over is not None
            rtc_info = f" | rtc_delay={delay} prev={'yes' if has_prev else 'no'}"

        self.crab_logger.info(
            f"Obs #{observation_t.get_timestep()} | "
            f"prep={prep_time * 1000:.0f}ms | "
            f"inference={inf_time * 1000:.0f}ms | "
            f"actions={len(action_chunk)}{rtc_info}"
        )
        return action_chunk


def main():
    parser = argparse.ArgumentParser(description="Async inference server for Crab robot")
    parser.add_argument("-m", "--model", required=True, help="Path to model.pt checkpoint")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS (default: 15)")
    parser.add_argument("--rtc", action="store_true",
                        help="Enable Real-Time Chunking")
    parser.add_argument("--rtc-horizon", type=int, default=10,
                        help="RTC execution horizon (default: 10)")
    parser.add_argument("--rtc-guidance", type=float, default=10.0,
                        help="RTC max guidance weight (default: 10.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model (same as run_inference.py)
    cfg = load_config(args.config)
    model = load_model(cfg, args.model, device)

    # RTC initialization
    if args.rtc:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.configs.types import RTCAttentionSchedule
        rtc_cfg = RTCConfig(
            enabled=True,
            execution_horizon=args.rtc_horizon,
            max_guidance_weight=args.rtc_guidance,
            prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        )
        model.smolvla.config.rtc_config = rtc_cfg
        model.smolvla.init_rtc_processor()
        logger.info(f"RTC enabled: horizon={args.rtc_horizon}, guidance={args.rtc_guidance}")

    # Start gRPC server with custom PolicyServer
    server_config = PolicyServerConfig(host=args.host, port=args.port, fps=args.fps)
    policy_server = CrabPolicyServer(server_config, model, cfg, rtc_enabled=args.rtc)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{server_config.host}:{server_config.port}")

    policy_server.crab_logger.info(f"CrabPolicyServer started on {server_config.host}:{server_config.port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
