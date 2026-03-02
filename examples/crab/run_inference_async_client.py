#!/usr/bin/env python3
"""Async inference client for Crab robot.

Connects to a PolicyServer, streams observations from the Crab robot,
receives action chunks, and executes them.

Usage:
    python run_inference_async_client.py \
        -m ~/crab_smolvla_6dof_right_arm_sim_real_cotrain/best \
        -t "Pick and place the can, medium hardness" \
        --server localhost:8080
"""

import argparse
import logging
import math
import threading

import torch

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.robots.crab import CrabClientConfig

CRAB_REMOTE_IP = "192.168.50.239"


class CrabRobotClient(RobotClient):
    """RobotClient that skips NaN actions (non-predicted joints stay put)."""

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {}
        for i, key in enumerate(self.robot.action_features):
            val = action_tensor[i].item()
            if not math.isnan(val):
                action[key] = val
        # Ensure base velocity defaults even if not predicted
        if "base_x.vel" not in action:
            action["base_x.vel"] = 0.0
        if "base_theta.vel" not in action:
            action["base_theta.vel"] = 0.0
        return action


def main():
    parser = argparse.ArgumentParser(description="Async inference client for Crab robot")
    parser.add_argument("-m", "--model", required=True, help="Path to pretrained model")
    parser.add_argument("-t", "--task", default="", help="Task description")
    parser.add_argument("--server", default="localhost:8080", help="PolicyServer address")
    parser.add_argument("--robot-ip", default=CRAB_REMOTE_IP, help="Crab robot IP")
    parser.add_argument("--device", default="cuda", help="Policy inference device (default: cuda)")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS (default: 15)")
    parser.add_argument("--actions-per-chunk", type=int, default=50, help="Actions per chunk (default: 50)")
    parser.add_argument(
        "--chunk-threshold", type=float, default=0.5, help="Chunk size threshold (default: 0.5)"
    )
    parser.add_argument(
        "--aggregate",
        default="weighted_average",
        choices=["weighted_average", "latest_only", "average", "conservative"],
        help="Aggregate function (default: weighted_average)",
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize action queue size on exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    robot_config = CrabClientConfig(remote_ip=args.robot_ip)

    config = RobotClientConfig(
        policy_type="smolvla",
        pretrained_name_or_path=args.model,
        robot=robot_config,
        actions_per_chunk=args.actions_per_chunk,
        task=args.task,
        server_address=args.server,
        policy_device=args.device,
        chunk_size_threshold=args.chunk_threshold,
        fps=args.fps,
        aggregate_fn_name=args.aggregate,
        debug_visualize_queue_size=args.visualize,
    )

    client = CrabRobotClient(config)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            client.control_loop(task=args.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            if args.visualize:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")
    else:
        logging.error("Failed to connect to policy server")


if __name__ == "__main__":
    main()
