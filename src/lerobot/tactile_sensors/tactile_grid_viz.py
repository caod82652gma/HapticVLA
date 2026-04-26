import argparse
import sys
import time

import numpy as np
'''
cd /home/data/fangyuxuan/projects/research/vtla_projects/crab
conda activate crab
python -m src.lerobot.tactile_sensors.tactile_grid_viz \
    --sensor-type crab \
    --port /dev/tactile_sensor

python -m src.lerobot.tactile_sensors.tactile_grid_viz \
    --sensor-type 8chips \
    --port /dev/tactile_sensor

python -m src.lerobot.tactile_sensors.tactile_grid_viz \
    --sensor-type 4chips \
    --left-port /dev/tactile_left \
    --right-port /dev/tactile_right

'''


from lerobot.tactile_sensors import (
    Tactile4ChipConfig,
    Tactile4ChipSensor,
    Tactile8ChipConfig,
    Tactile8ChipSensor,
    TactileSensor,
    TactileSensorConfig,
)


def _split_chips(matrix: np.ndarray, chips: int) -> list[np.ndarray]:
    return [matrix[:, i * 8 : (i + 1) * 8] for i in range(chips)]


def _build_sensor(args):
    if args.sensor_type == "crab":
        cfg = TactileSensorConfig(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
        )
        return TactileSensor(cfg)

    if args.sensor_type == "8chips":
        cfg = Tactile8ChipConfig(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
            header1=args.header1,
            header2=args.header2,
        )
        return Tactile8ChipSensor(cfg)

    cfg = Tactile4ChipConfig(
        left_port=args.left_port,
        right_port=args.right_port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        header1=args.header1,
        header2=args.header2,
    )
    return Tactile4ChipSensor(cfg)


def _apply_sensor_defaults(args):
    if args.sensor_type == "4chips":
        if args.left_port is None:
            args.left_port = "/dev/tactile_4chips_left"
        if args.right_port is None:
            args.right_port = "/dev/tactile_4chips_right"
        if args.baudrate is None:
            args.baudrate = 115200
    elif args.sensor_type == "8chips":
        if args.port is None:
            args.port = "/dev/tactile_8chips"
        if args.baudrate is None:
            args.baudrate = 2000000
    else:
        if args.port is None:
            args.port = "/dev/tactile_sensor"
        if args.baudrate is None:
            args.baudrate = 921600


def _build_panels(sensor_type: str, left: np.ndarray, right: np.ndarray):
    if sensor_type == "crab":
        return [left, right], ["left (10x10)", "right (10x10)"]

    if sensor_type == "8chips":
        combined = np.hstack([left, right])
        chips = _split_chips(combined, 8)
        titles = [f"chip {i}" for i in range(8)]
        return chips, titles

    left_chips = _split_chips(left, 4)
    right_chips = _split_chips(right, 4)
    chips = left_chips + right_chips
    titles = [f"left chip {i}" for i in range(4)] + [f"right chip {i}" for i in range(4)]
    return chips, titles


def main():
    parser = argparse.ArgumentParser(description="Grid visualizer for tactile sensors")
    parser.add_argument("--sensor-type", choices=["crab", "8chips", "4chips"], required=True)
    parser.add_argument("--port", default=None)
    parser.add_argument("--left-port", default=None)
    parser.add_argument("--right-port", default=None)
    parser.add_argument("--baudrate", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=0.1)
    parser.add_argument("--header1", type=lambda x: int(x, 0), default=0xAA)
    parser.add_argument("--header2", type=lambda x: int(x, 0), default=0x55)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    _apply_sensor_defaults(args)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    sensor = _build_sensor(args)
    sensor.connect()

    try:
        left, right = sensor.get_matrices()
        panels, titles = _build_panels(args.sensor_type, left, right)

        if len(panels) == 2:
            rows, cols = 1, 2
        else:
            rows, cols = 2, 4

        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
        axes = np.array(axes).reshape(-1)

        images = []
        for ax, panel, title in zip(axes, panels, titles):
            im = ax.imshow(panel, cmap="viridis", interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks(np.arange(-0.5, panel.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, panel.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.35)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            images.append(im)

        for idx in range(len(panels), len(axes)):
            axes[idx].axis("off")

        fig.suptitle(f"Tactile grid viz: {args.sensor_type}")
        plt.tight_layout()
        plt.ion()
        plt.show(block=False)

        dt = 1.0 / max(args.fps, 1e-3)
        while plt.fignum_exists(fig.number):
            start = time.time()
            left, right = sensor.get_matrices()
            panels, _ = _build_panels(args.sensor_type, left, right)

            for im, panel in zip(images, panels):
                vmax = max(float(np.max(panel)), 1.0)
                im.set_data(panel)
                im.set_clim(vmin=0.0, vmax=vmax)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        sensor.disconnect()


if __name__ == "__main__":
    main()
