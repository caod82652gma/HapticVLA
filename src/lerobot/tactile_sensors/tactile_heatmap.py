"""8 AD x 6 row x 8 ch heatmap viewer (FF 66 protocol).

Frame: [FF][66][rowId][AD0_CH0L][AD0_CH0H]...[AD7_CH7H][checksum]  (132 bytes)
  - 8 AD chips, 8 channels each, int16 little-endian, mv = raw * 5000 / 32768
  - ch0..5: tactile, ch6: VGND, ch7: temperature
  - checksum = XOR of bytes [3..130]
  - 6 rows complete one full sensor frame.

Usage:
    # live (needs DISPLAY)
python -m src.lerobot.tactile_sensors.tactile_heatmap

    # headless: dump latest.png + frame_NNNN.png every 2s
python -m src.lerobot.tactile_sensors.tactile_heatmap \
    --port /dev/tactile_8chips --mode snapshot --interval 2 --count 10
"""

import argparse
import os
import struct
import sys
import time

import numpy as np
import serial


FRAME_HEADER = b"\xff\x66"
FRAME_SIZE = 132
DATA_START = 3
CHECKSUM_INDEX = 131
CHECKSUM_COUNT = 128
AD_COUNT, ROW_COUNT, CH_COUNT = 8, 6, 8
CH_VGND, CH_TEMP = 6, 7
MV_SCALE = 5000.0 / 32768.0


class FrameParser:
    def __init__(self):
        self.buf = bytearray()
        self.frame = np.zeros((AD_COUNT, ROW_COUNT, CH_COUNT), dtype=np.float32)
        self.row_seen = [False] * ROW_COUNT
        self.ok = 0
        self.err = 0
        self.full = 0

    def feed(self, data: bytes) -> bool:
        if data:
            self.buf.extend(data)
        new_full = False
        while len(self.buf) >= FRAME_SIZE:
            i = self.buf.find(FRAME_HEADER)
            if i < 0:
                del self.buf[: -1]
                break
            if i > 0:
                del self.buf[:i]
                continue
            if len(self.buf) < FRAME_SIZE:
                break
            chunk = bytes(self.buf[:FRAME_SIZE])
            del self.buf[:FRAME_SIZE]
            if self._parse(chunk):
                new_full = True
        return new_full

    def _parse(self, frame: bytes) -> bool:
        row_id = frame[2]
        if row_id >= ROW_COUNT:
            return False
        cs = 0
        for b in frame[DATA_START : DATA_START + CHECKSUM_COUNT]:
            cs ^= b
        if cs != frame[CHECKSUM_INDEX]:
            self.err += 1
            return False

        values = struct.unpack_from("<64h", frame, DATA_START)
        arr = np.asarray(values, dtype=np.float32).reshape(AD_COUNT, CH_COUNT) * MV_SCALE
        self.frame[:, row_id, :6] = arr[:, :6]
        # VGND/temp broadcast to all rows of each AD
        self.frame[:, :, CH_VGND] = arr[:, CH_VGND : CH_VGND + 1]
        self.frame[:, :, CH_TEMP] = arr[:, CH_TEMP : CH_TEMP + 1]

        self.ok += 1
        if not self.row_seen[row_id]:
            self.row_seen[row_id] = True
            if all(self.row_seen):
                self.row_seen = [False] * ROW_COUNT
                self.full += 1
                return True
        return False


def chip_to_rgb(chip: np.ndarray) -> np.ndarray:
    """(6, 8) mV -> (6, 8, 3) uint8 RGB; bucket palette matches C# PageHeatmap."""
    rgb = np.empty((ROW_COUNT, CH_COUNT, 3), dtype=np.uint8)
    a = np.abs(chip)

    t, tac = a[:, :6], rgb[:, :6]
    tac[t < 21] = (100, 149, 237)
    tac[(t >= 21) & (t < 100)] = (144, 238, 144)
    tac[(t >= 100) & (t < 500)] = (255, 255, 0)
    tac[(t >= 500) & (t < 1000)] = (255, 165, 0)
    tac[t >= 1000] = (255, 99, 71)

    v, g = a[:, CH_VGND], rgb[:, CH_VGND]
    g[v < 50] = (200, 230, 200)
    g[(v >= 50) & (v < 200)] = (255, 255, 200)
    g[v >= 200] = (255, 200, 200)

    v, g = a[:, CH_TEMP], rgb[:, CH_TEMP]
    g[v < 100] = (173, 216, 230)
    g[(v >= 100) & (v < 500)] = (144, 238, 144)
    g[(v >= 500) & (v < 1000)] = (255, 218, 185)
    g[v >= 1000] = (255, 182, 193)
    return rgb


def build_figure(plt, fp: FrameParser):
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    axes = axes.reshape(-1)
    images = []
    for ad in range(AD_COUNT):
        ax = axes[ad]
        im = ax.imshow(chip_to_rgb(fp.frame[ad]), interpolation="nearest")
        ax.set_title(f"AD{ad + 1}", fontsize=10)
        ax.set_xticks(np.arange(-0.5, CH_COUNT, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, ROW_COUNT, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        images.append(im)
    fig.suptitle("Tactile heatmap | cols 0-5: tactile  col6: VGND  col7: temp")
    plt.tight_layout()
    return fig, images


def update_images(fp: FrameParser, images):
    for ad in range(AD_COUNT):
        images[ad].set_data(chip_to_rgb(fp.frame[ad]))


def resolve_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    return "live" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "snapshot"


def main():
    p = argparse.ArgumentParser(description="6x8 tactile heatmap (FF 66 protocol, 8 AD chips)")
    p.add_argument("--port", default="/dev/tactile_8chips")
    p.add_argument("--baudrate", type=int, default=2_000_000)
    p.add_argument("--timeout", type=float, default=0.05)
    p.add_argument("--mode", choices=["auto", "live", "snapshot"], default="auto")
    p.add_argument("--out-dir", default="tactile_heatmap")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between PNG snapshots")
    p.add_argument("--count", type=int, default=0, help="0 = run until Ctrl-C")
    p.add_argument("--fps", type=float, default=20.0)
    args = p.parse_args()

    mode = resolve_mode(args.mode)

    import matplotlib
    if mode == "snapshot":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mode == "live" and plt.get_backend().lower() == "agg":
        print(f"--mode live needs an interactive backend (got {plt.get_backend()}). "
              "Use --mode snapshot for headless.", file=sys.stderr)
        sys.exit(2)

    ser = serial.Serial(args.port, args.baudrate, timeout=args.timeout)
    fp = FrameParser()
    print(f"opening {args.port} @ {args.baudrate}, mode={mode}", flush=True)

    # Wait for first complete batch (6 rows) so the figure starts populated.
    while fp.full == 0:
        fp.feed(ser.read(4096))

    fig, images = build_figure(plt, fp)

    try:
        if mode == "live":
            plt.ion()
            plt.show(block=False)
            dt = 1.0 / max(args.fps, 1e-3)
            while plt.fignum_exists(fig.number):
                t0 = time.time()
                if fp.feed(ser.read(4096)):
                    update_images(fp, images)
                    fig.suptitle(f"full={fp.full}  ok={fp.ok}  err={fp.err}")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                rem = dt - (time.time() - t0)
                if rem > 0:
                    time.sleep(rem)
        else:
            os.makedirs(args.out_dir, exist_ok=True)
            print(f"snapshot -> {os.path.abspath(args.out_dir)} every {args.interval}s, "
                  f"count={args.count or 'unlimited'}, Ctrl-C to stop", flush=True)
            idx = 0
            last_save = 0.0
            while args.count <= 0 or idx < args.count:
                fp.feed(ser.read(4096))
                now = time.time()
                if now - last_save >= args.interval:
                    update_images(fp, images)
                    ts = time.strftime("%H:%M:%S")
                    fig.suptitle(f"frame {idx} ({ts})  ok={fp.ok}  err={fp.err}  full={fp.full}")
                    path = os.path.join(args.out_dir, f"frame_{idx:04d}.png")
                    fig.savefig(path, dpi=100)
                    fig.savefig(os.path.join(args.out_dir, "latest.png"), dpi=100)
                    peak = float(np.max(np.abs(fp.frame[:, :, :6])))
                    print(f"[{idx:04d}] {ts}  peak_tactile={peak:6.1f} mV  "
                          f"ok={fp.ok}  err={fp.err}  -> {path}", flush=True)
                    idx += 1
                    last_save = now
            print(f"saved {idx} frames to {args.out_dir} (latest.png is most recent)")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
