#!/bin/bash
# launch_crab.sh — One-command launcher for Crab robot inference on Orin.
# Usage:
#   ./launch_crab.sh distill                    # async, default task
#   ./launch_crab.sh rwfm_v3 -t "Pick waffle"   # async, custom task
#   ./launch_crab.sh v4 --sync                   # sync mode
#   ./launch_crab.sh --kill                      # stop everything
#   ./launch_crab.sh --list                      # list available models

set -e

CRAB_DIR="$HOME/anywhereVLA/workspaces/miilv_ws/lerobot/examples/crab"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="lerobot_vla"
SESSION="crab"

# Left arm is always v4
LEFT_MODEL="$HOME/crab_smolvla_6dof_left_arm_multitask_12v_v4/best/model.pt"
LEFT_CONFIG="$HOME/crab_smolvla_6dof_left_arm_multitask_12v_v4/config.yaml"

# Defaults
TASK="Open the carton, put the egg on the tray, soft hardness"
STEPS=10000
FPS=10
SYNC=0
RTC=0

# Model mapping
get_right_arm_paths() {
    case "$1" in
        v4)
            RIGHT_MODEL="$HOME/crab_smolvla_6dof_right_arm_multitask_12v_v4/best/model.pt"
            RIGHT_CONFIG="$HOME/crab_smolvla_6dof_right_arm_multitask_12v_v4/config.yaml"
            ;;
        rwfm_v3|rwfm)
            RIGHT_MODEL="$HOME/crab_smolvla_6dof_right_arm_rwfm_v3/best/model.pt"
            RIGHT_CONFIG="$HOME/crab_smolvla_6dof_right_arm_rwfm_v3/config.yaml"
            ;;
        distill)
            RIGHT_MODEL="$HOME/crab_smolvla_6dof_right_arm_distill_rwfm_v3/best/model.pt"
            RIGHT_CONFIG="$HOME/crab_smolvla_6dof_right_arm_distill_rwfm_v3/config.yaml"
            ;;
        sim_real|sim)
            RIGHT_MODEL="$HOME/crab_smolvla_6dof_right_arm_sim_real_cotrain/best/model.pt"
            RIGHT_CONFIG="$HOME/crab_smolvla_6dof_right_arm_sim_real_cotrain/config.yaml"
            ;;
        *)
            echo "Unknown model: $1"
            echo "Available: v4, rwfm_v3, distill, sim_real"
            exit 1
            ;;
    esac
}

# Parse args
MODEL_NAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kill)
            tmux kill-session -t "$SESSION" 2>/dev/null && echo "Stopped." || echo "Nothing running."
            exit 0
            ;;
        --list)
            echo "Available models:"
            echo "  v4        - Right arm v4 (baseline)"
            echo "  rwfm_v3   - Right arm RWFM v3 (tactile)"
            echo "  distill   - Right arm distilled from RWFM v3 (ours)"
            echo "  sim_real  - Right arm sim-real cotrain"
            echo ""
            echo "Left arm is always left_arm_v4."
            exit 0
            ;;
        --sync)
            SYNC=1
            shift
            ;;
        --rtc)
            RTC=1
            shift
            ;;
        -t)
            TASK="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        *)
            if [[ -z "$MODEL_NAME" ]]; then
                MODEL_NAME="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$MODEL_NAME" ]]; then
    echo "Usage: ./launch_crab.sh <model> [-t task] [--sync] [--rtc] [--kill] [--list]"
    echo "Run --list to see available models."
    exit 1
fi

get_right_arm_paths "$MODEL_NAME"

# Kill existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 1

ACTIVATE="source $CONDA_SH && conda activate $CONDA_ENV && cd $CRAB_DIR"

echo "=== Launching Crab Robot ==="
echo "  Right arm: $MODEL_NAME"
echo "  Left arm:  left_arm_v4"
echo "  Task:      $TASK"
echo "  Steps:     $STEPS"
echo "  FPS:       $FPS"
RTC_FLAG=""
if [[ $RTC -eq 1 ]]; then
    if [[ $SYNC -eq 1 ]]; then
        echo "ERROR: --rtc only works with async mode (not --sync)"
        exit 1
    fi
    if [[ "$MODEL_NAME" != "rwfm_v3" && "$MODEL_NAME" != "rwfm" ]]; then
        echo "ERROR: --rtc only works with rwfm_v3 model"
        exit 1
    fi
    RTC_FLAG="--rtc"
fi
echo "  Mode:      $([ $SYNC -eq 1 ] && echo 'SYNC' || echo 'ASYNC')$([ $RTC -eq 1 ] && echo ' + RTC')"
echo ""

if [[ $SYNC -eq 1 ]]; then
    # Sync mode: 3 windows (host + left + right)
    tmux new-session -d -s "$SESSION" -n host "$ACTIVATE && ./start_host.sh; read"

    tmux new-window -t "$SESSION" -n right \
        "$ACTIVATE && sleep 5 && python3 run_inference.py -m $RIGHT_MODEL -c $RIGHT_CONFIG -t \"$TASK\" --steps $STEPS; read"

    tmux new-window -t "$SESSION" -n left \
        "$ACTIVATE && sleep 5 && python3 run_inference.py -m $LEFT_MODEL -c $LEFT_CONFIG -t \"$TASK\" --steps $STEPS; read"

else
    # Async mode: 5 windows (host + 2 servers + 2 clients)
    tmux new-session -d -s "$SESSION" -n host "$ACTIVATE && ./start_host.sh; read"

    tmux new-window -t "$SESSION" -n right_srv \
        "$ACTIVATE && sleep 5 && python3 run_inference.py --async-server $RTC_FLAG -m $RIGHT_MODEL -c $RIGHT_CONFIG; read"

    tmux new-window -t "$SESSION" -n left_srv \
        "$ACTIVATE && sleep 5 && python3 run_inference.py --async-server $RTC_FLAG -m $LEFT_MODEL -c $LEFT_CONFIG --async-port 8081; read"

    tmux new-window -t "$SESSION" -n right_cli \
        "$ACTIVATE && echo 'Waiting 60s for servers to load models...' && sleep 60 && python3 run_inference.py --async-client -t \"$TASK\" --steps $STEPS --fps $FPS; read"

    tmux new-window -t "$SESSION" -n left_cli \
        "$ACTIVATE && echo 'Waiting 60s for servers to load models...' && sleep 60 && python3 run_inference.py --async-client --server-address localhost:8081 -t \"$TASK\" --steps $STEPS --fps $FPS; read"
fi

echo "tmux session '$SESSION' created. Attaching..."
tmux attach -t "$SESSION"
