# piper_lerobot 操作手册

[HuggingFace lerobot 文档](http://huggingface.co/docs/lerobot)

## 机器环境

| 机器 | 角色 | OS | GPU | CUDA |
|------|------|----|-----|------|
| **4090 本地机** | 数据采集 & 真机部署 | Ubuntu 22.04.5 LTS | RTX 4090 49GB | 12.4 / 570.211.01 |
| **A100 服务器** | 模型训练 | Ubuntu 20.04.6 LTS | A100 80GB × 8 | 12.4 / 550.163.01 |

---

# 一、4090 本地机（数据采集 & 部署）

## 1. 环境创建

```bash
conda create -y -n crab python=3.10
conda activate crab
conda install -c conda-forge ffmpeg=7.1.1 -y
pip install transformers --upgrade
git clone https://github.com/caod82652gma/HapticVLA.git
cd HapticVLA
pip install -e .
pip install python-can piper_sdk pyserial
```

---

## 2. 设备 udev 规则配置

配置完成后执行以下命令生效，无需重启：
```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 2.1 查找设备属性

```bash
# 相机
udevadm info --name=/dev/video0 --attribute-walk | grep -E "idVendor|idProduct|serial"

# 触觉串口（CH340）
ls /dev/ttyUSB*
udevadm info --name=/dev/ttyUSB0 --attribute-walk | grep -E "idVendor|idProduct|serial"
```

### 2.2 udev 规则文件

```bash
sudo nano /etc/udev/rules.d/99-tactile.rules
```

写入（根据实际 VID/PID/serial 填写）：
```
# 8chips 型（单串口，2000000 baud）
SUBSYSTEM=="tty", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SN>", SYMLINK+="tactile_8chips", TAG+="uaccess"

# 4chips 型（双串口，用 serial number 区分左右，115200 baud）
SUBSYSTEM=="tty", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SN_LEFT>",  SYMLINK+="tactile_4chips_left",  TAG+="uaccess"
SUBSYSTEM=="tty", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SN_RIGHT>", SYMLINK+="tactile_4chips_right", TAG+="uaccess"
```

---

## 3. 测试外设

### 3.1 相机

> 两个相机不能接在同一个扩展坞，否则读取异常。

```bash
conda activate crab
guvcview --device=/dev/video_wrist
guvcview --device=/dev/video_ground0
```

### 3.2 机械臂 & 触觉传感器
触觉传感器快速修复
```bash
lsusb | grep -i 1a86:7523

sudo systemctl stop ModemManager brltty brltty-udev.service 2>/dev/null || true
sudo pkill brltty 2>/dev/null || true

IFACE=$(for n in /sys/bus/usb/devices/*; do
  [ -f "$n/idVendor" ] || continue
  [ "$(cat "$n/idVendor" 2>/dev/null):$(cat "$n/idProduct" 2>/dev/null)" = "1a86:7523" ] || continue
  for i in "$n":*; do
    [ -d "$i" ] || continue
    basename "$i"
    break
  done
done | head -n1)

echo "IFACE=$IFACE"

if [ -L "/sys/bus/usb/devices/$IFACE/driver" ]; then
  CUR=$(basename "$(readlink -f "/sys/bus/usb/devices/$IFACE/driver")")
  [ "$CUR" = "usbfs" ] && echo "$IFACE" | sudo tee /sys/bus/usb/drivers/usbfs/unbind
fi

echo "$IFACE" | sudo tee /sys/bus/usb/drivers/ch341/bind

bash interface_up.sh 8chips
```

激活 CAN 接口（每次重启后执行）：
```bash
bash interface_up.sh 8chips   # 或 4chips
```

验证触觉传感器（可视化帧率、左右映射、丢帧）：
```bash
conda activate crab

# 8chips 型（/dev/tactile_8chips，2000000 baud）
python -m src.lerobot.tactile_sensors.tactile_heatmap

# 4chips 型（/dev/tactile_4chips_left + right，115200 baud）
python -m src.lerobot.tactile_sensors.tactile_grid_viz --sensor-type 4chips
```

---

## 4. 遥操作

```bash
conda activate crab
lerobot-teleoperate \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --teleop.type=piper_leader \
    --teleop.id=my_leader_arm \
    --display_data=true
```

---

## 5. 登录 HuggingFace

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token hf_YOUR_TOKEN_HERE --add-to-git-credential
# 验证
hf auth whoami
```

---

## 6. 采集数据集

相机已在 `PIPERFollowerConfig` 中预配置（`/dev/video_wrist`、`/dev/video_ground0`），无需额外传参。

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

lerobot-record \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --teleop.type=piper_leader \
  --teleop.id=my_leader_arm \
  --display_data=true \
  --robot.tactile_enabled=true \
  --robot.tactile.type=8chips \   
  --dataset.reset_time_s=5 \
  --dataset.episode_time_s=60 \
  --dataset.num_episodes=20 \
  --dataset.repo_id=FangYuxuan/record_test003 \
  --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the object and place it."
```

开启触觉传感器时追加：
```bash
  --robot.tactile_enabled=true \
  --robot.tactile.type=8chips \   
  # 或 4chips
```

数据保存至：`~/.cache/huggingface/lerobot/FangYuxuan/record_test003/`

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| → 右箭头 | 提前结束当前 episode，进入下一个 |
| ← 左箭头 | 取消当前 episode，重新录制 |
| ESC | 立即停止并编码视频 |

---

## 7. 上传数据集到 HuggingFace（含训练目录结构）

crab 训练流水线通过 `episodes_dir / {task}/{session}` 定位数据。**HuggingFace 仓库名不会出现在本地路径中**，只有仓库内部的两级子路径映射到本地。

命名规则：

| 层级 | 含义 | 示例 |
|------|------|------|
| `{task}` | 任务类型 | `pick_and_place_piper` |
| `{session}` | 采集批次 | `pick_place_soft_20260427_1430` |

会话命名格式：`pick_place_{难度}_{YYYYMMDD}_{HHMM}`

### 上传（指定仓库内子路径）

lerobot 录制数据默认保存为平坦结构，上传时通过第三个参数将其放入 `{task}/{session}` 子路径：

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf upload FangYuxuan/piper-tactile-dataset \
  ~/.cache/huggingface/lerobot/FangYuxuan/record_test003 \
  pick_and_place_piper/pick_place_soft_20260427_1430 \
  --repo-type dataset
```

追加更多批次时修改本地路径和 `{task}/{session}` 名重复执行。

上传后 HuggingFace 仓库结构：
```
FangYuxuan/piper-tactile-dataset
└── pick_and_place_piper/
    ├── pick_place_soft_20260427_1430/
    │   ├── data/chunk-000/file-000.parquet
    │   ├── meta/info.json, stats.json, tasks.parquet, episodes/
    │   └── videos/observation.images.main_camera/
    │              observation.images.wrist_camera/
    └── pick_place_soft_20260428_1000/
        └── ...
```

---

## 8. 全部失能

```bash
conda activate crab
python utils/teleop_disable.py
```

---

## 9. 部署策略（真机推理）

### pi05 真机推理（RTC）

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

python examples/rtc/eval_with_real_robot.py \
  --policy.path=FangYuxuan/pi05_catch_banana \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --task="Pick up the object and place it." \
  --duration=120 \
  --action_queue_size_to_get_new_actions=30 \
  --fps=50 \
  --rtc.execution_horizon=5 \
  --device=cuda
```

### 异步推理（模型跑在 A100）

在 4090 建立 SSH 端口转发：
```bash
ssh -L 8080:127.0.0.1:8080 fangyuxuan@<A100地址> -N
```

启动客户端：
```bash
conda activate crab
python -m src.lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --task="Pick up the object and place it." \
    --policy_type=pi05 \
    --pretrained_name_or_path=FangYuxuan/pi05_catch_banana \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

---

# 二、A100 服务器（模型训练）

## 1. 环境创建

```bash
conda create -y -n crab python=3.10
conda activate crab
conda install -c conda-forge ffmpeg=7.1.1 -y
pip install transformers --upgrade
git clone https://github.com/caod82652gma/HapticVLA.git
cd HapticVLA
pip install -e ".[train]"
```

---

## 2. 下载数据集

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download FangYuxuan/piper-tactile-dataset \
  --repo-type dataset \
  --local-dir /path/to/datasets
```

下载后本地结构：
```
/path/to/datasets/                     ← episodes_dir（填入训练 YAML）
└── pick_and_place_piper/
    ├── pick_place_soft_20260427_1430/ ← data/ meta/ videos/
    └── pick_place_soft_20260428_1000/
```

---

## 3. Crab 自定义训练

训练配置文件位于 `training/configs/`，参考现有 YAML 新建：

```yaml
# training/configs/train_piper_7dof_tactile.yaml
experiment:
  name: "piper_smolvla_7dof_tactile_v1"
  output_dir: "outputs/piper_smolvla_7dof_tactile_v1"

model:
  pretrained_path: "lerobot/smolvla_base"
  chunk_size: 50
  action_dim: 7
  state_dim: 7
  tactile_encoder:
    enabled: true

dataset:
  episodes_dir: "/path/to/datasets"
  episode_names:
    - "pick_and_place_piper/pick_place_soft_20260427_1430"
    - "pick_and_place_piper/pick_place_soft_20260428_1000"
  image_keys:
    - "observation.images.main_camera"
    - "observation.images.wrist_camera"
  action_indices: [0, 1, 2, 3, 4, 5, 6]
  state_indices:  [0, 1, 2, 3, 4, 5, 6]
  fps: 30
  image_size: [256, 256]
  val_ratio: 0.1
  num_workers: 4

training:
  batch_size: 8
  steps: 50000
  learning_rate: 1.0e-4
```

启动训练：
```bash
conda activate crab
cd training
python train.py --config configs/train_piper_7dof_tactile.yaml
```

---

## 4. 上传模型到 HuggingFace

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf upload FangYuxuan/piper_smolvla_7dof_tactile_v1 \
  outputs/piper_smolvla_7dof_tactile_v1/checkpoints/last/pretrained_model \
  --repo-type model
```

---

## 5. 异步推理服务端（配合 4090 真机）

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

python -m src.lerobot.async_inference.policy_server \
    --port=8080 \
    --policy_type=pi05 \
    --pretrained_name_or_path=FangYuxuan/pi05_catch_banana \
    --device=cuda \
    --gpu_id=0
```
