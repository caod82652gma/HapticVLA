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
sudo nano /etc/udev/rules.d/99-video-ground1.rules
cat /etc/udev/rules.d/99-video-ground1.rules
```
**当前系统已配置的udev规则：**
1. **99-piper-can.rules** - Piper机械臂CAN总线适配器
   - `can_master` (序列号: 003B00495246570620323934)
   - `can_follower` (序列号: 003C00455246570620323934)
   - `can_master2` (序列号: 002D001E4759530920353131)
   - `can_follower2` (序列号: 003F00214759530820353131)
2. **99-tactile.rules** - 触觉传感器串口 (CH340, 匹配VID:PID 1a86:7523)
   - `/dev/tactile_8chips` → 可插任意USB口
3. **99-video-wrist.rules** - 手腕相机
   - `/dev/video_wrist` (USB路径: pci-0000:80:14.0-usb-0:8.4.2:1.0)
4. **99-video-ground0.rules** - 地面相机0
   - `/dev/video_ground0` (USB路径: pci-0000:80:14.0-usb-0:13.4:1.0)
5. **99-video-ground1.rules** - 地面相机1
   - `/dev/video_ground1` (USB路径: pci-0000:80:14.0-usb-0:13.3:1.0)

### 配置新设备的udev规则模板

**触觉传感器（推荐用VID:PID，支持任意USB口）：**
```bash
# 8chips 型（单串口，2000000 baud）
SUBSYSTEM=="tty", KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="tactile_8chips", TAG+="uaccess"

# 如需多个CH340设备，可添加USB路径限制
# SUBSYSTEM=="tty", KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", ENV{ID_PATH}=="pci-0000:80:14.0-usb-0:13.4.4.1:1.0", SYMLINK+="tactile_8chips", TAG+="uaccess"
```

**相机（建议用USB路径，避免设备顺序变化）：**
```bash
SUBSYSTEM=="video4linux", KERNEL=="video*", ENV{ID_PATH}=="<USB_PATH>", ENV{ID_V4L_CAPABILITIES}==":capture:", SYMLINK+="video_wrist"
```

**机械臂CAN（必须用序列号，区分多个相同设备）：**
```bash
SUBSYSTEM=="net", ACTION=="add", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", ATTRS{serial}=="<SERIAL>", NAME="can_master"
```

---

## 3. 测试外设

### 3.1 相机

> 两个相机不能接在同一个扩展坞，否则读取异常。

```bash
conda activate crab
guvcview --device=/dev/video_wrist
guvcview --device=/dev/video_ground0

v4l2-ctl --device=/dev/video_wrist \
  --set-fmt-video=width=640,height=480,pixelformat=YUYV

```

### 3.2 机械臂 & 触觉传感器

**触觉传感器快速修复**（如果 /dev/tactile_8chips 未出现）：
```bash
# 停止干扰CH340的服务
sudo systemctl stop ModemManager brltty 2>/dev/null || true
sudo pkill brltty 2>/dev/null || true

# 重新触发udev规则
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty

# 验证设备
ls -la /dev/tactile_8chips
```

**激活 CAN 接口**（每次重启后执行）：
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
    --robot.port=can_follower \
    --teleop.type=piper_leader \
    --teleop.id=my_leader_arm \
    --teleop.port=can_master \
    --display_data=true
```

如需使用第二组机械臂，修改为：
```bash
    --robot.port=can_follower2 \
    --teleop.port=can_master2 \
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
bash interface_up.sh 8chips   # 或 4chips

lerobot-record \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --robot.port=can_follower \
  --teleop.type=piper_leader \
  --teleop.id=my_leader_arm \
  --teleop.port=can_master \
  --display_data=true \
  --robot.tactile_enabled=true \
  --robot.tactile.type=8chips \
  --dataset.reset_time_s=5 \
  --dataset.episode_time_s=60 \
  --dataset.num_episodes=20 \
  --dataset.repo_id=FangYuxuan/pick_place_soft_20260427_1019 \
  --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the object and place it."
```

> **多组机械臂说明**：如需使用第二组机械臂采集，改为 `--robot.port=can_follower2 --teleop.port=can_master2`

开启触觉传感器时追加：
```bash
  --robot.tactile_enabled=true \
  --robot.tactile.type=8chips \   
  # 或 4chips
```

数据保存至：`~/.cache/huggingface/lerobot/FangYuxuan/pick_place_soft_20260427_0001/`
rm -rf ~/.cache/huggingface/lerobot/FangYuxuan/pick_place_soft_20260427_0002
ls ~/.cache/huggingface/lerobot/FangYuxuan

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
| `{task}` | 任务类型 | `pick_and_place_task1` |
| `{session}` | 采集批次 | `pick_place_soft_20260427_0001` |

会话命名格式：`pick_place_{难度}_{YYYYMMDD}_{HHMM}`

### 上传（指定仓库内子路径）

lerobot 录制数据默认保存为平坦结构，上传时通过第三个参数将其放入 `{task}/{session}` 子路径：

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf upload FangYuxuan/piper-tactile-dataset \
  ~/.cache/huggingface/lerobot/FangYuxuan/pick_place_soft_20260427_1019 \
  pick_and_place_task1/pick_place_soft_20260427_1019 \
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
# 失能所有机械臂（默认）
python src/lerobot/utils/teleop_disable.py

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
  --robot.port=can_follower \
  --task="Pick up the object and place it." \
  --duration=120 \
  --action_queue_size_to_get_new_actions=30 \
  --fps=50 \
  --rtc.execution_horizon=5 \
  --device=cuda
```

> 使用第二组机械臂推理：`--robot.port=can_follower2`

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
    --robot.port=can_follower \
    --task="Pick up the object and place it." \
    --policy_type=pi05 \
    --pretrained_name_or_path=FangYuxuan/pi05_catch_banana \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

> 使用第二组机械臂：`--robot.port=can_follower2`

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
