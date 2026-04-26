# piper_lerobot 操作手册

[huggingface文档](http://huggingface.co/docs/lerobot)

---

## 机器环境

| 机器 | 角色 | OS | CPU | GPU | CUDA Toolkit / Driver |
|------|------|----|----|-----|-----------------------|
| **4090 本地机** | 数据采集 & 真机部署 | Ubuntu 22.04.5 LTS (kernel 6.8.0-107) | Intel Core Ultra 7 265K, 20 核 | RTX 4090 49GB | 12.4 / 570.211.01 |
| **A100 服务器** | 模型训练 | Ubuntu 20.04.6 LTS (kernel 5.15.0-56) | AMD EPYC 7H12 × 2 socket，256 线程 | A100 80GB × 8 | 12.4 / 550.163.01 |

---

# 一、4090 本地机（数据采集 & 部署）

## 1. 环境创建

### 安装 lerobot 依赖
```bash
conda create -y -n crab python=3.10
conda activate crab
conda install -c conda-forge ffmpeg=7.1.1 -y
pip install transformers --upgrade
git clone https://github.com/caod82652gma/HapticVLA.git
cd HapticVLA
pip install -e .
```

### 安装 piper 依赖
```bash
pip install python-can
pip install piper_sdk
```

### 安装触觉传感器依赖
```bash
pip install pyserial
```

---

## 2. 设备 udev 规则配置

> 配置完成后执行以下命令生效，无需重启：
> ```bash
> sudo udevadm control --reload-rules && sudo udevadm trigger
> ```

### 2.1 相机符号链接

查找相机设备属性（以 video0 为例）：
```bash
udevadm info --name=/dev/video0 --attribute-walk | grep -E "idVendor|idProduct|serial"
```

在 `/etc/udev/rules.d/99-cameras.rules` 中添加：
```
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SN_WRIST>",  SYMLINK+="video_wrist"
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SN_GROUND>", SYMLINK+="video_ground"
```

### 2.2 CAN 总线（机械臂）

```bash
# 确认 CAN 网卡是否识别
ip link show | grep can

# 激活（每次重启后需执行）
bash can_up.sh
```

### 2.3 触觉传感器符号链接

查找串口设备属性：
```bash
ls /dev/ttyUSB* /dev/ttyACM*
udevadm info --name=/dev/ttyUSB0 --attribute-walk | grep -E "idVendor|idProduct|serial"
```

在 `/etc/udev/rules.d/99-tactile.rules` 中添加（根据实际 VID/PID/serial 填写）：
```
# crab 型（CP2102N，双 10x10，单串口）
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", ATTRS{serial}=="<SN>", SYMLINK+="tactile_sensor"

# 8chips 型（单串口）
SUBSYSTEM==" tty", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SN>", SYMLINK+="tactile_8chips"

# 4chips 型（双串口，用 serial number 区分左右）
SUBSYSTEM=="tty", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SN_LEFT>",  SYMLINK+="tactile_4chips_left"
SUBSYSTEM=="tty", ATTRS{idVendor}=="<VID>", ATTRS{idProduct}=="<PID>", ATTRS{serial}=="<SN_RIGHT>", SYMLINK+="tactile_4chips_right"
```

将当前用户加入 `dialout` 组（避免每次 sudo，重新登录后生效）：
```bash
sudo usermod -aG dialout $USER
```

验证符号链接：
```bash
ls -l /dev/video_* /dev/tactile_*
```

---

## 3. 测试外设

### 3.1 测试相机

> 注意：两个相机不能从同一个扩展坞连接电脑，否则可能读取出问题

```bash
sudo apt install guvcview
guvcview --device=/dev/video_wrist   # 测试 wrist 相机
guvcview --device=/dev/video_ground  # 测试 ground 相机
```

### 3.2 连接机械臂

```bash
conda activate crab
bash can_up.sh
```

### 3.3 测试触觉传感器

串口接好并配置 udev 规则后，用可视化工具验证帧率、左右映射及丢帧情况：

```bash
conda activate crab

# crab 型（默认 /dev/tactile_sensor，921600 baud）
python -m src.lerobot.tactile_sensors.tactile_grid_viz --sensor-type crab

# 8chips 型（默认 /dev/tactile_8chips，2000000 baud）
python -m src.lerobot.tactile_sensors.tactile_grid_viz --sensor-type 8chips

# 4chips 型（默认 /dev/tactile_4chips_left + /dev/tactile_4chips_right，115200 baud）
python -m src.lerobot.tactile_sensors.tactile_grid_viz --sensor-type 4chips

# 覆盖默认端口（可选）
python -m src.lerobot.tactile_sensors.tactile_grid_viz \
    --sensor-type crab \
    --port /dev/ttyUSB0
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
```

验证登录：
```bash
HF_USER=$(hf auth whoami | head -n 1)
echo $HF_USER
```

---

## 6. 采集数据集

> 相机已在 `PIPERFollowerConfig` 中预配置（`/dev/video_wrist`、`/dev/video_ground`），无需手动传 `--robot.cameras`。

### 6.1 不带触觉（默认）

`PIPERFollowerConfig` 默认 `tactile_enabled=false`，直接运行：

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

lerobot-record \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --teleop.type=piper_leader \
  --teleop.id=my_leader_arm \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=FangYuxuan/record_banana \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the banana and put it into the basket."
```

### 6.2 带触觉

触觉传感器验证通过后，通过命令行参数开启：

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

# crab 型触觉（双 10x10，/dev/tactile_sensor）
lerobot-record \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --teleop.type=piper_leader \
  --teleop.id=my_leader_arm \
  --robot.tactile_enabled=true \
  --robot.tactile.type=crab \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=FangYuxuan/record_banana_tactile \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the banana and put it into the basket."

# 8chips 型触觉（/dev/tactile_8chips）
# 将上面 --robot.tactile.type=crab 改为 --robot.tactile.type=8chips

# 4chips 型触觉（/dev/tactile_4chips_left + right）
# 将上面 --robot.tactile.type=crab 改为 --robot.tactile.type=4chips
```

数据保存在：`~/.cache/huggingface/lerobot/FangYuxuan/record_banana`

### 可选参数
```
--dataset.episode_time_s=60   每个 episode 的持续时间（默认60秒），可提前结束
--dataset.reset_time_s=60     每 episode 之后重置环境的时长（默认60秒）
--dataset.num_episodes=50     记录的总 episode 数（默认50）
```

### 键盘快捷键
| 按键 | 功能 |
|------|------|
| → 右箭头 | 提前结束当前 episode，进入下一个 |
| ← 左箭头 | 取消当前 episode，重新录制 |
| ESC | 立即停止，编码视频 |

---

## 7. 合并数据集

```bash
lerobot-edit-dataset \
  --repo_id FangYuxuan/pick_and_place \
  --operation.type merge \
  --operation.repo_ids "['FangYuxuan/record_apple', 'FangYuxuan/record_banana']" \
  --push_to_hub true
```

---

## 8. 可视化数据集

```bash
python src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id FangYuxuan/record_banana \
    --episode-index 0
```

或直接用 VLC 看视频文件：
```bash
vlc *.mp4
```

---

## 9. 上传数据集到 HuggingFace

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf upload FangYuxuan/record_banana \
  ~/.cache/huggingface/lerobot/FangYuxuan/record_banana \
  --repo-type dataset \
  --revision "main"
```

---

## 10. 全部失能

```bash
conda activate crab
python utils/teleop_disable.py
```

---

## 11. 部署策略（真机推理）

### ACT 真机测试

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

lerobot-record \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --display_data=true \
  --dataset.repo_id=FangYuxuan/eval_act_banana \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=120 \
  --dataset.push_to_hub=false \
  --dataset.single_task="Pick up the banana and put it into the basket." \
  --policy.path=FangYuxuan/act_pick_banana
```

### pi05 真机推理（RTC）

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

python examples/rtc/eval_with_real_robot.py \
  --policy.path=FangYuxuan/pi05_catch_banana \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --task="Pick up the banana and put it into the basket." \
  --duration=120 \
  --action_queue_size_to_get_new_actions=30 \
  --fps=50 \
  --rtc.execution_horizon=5 \
  --device=cuda
```

### 异步推理（本地显存不够，模型跑在 A100）

安装依赖：
```bash
pip install -e ".[async]"
```

**在 4090 本地建立 SSH 端口转发：**
```bash
ssh -L 8080:127.0.0.1:8080 fangyuxuan@<A100服务器地址> -N
# 验证
nc -zv 127.0.0.1 8080
```

**在 4090 本地启动客户端：**
```bash
python -m src.lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=piper_follower \
    --robot.id=my_follower_arm \
    --task="Pick up the banana and put it into the basket." \
    --policy_type=pi05 \
    --pretrained_name_or_path=FangYuxuan/pi05_catch_banana \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

---

# 二、A100 服务器（模型训练）

**机器规格：** Ubuntu 20.04.6 LTS｜AMD EPYC 7H12 × 2 socket（256 线程）｜A100 80GB × 8｜CUDA Toolkit 12.4 / Driver 550.163.01

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

huggingface-cli download FangYuxuan/record_banana \
  --repo-type dataset \
  --local-dir ~/.cache/huggingface/lerobot/FangYuxuan/record_banana
```

---

## 3. 训练 ACT

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

python -m lerobot.scripts.train \
  --policy.type=act \
  --dataset.repo_id=FangYuxuan/record_banana \
  --output_dir=outputs/act_banana \
  --device=cuda \
  --wandb.enable=true
```

多卡训练（torchrun）：
```bash
torchrun --nproc_per_node=4 -m lerobot.scripts.train \
  --policy.type=act \
  --dataset.repo_id=FangYuxuan/record_banana \
  --output_dir=outputs/act_banana \
  --device=cuda \
  --wandb.enable=true
```

---

## 4. 上传模型到 HuggingFace

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf upload FangYuxuan/act_pick_banana \
  outputs/act_banana/checkpoints/last/pretrained_model \
  --repo-type model \
  --revision "main"
```

---

## 5. 异步推理服务端（配合 4090 真机推理）

```bash
conda activate crab
export HF_ENDPOINT=https://hf-mirror.com

python -m src.lerobot.async_inference.policy_server \
    --port=8080 \
    --policy_type=pi05 \
    --pretrained_name_or_path=FangYuxuan/pi05_catch_banana \
    --device=cuda \
    --gpu_id=3
```
