# piper_lerobot 操作手册

[huggingface文档](http://huggingface.co/docs/lerobot)

---

# 一、4090 本地机（数据采集 & 部署）

## 1. 环境创建

### 安装 lerobot 依赖
```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install -c conda-forge ffmpeg=7.1.1 -y
pip install transformers --upgrade
git clone https://github.com/jokeru8/piper_lerobot.git
cd piper_lerobot
pip install -e .
```

### 安装 piper 依赖
```bash
pip install python-can
pip install piper_sdk
```

---

## 2. 测试相机

> 注意：两个相机不能从同一个扩展坞连接电脑，否则可能读取出问题

```bash
sudo apt install guvcview
guvcview --device=/dev/video0   # 测试 wrist 相机
guvcview --device=/dev/video2   # 测试 ground 相机
```

---

## 3. 连接机械臂

> `"1-8.2:1.0"` 根据 find_all_can_port.sh 输出的 CAN 端口号修改

```bash
conda activate lerobot
bash find_all_can_port.sh
bash can_activate.sh can_master   1000000 "1-8.2:1.0"
bash can_activate.sh can_follower 1000000 "1-8.3:1.0"
```

---

## 4. 遥操作

```bash
conda activate lerobot
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
conda activate lerobot
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

> /dev/video0 等参数改为自己对应的端口

```bash
conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com

lerobot-record \
  --robot.type=piper_follower \
  --teleop.type=piper_leader \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video2",
      "width": 640,
      "height": 480,
      "fps": 30,
      "rotation": 0
    }
  }' \
  --display_data=true \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=FangYuxuan/record_banana \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the banana and put it into the basket."
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
conda activate lerobot
python utils/teleop_disable.py
```

---

## 11. 部署策略（真机推理）

### ACT 真机测试

```bash
conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com

lerobot-record \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": -90
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video2",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    }
  }' \
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
conda activate lerobot
export HF_ENDPOINT=https://hf-mirror.com

python examples/rtc/eval_with_real_robot.py \
  --policy.path=FangYuxuan/pi05_catch_banana \
  --robot.type=piper_follower \
  --robot.cameras='{
    "wrist": {
      "type": "opencv",
      "index_or_path": "/dev/video0",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": -90
    },
    "ground": {
      "type": "opencv",
      "index_or_path": "/dev/video2",
      "width": 480,
      "height": 640,
      "fps": 30,
      "rotation": 90
    }
  }' \
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
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "width": 480, "height": 640, "fps": 30, "rotation": -90}, "ground": {"type": "opencv", "index_or_path": "/dev/video2", "width": 480, "height": 640, "fps": 30, "rotation": 90}}' \
    --task="Pick up the banana and put it into the basket." \
    --policy_type=pi05 \
    --pretrained_name_or_path=FangYuxuan/pi05_catch_banana \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

---

## 附录：Piper 系统集成总结

### 系统移植过程

如需将其他类型的机器人（如 Piper）集成到 LeRobot 框架，需要在以下几个关键位置进行修改：

#### 1. 机器人工厂集成（`src/lerobot/robots/utils.py`）

在 `make_robot_from_config()` 函数的 robot type 分支判断中添加：

```python
if robot_type == "piper_follower":
    from lerobot.robots.piper_follower import PiperFollower
    robot = PiperFollower(
        robot_type=robot_type,
        robot_id=robot_config.id,
        ...
    )
```

#### 2. 遥操作工厂集成（`src/lerobot/teleoperators/utils.py`）

在 `make_teleoperator_from_config()` 函数的 teleop type 分支判断中添加：

```python
if teleop_type == "piper_leader":
    from lerobot.teleoperators.piper_leader import PiperLeaderTeleop
    teleop = PiperLeaderTeleop(
        teleop_type=teleop_type,
        teleop_id=teleop_config.id,
        ...
    )
```

#### 3. 脚本入口导入（确保解析器能识别类型）

在以下脚本的导入部分添加 piper 模块导入：

- **`src/lerobot/scripts/lerobot_teleoperate.py`**：需要导入 `piper_leader`，使得 `--teleop.type=piper_leader` 参数能被识别
- **`src/lerobot/scripts/lerobot_record.py`**：需要导入 `piper_follower`，使得 `--robot.type=piper_follower` 参数能被识别  
- **`src/lerobot/scripts/lerobot_replay.py`**：需要导入 `piper_follower`，便于回放时接管机器人

建议在各脚本开头或配置加载之前添加：

```python
# 确保所有机器人类型被注册到配置系统
from lerobot.robots.piper_follower import PiperFollower
from lerobot.teleoperators.piper_leader import PiperLeaderTeleop
```

#### 4. 文件结构要求

完整的 Piper 集成需要以下模块文件：

```
src/lerobot/
├── robots/
│   └── piper_follower/
│       ├── __init__.py           # 导出 PiperFollower
│       ├── robot.py              # 主类实现
│       └── config.py             # 配置类（通过 @dataclass 注册）
├── motors/
│   └── piper/
│       ├── __init__.py
│       └── motor.py              # 电机控制接口
└── teleoperators/
    └── piper_leader/
        ├── __init__.py           # 导出 PiperLeaderTeleop
        ├── teleop.py             # 主类实现
        └── config.py             # 配置类
```

#### 5. 数据格式要求

- 机器人状态向量：符合 LeRobot 约定的 observation（图像、state）和 action
- 相机参数：需在 robot config 中指定 `cameras` 字段（型号、分辨率、FPS）
- 电机参数：在 motor config 中指定关节数、速度限制等

#### 6. 测试验证

移植完成后，依次验证：

```bash
# 1. 导入检查
python -c "from lerobot.robots.piper_follower import PiperFollower; from lerobot.teleoperators.piper_leader import PiperLeaderTeleop; print('✓ Imports OK')"

# 2. 配置解析
lerobot-teleoperate --help | grep piper_leader  # 应该出现在支持的类型中

# 3. 实际遥操作（需连接硬件）
lerobot-teleoperate --robot.type=piper_follower --teleop.type=piper_leader
```

#### 7. 常见问题

- **"Unknown robot type: piper_follower"**：检查是否导入了模块且配置类已注册
- **"No attribute 'piper_leader' in teleop_config"**：检查脚本中是否导入了遥操作模块
- **硬件连接失败**：检查 CAN 总线是否已激活（参见第3节"连接机械臂"）

---

---

# 二、A100 服务器（训练模型）

## 1. 环境准备

```bash
conda activate lerobot
cd /home/data/fangyuxuan/projects/research/vtla_projects/piper_lerobot
export HF_ENDPOINT=https://hf-mirror.com
```

登录 HuggingFace：
```bash
hf auth login --token hf_YOUR_TOKEN_HERE --add-to-git-credential
```

登录 wandb：
```bash
conda run -n lerobot wandb login
```

---

## 2. 训练 ACT

> GPU 6，单卡，batch_size=128，30000 步

```bash
cat > /tmp/train_act.sh << 'EOF'
cd /home/data/fangyuxuan/projects/research/vtla_projects/piper_lerobot
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=5
mkdir -p outputs
nohup conda run -n lerobot lerobot-train \
  --job_name=act_finetune_pick_banana \
  --policy.type=act \
  --dataset.repo_id=FangYuxuan/record_banana \
  --dataset.revision=main \
  --output_dir=outputs/train/act_record_banana \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=FangYuxuan/act_pick_banana \
  --policy.push_to_hub=false \
  --steps=30000 \
  --batch_size=32 \
  --num_workers=16 > outputs/act_banana.log 2>&1 &
echo "PID: $!"
EOF
bash /tmp/train_act.sh && tail -f outputs/act_banana.log
```

> 如果报 FileExistsError，先执行：`rm -rf outputs/train/act_record_banana`

---

## 3. 训练 pi05（多卡）

> GPU 3,4，双卡，batch_size=64，30000 步
> 需先安装 pi 依赖：`conda run -n lerobot pip install -e ".[pi]"`

```bash
cat > /tmp/train_pi05.sh << 'EOF'
cd /home/data/fangyuxuan/projects/research/vtla_projects/piper_lerobot
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3,4
mkdir -p outputs
nohup conda run -n lerobot python -m accelerate.commands.launch --num_processes=2 \
  src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=FangYuxuan/record_banana \
  --dataset.revision=main \
  --policy.type=pi05 \
  --output_dir=outputs/train/pi05_record_banana \
  --job_name=pi05_catch_banana \
  --policy.repo_id=FangYuxuan/pi05_catch_banana \
  --policy.pretrained_path=lerobot/pi05_libero \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --wandb.enable=true \
  --policy.dtype=bfloat16 \
  --steps=30000 \
  --policy.device=cuda \
  --batch_size=64 \
  --num_workers=16 \
  --policy.push_to_hub=false > outputs/pi05_banana.log 2>&1 &
echo "PID: $!"
EOF
bash /tmp/train_pi05.sh && tail -f outputs/pi05_banana.log
```

> 如果报 FileExistsError，先执行：`rm -rf outputs/train/pi05_record_banana`

---

## 4. 启动异步推理服务器（供 4090 客户端调用）

```bash
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 conda run -n lerobot python -m src.lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=1
```

---

## 5. 上传模型到 HuggingFace

上传 ACT 模型：
```bash
export HF_ENDPOINT=https://hf-mirror.com
hf upload FangYuxuan/act_pick_banana \
  outputs/train/act_record_banana \
  --repo-type model \
  --revision "main"
```

上传 pi05 模型：
```bash
export HF_ENDPOINT=https://hf-mirror.com
hf upload FangYuxuan/pi05_catch_banana \
  outputs/train/pi05_record_banana \
  --repo-type model \
  --revision "main"
```

---

## 6. 常用排查

| 问题 | 解决方案 |
|------|----------|
| `lerobot-train: command not found` | 用 `conda run -n lerobot lerobot-train ...` |
| `accelerate: command not found` | 用 `conda run -n lerobot python -m accelerate.commands.launch ...` |
| `RevisionNotFoundError: v3.0` | 加 `--dataset.revision=main` |
| `FileExistsError: output dir exists` | 删除旧目录 `rm -rf outputs/train/<目录名>` |
| wandb API key 错误 | `pip install wandb --upgrade` 后重新 `wandb login` |
| protobuf 版本冲突 | `pip install protobuf --upgrade` |
