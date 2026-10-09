# R1-A7、LeRobot 与 PI0.5 运维流程

本文描述队友训练服务器与 `admi` 本机的 LeRobot 环境、R1-A7 四相机数据、PI0.5 全参数微调和模型交付边界。所有命令均从 `r1a7-system` 仓库根目录执行。路径必须先按机器角色选择，不能把队友服务器的 `/home/jieguo` 或 `/media/jieguo` 路径直接复制到本机运行。

> 2026-08-27契约变更：模型已移除未示教的腰部自由度，当前训练和推理契约为
> 16维（双臂14维、双夹爪2维）。本文中标为“旧17维基线”的数据、4K/8K结果和
> 数值仅作历史实验记录，不能用于当前16维真机入口。

## 1. 机器配置与固定契约

### 1.1 队友训练服务器

| 项目 | 约定 |
| --- | --- |
| 旧环境 | `/home/jieguo/anaconda3/envs/unitree_lerobot` |
| 旧源码 | `data/unitree-lerobot/unitree_lerobot/lerobot` |
| 新环境 | `/home/jieguo/anaconda3/envs/lerobot` |
| 新源码 | `/media/jieguo/disk/zyc/lerobot` |
| 新版 R1 插件 | `/media/jieguo/disk/zyc/lerobot/integrations/lerobot_robot_unitree_r1a7` |
| 原始 LeRobot 数据 | `/home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30` |
| PI0.5 训练数据 | `/home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30_pi05_stats` |
| PI0.5 base | `/media/jieguo/disk/zyc/lerobot/pretrained/pi05_base` |
| PaliGemma tokenizer | `/media/jieguo/disk/zyc/lerobot/pretrained/google/paligemma-3b-pt-224` |

上述路径只对应队友的双 4090 训练服务器。

### 1.2 `admi` 本机离线推理

本机当前配置为：

| 项目 | 本机路径 | 2026-08-26 状态 |
| --- | --- | --- |
| 仓库 | `/home/user/r1-a7-autonomous-checkout` | 已存在，`main` 已包含 `pi05_deploy` 合并内容 |
| LeRobot Python | `/home/user/miniconda3/envs/lerobot/bin/python` | 已存在，LeRobot 0.4.1 |
| vendored LeRobot | `data/unitree-lerobot/unitree_lerobot/lerobot` | 已存在 |
| 旧17维四相机数据 | `/home/user/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30` | 已存在，约 531 MB；仅作历史对照 |
| PI0.5 派生训练数据 | `/home/user/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30_pi05_stats` | 缺失；离线推理可先使用原始数据，继续训练前必须补齐 |
| 旧17维 legacy 权重 | `data/unitree-lerobot/unitree_lerobot/lerobot/outputs/train/r1a7_pi05_single_gpu_bs32_20260826-010706/checkpoints/008000/pretrained_model_legacy` | 约 8.3 GB；仅作历史对照，不用于当前入口 |
| PaliGemma tokenizer | `/home/user/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c` | 已存在，脚本通过 `refs/main` 自动解析，不复制资产 |

本机路径可在当前终端统一定义：

```bash
export R1A7_ROOT=/home/user/r1-a7-autonomous-checkout
export CONDA_ROOT=/home/user/miniconda3
export R1A7_PI05_SOURCE_DATASET=/home/user/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30
export R1A7_PI05_DATASET_ROOT=/home/user/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30_pi05_stats
export R1A7_PI05_LEGACY_CHECKPOINT="$R1A7_ROOT/data/unitree-lerobot/unitree_lerobot/lerobot/outputs/train/r1a7_pi05_single_gpu_bs32_20260826-010706/checkpoints/008000/pretrained_model_legacy"
export R1A7_PI05_TOKENIZER=/home/user/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c
cd "$R1A7_ROOT"
```

### 1.3 数据与动作契约

数据协议固定为：

- LeRobot Dataset v3.0，15 FPS；
- 60 个 episode，13,222 帧，两个英文 task；
- `observation.state` 和 `action` 均为16维；
- 四路 AV1 视频：全局、左腕、右腕、扫码区；
- 16维顺序固定为左臂7、右臂7、左夹爪、右夹爪。

任何训练、离线回放和真机推理都不得改变这个关节顺序。原始 Episode 可继续记录
`waist.qpos`用于审计，但转换时不进入模型。PI0.5 内部会把状态和动作补齐到32维，
机器人接口只能接收原始16维语义，输出必须在下发前裁回16维。

## 2. 为什么必须使用派生训练数据

原始数据的 Parquet、episode 和视频是完整的，但 `meta/stats.json` 中的 `q01/q99` 不是对全部13,222行计算得到的真实分位数。已审计到的最大绝对误差约为：

- `observation.state`：`q01` 约1.131，`q99` 约1.064；
- `action`：`q01` 约1.122，`q99` 约1.060。

PI0.5 默认以 `QUANTILES` 归一化状态和动作，因此这些误差会直接改变训练目标。不要修改原始数据；使用准备脚本创建派生副本：

```bash
python scripts/lerobot/prepare_r1a7_dataset.py \
  --source /home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30 \
  --output /home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30_pi05_stats
```

脚本会：

1. 复制完整 v3 数据集；
2. 逐行扫描全部 Parquet；
3. 仅替换派生副本中 state/action 的数值统计；
4. 写入 `meta/r1a7_exact_stats_audit.json`；
5. 校验原始 `stats.json` 哈希没有变化。

输出目录已存在时脚本默认拒绝覆盖。只有明确确认该目录就是先前生成的派生副本时，才使用 `--reuse-output` 更新统计。

## 3. 中央只读检查

只检查当前训练链路时运行：

```bash
./scripts/lerobot/check_r1a7_setup.py --training-only
```

`--training-only`（别名 `--skip-control`）只检查新版 `lerobot` 环境、派生训练数据、PI0.5 base、tokenizer 和四相机训练配置。它不会启动旧 `unitree_lerobot` 环境，不导入机器人插件、Unitree SDK、CycloneDDS 或 OpenCV，也不会查询或创建 CUDA context。`train-pi05-full.sh` 的 preflight 默认使用此模式；GPU 编号只由 `nvidia-smi` 做只读存在性检查，Torch 仅在正式训练进程中使用 CUDA。

需要同时检查两套环境时再运行完整模式：

```bash
./scripts/lerobot/check_r1a7_setup.py
```

检查器不会下载或修改文件。它会验证：

- 两个 Conda 环境的 Python、LeRobot distribution、module 和源码 `pyproject.toml` 版本一致；
- editable import 实际指向预期源码目录；
- 新环境能导入 PI0.5 和 Transformers；
- Unitree Python SDK 与 CycloneDDS 的安装状态（默认只警告，不阻断纯训练）；
- 同一 `lerobot_robot_unitree_r1a7` 插件在两套环境中的版本和 import 落点一致；
- 两张 CUDA GPU 均可见并支持 BF16；
- 原始和派生数据均为 v3、16维、四相机，视频元字段和帧数闭合；
- 派生数据的存储分位数与全部 Parquet 行重新计算结果一致；
- 本地 PI0.5 safetensors、配置、pre/postprocessor 和 tokenizer 完整；
- `--policy.input_features=null` 的四相机全参配置能够被当前源码解析。

原始数据的分位数误差会显示为 `WARN`，因为原始目录只作证据保留；派生数据出现同类误差则是 `FAIL`。任何必需项失败都会返回非零退出码。纯训练不依赖 Unitree SDK 或 CycloneDDS；控制环境验收时必须显式增加 `--require-hardware`。

常用覆盖：

```bash
R1A7_PI05_DATASET_ROOT=/path/to/derived \
R1A7_PI05_BASE=/path/to/pi05_base \
R1A7_PI05_TOKENIZER=/path/to/tokenizer \
  ./scripts/lerobot/check_r1a7_setup.py --require-plugin
```

控制侧完整检查使用：

```bash
./scripts/lerobot/check_r1a7_setup.py \
  --require-plugin \
  --require-hardware \
  --require-legacy-pi
```

使用 `--json` 可获得机器可读报告。

## 4. 刷新两套环境

先查看计划，不执行安装：

```bash
./scripts/lerobot/setup-r1a7-envs.sh
```

确认路径和命令后才允许执行：

```bash
./scripts/lerobot/setup-r1a7-envs.sh \
  --apply
```

该脚本执行以下受控操作：

1. 在两套环境中固定 CycloneDDS；
2. 在两套环境中 editable 安装仓库内 `vendor/unitree-sdk2-python`；
3. 刷新旧环境中的 vendored LeRobot 和 `unitree_lerobot`；
4. 刷新新环境中的 `/media/jieguo/disk/zyc/lerobot`；
5. 在新环境中固定源码要求的 `draccus>=0.11.6,<0.12`；
6. 在两套环境中 editable 安装同一个 `lerobot_robot_unitree_r1a7` 插件；
7. 运行两套环境的 `pip check` 和中央检查。

所有 editable 命令都带 `--no-deps --no-build-isolation`，没有 `--upgrade`。脚本会记录安装前后的 Torch 版本；若 Torch 发生变化则报错。不要为了修复 editable 版本号而执行不受控的 `pip install -U lerobot`。

源码0.6.2要求 `draccus>=0.11.6,<0.12`，旧的0.10.0会导致 `pip check` 失败。setup 对该包使用精确范围和 `--no-deps` 单独安装，不让解析器顺带升级 Torch 或其他机器学习栈。

若现场有离线 CycloneDDS 源码或 wheel：

```bash
./scripts/lerobot/setup-r1a7-envs.sh \
  --cyclonedds-source /path/to/cyclonedds.whl
```

未提供本地资产时默认使用 `cyclonedds==0.10.2`，实际 `--apply` 可能访问配置的 pip index 或本机 cache。

## 5. PI0.5 本地资产

可用于 LeRobot 的 base 目录必须整体保留：

```text
pi05_base/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
└── policy_postprocessor.json
```

当前 base 的 `model.safetensors` 约14.47 GB，包含812个实际存储的 tensor key；PaliGemma 输入 embedding 通过 safetensors metadata 与 `lm_head` 共享权重。审计时已在当前源码下以 `strict=True`、BF16 完整加载，未发现 missing 或 unexpected key；去重后的模型参数量为3,616,757,520（约36.17亿）。

以下资产不能替代它：

- Hugging Face cache 中 `models--lerobot--pi05_base` 只有 ref、没有完整 snapshot/blob；
- OpenPI 的 `pi05_base_pytorch` 是另一套交付布局，缺少 LeRobot pre/postprocessor；
- `pi05_libero_base` 面向 LIBERO，不是当前 R1-A7 微调的默认 base。

base 中保存的 `device` 是 `mps`，训练脚本会覆盖为 CUDA；preprocessor 中 tokenizer 是本机绝对路径，移动 checkpoint 后必须同步修正资产布局并重新检查。

## 6. 三相机 base 到四相机 R1-A7

预训练配置声明三个图像 key：

```text
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
observation.images.right_wrist_0_rgb
```

R1-A7 数据有第四路扫码相机，key 名也不同。只做三个 rename 会导致扫码相机不进入模型。中央训练脚本不使用 `--policy.path` 继承 base 的三相机配置，而是固定传入：

```text
--policy.type=pi05
--policy.pretrained_path=/media/jieguo/disk/zyc/lerobot/pretrained/pi05_base
--policy.input_features=null
--policy.tokenizer_name=/media/jieguo/disk/zyc/lerobot/pretrained/google/paligemma-3b-pt-224
--policy.n_action_steps=10
```

这会让当前 LeRobot 从派生数据元信息重新推导输入 feature，最终得到：

```text
observation.images.cam_global
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.images.cam_scanner
observation.state                 # 16D
action                            # 16D
```

PI0.5 对相机共享视觉编码参数，增加输入 key 不会改变 base safetensors 的参数形状，但会改变 token 序列和数据契约。因此训练保存的完整 `config.json` 与 processor 才是推理端权威来源，不能在推理时退回 base 的三相机配置。

## 7. 双 GPU DDP smoke

先只打印最终命令并执行 training-only 只读 preflight：

```bash
./scripts/lerobot/train-pi05-full.sh smoke --dry-run
```

再执行两步 smoke：

```bash
./scripts/lerobot/train-pi05-full.sh smoke \
  --output-dir /media/jieguo/disk/zyc/lerobot/outputs/train/r1a7_pi05_smoke_01
```

smoke 固定为：

- `torchrun --nproc-per-node=2`；
- 普通 DDP：`dp_replicate=2`、`dp_shard=1`；
- 每卡 batch 1；
- 两个 micro-batch step；
- BF16、gradient checkpointing；
- 无 PEFT、无冻结、无 EMA；
- 不保存 checkpoint，不访问 Hub，不运行仿真 eval；
- 视频后端明确使用 PyAV。

这里的目标不是得到可用模型，而是确认两卡建模、数据解码、processor、前向、反向、梯度同步和优化器 step 能完成。必须看到至少一个完整 optimizer update 才算通过。

当前环境的 TorchCodec 存在 FFmpeg/C++ ABI 加载告警，因此训练固定使用 `pyav`。不要为了消除该告警擅自升级 Torch；TorchCodec 修复应作为独立环境变更评审。

2026-08-25 已完成一次真实的两卡 DDP optimizer update：去重后的3,616,757,520个参数均未被配置主动冻结，loss 为0.260，梯度范数为5.087，每卡峰值显存36.68 GB。该结果证明每卡 batch 1 的普通 DDP 可用，不需要为当前硬件强行切换到未经验证的 FSDP。

## 8. Production 全参数微调

smoke 通过后才运行：

```bash
./scripts/lerobot/train-pi05-full.sh production \
  --output-dir /media/jieguo/disk/zyc/lerobot/outputs/train/r1a7_pi05_full_01
```

默认值：

| 参数 | 默认值 |
| --- | ---: |
| steps | 30,000 |
| 每卡 batch | 1 |
| GPU 数量 | 2 |
| gradient accumulation | 8 |
| effective batch | 16 |
| save frequency | 1,000 |
| log frequency | 10 |

当前 safetensors checkpoint 的 policy 约8.30 GB，完整 AdamW training state 约15.13 GB，单个可恢复 checkpoint 合计约23.43 GB。默认30,000步、每1,000步保存且不自动清理历史产物时，最多会产生30份、约703 GB（十进制）的 checkpoint；`checkpoints/last` 只是符号链接，不是额外副本。按已测3.33秒/micro-step和约147秒/次保存粗略估算，完整作业约29小时，实际时间取决于视频解码和磁盘写入。启动前应确认磁盘预算，或显式调大 `--save-freq`。

可从脚本参数覆盖：

```bash
./scripts/lerobot/train-pi05-full.sh production \
  --output-dir /media/jieguo/disk/zyc/lerobot/outputs/train/r1a7_pi05_full_02 \
  --steps 10000 \
  --gradient-accumulation 4 \
  --save-freq 500
```

可在 `--` 后追加非契约参数：

```bash
./scripts/lerobot/train-pi05-full.sh production \
  --output-dir /media/jieguo/disk/zyc/lerobot/outputs/train/r1a7_pi05_full_lr_test \
  -- \
  --policy.optimizer_lr=1e-5 \
  --policy.scheduler_decay_steps=10000
```

脚本会拒绝追加参数覆盖下列契约：数据路径、base、输出目录、policy 类型、四相机 feature 推导、CUDA/BF16、冻结状态、PEFT、DDP 拓扑和 resume。

“全参数”在这里明确表示训练配置没有使用参数高效微调或主动冻结：

- `freeze_vision_encoder=false`；
- `train_expert_only=false`；
- `use_peft=false` 且没有 `peft` 配置；
- VLM、视觉塔、action expert 和投影层的实际前向路径均参与训练。

这一定义不承诺每个已注册参数在每个 batch 都有非空梯度。PI0.5 当前使用 suffix-only loss，expert `lm_head` 以及部分位于损失不可达路径上的参数可能结构性地没有梯度；这与 LoRA 或主动冻结不同，也不能据此宣称“每个存储参数都被更新”。

每次运行必须使用新的 `output_dir`。目录只要已经存在，脚本就拒绝启动，不会隐式覆盖。恢复训练应按该 checkpoint 中的 `train_config.json` 使用 LeRobot 官方 resume 流程，并先在独立输出目录验证；中央脚本不把 resume 混入首次训练入口。

已实测的两卡恢复命令形态如下，`--steps` 是恢复后希望达到的总 micro-batch step：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/jieguo/anaconda3/envs/lerobot/bin/torchrun \
  --standalone --nproc-per-node=2 \
  /home/jieguo/anaconda3/envs/lerobot/bin/lerobot-train \
  --config_path=/absolute/checkpoint/pretrained_model/train_config.json \
  --resume=true \
  --output_dir=/new/output/directory \
  --steps=30000
```

实测从 tied step-1 checkpoint 恢复了15.13 GB optimizer state、scheduler 和已保存 RNG；日志明确显示 `Resuming data order at epoch 0, sample 2`，随后完成 step 2，loss 为0.263、梯度范数为4.915，峰值为36.68 GB/GPU。map-style sampler 没有单独的 cursor 文件，而是根据 `training_step.json` 中的 step、DP world size 和 batch size 推导恢复位置。

当前 LeRobot 的 `rng_state.safetensors` 是 rank 0 的单份 Python/NumPy/Torch RNG 快照，DataLoader worker RNG 和预取队列也不单独序列化。因此同拓扑、`num_workers=0`、关闭图像增强的 smoke 能验证 sample-exact 的数据顺序和完整训练状态恢复，但不能宣称多 rank、多 worker 续训与不中断训练逐 bit 等价。production 数据本身仍可按确定性 sampler 续接；需要逐 bit 复现实验时应另行实现 per-rank/per-worker RNG checkpoint。

## 9. 显存与分布式边界

两张 GPU 各约48 GB。DDP 会在每张卡复制完整模型、梯度和优化器状态，两张卡的显存不会自动合并。tied checkpoint 单卡 strict load 时 `nvidia-smi` 记录约8530 MiB；实测完整 AdamW 更新的 Torch 峰值为36.68 GB/GPU，约保留11 GB设备余量。该余量以每卡 batch 1、BF16、gradient checkpointing、DDP gradient bucket view、四相机输入和无 EMA 为前提。

因此：

1. production 保持每卡 batch 1，通过 gradient accumulation 增大有效 batch；
2. 修改相机数量、分辨率、模型结构或关闭 gradient checkpointing 后必须重新测峰值；
3. 不得把 `train_expert_only=true` 或 LoRA 宣称为“全参数微调”；
4. FSDP/ZeRO 只能作为 DDP OOM 后的实验分支。

当前项目尚未验证 PI0.5 在这两张卡上的 FSDP wrap、checkpoint 保存、resume 和推理导出。因此本文不提供“已可用”的 FSDP 命令，也不允许在 production 脚本附加 `parallelism.dp_shard`。若后续实验 FSDP，必须单独验证 strict checkpoint 重载、processor 文件、16维输出和离线推理等价性。

## 10. Checkpoint 交付与两套框架

训练产物必须以 `checkpoints/<step>/pretrained_model/` 整个目录交付，至少包括：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_*_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_*_unnormalizer_processor.safetensors
train_config.json
```

只复制 `model.safetensors` 会丢失四相机 feature、精确 quantile、16维裁剪、tokenizer 和动作反归一化，禁止用于真机。

### 10.1 本机现有的 4000-step legacy 权重

本机目前找到的 R1-A7 PI0.5 权重位于：

```text
data/unitree-lerobot/unitree_lerobot/lerobot/outputs/train/
└── r1a7_pi05_single_gpu_bs32_20260826-010706/
    └── checkpoints/004000/pretrained_model_legacy/
        ├── config.json
        ├── legacy_export.json
        ├── model.safetensors                         # 8,300,756,168 bytes
        ├── policy_preprocessor.json
        ├── policy_preprocessor_step_2_normalizer_processor.safetensors
        ├── policy_postprocessor.json
        └── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

该目录的事实边界：

- `legacy_export.json` 记录其来源为队友服务器的 `004000/pretrained_model`；
- 权重包含 812 个实际 tensor key，并带 PaliGemma tied-weight metadata；
- 以下2026-08-26结果属于旧17维基线：配置是四相机输入、17维状态、17维动作、BF16、10步 action 输出；
- 它没有 `train_config.json`、optimizer、scheduler、RNG 或 `training_step.json`，因此是推理交付物，不是可恢复训练的完整 checkpoint；
- checkpoint 保存的是旧绝对 tokenizer 路径；本机 Hugging Face cache 已有完整 snapshot，shadow 脚本会显式覆盖该路径；
- 该目录被 vendored LeRobot 的 `outputs/` 规则忽略，不在 Git 中。Git 分支只管理代码与文档，模型权重必须单独传输和校验。

若要保存独立于 Hugging Face cache 的完整交付包，应从队友机器整体复制以下目录，而不是只复制单个文件：

```text
/media/jieguo/disk/zyc/lerobot/pretrained/google/paligemma-3b-pt-224/
/media/jieguo/disk/zyc/lerobot/outputs/train/r1a7_pi05_single_gpu_bs32_20260826-010706/checkpoints/004000/pretrained_model/
```

当前本机 legacy 推理不再依赖第一项复制，因为 cache snapshot 已完整。第二项是训练服务器上的原始完整 checkpoint；只有它包含训练配置和可用于交付审计的完整上下文。若还要 resume，必须连同该 step 对应的 optimizer、scheduler、RNG 和 training state 一并复制，不能从本机 legacy 目录恢复训练。

### 10.2 本机离线推理

先验证数据，不加载模型，也不连接机器人：

```bash
cd /home/user/r1-a7-autonomous-checkout
./scripts/lerobot/run-r1a7-pi05-local-shadow.sh --dry-run
```

本机 2026-08-26 已验证旧17维基线可读取 episode 0/frame 0、四相机数据契约和17维参考动作，并报告 `hardware_commands_sent=false`；该结果不验收当前16维契约。

执行真实模型的单帧离线推理：

```bash
cd /home/user/r1-a7-autonomous-checkout
./scripts/lerobot/run-r1a7-pi05-local-shadow.sh \
  --episode 0 \
  --frame 0 \
  --device cuda \
  --seed 1000
```

脚本会读取 Hugging Face cache 的 `refs/main`，解析具体 snapshot，检查 checkpoint 和 tokenizer 是否完整，随后强制 Hugging Face/Transformers 离线运行。它只读取已录制的数据帧，不导入硬件 gateway，也没有 DDS publisher。当前本机是 16 GB RTX 4070 Ti SUPER；8.3 GB 权重不等于一定能在 16 GB 显存中完成四相机推理，仍需以实际单帧峰值为准，不能从权重文件大小直接断言显存足够。

2026-08-26 本机实测结果：

- 旧17维 episode 0/frame 144：strict 权重加载成功，约35秒完成首次加载和单帧推理，输出17维有限动作且均在硬限位内，`hardware_commands_sent=false`；
- episode 0/frame 0：模型完成前向，但左夹爪预测为`5.4361262`，超过`5.4`硬上限约`0.0361`，被动作契约拒绝；
- frame 0 测试期间 `nvidia-smi` 记录整卡峰值约11,371 MiB，证明当前16 GB显卡能够完成模型加载与前向，但这不是实时推理吞吐测试；
- 推理日志中的 vision embedding warning 没有造成 missing/unexpected key，权重 loader 报告全部 key 成功加载。该 warning 仍需与训练端版本对照，不能仅凭一次有限输出判定模型已具备真机部署质量。

因此，本机离线推理链路已打通，但 checkpoint 尚未通过多帧越界率、动作误差和闭环 shadow 验收。真机适配层必须继续保留硬限位与越界拒绝，不能因为 frame 144 成功就直接发布动作。

补充的 place 阶段代表帧 episode 5/frame 43 也已成功完成 strict load 和离线推理：双臂 MAE 为`0.01582 rad`，最大关节误差为`0.03916 rad`，腰部误差为`1.55e-5 rad`，左右夹爪误差分别为`0.0381`和`0.4462`。这只能证明 scan/place 两种 task 文本都能进入模型并产生合法动作，样本数仍不足以估计整体成功率。

随后补测 episode 2/frame 273（scan）和 episode 9/frame 39（place）：前者预测左腕 Roll 为`-1.9378177 rad`，越过`-1.9199 rad`硬下限并被拒绝；后者合法，双臂 MAE 为`0.01952 rad`、最大误差为`0.04390 rad`，左右夹爪误差分别为`0.0211`和`0.4445`。截至当前5个有记录的代表帧为3个合法、2个越界，但这些帧不是随机抽样，不能把`3/5`解释为模型成功率。它只说明越界不是单个 frame 0 的偶发现象。

### 10.3 扫码事件与推理 task 切换

扫码枪继续发送现有 ZMQ `item_accepted` 事件，不需要为 PI0.5 修改消息格式。当前推理任务状态契约为：

1. 推理 run 从 scan task、generation 0 开始；
2. 只接受当前 checkpoint 对应的条码`6933949903337`；
3. 首个有效 `item_accepted` 把 task 切换为 place、generation 增加到1；
4. 回复包含`policy_reset_required=true`，实时推理循环必须立即调用`policy.reset()`，丢弃扫码前缓存的 action chunk；
5. 同一 run 的后续扫码作为 duplicate 回复，不再增加 generation，也不重复 reset；
6. 错误条码返回`status:error`，task 保持 scan。

先在终端 A 启动无模型、无机器人控制的诊断接收器：

```bash
cd /home/user/r1-a7-autonomous-checkout

PYTHONPATH="$PWD/robot/xr-teleoperate:$PWD/data/unitree-lerobot" \
  /home/user/miniconda3/envs/lerobot/bin/python -m \
  unitree_lerobot.r1a7_inference_task_state \
  --bind tcp://0.0.0.0:55580 \
  --expected-barcode 6933949903337
```

看到`"status": "ready"`后，在终端 B 启动现有扫码枪桥：

```bash
cd /home/user/r1-a7-autonomous-checkout

/home/user/miniconda3/envs/tv_official/bin/python \
  robot/xr-teleoperate/teleop/tools/barcode_task_event_bridge.py \
  --device /dev/supermarket-scanner \
  --database /home/user/r1-a7-autonomous-checkout/apps/r1-supermarket/data/supermarket.db \
  --endpoint tcp://127.0.0.1:55580 \
  --duplicate-window 0
```

扫描正确商品后，接收器应返回`phase=place`、place task 和`policy_reset_required=true`；扫码桥应打印`ACCEPTED`。以上只验证 HID、商品数据库、ZMQ 协议和推理任务状态切换，不加载 PI0.5，也不发送硬件命令。

实时 shadow runner 已把`InferenceTaskState`嵌入进程本身；运行实时 shadow 时不要同时启动上述诊断接收器，因为两个进程不能绑定同一个`55580`端口。runner 每轮推理前读取 snapshot，发现 generation 变化时先 reset policy、丢弃扫码前缓存的 action chunk，再使用新的 task 文本请求下一组动作。

### 10.4 本机四相机实时 shadow

该入口读取四路实时图像、R1-A7 双臂与 Dex1 左右位置，并运行 PI0.5；它只把16维预测写入 JSONL，不创建 DDS command publisher，也不控制机器人。模型进程和 DDS 进程使用不同 Conda 环境：启动脚本会在`tv_official`中托管只读状态桥，在`lerobot`中加载模型，退出时自动回收状态桥。

终端 A 启动已配置的四相机服务：

```bash
export R1A7_ROOT=/home/user/r1-a7-autonomous-checkout
TV_PYTHON=/home/user/miniconda3/envs/tv_official/bin/python \
  "$R1A7_ROOT/scripts/start-teleimager.sh"
```

终端 B 启动 Dex1 服务。该服务需要同时发布左右夹爪状态，否则 shadow preflight 会明确报告缺少 topic：

```bash
cd /home/user/r1-a7-autonomous-checkout/robot/dex1-service/bin
sudo ./dex1_1_gripper_server --network enx00e04c187010
```

终端 C 首先执行一次不含扫码的单步 smoke，确认模型可以加载并生成一条合法预测或明确的限位拒绝：

```bash
cd /home/user/r1-a7-autonomous-checkout

./scripts/lerobot/run-r1a7-pi05-live-shadow.sh \
  --network-interface enx00e04c187010 \
  --image-host 127.0.0.1 \
  --expected-barcode 6933949903337 \
  --max-steps 1 \
  --output /home/user/r1-a7-autonomous-checkout/runtime/logs/pi05_black_truffle_live_shadow.jsonl
```

人工扫码测试需要重新启动 runner 并删掉`--max-steps 1`，否则 action chunk 很快消费完，扫码枪可能来不及在进程退出前发送事件。无上限模式持续运行到 Ctrl-C。

启动顺序不能颠倒。runner 会先验证四相机 shape、图像年龄/跨相机偏差、16维状态和三个 DDS 状态源，再加载 checkpoint；只有看到以下 JSON 行才表示模型和扫码事件端口都已就绪：

```json
{"event":"ready","hardware_commands_sent":false}
```

终端 D 在 ready 后启动现有扫码桥。它无需增加 PI0.5 专用参数，仍发送同一个`item_accepted`事件：

```bash
export R1A7_ROOT=/home/user/r1-a7-autonomous-checkout
cd "$R1A7_ROOT/robot/xr-teleoperate"

/home/user/miniconda3/envs/tv_official/bin/python \
  teleop/tools/barcode_task_event_bridge.py \
  --device /dev/supermarket-scanner \
  --database "$R1A7_ROOT/apps/r1-supermarket/data/supermarket.db" \
  --endpoint tcp://127.0.0.1:55580 \
  --duplicate-window 0
```

扫到`6933949903337`后，扫码桥应打印`ACCEPTED`；实时日志后续记录应从`phase=scan`变为`phase=place`，generation 从0变为1。相同 run 后续扫码会按 episode duplicate 忽略，不会再次 reset。扫码枪事件只能切换语言 task，不能证明机器人已经完成动作，也不能把 shadow 预测变成真机命令。

PI0.5 是连续生成模型，夹爪输出在训练上限`5.4`附近可能轻微过冲。实时 shadow 保留`raw_action`，把夹爪安全候选裁剪到`[0, 5.4]`并写入`action`，同时记录`clipping`且将事件标记为`prediction_clipped`；它不会提高硬限位或隐瞒原始输出。双臂越过硬限位仍拒绝整条动作。真机控制层仍需再次执行独立硬限位，不能只依赖 shadow 后处理。

2026-08-26 已完成只读 DDS 预检：Domain 0、网卡`enx00e04c187010`和`rt/lowstate`可用；当时 Dex1 服务未运行，因此左右 gripper state topic 缺失，完整四相机实时模型循环尚未执行。必须区分“代码和协议测试通过”与“现场四输入源已全部在线”。

每个训练 checkpoint 先运行静态检查，再运行单帧离线推理：

```bash
conda run -n lerobot python scripts/lerobot/verify_r1a7_pi05_checkpoint.py \
  --checkpoint /absolute/checkpoint/pretrained_model

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run -n lerobot python scripts/lerobot/verify_r1a7_pi05_checkpoint.py \
  --checkpoint /absolute/checkpoint/pretrained_model \
  --run-inference \
  --seed 1000
```

当前验证器检查四相机、16D关节顺序、全参标志、自包含 tokenizer、processor quantile、实际 tensor、PaliGemma tied-weight metadata 和 `train_config.json`；推理只读取数据集帧，不导入机器人插件或 Unitree SDK。2026-08-25 的 tied step-2 数值属于旧17维基线：checkpoint 为8,300,756,168字节，processor 最大统计误差约`1.05e-7`，输出为有限的`(1, 17)`离线动作。

两套框架的职责应明确：

- 新版 LeRobot：推荐的 PI0.5 production 训练框架，支持已验证的双卡 DDP、checkpoint 标准加载和新版 R1-A7 插件推理；
- `unitree_lerobot`：支持单卡 PI0.5 全参训练、旧项目的数据工具、R1-A7 DDS/控制、legacy checkpoint 导出和无发布器的离线 shadow 入口；
- 跨版本使用同一个 checkpoint 时，必须由兼容层使用训练 checkpoint 自带的 config 和 processor，不能让旧 loader 猜测新版格式。

### 旧框架原生训练

旧版 LeRobot 0.4.1 的 R1-A7 单卡训练入口为：

```bash
./scripts/lerobot/train-pi05-unitree-legacy.sh production \
  --cuda-device 1 \
  --batch-size 1 \
  --steps 30000 \
  --save-freq 4000 \
  --wandb
```

该 launcher 固定使用 `unitree_lerobot` conda 环境和当前项目中的 vendored LeRobot 源码，训练物理 GPU 由 `--cuda-device` 指定；单卡进程内部会把所选物理卡映射为`cuda:0`。它在启动前严格检查四相机、16维 state/action、两个 task、精确 quantile 派生数据、PI0.5 base、本地 tokenizer 和 LeRobot 0.4.1 源码归属，不导入 Unitree SDK 或创建 DDS publisher。

先执行不会启动训练的 preflight：

```bash
./scripts/lerobot/train-pi05-unitree-legacy.sh production \
  --cuda-device 1 \
  --dry-run
```

恢复训练时传入具体 checkpoint 的整个`pretrained_model`目录；旧框架继续使用该`train_config.json`中保存的 output directory、四相机 feature 和 processor：

```bash
./scripts/lerobot/train-pi05-unitree-legacy.sh resume \
  --cuda-device 1 \
  --checkpoint /absolute/run/checkpoints/000004/pretrained_model \
  --steps 30000 \
  --save-freq 4000 \
  --wandb
```

旧训练器的`steps`是 optimizer update 数，当前 launcher 不提供新版训练器的 gradient accumulation 或 DDP/FSDP 参数。需要这些能力时仍使用新版 LeRobot，不要把两套 CLI 参数混用。

2026-08-26 已在物理 GPU 1 完成旧17维框架 batch 1 的 step-1 训练与完整 checkpoint 保存、step 1 到 step 2 resume、step-2 strict load 和固定 seed 的四相机 shadow inference。两次更新的 loss 分别为0.259和0.198，模型总参数与可训练参数均为3,616,757,520，推理输出为有限17维动作且报告`hardware_commands_sent=false`。该结果只证明旧链路当时可运行，不可验收当前16维训练链路。

旧版0.4.1不能直接加载新版 checkpoint。先在旧环境导出受限 legacy 目录：

```bash
conda run -n unitree_lerobot python -m unitree_lerobot.export_pi05_legacy \
  --source-policy /absolute/path/to/checkpoint/pretrained_model \
  --dataset-root /home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30_pi05_stats \
  --output /absolute/path/to/legacy-export \
  --weights-mode hardlink \
  --device cuda \
  --tokenizer-name /media/jieguo/disk/zyc/lerobot/pretrained/google/paligemma-3b-pt-224
```

exporter 不加载权重，会过滤旧 `PI05Config` 不认识的字段、按 R1 数据集重建 processor，并 hardlink 或复制 safetensors。只支持 full checkpoint，且要求 `use_peft=false`、MEM=false、relative actions=false、RTC=false；遇到新版独有能力会 fail fast。旧 loader 会读取 safetensors metadata，只对 metadata 明确声明的 PaliGemma tied-weight alias 复用同一 tensor storage；alias 目标缺失或映射异常仍会在 strict load 前失败，不会用猜测的权重掩盖损坏文件。

旧环境已安装定制 Transformers 4.53.3，editable 源码位于`/media/jieguo/disk/zyc/transformers_lerobot_fix`，旧 PI0.5 import 检查通过。仍必须对每个 production checkpoint 分别执行 legacy export 和离线 loader 测试；不要从新版环境复制 site-packages。

无需 policy 的纯数据契约 shadow：

```bash
conda run -n unitree_lerobot python -m unitree_lerobot.r1a7_shadow_inference \
  --dataset-root /home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30_pi05_stats \
  --episode 0 \
  --frame 0 \
  --dry-run
```

当前16维 PI0.5 依赖与 legacy export 验证完成后，再增加 `--policy-path /absolute/path/to/legacy-export --device cuda --seed 1000`。该入口不包含 hardware publisher，不能用于直接控制真机。相同 checkpoint、数据帧和 seed 可用于比较新旧 loader 的完整16维输出。

2026-08-25 对 tied step-2 checkpoint 的 hardlink legacy export 数值属于旧17维基线：新版与旧版输出的平均绝对差为0.001634，双臂最大差0.005487，腰部差`5.81e-5 rad`，夹爪最大差0.004943；不能外推到当前16维模型。

旧版 LeRobot 的正式 policy 推理入口是 `lerobot-record`，robot 仍由同一个插件注册；它只能接收已通过上述 exporter 生成并验证的 legacy checkpoint。先执行 shadow/dry-run，再按插件兼容文档构造 record 命令，禁止把新版 `pretrained_model` 直接传给0.4.1 loader。

“两个环境都能读数据”不等于“旧 loader 能直接读新版 checkpoint”。每个具体 checkpoint 都需要以下离线门禁：

1. `strict=True` 加载，无 missing/unexpected key；
2. 相同四相机图像、16维 state 和 task 文本输入；
3. 新旧入口分别输出完整 action chunk；
4. 比较首个 chunk 的形状、有限值和数值误差；
5. processor 反归一化后仍保持固定16维关节顺序。

若旧框架不具备与训练版本一致的 PI0.5 loader，使用“新版 policy server + 旧版 R1 控制 client”进程边界，而不是复制模型主体代码。

## 11. 真机推理门禁

本机 PI0.5 真机入口为`scripts/lerobot/run-r1a7-pi05-real.sh`。它与已验证的
官方数采控制配置一致，不再叠加早期的保守工作空间包络或速度配置：

| 控制对象 | 真机配置 |
|---|---|
| 双臂目标 | 官方14关节硬限位，`joint_limit_margin=0` |
| 双臂速度 | `arm_velocity_limit=30 rad/s`，`rate_limit_mode=official` |
| 左臂硬限位 | `[-3.1416/-0.22689/-1.9199/-0.97564/-1.9199/-1.6144/-1.6144, 2.0944/2.4784/1.9199/2.1852/1.9199/1.6144/1.6144] rad` |
| 右臂硬限位 | `[-3.1416/-2.47849/-1.9199/-0.97564/-1.9199/-1.6144/-1.6144, 2.0944/0.2268/1.9199/2.1852/1.9199/1.6144/1.6144] rad` |
| 腰部 yaw | 不进入模型；真机控制桥固定在推理激活时的实测位置 |
| Dex1 | 位置`[0, 5.4]`，`kp=2.0`、`kd=0.05`、`command_step=0.18` |

当前示教没有主动控制腰部，因此新转换器同时从`observation.state`和`action`删除
`waist.qpos`。真机桥在第一次有效命令激活时锁定当前实测腰位，后续持续发送零
腰速；腰部实测值只作为独立硬件遥测记录，不参与模型归一化、tokenizer或动作输出。

PI0.5 的夹爪预测允许在模型后处理层裁到`[0, 5.4]`。双臂预测轻微越界不超过
`0.05 rad`时裁到对应硬限位，更大的越界仍拒绝；原始值和所有裁剪量都写入
审计日志。控制桥会再次独立校验16维硬限位。手臂没有额外`0.8 rad/s`限速、
5°余量、笛卡尔工作空间包络
或旋转包络。双臂的起始姿态对齐、策略动作和实测姿态保持均使用 R1-A7 URDF
在目标关节角、零速度、零加速度下计算的 RNEA 重力前馈力矩，与官方 XR 遥操
的动力学前馈计算一致；模型加载、关节顺序或力矩数值异常时控制桥直接拒绝
启动或进入故障锁定，不会退回零前馈继续运动。未启用额外 Dex1 力矩软限制。
腰部保持不是保守限速，而是匹配这批数据没有腰部控制示教的事实。

PI0.5 模型每次生成50条候选动作，部署只执行其中10条；目标输出频率为15 Hz，
即每66.7 ms发送一条命令。真机入口默认在当前块剩余5条时，用最新四路图像和
16维状态异步预取下一组；任务 generation 改变时
立即丢弃旧任务当前块和预取结果。若 GPU 推理未在块结束前完成，控制桥保持实测
姿态并记录`real_async_prefetch_wait`，不会继续执行过期动作。正常预取结果从
完整50条候选的索引5开始截取索引5到14（即第6到15条），使候选动作时间与
当前块已经执行的5个时间步对齐，不会延迟补发新块开头的历史目标。实测日志显示动作块
交界处的双臂目标跳变明显大于块内变化，因此仍默认用3条命令从上一条已发送
目标过渡到新动作块（15 Hz 下约0.2秒）。这只是动作块接缝插值，不是关节速度限制；可用
`--chunk-blend-steps 0`关闭；预取阈值可用`--async-prefetch-actions`调整，必须小于
checkpoint 的`n_action_steps=10`。它只能减少块边界抽动，不能把持续预测为全开的夹爪
改成闭合，也不能证明当前8000步 checkpoint 已具备场景泛化能力。

正常动作参数不保守，但以下运行保护仍保留：必须同时提供环境变量和精确确认
字符串；DDS lowstate 超过250 ms停止发布；策略命令流超过500 ms进入锁定保持，
必须重启；退出时保持实测位置，不自动回零。它们处理断流和误启动，不改变正常
动作的速度与范围。模型动作被拒绝时 runner 会显式要求控制桥保持实测关节并
刷新健康心跳，因此不会再把“模型动作无效”误判成通信断流；真正失去 runner
通信时仍在500 ms后锁定。

启动时不能从任意残留姿态直接交给策略。runner 从 task 文本对应的30个 scan
Episode 首帧读取16维状态并取逐关节中位数；双臂在3秒内平滑移动到该起点，
腰部仍保持实测位置，双夹爪同时命令为`5.4`全开。反馈达到双臂`0.08 rad`、
夹爪`0.20`容差后，才清空 action queue 并启动策略。3秒轨迹结束后最多再等待
12秒实际关节收敛，并每秒输出当前误差最大的关节；初始化过程本身会移动真机，
不是只读检查，也不会通过放宽到位容差掩盖跟踪误差。

初始化命令与正式策略命令在控制协议中分开标记。起始姿态对齐、收敛等待以及
首个 PI0.5 动作块生成期间，控制桥以最后一个初始化目标和 RNEA 重力前馈持续
保持，不启动策略断流计时。第一条策略动作成功下发后才将
`policy_watchdog_active`切换为`true`，此后仍严格执行500 ms断流故障锁定；该
分阶段处理不会放宽正式推理的超时保护。首条策略动作仍必须在最后一条外部
初始化命令后的5秒内到达，否则启动阶段也会故障锁定，不会无限保持。

### 11.1 首条真机命令

先启动四相机服务和 Dex1 服务，确认实体急停可用、机器人周围清场。首次运行只
发送1条策略命令，用于核对左右映射和初始运动方向；`--max-commands`限制的是
策略接管后的命令数，不包含前面的起始姿态初始化命令，也不改变上述动作限位：

```bash
cd /home/user/r1-a7-autonomous-checkout
export R1A7_REAL_INFERENCE=YES

./scripts/lerobot/run-r1a7-pi05-real.sh \
  --network-interface enx00e04c187010 \
  --image-host 127.0.0.1 \
  --expected-barcode 6933949903337 \
  --max-commands 1 \
  --output /home/user/r1-a7-autonomous-checkout/runtime/logs/pi05_real_first_command.jsonl \
  --confirm-real-robot I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE
```

runner 加载模型期间机器人保持实测位置；随后出现
`Aligning to median scan start pose from 30 episodes...`时，机器人已经开始执行
3秒起始姿态初始化。`"event":"real_inference_ready"`表示初始化完成、策略开始
接管，不表示此前没有发送硬件命令。完成1条策略命令后自动禁用跟踪并关闭控制器。
控制桥启动状态中的`"arm_feedforward":"r1a7_model_gravity_rnea"`表示上述
重力前馈已经成功加载。此时`"policy_watchdog_active":false`是正常状态，首条
策略动作成功下发后其响应会变为`true`。
日志中的`state`和`policy_state`都是16维模型输入；`raw_action`是模型原始输出，
`unblended_policy_action`是夹爪裁剪后的输出，`policy_action`是动作块过渡后的下发
目标，`action`是真正执行的16维动作。腰部实测值由控制桥单独写入
`measured_waist_yaw`。`clipping`记录夹爪和轻微手臂越界
裁剪，`chunk_blending`记录动作块过渡；初始化记录为`initialization_started`和
`initialization_complete`。

### 11.2 连续短测和扫码切换

首条方向正确后，先做30条短测：

```bash
cd /home/user/r1-a7-autonomous-checkout
export R1A7_REAL_INFERENCE=YES

./scripts/lerobot/run-r1a7-pi05-real.sh \
  --network-interface enx00e04c187010 \
  --image-host 127.0.0.1 \
  --expected-barcode 6933949903337 \
  --max-commands 30 \
  --output /home/user/r1-a7-autonomous-checkout/runtime/logs/pi05_real_30_commands.jsonl \
  --confirm-real-robot I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE
```

需要完整 scan 到 place 流程时把`--max-commands`设为`0`，看到 ready 后再启动
第10.4节的扫码桥。正确条码会触发`item_accepted`、清空旧 action chunk 并切换
到 place task；按`Ctrl-C`结束，不能同时启动占用`55580`的 shadow 或诊断接收器。

队友分支中的新版插件 distribution 和 import 名为 `lerobot_robot_unitree_r1a7`，注册地址是 `robot.type=unitree_r1a7`，策略入口是 `lerobot-rollout`。它是另一条部署入口；不要把其参数与本节已经落地的本机 launcher 混用。

该新版插件的真实 backend 使用`--robot.enable_real_robot=true`与环境变量
`LEROBOT_R1A7_ALLOW_REAL=YES`；本机 launcher 使用
`R1A7_REAL_INFERENCE=YES`与`--confirm-real-robot`。两套授权不能互相替代。

中央环境和训练脚本只负责安装、检查和离线训练，不会连接 DDS、启动相机或向真机发布任何命令。只有本节的显式真机 launcher 同时通过两项授权后，才会启动本机硬件控制桥。

旧数据的腰部动作范围仅约`[-0.04234, -0.03114] rad`，总跨度约0.0112 rad，标准差约0.00416 rad，不足以学习有意义的主动腰部运动；新模型因此删除该自由度，当前真机入口在模型之外固定腰部。需要主动腰部能力时必须补采包含受控腰部变化的数据、重新定义模型契约并实现经过验证的绝对位置控制。

## 12. 常见失败

`env.lerobot` 显示源码0.6.2、安装元数据0.6.1：

```bash
./scripts/lerobot/setup-r1a7-envs.sh
./scripts/lerobot/setup-r1a7-envs.sh --apply
```

`dataset.training` 报 quantile error：重新运行派生准备脚本；不要把原始 `stats.json` 覆盖回派生目录。

`pi05.assets` 报 tokenizer 路径错误：确认 base 的 `policy_preprocessor.json` 指向当前本地 tokenizer，或重新准备一个自包含、可审计的 checkpoint 目录。

训练看不到扫码相机：检查 checkpoint 的 `config.json`，训练后应保存四个 R1-A7 image key；若仍是三个 base key，说明没有应用 `--policy.input_features=null`。

DDP 一开始就 OOM：保留错误日志和 `nvidia-smi` 峰值，不要直接切到 LoRA 后仍称为全参训练；按第9节建立独立的分片训练实验。
