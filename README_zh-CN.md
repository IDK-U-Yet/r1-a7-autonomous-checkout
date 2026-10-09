# R1-A7 无人收银系统

项目自有原创代码采用 [Apache-2.0](LICENSE)，第三方组件沿用各自许可证，详见 [许可范围](LICENSE_SCOPE.md) 和 [第三方来源](THIRD_PARTY_NOTICES.md)。结账上位机不包含在公开副本中。

**当前运行缺口：**扫码桥仍依赖私有上位机中的 HID 读取模块和商品数据库结构。公开版需要独立扫码读取实现及商品目录初始化工具，才能完整运行扫码驱动的工作流。其余文档中的历史命令不能视为本副本已独立验证的操作流程。

本公开候选副本包含 R1-A7 遥操作、Dex1 夹爪、数采监控、数据回放标注、数据转换及模型训练推理代码。结账上位机不公开，不包含其源码及专用启动脚本；数采监控与回放工具是独立模块，仍保留。日志、数据集、数据库、密钥和面试资料不进入 Git。部分原有操作文档描述完整私有系统，其中涉及结账上位机的章节仅供理解接口，不代表本副本包含该组件。

## 目录

```text
apps/r1-collection-console   数采监控台
apps/r1-dataset-review       数据回放及审核标注工具
robot/xr-teleoperate         R1-A7 遥操作源码
robot/dex1-service           Dex1 串口/DDS 服务源码
data/unitree-lerobot         R1-A7 数据转换源码
vendor/unitree-sdk2          Unitree C++ SDK 源码快照
vendor/unitree-sdk2-python   Unitree Python SDK 源码快照
runtime/                     本机数据、日志、部署快照（Git 忽略）
scripts/                     集成仓库入口脚本
docs/                        架构、部署和现场配置说明
```

## 初始化

本项目采用单体仓库，克隆主仓库后已经包含运行所需的全部源码，不需要初始化子模块。部分机器人网格和测试夹具由 Git LFS 管理，因此克隆前必须先安装 `git-lfs`。在仓库根目录执行：

```bash
./scripts/bootstrap.sh
./scripts/check.sh
```

从任意路径调用根脚本都可以；脚本会根据自身位置计算仓库根目录，并导出 `R1A7_ROOT`。`versions.lock` 记录各个外部组件导入时的上游来源和基准提交；主仓库自身的提交号是整个系统实际生效的唯一版本。

移动已有工作区后，Python 的 editable install 仍可能记录旧绝对路径。分别激活对应环境并修复，不会下载或更新依赖：

```bash
conda activate tv_official
./scripts/setup-editable.sh teleop

conda activate lerobot
./scripts/setup-editable.sh lerobot

```

## 启动

先激活相应 Python 环境，再从仓库根目录运行：

```bash
./scripts/start-teleop.sh \
  --arm R1_A7 \
  --ee dex1 \
  --input-mode controller \
  --no-camera \
  --enabled-arms both \
  --network-interface enx00e04c187010 \
  --r1-control-profile official \
  --confirm-official-parameters \
  --confirm-real-robot
```

真机操作、安全检查、遥操与数采流程见 [`使用手册.md`](使用手册.md)。不要将示例启动命令视为真机安全确认的替代品。

从 PICO 接入、遥操作、数采、训练到推理的整体路线、历史问题和后续建议见
[`docs/项目全流程与问题复盘.md`](docs/项目全流程与问题复盘.md)。

## 仓库边界

- 公开范围内的 `apps/`、`robot/`、`data/` 和 `vendor/` 源码由本仓库直接跟踪；结账上位机不在公开范围内。
- `versions.lock` 只记录外部源码的来源，不驱动自动下载或更新。
- 更新外部源码时必须在独立分支中审查差异、完成测试，并同步更新 `versions.lock`。
- `runtime/` 不是项目源码，也不是备份仓库；数据库与现场配置应另行加密备份。

部分导入的第三方组件保留了各自上游 README 和 CHANGELOG，其中可能描述其原仓库的克隆或子模块安装方式。这些内容仅作为上游资料保留；本系统的安装、检查和启动一律以本 README、`docs/` 和根目录 `scripts/` 为准。
