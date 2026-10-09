# R1-A7 人工接管式 DAgger（第一版）

目标：模型运行到偏差状态，操作者接管并正确完成；连续记录模型前缀和人工
后缀，但训练只用人工后缀。这不是给所有模型访问状态都补专家标签的完整
DAgger 实现。先少量采集与复核，再与合格的原始示教聚合训练。

当前实现完成软件验证，尚未验证真机接管延迟、实际运动连续性和任务成功率。
首次用空夹爪、清空工作区验证 A/B 接管与停机；不要直接以商品抓取作为首测。
保持机器人高度、腰姿、桌面、四相机和照明与训练数据一致。

## 启动顺序

以下使用本机现有 Oreo v2 全参数 8K 检查点作为示例，不表示它是最佳模型。
模型、数据和 task 文本必须匹配。换商品不需要改程序，使用自己的检查点、
数据目录、task 文本和 `--expected-barcode` 即可。
不要同时启动普通 PICO 遥操、其他推理程序或关节控制程序。

### 终端 1：夹爪服务

已运行且左右两个电机都正常时，不要重复启动。

```bash
sg dialout -c '/home/user/r1-a7-autonomous-checkout/robot/dex1-service/bin/dex1_1_gripper_server --network enx00e04c187010'
```

若无法发现两个电机，停止，不进入下一步。

### 终端 2：四相机

```bash
TV_PYTHON=/home/user/miniconda3/envs/tv_official/bin/python \
  /home/user/r1-a7-autonomous-checkout/scripts/start-teleimager.sh --no-affinity
```

四路都 ready 且相机配置成功后继续；已运行时保留现有服务。

### 终端 3：模型 + 接管 + 录制

```bash
cd /home/user/r1-a7-autonomous-checkout
export R1A7_REAL_INFERENCE=YES
CKPT=/home/user/r1-a7-autonomous-checkout/runtime/checkpoints/pi05_oreo_v2v3_standard160_a100_gpu3_full_b24_lr1e5_8k_01/checkpoints/008000/pretrained_model
DATA=/home/user/r1-a7-autonomous-checkout/runtime/datasets/lerobot/r1a7_oreo_4cam_16d_v2v3_standard160_train

./scripts/lerobot/run-r1a7-pi05-dagger.sh \
  --checkpoint "$CKPT" \
  --dataset-root "$DATA" \
  --tokenizer-root "$CKPT/tokenizer" \
  --expected-barcode 6901668005939 \
  --task-before 'Bring the product from its current location to the scanner in the scanning area, present its barcode, and wait until the scan is accepted.' \
  --task-after 'Place the scanned product in the black tray on the right.' \
  --network-interface enx00e04c187010 \
  --image-host 127.0.0.1 \
  --confirm-real-robot I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE
```

启动前检查端口 55580/55582/8012、16D 模型、四相机、task 文本、可写存储。
夹爪参数与 Oreo 数采一致：kp=3.0、kd=0.05、step=0.24。
DAgger 模型输出的有限关节越界目标会裁剪到硬限位后继续执行，不因普通
限幅进入 HOLD；原始目标及裁剪量保留在审计中。普通推理已恢复原来的
0.05 rad 越界容差，超出时跳过该周期。两者等待/暂停时均保留之前的夹持目标，
而不是把夹爪实测接触开度当作新的保持目标；无历史目标时使用实测开度。
人工 IK 与普通数采一样使用 solve_ik 返回的平滑/回退结果，不再因额外的
8 cm 位置残差或 0.5 rad 姿态残差进入持续 HOLD。不可达目标可能不能跟随到位，
这不代表碰撞安全。NaN/无效动作、硬关节限位、XR 失联及硬件故障保护仍保留。
限幅不等于碰撞检测；频繁触及边界仍需要操作者暂停检查。
明确按 Y/Q 废弃或按 R 放弃当前轮时，完成安全停止与文件收尾后直接删除该轮
视频和轨迹，不保留 rejected 目录。故障/中断导致的收尾异常仍保留用于排查，
不会把尚未结束的正在录制目录自动删除。
腰部仍是伺服保持，不是机械锁死；不属于模型的16维动作。

后续轮次按 R／右内侧握持键回起点时，如果右手柄 B 中断运动，拒绝旧控制权
命令并留在轮间 HOLD，不退出脚本、不自动续动、不创建新轮记录或重新启用扫码。
清空工作区后，再次 R／释放后重新按右内侧握持键才能重试。此恢复只处理健康的
operator B 中断；XR 断流、故障锁定和其他启动失败仍按原故障处理。首次 G 启动
流程不属于本次轮间恢复修改。

启动后 PICO 网站已待命，模型加载完成会等待 `G + 回车`，此前不会启动
策略或回起点运动。查本机 Wi-Fi IPv4：

```bash
ip -br -4 addr
```

在 PICO 中访问 `https://电脑WiFi地址:8012/?ws=wss://电脑WiFi地址:8012`，
进入 pass-through，保持两只控制器跟踪正常。不要再启动 `start-teleop.sh`。
确认夹爪为空、起点路径和工作区无人后，在终端 3 输入 `g` 回车。
机器人先回示教起点，然后开始模型运行及15 Hz四相机录制。
初始化期间 A/B 或 XR 失联会中止启动，不会绕过初始化继续下发动作。

### 终端 4：扫码桥

等待终端 3 出现 `DAgger: PICO A = takeover` 后启动：

```bash
cd /home/user/r1-a7-autonomous-checkout/robot/xr-teleoperate
/home/user/miniconda3/envs/tv_official/bin/python \
  teleop/tools/barcode_task_event_bridge.py \
  --device /dev/supermarket-scanner \
  --database /home/user/r1-a7-autonomous-checkout/apps/r1-supermarket/data/supermarket.db \
  --endpoint tcp://127.0.0.1:55580 \
  --duplicate-window 0 \
  --log-path /home/user/r1-a7-autonomous-checkout/runtime/logs/dagger.scanner.jsonl
```

确认 `Desktop input: BLOCKED` 与 `SCANNER READY`。不要同时运行两个扫码桥。
扫码接受才切换 scan -> place；相机或策略不能自行确认扫码成功。

## 一轮怎么操作

| 输入 | 行为 |
| --- | --- |
| 右手柄内侧握持键或 `g` + 回车 | PICO 已新鲜连接、清空工作区后，回起点并开始模型运行 |
| PICO 右 A | 暂停模型、按当前腕姿对齐，再平滑过渡到普通手柄映射 |
| PICO 右 B | HOLD，保留夹持目标；不会自动恢复模型 |
| `p` 或 `t` + 回车 | 先 HOLD，摆好控制器后按 A 对齐接管 |
| PICO 左 X 或 `s` + 回车 | 操作者确认操作完成，先 HOLD 再保存；须已有扫码接受事件才记为完成 |
| PICO 左 Y 或 `q` + 回车 | 先 HOLD，关闭本轮写入，再删除当前轮视频、轨迹及事件目录，不导出训练 |
| 右手柄内侧握持键或 `r` + 回车 | 仅在 HOLD 或本轮结束后：确认空夹爪、路径无人，废弃未结束本轮，清空旧动作并回起点，新建一轮 scan |
| `Ctrl+C` | 停止，本轮视为未完成，不自动导出训练 |

接管时双臂按当前测量腕姿建立参考，不在按 A 瞬间跳回遥操固定起点。
默认初始位置和姿态偏移逐渐消退，最后完全使用普通遥操的手柄映射。
`--takeover-blend-seconds 3` 设置最短过渡时间（默认 3 秒）；偏移较大时
自动延长，使偏移消退部分不超过 0.10 m/s、0.6 rad/s。这不是总运动速度限制，
手柄主动运动仍通过原有 IK、关节限制和新鲜度检查。
过渡时间按有效 HUMAN 控制更新累计，卡顿后不会追赶跳跃。每次 A 重新开始过渡，
B/P HOLD 停止过渡。即使手柄静止，过渡也会移动机械臂；先空夹爪、空工作区测试，
不要在持物交接时首次试用。`--takeover-blend-seconds 0` 恢复旧的固定相对参考。
过渡进度写入原始审计；过渡结束前 `expert_action_valid=false`，不导出为专家
训练动作。原始视频与轨迹仍保留，X/Y 保存和删除规则不变。
X/Y 不需要回车，按一次即可；初始化期间不接受结束请求，同时按下时 Y 优先。
右手柄内侧握持键（Grip，不是食指扳机）也不需要回车。模型加载完成并显示
启动提示后，先松开再按下才有效；提前按住不会自动启动。按下即确认工作区
无人并允许回起点运动，程序不能自行检测人员。模型或人工正在运动时，该键
不重启；先右 B 或 P 暂停，再松开并重新按内侧键，才执行与 R 相同的复位。
结束轮次后的内侧键也执行 R。持续握住不重复触发，保存/删除期间的按键
不会被缓存为之后的自动重启；看到 HOLD 就绪提示后再松开、按下。
本轮模型已扫码成功、切到 place 后，也可按 A 接管恢复放置，完成后按 X。
不用再次扫码；已有的本轮扫码接受事件保留，人工放置片段按 place 导出。
如果扫码事件来自上一次启动，本轮并不会继承，不能据此确认完成。
夹爪使用“接管位置拾取”：先保持此前夹持目标，慢慢移动对应扳机到当前
夹持开度，匹配或经过该位置后才启用该扳机；随后可完整开合。
这可避免接管瞬间松爪。它与普通遥操扳机立即生效的手感不同。

每次按 A 会重新对齐。XR 超过250 ms未更新会 HOLD，需要再次按 A；
超过1 s、控制器健康故障或录制协调心跳丢失会锁故障，须重启本轮。
人工控制约30 Hz，IK提交前再核验输入时效；这些是项目接管保护，不是机械
急停或碰撞检测。出现危险时使用机器人急停，不依赖终端或网络按键。

一次启动可连续采集多轮，不重复加载模型。X/Y 结束后保留夹持目标并暂停，
不会自动回起点；你完成空夹爪、商品复位并离开工作区后，输入 R 回车或
重新按右手柄内侧握持键。
运行中先 P 暂停再 R；正在人工操控时不能直接 R。R 会废弃尚未结束的本轮。
X 在未扫码时只暂停并提示拒绝保存，本轮不结束，可按 A 继续完成操作。
本轮扫码后接管仍可 X 保存；新一轮不继承上一轮的扫码事件、任务或动作块。
保存或删除期间保持 HOLD 和协调心跳；写盘/删除失败会停机，不会开始下一轮。
Y/Q 仅删除当前录制实例的独立目录，不清理以前的轮次；R 废弃的未完成轮次
和 Ctrl+C 中断轮次仍留作排查。共享运行总日志不删除，不属于训练导出数据。
相机、夹爪服务和扫码桥保持运行。只有 Ctrl+C 退出整个采集进程。

接管应该发生在模型真实偏差状态，不要由人工故意松爪制造掉落后，把那段
错误动作也当专家标签。需要制造故障的实验须另行标出错误区间。

## 数据在哪里，怎么导出

每轮独立目录：
`runtime/datasets/r1a7_dagger_raw/r1a7_oreo_4cam_16d_v2v3_standard160_train/dagger_<编号>/`

批次标识默认取 `--dataset-root "$DATA"` 的目录名，无需另填参数，不写死商品名
或数量。旧 `--collection-name` 仅保留为可选覆盖项。manifest 和每帧另存真实
dataset_root 与 checkpoint；同一数据集使用不同模型仍可区分。旧目录不改名、不迁移。

每轮回到示教起点、开始模型运行时即录制四路画面、状态与动作。模型运行、
HOLD 和人工接管都会进入本轮原始记录；按 A 不会另开文件或重置录制起点。
训练导出只选成功保存轮次中的连续有效 HUMAN 控制片段，模型和 HOLD 动作
只用于排查，不会作为专家标签。扫码后接管的人工片段按 place 任务导出。

- `frames.jsonl`、四路 `colors/`：连续审计，包含 policy/hold/human。
- `events.jsonl`：接管边界、画面断档、扫码与操作者完成事件。
- `data.json`、`manifest.json`：正常关闭后生成；缺少 manifest 表示未完整写完。
- 每帧保留16D状态、臂目标、夹爪限幅器实际输出、原始请求目标、控制权编号、
  单调时间、相机帧信息、task/phase、检查点和原始试验编号。

完成若干轮后导出人工片段，不覆盖旧导出：

```bash
cd /home/user/r1-a7-autonomous-checkout
PYTHONPATH="$PWD/data/unitree-lerobot" \
  "$PWD/runtime/envs/pi05-262d2fd/bin/python" \
  -m unitree_lerobot.r1a7_dagger_recording \
  --trials-root "$PWD/runtime/datasets/r1a7_dagger_raw/r1a7_oreo_4cam_16d_v2v3_standard160_train" \
  --output "$PWD/runtime/datasets/r1a7_dagger_expert_01" \
  --min-frames 50
```

导出条件：完整写入 + 操作者确认完成 + 扫码接受；片段必须为连续有效人工
控制，同一 task、控制权 epoch、无画面断档，长度至少50帧。
太短的片段会跳过，故障/未完成试验也不会自动入训练。
原始审计不能直接转换训练，转换器会拒绝，避免学习模型错误前缀。

`expert_export.json` 保留父试验编号。转换后的 source_segments 也会保留该编号，
划分脚本将同一试验的多个片段放到同一集合，不拆开导致泄漏。
每个连续片段是独立 LeRobot episode；50步动作块不可跨片段边界。
后续正式训练还需检查 episode 尾部 padding/mask 和采样权重，不直接把
片段长度当成独立示教数量。人工数据先审核动作、画面与扫码时点，再重算
聚合数据的 stats，和原始合格示教共同训练。

## 本次软件验证

2026-10-01：71项针对性测试通过，包含模拟两轮独立采集、未扫码拒绝保存、
批次名称路径验证、R故障/失联拒绝和“模型 -> 人工 -> 扫码 -> 完成录制”全链路、
迟到模型命令拒绝、夹持保留、永久停机、IK后时效校验、磁盘写入失败、队列
阻塞、只导人工、任务/断档切段、父试验分组和旧功能回归。
实际 v2 8K 资产/端口/存储预检通过，未发硬件命令。
上述资产预检是第一版历史结果；本次只做软件测试和启动器语法检查，未启动真机。
尚未完成：操作者空夹爪真机门禁、实测A接管延迟、抓取与恢复成功率。
