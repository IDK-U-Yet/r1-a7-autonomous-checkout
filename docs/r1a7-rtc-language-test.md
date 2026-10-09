# 独立 RTC 语言测试

入口：`scripts/lerobot/run-r1a7-pi05-rtc-language-test.sh`。
原来的 RTC 启动脚本、运行模块及数采入口不变。实验运行模块为
`data/unitree-lerobot/unitree_lerobot/r1a7_rtc_language.py`，是当前 RTC 入口的独立快照，后续原入口更新不会自动同步此实验快照。

## 测试指令

保持相机与 Dex1 服务开启，先退出其他遥操/推理发布程序。
以下选用 standard405 的 16K 检查点，固定 s=10、n=10、预取/初始延迟7、β=10、overlap；仅改提示词。

```bash
export R1A7_ROOT=/home/user/r1-a7-autonomous-checkout
export R1A7_REAL_INFERENCE=YES
TASK_BEFORE="Pick up the product, bring its barcode in front of the scanner, and hold it there until the scan is confirmed."
TASK_AFTER="Put the scanned product into the black tray on the right."

bash "$R1A7_ROOT/scripts/lerobot/run-r1a7-pi05-rtc-language-test.sh" \
  --policy-path "$R1A7_ROOT/runtime/checkpoints/pi05_oreo_standard405_a100_gpu3_full_b24_lr1e5_16k_01/checkpoints/016000/pretrained_model" \
  --dataset-root "$R1A7_ROOT/runtime/datasets/lerobot/r1a7_oreo_4cam_16d_standard405_train" \
  --network-interface enx00e04c187010 \
  --image-host 127.0.0.1 \
  --execution-steps 10 \
  --num-inference-steps 10 \
  --prefetch-steps 7 \
  --initial-delay-steps 7 \
  --max-guidance-weight 10 \
  --rtc-mask-mode overlap \
  --allow-task-override \
  --task-before "$TASK_BEFORE" \
  --task-after "$TASK_AFTER" \
  --expected-barcode 6901668005939 \
  --max-commands 0 \
  --output "$R1A7_ROOT/runtime/logs/rtc_16k_language_$(date +%Y%m%d_%H%M%S_%N).jsonl" \
  --confirm-real-robot I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE
```

增加 `--preflight` 可仅检查参数和本地模型/数据元信息，不启动硬件。
出现 `RTC READY` 后，按原流程启动扫码桥（已运行则不重复启动）：

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

P+回车暂停，N+回车继续，R+回车空夹爪受控回起点，Ctrl-C退出。
恢复起点使用原训练任务参考，不使用新提示词查询轨迹；需要时用
`--reference-task-before` 明确指定训练中存在的文本。

## 实验记录与解释

- 不加 `--allow-task-override` 仍拒绝训练数据中没有的提示词；开关只放行文本，不放宽硬件、相机、16维及15Hz检查。
- 日志 `rtc_task_contract` 和 `rtc_ready.task_contract` 保存两段实际提示词、参考任务、是否匹配训练文本。
- 先原文本10次、同义英文改写10次，使用相同位置/朝向与模型参数，记录抓取、衔接、扫码、放置及完整无介入成功数、耗时。中文另作一组，不与措辞实验混合。
- 同义改写成功只能证明措辞鲁棒性；不能排除模型忽略语言。扫码成功切换放置仍由外部事件驱动，不是模型自动预测子任务。
- 改目的地或改变动作顺序不是单纯同义改写；未训练的语义和反事实提示应先离线/影子验证，不直接让真机执行危险指令。

这里只验证软件参数与离线行为，不代表新提示词已有真机成功率。
