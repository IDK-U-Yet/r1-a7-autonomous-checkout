# R1-A7 独立 guided RTC 规格

## 目标

为原始 standard160 8K π0.5 增加独立 guided RTC 推理入口。
普通推理、数采、DAgger 入口和检查点保持不变；本阶段不启动真机。

## 技术与接口

- 使用已有 `runtime/lerobot-a100-262d2fd` 的 PyTorch π0.5/RTC，不升级依赖。
- 使用 `pi05_oreo_v2v3_standard160_a100_gpu3_full_b24_lr1e5_8k_01/checkpoints/008000/pretrained_model`。
- 保持 16D 动作契约、四相机输入、15 Hz 策略执行时间尺度。
- 每次生成 50 步；执行窗口为 10 步，独立入口预取阈值为 7 步，可配置。
  普通入口仍为 5 步；离线 5 次引导推理为 359–378 ms，7 步提供约 467 ms 窗口。
- 模型采样接收同一处理器归一化后的剩余动作前缀，使用官方 soft-mask guidance。
- 延迟按实际控制步进记录，不把固定预取步数直接当成实测延迟。
- 区分已承诺执行前缀和可替换未来动作；限幅或拒绝后的动作不得冒充原始预测已执行。
- `item_accepted`、暂停、重启、故障均使队列失效，并拒绝旧 generation 的晚到结果。
- 队列耗尽时保持，不重复消费旧动作；真正反馈故障仍停止。

## 工程结构

- 独立 `r1a7_rtc_*` 模块：模型适配、队列/时间轴、推理入口。
- `scripts/lerobot/run-r1a7-pi05-rtc-real.sh`：独立启动器。
- `data/unitree-lerobot/test/test_r1a7_rtc*.py`：必要离线测试。
- 使用手册只新增 RTC 实验章节，不改现有数采命令。

## 风格

沿用 Python 类型注解、dataclass 和现有 JSONL 事件风格，例如：

```python
{"event": "rtc_result_discarded", "generation": generation, "reason": "task_changed"}
```

## 验证策略与命令

实施后仅执行对应离线测试：

```bash
export R1A7_ROOT=/home/user/r1-a7-autonomous-checkout
PYTHONPATH="$R1A7_ROOT/data/unitree-lerobot:$R1A7_ROOT/runtime/lerobot-a100-262d2fd/src:$R1A7_ROOT/robot/xr-teleoperate:$R1A7_ROOT/robot/xr-teleoperate/teleop/teleimager/src" \
  "$R1A7_ROOT/runtime/envs/pi05-262d2fd/bin/python" -m pytest \
  "$R1A7_ROOT/data/unitree-lerobot/test" -k r1a7_rtc -q
```

测试覆盖归一化往返、延迟跨步、前缀/窗口对齐、task 切换、暂停和旧结果丢弃。
实际模型离线测试使用独立日志；记录推理耗时、队列长度、等待次数和块间目标变化。
真实机器人操作另经用户确认，不以离线通过宣称真机效果改善。

## 边界

- 必须：保留用户现有改动、来源数据、检查点和硬件反馈保护。
- 先确认：新增依赖、升级 runtime、修改现有入口、启动真机。
- 禁止：修改训练权重、改动作播放频率来加速轨迹、把插值称作 RTC、推送无关改动。

## 验收

独立入口确实调用 guided RTC；延迟/前缀对齐测试通过；切换后不执行旧任务动作；
普通入口不启用 RTC；提供独立启动命令和必要诊断记录。

## 未决事项

无额外硬件参数修改。上述为首版实现边界，真机表现尚未验证。

## 真机卡顿修复：2026-10-03

首轮日志 170 次动作中 167 次夹爪限幅触发整队列失效，所有 167 次合并
均为无前缀 bootstrap，实际目标流约 2.88Hz。不得把该结果归因于 guided 推理慢。
独立 RTC 现在在 worker 内按检查点 action 统计投影夹爪归一化目标，再生成硬件目标；
缓存前缀与执行目标一致，不通过 state 统计重新编码。手臂输出不投影，保留严格校验。
派发阶段比较桥接反馈与合法目标，边界浮点误差不清队列；意外修改、拒绝、故障仍失效。
掩码截止位置按实际前缀长度设置（H-s），软区长度 H-s-d，不再把调度窗口 10
直接作为掩码终点。初次预取剩余 47 步，正常后续预取剩余 40 步；超时按实际长度变化。
该修复不改共享 runtime、普通推理、数采或权重；真机改善须重新验证。
修复后专项 22 项测试通过，包括夹爪浮点边界限幅时连续派发 20 个动作且进入
guided 合并；原始 8K 保存快照模拟消费 40 步，仅启动等待 4 周期，随后没有断供。
三次 guided 合并耗时 390.37、648.64、477.43 ms，实际消费延迟 7、10、8 步；
超出预留窗口并不表示执行停顿，但会增加观测滞后，需真机日志进一步量化。
日志 `runtime/logs/rtc_8k_projection_overlap_fix_20261003.jsonl` 全部硬件发送为 false。

### 单变量比较：暂缓掩码对齐

2026-10-04：跨相机时间差超限不立即退出，而是先失效队列、保持当前位置，
最多等待 0.5 秒重新满足既有观测校验，然后从新观测重新推理。等待中仍验证
硬件反馈，超时停止；独立旧帧、相机形状错误、反馈故障不进入运行恢复。
此为针对扫码相机短时滞后的容错，不扩大观测有效范围，不恢复旧动作。

后续按用户批准增加可选 `--rtc-mask-mode configured|overlap` 对照。默认仍保持
configured；overlap 每块以实际前缀长度作为 mask endpoint。两组其他控制逻辑
相同，日志记录模式与有效截止位置。bootstrap 无前缀时不启用掩码。
该开关不改变调度方式，不能称为完整论文算法严格复现。

后续真机日志 `rtc_8k_20261003_232241.jsonl` 显示左腕俯仰 38 次拒绝、40 次
限幅，使队列再次频繁重置。按用户批准，将 worker 的合法目标投影扩展到全部
16 维，归一化上下界仍由原硬限位与检查点 action 统计计算，不改变实际关节范围。
仅浮点边界修正（<=1e-6）可保留队列，非有限数据、意外派发修改和反馈故障仍保护。
记录 target_projection 分析饱和频率；投影维持流畅不等于解决模型策略或碰撞问题。
26 项 RTC 专项回归通过；原始 8K 保存的 action 归一化器 CPU 往返检查通过，
腕部越界输出投影后的硬件目标为 1.614400029 rad（仅浮点舍入，不扩展硬限位）。
未进行本轮真机操作。

启动就绪门控：先等相机新鲜观测，完成无预测命令派发的普通/引导推理预热，
再返回起始姿态并重新检查观测，之后才发布 READY。仅启动阶段在保持状态下
最多等相机 5 秒，帧龄保护不变；运行时旧帧仍停止，硬件状态故障从不重试。
先读状态 RPC、再取相机快照，避免慢 RPC 使之前抓取的帧人为变旧。
日志增加读取耗时与循环超时，不以预热保证长期调度或摄像头绝不断流。

用户随后要求保持原掩码参数。当前恢复使用配置中的截止位置 10，不再逐块传入
旧块剩余长度；保留归一化夹爪合法投影与队列连续性修复。H-s 对齐留待独立实验。
`prefix_overlap_steps` 只表示旧块剩余长度。上面的真实权重离线测量是在 H-s
掩码版本上取得的，不能作为当前恢复配置的新测量结果。

## 2026-10-03 软件证据

- RTC 专项 17 项测试通过，包含模拟完整主循环/硬件反馈故障后的撤销控制。
- 加上普通异步推理、task 状态和动作处理回归，共 36 项通过。
- launcher `--preflight` 通过，仅检查配置，无硬件连接。
- 原始 8K 和保存的四路 RGB 诊断快照：6 次真实离线前向输出均为有限 `(50,16)`。
  首次 2663 ms；5 次带前缀为 361.9、359.2、378.4、371.9、376.9 ms。
  样本量小、静态图像，没有进行配对普通推理对照，不代表真机成功率或长期 P95。
- pinned RTC 在去噪器前向之后才设置 `requires_grad`；独立 `R1RTCProcessor`
  把设置提前，用线性去噪器解析 Jacobian 测试确认；共享 runtime 未修改。
- 实际日志原始文件：`/tmp/r1a7_rtc_8k_probe6_20261003.jsonl`；本节保存摘要，
  `/tmp` 清理不影响源数据或权重。

## 后续异步离线验证

使用同一原始 8K 权重与保存快照，模拟 15Hz 消费 40 个动作，没有实例化机器人通信客户端。
3 次带前缀接管耗时 372.47、359.19、371.66 ms；每次引导前缀为 7 步、接管时实际
消费也为 7 步；仅 bootstrap 等待 4 个周期，后续队列未耗尽。
日志：`runtime/logs/rtc_8k_offline_async_20261003.jsonl`，全部记录
`hardware_commands_sent=false`。它验证软件时序，不是闭环物理仿真，也不证明抓取成功。

离线入口新增 `--async-steps N` 可复现模拟时间轴。bootstrap 冷启动耗时不用于更新
下一块的硬前缀长度，避免首次 CUDA 延迟污染稳定引导参数；该行为有独立回归测试。
# RTC comparison instrumentation (2026-10-04)

Runtime recovery now includes a stale camera frame as well as cross-camera
skew. Both invalidate the old generation and HOLD before reacquisition, bounded
to 0.5 seconds with all original age/skew checks retained. Feedback, shape and
timestamp faults remain fatal. No old queued action is resumed. Compare new
s=10/9 runs under this same behavior; pre-change runs are historical only.

The independent RTC path measures synchronized model-call latency separately
from total worker processing. Legacy `inference_ms` retains its meaning and
delay adaptation still uses total worker latency; dispatch/queue behavior is
unchanged. Merged results capture conditioned, reserved and consumed delays from
their own job. READY captures the comparison configuration. Dispatch logs add a
monotonic timestamp; older logs fall back to wall time and cannot supply isolated
model latency.

`r1a7_rtc_compare` analyzes guided chunks separately from bootstrap and reports
target seam continuity, not measured robot jitter. Operator trial annotations
are kept separately; unknown trials are not manufactured as successes/failures.
Compare overlap execution windows 10, 9, optionally 8 with other parameters
fixed. Actual mask length also changes with the remaining prefix, so this tests
the coupled scheduling/overlap configuration rather than isolated reactivity.
See 使用手册 §12.5 for commands and metric denominators.

## Independent denoising-step experiment

The real launcher also accepts explicit `--policy-path` and `--dataset-root`;
tokenizer follows the selected checkpoint. Default standard160 8K is retained.
standard405 16K experiments select their matching dataset for metadata,
processor statistics loading and return-pose references. Existing 16D/four-view/
15Hz/task checks apply before hardware startup; preflight and READY record both
paths. This does not upgrade shared dependencies or modify checkpoint files.

`--num-inference-steps` defaults to10 and accepts positive integers. RTCSession
passes it as per-call `num_steps` into PI05 predict_action_chunk/sample_actions;
checkpoint and policy configuration files remain unchanged. Bootstrap, warmup
and guided calls use the same override. READY/preflight report the requested
effective sampling count. Compare 10→7, optionally5, with s8, β10, prefetch7
unchanged. Reduced compute latency does not bypass the committed old-action
window. Fewer integration steps can change action/guidance quality; benchmark
before changing scheduling. Only the independent real RTC CLI exposes this flag.
