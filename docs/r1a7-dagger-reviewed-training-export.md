# DAgger 人工审核训练导出

审核目录不是训练目录。旧 `dagger_audit_only=true` 数据继续禁止直接转换；不能只翻转这个开关。

## 审核契约

- 只选择回放工具标注为 `approved` 的轨迹，`excluded` 和未审核轨迹不导出。
- 每条保留审核快照、修订号、原始 trial/frame 来源，以及 JSON SHA256。
- 明确使用 `--include-reviewed-hold` 和 `--include-reviewed-transition` 时，保留人工确认的 HOLD、接管渐变帧，包括 `expert_action_valid=false`。该标记和 `control_source` 不改写。
- 即使审核通过，`policy` 动作、`sample_valid!=true` 仍拒绝导出。尚未应用的训练排除区间也会拦截，避免忽略标注。
- 独立复制四路图片；源审核文件不修改。目标已存在时拒绝覆盖。清单最后生成，未完成副本不能转换。
- 清单/校验用于一致性检查，不代表具有防篡改签名；媒体复制不是逐图哈希校验。

## 命令

在工程目录执行，目标必须是新的目录：

```bash
PYTHONPATH=data/unitree-lerobot:apps/r1-dataset-review/backend \
runtime/envs/pi05-262d2fd/bin/python -m unitree_lerobot.utils.reviewed_dagger \
  --source 'runtime/datasets/r1a7_dagger_candidates/dagger—r1a7_oreo_4cam_16d_v2v3_standard160_train/20261002_reviewed34_pause_trimmed' \
  --destination runtime/datasets/r1a7_dagger_training_exports/r1a7_oreo_4cam_16d_v2v3_standard160_train/20261002_reviewed33_v1 \
  --annotations runtime/annotations/dataset-review \
  --include-reviewed-hold --include-reviewed-transition
```

生成的 `reviewed_training_export.json` 是转换器的明确授权入口。转换器核对完整 episode 集合、JSON 哈希、批准状态及每帧来源，不接受仅添加 `dagger_reviewed_export` 标记的目录。

## 回放 episode 与训练序列

原始导出仍是从首次接管到本轮结束的完整 episode，不为回放拆 scan/place。
LeRobot 转换沿用每帧 `active_task` 分任务训练；审核导出额外在 `continuity_id` 变化或 `edit_discontinuity_before` 处分割训练序列。
因此删除暂停后的拼接仍可回放，但未来动作块不能跨越剪辑/重对齐边界，所有批准帧均保留。
训练 episode 数可能大于原始 33 条，不能将它误报为更多独立示教。
输出 `meta/source_segments.jsonl` 保留原 trial 身份与边界规则，后续 train/val/test 必须按 `source_trial_id` 分组。

2026-10-02 本批：审核目录共 34 条，排除 `episode_0009`，导出 33 条、9,809 帧。
正式 LeRobot 视频转换和训练需另行执行，此处只生成并验证训练原始副本。
