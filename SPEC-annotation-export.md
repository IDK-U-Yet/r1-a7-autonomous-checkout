# Spec: annotation-export

## 状态与目标

2026-10-01 用户确认实施；已实现独立导出、区间映射及兼容训练过滤，依赖 annotation-editor。
导出可追溯人工标注和训练过滤清单，使人工标注可供后续数据处理使用。
不启动训练，不把新标签默认写入 LeRobot task，也不重编码视频。

## 技术与结构

backend/dataset_review/export.py：标注导出和原始帧到训练分段帧的映射。
backend/tests/test_export.py：边界和训练 action chunk 测试。
runtime/annotations/dataset-review/<dataset_id>/exports/：显式生成的新输出版本。
沿用 Python JSON 与现有训练过滤合同，缺依赖不自行安装。

## 预定命令

网页「导出标注」内显式选择是否生成训练过滤；实际命令：

```bash
bash /home/user/r1-a7-autonomous-checkout/scripts/export-dataset-review.sh \
  --dataset-root /home/user/r1-a7-autonomous-checkout/runtime/datasets/lerobot/r1a7_oreo_4cam_16d_v2v3_standard160_train \
  --annotation-root /home/user/r1-a7-autonomous-checkout/runtime/annotations/dataset-review \
  --output-dir /home/user/r1-a7-autonomous-checkout/runtime/annotations/dataset-review/DATASET_ID/exports/REVIEW_VERSION \
  --include-training-filter --action-chunk-size 50

/home/user/r1-a7-autonomous-checkout/runtime/envs/pi05-262d2fd/bin/python -m pytest \
  /home/user/r1-a7-autonomous-checkout/apps/r1-dataset-review/backend/tests/test_export.py -q
```

CLI 由独立导出脚本设置 PYTHONPATH；DATASET_ID/REVIEW_VERSION 是占位值，不是现有路径。--annotation-root 指标注根目录，不是 dataset_id 子目录。
50 是示例，必须与准备训练的模型 action chunk 配置一致，不能用 n_action_steps 替代。

## 导出合同

- annotations.json：完整标注，包含来源身份、修订版本、轨迹类型、朝向、阶段、质量区间、备注。
- source mapping：使用 source_segments.jsonl 的 source_start/end 和 output_episode_index 对齐，不能按 episode 目录名排序推算。
- 有来源清单时，完整轨迹区间与每个 segment 求交，再转换到分段局部和 dataset 全局坐标。
- 无来源清单仅支持训练分段自身的标注和过滤，拒绝猜测跨段对应关系。
- training_filter.json：兼容项目现有逻辑过滤结构；实现前阅读实际 sampler/filter 代码核实字段。
- 严格绑定数据集指纹、总帧数和 chunk_size；错数据集/越界/缺映射阻止生成过滤，不静默跳过。
- 对错误动作 [a,b)，排除 action chunk 与其相交的起始帧，并截断到所属 Episode；不能误排前一 Episode。
- 恢复段默认不排除，除非操作员明确设置 exclude_from_training。
- 建议排除整条轨迹时排除全部关联分段，不丢另一任务阶段。
- 同时输出 excluded/allowed counts、源轨迹清单、未审核条数、警告与排除理由，支持空排除集合。
- 所有导出生成新目录；源数据和原标注不修改，已存在输出不默认覆盖。

## 风格示例

```python
def intersect(start: int, end: int, segment_start: int, segment_end: int):
    left, right = max(start, segment_start), min(end, segment_end)
    return (left, right) if left < right else None
```

保持原始标签与派生过滤结果分离，不将训练采样起始帧误称为错误动作帧。

## 验证与成功标准

pytest 验证跨 scan/place 区间、action chunk 相交、首尾边界、同名跨批次来源、空标注、修订变更和旧数据不覆盖。
一个小型合成样本检查导出允许起始帧数与既有训练 sampler 一致，不运行训练验证此工具。
UI 明示：导出过滤不等于已经启用；训练需显式传入过滤文件，并按实际使用样本重新计算统计。
子任务标签仅供研究和分析；当前 PI0.5 连续动作训练不会因出现新标签自动预测子任务。

## 边界与待确认

- Always：保留来源、按 Episode 截断、报告未审核状态、验证实际训练过滤字段。
- Ask first：改变任务提示词、升级到 v3.1、新训练特征、服务器上传或启动训练。
- Never：物理删除错误数据、自动改原始 stats、把标注完成等同训练效果改善。

本模块只导出标注与过滤；基于子任务改 task/高层规划训练不是第一版范围。
