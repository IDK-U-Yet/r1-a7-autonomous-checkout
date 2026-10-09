# 数据回放与标注工具：能力地图

2026-10-01。状态：能力边界、模块规格和 A 版 UI 已获用户确认；实施计划待审阅。

| 模块 ID | 职责 | 依赖 | 规格 |
| --- | --- | --- | --- |
| dataset-playback | 原始数据和 LeRobot v3 读取、四路同步回放、来源定位 | 无 | SPEC-dataset-playback.md |
| annotation-editor | 轨迹标签、帧区间、审核状态、持久化 | dataset-playback | SPEC-annotation-editor.md |
| annotation-export | 标注导出、训练过滤区间映射 | annotation-editor | SPEC-annotation-export.md |

构建顺序：dataset-playback → annotation-editor → annotation-export。

画面顺序固定为上排「全局 / 扫码区」，下排「左腕 / 右腕」。
视觉方案：A 版深色专业工作台，炭黑/深灰绿色底、白色文字、绿色选中态和保存按钮。
独立本地网页，不改变现有实时数采监控台，不接入 DDS、夹爪、PICO 或相机采集进程。
原始数据只读，标注与导出结果独立保存。第一版不做 AI 自动标注、目标框、多人协作或手机适配。

相关既有代码：

- apps/r1-collection-console：复用视觉风格与已有前端依赖，不复用实时相机连接逻辑。
- data/unitree-lerobot/unitree_lerobot/utils/annotate_r1a7_recovery_intervals.py：参考帧区间与恢复标注语义。
- runtime/lerobot-a100-262d2fd/src/lerobot/scripts/lerobot_dataset_viz.py：官方本地可视化实现参考，不修改固定的 LeRobot 源码。

默认开发位置：apps/r1-dataset-review；标注：runtime/annotations/dataset-review；启动脚本：scripts/start-dataset-review.sh。
首次上线后再更新使用手册中的真实可运行命令。现有 tasks/plan.md 属于另一项 DAgger 工作，不覆盖。
用户已同意本功能使用 tasks/dataset-review/plan.md 和 tasks/dataset-review/todo.md，作为本功能唯一计划/任务清单。
