# Implementation Plan: R1-A7 数据回放与标注工作台

日期：2026-10-01。状态：回放与标注获用户确认继续；导出、独立入口和文档已实现，通过定向桌面验收，等待用户审阅交付。
任务清单：tasks/dataset-review/todo.md。旧 tasks/plan.md 和 tasks/todo.md 不修改。

## 范围与视觉

独立本地网页，默认 127.0.0.1:8766。左侧轨迹列表、中部 2×2 视频及统一时间轴、右侧大字号标注编辑。
上排全局/扫码区，下排左腕/右腕。A 版：炭黑/深灰绿色背景、浅色文字、绿色操作按钮。
参考图：/home/user/.codex/generated_images/01a02df2-b5b2-7b80-8e40-fafd4f1252b0/exec-956d86ad-194b-4038-b878-d6acf756013f.png。
图中图片、数值、分辨率和阶段边界均为示意，实际 UI 从数据读取，不能照图硬编码。

依赖图：dataset-playback → annotation-editor → annotation-export。
实现时沿用 SPEC-dataset-playback.md、SPEC-annotation-editor.md、SPEC-annotation-export.md。

## 实现方式与取舍

1. 独立 apps/r1-dataset-review，不改现有数采监控台，也不复用其 ImageClient/V4L2 连接。
2. React/TypeScript/Vite/Ant Design 使用已有依赖版本；不更换前端组件体系。
3. Python aiohttp + PyAV + PyArrow，复用 runtime/envs/pi05-262d2fd。所需包已发现，不要求安装/升级 LeRobot。
4. 后端按元数据读取四路当前帧包，浏览器预解码四路后统一更新。不能把四个独立 video 元素的播放时间当同步保证。
5. 原始批次直接取 JSON/JPEG；LeRobot 按 episodes/data Parquet 定位视频片段。首版按关键帧定位并解码，缓存上限 32 帧包、8 个视频容器；不先转完整视频库。自动播放按时间推进，慢解码可能跳显示帧，逐帧仍逐一定位。
6. 来源清单重组完整轨迹；无清单只显示分段。原始源路径已失效仍可从 LeRobot 文件回放。
7. 标注单独保存为带身份/指纹/revision 的 JSON。单用户、原子更新；不增加数据库迁移。
8. 字段与阶段标签可扩展，商品/SKU/批次/task 均从数据读取，不写死奥利奥或某个左/右臂策略。
9. 标注和训练过滤独立导出。子任务不自动改 task，不声称标注后 PI0.5 自动具备高层规划。

## 实施顺序

### 第一阶段：先交付能看的同步回放

1. 建立只读数据索引和可追溯帧坐标合同，验证原始/训练分段来源关系。
2. 验证高风险视频读取路径：四路不同视频偏移、跨 scan/place 和 MP4 文件边界。
3. 实现本机服务和 A 版基础页面，选一条轨迹可看到同帧四路画面。
4. 增加播放/暂停、倍速、逐帧、拖动、上一条/下一条、放大和事件时间轴。

检查点：用户可看真实数据；人工抽查首条、v2/v3 交界、末条，不遍历解码全库。

### 第二阶段：人工标注与续标

5. 建立独立标注存储及字段/区间/来源/revision 校验。
6. 接入类型/朝向/审核状态/阶段/质量区间/备注编辑、I/O 打点、Ctrl+S、保存并下一条。
7. 增加未审核过滤、完成计数、刷新重载和未保存内容确认。

检查点：标一条、保存、刷新重载、再修改；源数据不变。

### 第三阶段：导出与正式入口

8. 导出标注与来源清单，跨完整轨迹/训练分段进行区间映射。
9. 导出兼容既有准备脚本的 training_filter.json；核对 action chunk 相交和允许起始帧数。
10. 完成独立启动脚本、README、使用手册命令，必要桌面浏览器验收与定向测试。

检查点：真实首条回放与标注可用，合成样本过滤结果正确，启动不接触硬件。

详细候选任务及每项验收见 todo.md；计划批准前不执行任务。

## 当前训练过滤合同（已检查现有代码）

scripts/lerobot/prepare_r1a7_dataset.py 的 _allowed_mask 读取 total_frames、excluded_ranges 中的
dataset_start/dataset_end_exclusive，以及 allowed_training_starts。
导出必须按 Episode 范围截断，并排除未来 action chunk 与错误区间相交的采样起点。
chunk_size 来自待训练策略配置，不能将实际下发的 n_action_steps 当作 chunk_size。
过滤启用和重算统计是后续显式操作，本工具不自动启动训练或覆盖统计。

## 验证命令（实现后使用）

```bash
export R1A7_ROOT=/home/user/r1-a7-autonomous-checkout
export REVIEW_PYTHON="$R1A7_ROOT/runtime/envs/pi05-262d2fd/bin/python"
PYTHONPATH="$R1A7_ROOT/apps/r1-dataset-review/backend" \
  "$REVIEW_PYTHON" -m pytest "$R1A7_ROOT/apps/r1-dataset-review/backend/tests" -q
cd "$R1A7_ROOT/apps/r1-dataset-review/frontend"
pnpm lint
pnpm build
bash -n "$R1A7_ROOT/scripts/start-dataset-review.sh"
```

浏览器只做桌面端核心流程，不做手机适配、全库视频重新验证或真机回放。

## 风险与处理

| 风险 | 处理 |
| --- | --- |
| 各相机视频文件偏移不同，MP4 内含多段 | 按每路元数据 from_timestamp 和数据帧 timestamp 定位，不统一假设文件起点 |
| 同名 episode_0000 来自不同批次 | 身份包含数据集和来源批次，核对 source_segments |
| 缺路、乱序解码或旧帧留在画面 | 单帧包编号及预解码统一更新，缺路显示缺失，不伪装同步 |
| 首次拖动时 AV1 解码慢 | 优先验证真实偏移读取，有界缓存/顺序预取；无法保持倍率时显示实际播放状态 |
| 每次按原始 FPS 解码但前端慢 | 统一时钟允许跳显示帧，暂停逐帧始终精确，不虚报完整帧率 |
| 标注落到错误数据版本 | 元数据指纹+源帧数+revision，错版本拒绝保存/导出 |
| 滑动操作清掉未保存标注 | 标注草稿与播放位置分离，切条时明确确认 |
| 标注引入语言训练矛盾 | 不自动改现有 task，导出字段与训练输入分离 |

## 边界与批准

不改 LeRobot 0.6.2 固定源码，不接入机器人控制，不改相机参数，不安装新依赖，不删除/覆写数据。
不覆盖其他工作树修改，不做服务器上传/训练/真机测试。
如需要添加依赖或改变已确认数据合同，先说明具体原因与影响。
待用户确认本实施顺序后，再按任务逐项编码；检查点不因纯测试通过而宣称真机或模型效果已验证。
