# Spec: dataset-playback

## 状态与目标

2026-10-01 用户确认规格与 A 版 UI，尚未实现。能力地图见 CAPABILITY-MAP-dataset-review.md。
为本机人工审核提供四路同步回放，支持当前原始批次和 LeRobot v3.0 数据。
不播放机器人动作到真机。「回放」仅指视频/数据查看，不调用 lerobot-replay。

## 已确认要求与假设

- 本地浏览器使用，默认监听 127.0.0.1:8766，不占用数采监控台的 8765。
- 独立应用，采用用户选定的 A 版深色专业工作台：炭黑/深灰绿色、白字、绿色按钮，较大字号；上排全局/扫码区，下排左腕/右腕。
- 左侧轨迹列表，中部四路画面与统一时间轴，右侧轨迹信息和标注面板。
- 原始 JSON 和 JPEG 只读；LeRobot Parquet 与分片 MP4 只读。不预先重编码整个数据集。
- 默认 15 FPS 数据，但实际时间轴读取文件中的 FPS/时间戳，不硬编码 15。
- 原始和训练分段是两种坐标系，必须在 UI 明示，不能混用帧编号。

## 技术与结构

拟复用 React 18 / TypeScript / Vite / Ant Design；Python 后端优先复用 aiohttp、PyAV、PyArrow。
实现前核实运行环境中已有依赖，缺依赖先提出，不升级 LeRobot 0.6.2 或修改 vendor。

```text
apps/r1-dataset-review/backend/dataset_review/  # 索引、解码、HTTP 服务
apps/r1-dataset-review/backend/tests/          # 纯数据和接口测试
apps/r1-dataset-review/frontend/src/          # 回放与标注 UI
scripts/start-dataset-review.sh               # 独立启动入口
```

## 预定命令

以下是拟实现接口，当前不能执行，不代表已存在的启动程序。

```bash
cd /home/user/r1-a7-autonomous-checkout/apps/r1-dataset-review/frontend
pnpm build
pnpm lint

export R1A7_ROOT=/home/user/r1-a7-autonomous-checkout
REVIEW_PYTHON="$R1A7_ROOT/runtime/envs/pi05-262d2fd/bin/python" \
  "$R1A7_ROOT/scripts/start-dataset-review.sh" \
  --dataset-root "$R1A7_ROOT/runtime/datasets/lerobot/r1a7_oreo_4cam_16d_v2v3_standard160_train" \
  --host 127.0.0.1 --port 8766

"$R1A7_ROOT/runtime/envs/pi05-262d2fd/bin/python" -m pytest \
  "$R1A7_ROOT/apps/r1-dataset-review/backend/tests/test_playback.py" -q
```

--dataset-root 可重复指定；默认只允许启动时注册的数据根，不开放任意路径读取。

## 数据与模块边界接口

- DatasetIdentity：由规范化数据根生成稳定 ID，记录格式、FPS、元数据指纹、总帧数。
- TrajectoryIdentity：dataset_id + 来源批次 + 原始 episode ID；同名 episode_0000 不跨批次合并。
- Segment：output_episode_index、source_start_frame、source_end_frame_exclusive、task、视频文件及每路 from/to_timestamp。
- FrameBundle：trajectory_id、frame_index、timestamp_s、每路帧及其时间戳/缺失状态。四路必须属于同一请求编号，完整后统一更新画面。
- 提供数据集/轨迹列表、轨迹元数据、指定帧包、事件读取接口；保存标注不属于本模块。
- 原始数据从 batch_state 的 accepted 清单读取，区分序号和实际目录 ID，不读取 rejected 作为默认合格数据。
- LeRobot 从 meta/info.json、episodes Parquet 和 data Parquet 定位，不能按 MP4 文件名猜 Episode。
- source_segments.jsonl 有可靠来源时按来源批次和轨迹分组，按源帧范围拼接 scan/place；严格检查连续性。
- 没有来源清单的 LeRobot 数据只提供训练分段视图，不能猜测哪两段属于同一原始轨迹。
- 来源原始目录不存在时仍能按映射从 LeRobot 视频回放，不因历史绝对路径失效而整体报错。
- 扫码事件优先来自 events.jsonl，使用 last_enqueued_frame_index + 1 显示任务边界；仅有 task 切换时注明是任务边界而非已验证扫码事件。

## 播放行为

播放/暂停、0.25/0.5/1/2 倍速、上一帧/下一帧、上一条/下一条、拖动定位、单相机放大。
Space 播放，左右箭头逐帧；输入框聚焦时不触发全局快捷键。
暂停/逐帧不得混合不同时刻画面。播放按统一时钟，可跳过显示帧赶进度，但当前帧编号必须真实。
乱序解码响应丢弃；缺路或解码失败明确显示，不用旧帧伪装当前帧。
采用有界缓存和预取，不一次加载整个数据集，不将图像张量移到 GPU。

## 风格示例

```python
def validate_frame_index(frame_index: int, total_frames: int) -> None:
    if not 0 <= frame_index < total_frames:
        raise ValueError("frame index is outside this trajectory")
```

Python snake_case / 类型提示；TypeScript 类型化接口；错误清楚区分数据损坏、映射缺失与解码失败。

## 验证与成功标准

- 小型合成视频测试四路不同起始时间、跨 MP4 分片、首尾帧及相邻 Episode 边界。
- 来源跨批次同名 ID、缺失来源目录、缺路、乱序请求、路径穿越拒绝测试。
- 使用现有合并数据的首条/跨 v2-v3 边界/末条做必要人工抽查，不解码遍历全部数据。
- 2×2 顺序正确，暂停逐帧四路按同一数据帧展示，不能播到相邻 Episode。
- 启动/浏览不创建机器人命令发布者，不打开 V4L2，不修改源数据。

## 边界

- Always：验证索引/路径/来源映射，保持数据只读，测试和构建后再提交。
- Ask first：新依赖、全库重编码、局域网暴露服务、扩大设备控制范围。
- Never：真机回放、上传私人数据、覆盖固定 LeRobot 源码、改已有数采启动配置。

## 待确认

本规格已获确认。实施计划保存在 tasks/dataset-review/plan.md，待用户审阅后编码。
缺来源映射时不能保证完整轨迹视图，须明确提示。
