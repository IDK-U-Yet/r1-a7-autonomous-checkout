# R1-A7 数据回放与标注工作台

已完成四路回放、独立标注保存和可追溯导出；不接入机器人控制，不自动改变训练配置。

## 启动

```bash
export R1A7_ROOT=/home/user/r1-a7-autonomous-checkout
bash "$R1A7_ROOT/scripts/start-dataset-review.sh"
```

打开 http://127.0.0.1:8766/ 。默认注册合并后的 160 条普通轨迹及两批原始数据；不启动相机、扫码桥、遥操或 DDS。
Ctrl+C 停止回放服务，不影响原始数据。

指定其他数据集（可重复注册；指定后替代默认列表）：

```bash
bash "$R1A7_ROOT/scripts/start-dataset-review.sh" \
  --dataset-root "$R1A7_ROOT/runtime/datasets/lerobot/r1a7_oreo_4cam_16d_v2v3_standard160_train"
```

仅注册 LeRobot 时仍可回放视频；如果原始来源未注册/不存在，不能显示原始商品信息或扫码事件。任务分段边界会明确显示为“任务切换（非扫码事件）”。

## 操作

上排全局/扫码区，下排左腕/右腕。左侧选择轨迹，右侧查看商品、任务和事件。

| 操作 | 方法 |
|---|---|
| 播放/暂停 | 按钮或 Space |
| 前/后一帧 | 按钮或 ← / → |
| 任意位置 | 拖动时间轴；帧号从 0 起 |
| 扫码位置 | 点击扫码事件 |
| 倍速 | 0.25× / 0.5× / 1× / 2× |
| 上/下一条 | 时间轴上方按钮 |
| 单路放大 | 相机右下角放大按钮 |
| 完整/训练分段 | 顶部视图切换；无来源清单禁用完整轨迹 |

同步指四路对应同一录制样本，不代表四台相机硬件曝光同时发生。后端依据各路视频偏移定位画面；四路加载后整体更新，缺帧明确提示。
播放以录制时间推进，解码较慢时可能跳过显示帧；逐帧模式不会跳帧。FPS 标签是数据集采样率，不是浏览器实测渲染帧率。

## 标注操作

建议在「完整轨迹」视图标注，不必对所有 Episode 细分动作阶段。

1. 在右侧选择轨迹类型（standard_pick / tabletop_recovery）、朝向 O1～O8 和审核状态。不能确定时保留未指定；不会按每 10 条自动推测朝向。
2. 回放到区间起点，按 I 或点「设起点」。回放到最后一帧，按 O 或点「设终点」。O 包含当前帧，内部保存 `[start, current+1)`。
3. 选择动作阶段或质量区间，填写标签，点「添加区间」。支持自定义标签、编辑和移除；同层区间不允许重叠，阶段与质量区间可以交叉。
4. 质量类型「恢复」默认保留训练；错误动作/不必要停顿默认标记排除，可手动修改。此时只是标注，尚未产生训练过滤文件。
5. 写备注后按 Ctrl+S 或点「保存」。未加入列表的区间会阻止保存，先添加或取消打点。
6. 「保存并下一条」跳到后续未审核轨迹；可勾选左侧「仅未审核」继续工作。刷新/重启可重载已保存标签，不承诺恢复上次播放帧。

标注保存于 `runtime/annotations/dataset-review/<dataset_id>/<trajectory_id>.json`，不在原始数据或 LeRobot 目录内。
LeRobot 完整轨迹和训练分段各自标注，**不会自动互相投射**；同一原始批次与其转换后数据集也不会自动共用标签。导出完整轨迹时会映射到训练分段，但不覆写另一视图的标注，请避免多处重复标注。

「建议排除」不删除数据。扫码事件只读；任何人工阶段、质量标签都不会改写 task 或伪造 item_accepted。

### 保存协议与边界

独立 JSON 比改写数据集/增加数据库更适合当前单人离线审核；写入采用原子替换和 revision 冲突校验。
身份包含数据集、轨迹、源帧数、元信息指纹；运行中源元信息变化或重启后指纹不匹配会拒绝保存。指纹基于元信息内容及数据/索引文件大小和修改时间，不是整库视频内容校验。
GET/PUT `/api/datasets/{dataset_id}/trajectories/{trajectory_id}/annotation`，PUT 需同源 Origin、`X-R1-Annotation: 1` 和完整记录；409 表示来源/版本或 revision 冲突。
仅支持本机单进程保存，不提供多服务进程共享写入或多人协作。写操作拒绝跨站请求，用户不需要填写文件路径。

## 导出与训练过滤

1. 先保存当前标注，再点顶部「导出标注」。范围为当前数据集、当前视图的全部轨迹，不受左侧搜索/仅未审核筛选影响。
2. 默认只导出标注。转换后的 LeRobot 数据集可勾选「同时生成 training_filter.json」，动作块长度必须与策略 `chunk_size` 一致，不能用 `n_action_steps` 替代。
3. 点「预览计数」查看审核状态、允许/排除的训练起始帧数及警告。未审核、待复核不会自动排除；恢复区间默认保留。
4. 点「导出到新目录」。输出在 `runtime/annotations/dataset-review/<dataset_id>/exports/<UTC时间_随机后缀>/`，不覆盖旧版本。预览后若标注已变化，需重新预览。

| 文件 | 用途 |
|---|---|
| `annotations.json` | 原始人工标签与 revision、数据指纹、视图 |
| `source_mapping.jsonl` | 每段的来源批次、完整源路径、源/轨迹/全局帧范围 |
| `interval_mapping.jsonl` | 每个阶段或质量区间映射到分段的结果 |
| `summary.json` | 审核状态、计数、警告、快照哈希 |
| `training_filter.json` | 仅显式勾选时生成，兼容既有训练过滤接口 |
| `COMPLETE.json` | 整次导出完成标志；缺失时不要使用该目录 |

错误动作区间 `[a,b)` 对长度 N 的动作块，排除所有与其相交的起始帧：先与每个训练分段求交，再扩展到 `[max(0,a-N+1),b)`。不会跨 scan/place 边界误排上一分段。审核状态为「整条废弃」则排除全部关联分段；源数据依然保留。

命令行也可导出（输出目录必须是不存在的新目录）：

```bash
export R1A7_ROOT=/home/user/r1-a7-autonomous-checkout
bash "$R1A7_ROOT/scripts/export-dataset-review.sh" \
  --dataset-root "$R1A7_ROOT/runtime/datasets/lerobot/r1a7_oreo_4cam_16d_v2v3_standard160_train" \
  --annotation-root "$R1A7_ROOT/runtime/annotations/dataset-review" \
  --output-dir "$R1A7_ROOT/runtime/annotations/review_export_v1" \
  --view full \
  --include-training-filter \
  --action-chunk-size 50
```

50 是示例，不是对所有策略的保证。删去 `--include-training-filter` 可仅导出标注。
原始 JSON/JPEG 数据可导出标签，但本版不自动将原始批次标签搬到另一个转换数据集；训练过滤请在目标 LeRobot 数据集上标注/导出。

导出时严格核对当前元信息指纹、标注 revision 和帧范围；指纹不是全视频内容校验。**现有训练器只校验过滤文件的总帧数，不能独自防止同帧数的错数据集或错误 chunk_size。** 使用时须核对 `dataset_root`、指纹与策略长度，显式传入 `dataset.training_filter_path`，并基于过滤后的样本重新计算归一化统计。该工作台不会替你启用过滤、重算统计或启动训练。
自定义阶段/子任务标签用于分析和后续研究，不自动改现有 task，也不启用 PI0.5 的高层任务预测。

## 开发与检查命令

Python 复用 `runtime/envs/pi05-262d2fd/bin/python`，可通过 `REVIEW_PYTHON` 指定其他已安装 requirements 的环境。
前端沿用数采监控台已有依赖，首次在本机复用：

```bash
cd "$R1A7_ROOT/apps/r1-dataset-review/frontend"
ln -s "$R1A7_ROOT/apps/r1-collection-console/frontend/node_modules" node_modules
```

已有链接时不重复执行。正常环境中执行 `npm run build`；本机验证使用已配置的 Codex Node 运行时：

```bash
export PATH=/home/user/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin:$PATH
node node_modules/typescript/bin/tsc --noEmit
node node_modules/vite/bin/vite.js build
node node_modules/eslint/bin/eslint.js .
cd "$R1A7_ROOT"
PYTHONPATH=apps/r1-dataset-review/backend \
  runtime/envs/pi05-262d2fd/bin/python -m pytest apps/r1-dataset-review/backend/tests -q
```

测试覆盖 accepted 目录、来源映射、不同相机视频偏移、分段首尾、反向跳转、缺帧、路径越界及只读 API。没有全库转码或手机测试。
