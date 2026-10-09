# Spec: annotation-editor

## 状态与目标

2026-10-01 存储与人工编辑已实现并通过定向测试；用户确认继续导出。依赖 dataset-playback。人工标注整条轨迹和帧区间，保存后能继续审核未标轨迹。
提供标注能力，不强制所有 Episode 都做细粒度阶段标注。

## 技术与结构

沿用 apps/r1-dataset-review 的 React/TypeScript 和 Python 后端。
backend/dataset_review/annotations.py 负责存储校验；frontend/src 的标注组件负责交互。
标注文件保存在 runtime/annotations/dataset-review/<dataset_id>/，源数据不变。

## 预定命令

后端测试已验证可用；前端构建使用 README 中本机 Node 指令。

```bash
cd /home/user/r1-a7-autonomous-checkout/apps/r1-dataset-review/frontend
pnpm lint
pnpm build
/home/user/r1-a7-autonomous-checkout/runtime/envs/pi05-262d2fd/bin/python -m pytest \
  /home/user/r1-a7-autonomous-checkout/apps/r1-dataset-review/backend/tests/test_annotations.py -q
```

应用运行命令同 SPEC-dataset-playback.md，不增加另一套后台服务。

## 标注合同

每条记录包含 schema_version、dataset_id、trajectory_id、源帧数、元数据指纹、revision、updated_at、review_status。

- episode_type：standard_pick / tabletop_recovery / 未指定，不依帧位置自动猜恢复类型。
- orientation：O1～O8 / 未指定，仅记录用户选择，不推断每 10 条必是一个朝向。
- review_status：未审核 / 通过 / 待复核 / 建议排除；建议排除不物理删除数据。
- stages：任意数量的带 label 的 [start_frame, end_frame_exclusive)；内置抓取、重新定向/翻转、扫码、放置，可自定义，不写死手臂角色。
- quality_intervals：错误动作/不必要停顿/恢复等标注，带独立 exclude_from_training 开关；“恢复”默认保留。
- notes：自由备注。全局主任务和现有 scan/place 文本只展示，不自动改写。

区间边界是 0 起始帧号和半开区间，不用浮点秒作为唯一定位依据。
阶段层允许未标部分；同层重叠须明确校验/提示，错误区间可与阶段重叠。
扫码 item_accepted 属于只读事件，不让人工标签伪造成功事件。

## 交互与接口

当前帧「设起点」「设终点」，编辑/删除区间，时间轴点击跳到区间。
I/O 对应起点/终点；结束按钮包含当前显示帧，内部保存 end=current_frame+1。
Ctrl+S 保存；未完成区间或非法区间阻止保存并提示。输入框聚焦不截获 I/O。
保存+下一条、未审核筛选、标注完成计数；切换/关闭有未保存内容时确认，不静默丢失。
单用户本机首版。GET/PUT annotation 带 revision；版本不符返回冲突，不覆盖较新的修改。
原子写入临时文件后 replace；更新失败保持旧文件有效。
写操作限定本机服务和允许的 Origin，拒绝跨站写入和任意文件路径。

## 风格示例

```python
def validate_interval(start: int, end: int, total_frames: int) -> None:
    if not 0 <= start < end <= total_frames:
        raise ValueError("invalid half-open frame interval")
```

接口使用整数帧号；未知 schema/revision 冲突明确报错，不自动修正人工输入。

## 验证与成功标准

pytest 检查空/越界区间、首末帧、同层重叠、来源指纹变化、保存失败、修订冲突、路径穿越。
浏览器做一条轨迹的打点、保存、刷新重载、修改、切换未保存提示；只测桌面。
源 JSON、Parquet、视频不变；刷新重启后能恢复标注，不假称恢复播放位置等同恢复已保存标签。

## 边界与待确认

- Always：标注独立存储，检查源身份/帧数，失败时保留已有记录。
- Ask first：修改训练 task、自动标注、多人协作、改数据格式。
- Never：直接删除 Episode、凭视觉宣称扫码成功、把阶段标签当作模型已启用的高层规划。

标签字段、手工打点快捷键及保存策略按已确认规格实施。计划见 tasks/dataset-review/plan.md。
