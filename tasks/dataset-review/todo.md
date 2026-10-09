# Dataset review：候选任务清单

日期：2026-10-01。状态：回放已验收；标注阶段完成定向验证，等待用户确认后继续导出。
唯一计划：tasks/dataset-review/plan.md。代码根以下简称 apps/r1-dataset-review。

## Task 1：建立只读数据目录

- [x] 提供原始批次/LeRobot 数据集及轨迹索引，接受启动时指定的多个数据根。
- [x] 稳定区分批次、原始 Episode、训练分段，来源范围连续且不跨批次误合并。
- [x] 验证缺来源目录、同名 Episode、错帧数与路径越界；不解码整库。
- 验证：pytest tests/test_catalog.py。
- 依赖：无。
- 文件：backend/dataset_review/__init__.py、catalog.py、models.py、backend/tests/test_catalog.py（4）。

## Task 2：读取指定帧的四路画面

- [x] 原始图片和 LeRobot 视频都输出带帧号/每路时间戳的统一帧包。
- [x] 跨视频片段时首尾正确；有界缓存、关键帧定位解码，缺路明确报错（首版不预取）。
- [x] 合成多路不同偏移视频测试，不误读相邻 Episode。
- 验证：pytest tests/test_frames.py；真实首条首/中/尾帧抽查。
- 依赖：Task 1。
- 文件：backend/dataset_review/frames.py、backend/tests/test_frames.py、tests/conftest.py（3）。

## Task 3：建立本机服务

- [x] 数据集、轨迹、元数据、帧包接口可读，指定数据根之外不能读文件。
- [x] 监听本机，慢解码不阻塞整个接口；不引入 DDS/V4L2/硬件客户端。
- [x] API 错误包含明确数据/索引/解码原因。
- 验证：pytest tests/test_server.py，导入不打开硬件。
- 依赖：Task 1、2。
- 文件：backend/dataset_review/server.py、backend/tests/test_server.py、backend/requirements.txt（3）。

## Task 4：建立可构建的前端入口

- [x] 沿用已有 React/Ant Design 版本和构建约定，不安装新组件体系。
- [x] TypeScript/Vite 配置与本机开发代理就绪；页面构建随 Task 5 的入口组件验证。
- 验证：pnpm exec vite --version、pnpm exec tsc --version；不更改数采监控台工程。
- 依赖：Task 3。
- 文件：frontend/package.json、tsconfig.json、vite.config.ts、index.html、eslint.config.js（5）。

## Task 5：交付 A 版静态帧浏览

- [x] 左侧选择真实轨迹，中部上排全局/扫码区、下排左腕/右腕，右侧真实元数据。
- [x] A 版配色、字号、绿色选中/按钮，大图优先；无图中硬编码商品、分辨率或统计。
- [x] 四路加载完成统一更新，旧请求不覆盖新选择。
- 验证：pnpm lint/build；桌面浏览器选择一条真实轨迹。
- 依赖：Task 4。
- 文件：frontend/src/main.tsx、App.tsx、styles.css、types.ts、api.ts（5）。

## Task 6：回放操作和统一时间轴

- [x] 播放/暂停、倍速、逐帧、拖动、切条、单路放大可用。
- [x] 显示真实当前帧/时间、来源视图、扫码事件和阶段边界；输入框不触发全局快捷键。
- [x] 缺路/慢解码/乱序响应不伪装同步，无来源清单不猜完整轨迹。
- 验证：构建；首条、v2/v3 交界、末条的核心浏览器流程。
- 依赖：Task 5。
- 文件：frontend/src/Playback.tsx、Timeline.tsx、usePlayback.ts、App.tsx、styles.css（5）。

## Checkpoint：同步回放

- [x] 后端 9 项测试、前端类型检查/ESLint/构建通过；真实第 1/80/81/160 条首帧、分段边界、尾帧四路解码通过。
- [x] 用户已确认 A 版布局和回放流程，可以推进标注。

## Task 7：独立标注存储

- [x] 类型/朝向/审核状态/阶段/质量区间/备注按已确认合同保存。
- [x] 身份指纹、帧区间、原子更新、revision 冲突校验有效；无写源数据接口。
- [x] 写操作拒绝跨站与任意路径，未知版本不默认迁移。
- 验证：pytest tests/test_annotations.py 与保存接口测试。
- 依赖：Task 6。
- 文件：backend/dataset_review/annotations.py、server.py、backend/tests/test_annotations.py、test_server.py（4）。

## Task 8：人工标注和续标 UI

- [x] 右侧表单和时间轴区间编辑；I/O 打点、Ctrl+S、保存并下一条。
- [x] 刷新重载已保存标签，切条确认未保存内容；未审核过滤与完成计数按已保存状态更新。
- [x] 阶段可自定义，错误排除与恢复保留独立；只读事件不可伪造扫码成功。
- 验证：一条轨迹打点、保存、刷新、修改，测试冲突/非法区间反馈。
- 依赖：Task 7。
- 文件：frontend/src/AnnotationPanel.tsx、types.ts、api.ts、App.tsx、Timeline.tsx（5）。

## Checkpoint：标注

- [x] 20 项后端测试通过；浏览器在独立临时标注目录完成打点、保存、刷新重载、未保存确认和保存并下一条，未写入正式标注目录。
- [x] 用户确认继续，推进导出。

## Task 9：标注与训练过滤导出

- [x] 新目录输出 annotations.json、来源清单和显式要求的 training_filter.json。
- [x] 原始帧到 scan/place 及全局帧映射正确；action chunk 相交排除不越 Episode。
- [x] 显示允许/排除计数、理由、未审核条数；输出已存在/错版本/缺映射时拒绝完整轨迹导出，无映射允许分段独立过滤。
- 验证：pytest tests/test_export.py，对照现有 _allowed_mask 的样本计数。
- 依赖：Task 8。
- 文件：backend/dataset_review/export.py、server.py、backend/tests/test_export.py、frontend/src/ExportPanel.tsx、App.tsx（5）。

## Task 10：启动入口与使用说明

- [x] 独立启动脚本 --dataset-root 可重复，--help 清楚，默认 127.0.0.1:8766。
- [x] README 和使用手册提供实际验证过的启动/标注/导出命令及快捷键。
- [x] 最终定向测试、构建、一次真实桌面核心流程通过；不做手机或真机测试。
- 验证：bash -n、--help、pnpm lint/build、后端 tests、桌面验收。
- 依赖：Task 9。
- 文件：scripts/start-dataset-review.sh、apps/r1-dataset-review/README.md、使用手册.md（3）。

## Checkpoint：完成

- [x] 同步、保存、导出均实际验证；用户确认并要求 Git 保存上传。
- 验收记录：真实 160 轨迹/56973 帧预览与独立导出成功，正式标签仍为未审核，未替用户自动标注；输出含 COMPLETE.json，浏览器无 error。合成坏动作案例对照既有 _allowed_mask 和 EpisodeAwareSampler。
- [x] 仅暂存本功能变更及使用手册对应章节；不混入 DAgger/相机/旧训练流程修改。用户已明确授权提交并 push。
