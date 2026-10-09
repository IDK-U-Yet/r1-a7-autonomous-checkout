import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { ConfigProvider, Modal, Progress, Skeleton, Tooltip, theme } from 'antd'
import {
  Bot,
  CheckCircle2,
  CircleAlert,
  CircleDot,
  CircleX,
  Copy,
  Expand,
  HardDrive,
  PauseCircle,
  ScanLine,
  WifiOff,
} from 'lucide-react'
import type { CameraStatus, ConsoleEvent, EventLevel, Snapshot } from './types'

const CAMERA_ORDER = ['global', 'left_wrist', 'right_wrist', 'scanner']

const PHASE_LABELS: Record<string, string> = {
  waiting: '等待采集器',
  ready: '就绪',
  scanning: '拾取并扫码',
  placing: '已扫码，执行放置',
  saving: '保存中',
  aborting: '废弃中',
  resetting: '复原中',
  complete: '批次完成',
  unknown: '状态未知',
}

const PHASE_DESCRIPTIONS: Record<string, string> = {
  waiting: '批次状态文件尚未创建。',
  ready: '场景已就绪，可以开始下一条 Episode。',
  scanning: '正在录制。等待扫码器返回有效的 item_accepted。',
  placing: '扫码已经成功，但本条尚未保存和计入批次。',
  saving: '采集器正在写入 Episode，请等待最终结果。',
  aborting: '当前 Episode 正在移入 rejected。',
  resetting: '上一条已经完成，请复原商品和机械臂。',
  complete: '当前批次已经达到目标数量。',
  unknown: '无法识别当前批次状态，请检查采集器终端。',
}

function formatBytes(value: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = value
  let index = 0
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024
    index += 1
  }
  return `${amount >= 100 || index === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[index]}`
}

function formatTime(wallTimeNs: number): string {
  if (!wallTimeNs) return '--:--:--'
  return new Date(wallTimeNs / 1_000_000).toLocaleTimeString('zh-CN', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function episodeLabel(value?: number | null): string {
  return typeof value === 'number' ? `episode_${String(value).padStart(4, '0')}` : '等待下一条'
}

function useSnapshot() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    let closed = false
    let socket: WebSocket | null = null
    let retryTimer: number | null = null

    const applyPayload = (text: string) => {
      try {
        const value = JSON.parse(text) as Snapshot
        if (!closed) setSnapshot(value)
      } catch {
        // Keep the last valid state if one message is incomplete.
      }
    }

    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${protocol}//${window.location.host}/ws/status`)
      socket.onopen = () => !closed && setConnected(true)
      socket.onmessage = (event) => applyPayload(String(event.data))
      socket.onerror = () => socket?.close()
      socket.onclose = () => {
        if (closed) return
        setConnected(false)
        retryTimer = window.setTimeout(connect, 1500)
      }
    }

    fetch('/api/snapshot', { cache: 'no-store' })
      .then((response) => response.ok ? response.text() : Promise.reject(new Error('snapshot')))
      .then(applyPayload)
      .catch(() => undefined)
    connect()

    return () => {
      closed = true
      if (retryTimer !== null) window.clearTimeout(retryTimer)
      socket?.close()
    }
  }, [])

  return { snapshot, connected }
}

function StatusItem({ online, warning, icon, children }: {
  online?: boolean
  warning?: boolean
  icon?: ReactNode
  children: ReactNode
}) {
  const state = warning ? 'warning' : online ? 'online' : 'offline'
  return (
    <span className={`status-item status-item--${state}`}>
      {icon ?? <span className="status-light" aria-hidden="true" />}
      {children}
    </span>
  )
}

function CameraTile({ cameraKey, camera, onExpand }: { cameraKey: string; camera?: CameraStatus; onExpand: () => void }) {
  const online = Boolean(camera?.online)
  return (
    <section className="camera-tile">
      <div className="camera-viewport">
        {online && <img src={`/stream/${camera?.key ?? cameraKey}`} alt={`${camera?.title ?? '相机'}实时画面`} />}
        {!online && (
          <div className="camera-offline" role="status">
            <WifiOff size={22} strokeWidth={1.8} />
            <span>等待视频流</span>
          </div>
        )}
        {online && <span className="live-indicator"><span />LIVE</span>}
        {online && (
          <Tooltip title="放大画面">
            <button className="icon-button camera-expand" type="button" aria-label={`放大${camera?.title ?? '相机'}`} onClick={onExpand}>
              <Expand size={16} strokeWidth={1.8} />
            </button>
          </Tooltip>
        )}
      </div>
      <footer className="camera-footer">
        <strong>{camera?.title ?? '相机'}</strong>
        <span className={camera?.low_fps ? 'metric-warning' : ''}>
          {camera?.width || 0}×{camera?.height || 0} · {camera?.fps?.toFixed(1) ?? '0.0'} FPS · {camera?.age_ms == null ? '--' : Math.round(camera.age_ms)} ms
        </span>
      </footer>
    </section>
  )
}

function EventIcon({ level, type }: { level: EventLevel; type: string }) {
  if (type.includes('duplicate')) return <Copy size={17} strokeWidth={1.8} />
  if (type === 'item_accepted') return <ScanLine size={17} strokeWidth={1.8} />
  if (level === 'success') return <CheckCircle2 size={17} strokeWidth={1.8} />
  if (level === 'error') return <CircleX size={17} strokeWidth={1.8} />
  if (level === 'warning') return <CircleAlert size={17} strokeWidth={1.8} />
  return <CircleDot size={17} strokeWidth={1.8} />
}

function EventList({ events, emptyText }: { events: ConsoleEvent[]; emptyText: string }) {
  if (events.length === 0) {
    return <div className="event-empty">{emptyText}</div>
  }
  return (
    <div className="event-list">
      {events.slice(0, 40).map((event) => (
        <div className="event-row" key={event.id}>
          <time>{formatTime(event.wall_time_ns)}</time>
          <span className={`event-icon event-icon--${event.level}`}><EventIcon level={event.level} type={event.type} /></span>
          <span className="event-copy">
            <strong>{event.title}</strong>
            <small>{event.detail}</small>
          </span>
        </div>
      ))}
    </div>
  )
}

function App() {
  const { snapshot, connected } = useSnapshot()
  const [expanded, setExpanded] = useState<CameraStatus | null>(null)
  const cameraItems = snapshot?.cameras.items ?? {}
  const cameras = CAMERA_ORDER.map((key) => cameraItems[key])
  const onlineCameras = cameras.filter((camera) => camera?.online).length
  const scannerEvents = useMemo(() => snapshot?.events.filter((event) => event.source === 'scanner') ?? [], [snapshot])
  const collectionEvents = useMemo(() => snapshot?.events.filter((event) => event.source === 'collection') ?? [], [snapshot])
  const batch = snapshot?.batch
  const attempt = batch?.active_attempt
  const phase = batch?.phase ?? 'waiting'
  const lastEpisode = attempt?.episode_id ?? (phase === 'resetting' ? batch?.last_finalized?.episode_id : null)
  const activeProduct = attempt?.name ?? (
    batch?.selection_mode === 'flexible'
      ? undefined
      : batch?.products.find((product) => product.accepted < product.target)?.name
  )
  const progress = batch?.target ? Math.round((batch.accepted / batch.target) * 100) : 0
  const cameraWarning = cameras.some((camera) => camera?.low_fps)
  const alertText = batch?.error
    ?? snapshot?.collector.fault_reason
    ?? (cameraWarning ? '至少一路相机帧率持续低于 12 FPS，请检查 USB 带宽或相机服务。' : PHASE_DESCRIPTIONS[phase])

  return (
    <ConfigProvider theme={{ algorithm: theme.darkAlgorithm, token: { colorPrimary: '#45a978', borderRadius: 5, fontFamily: '"Noto Sans SC", "Microsoft YaHei", Arial, sans-serif' } }}>
      <div className="console-shell">
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark"><Bot size={19} strokeWidth={1.8} /></span>
            <span className="brand-copy"><strong>R1-A7 数采监控台</strong><small>{batch?.batch_id ?? '等待批次'}</small></span>
          </div>
          <div className="system-status" aria-label="系统状态">
            <StatusItem online={onlineCameras === 4} warning={onlineCameras > 0 && onlineCameras < 4}>四相机 {onlineCameras}/4</StatusItem>
            <StatusItem online={snapshot?.collector.online}>采集器{snapshot?.collector.online ? '在线' : '离线'}</StatusItem>
            <StatusItem online={snapshot?.scanner.online}>扫码器{snapshot?.scanner.online ? '就绪' : '离线'}</StatusItem>
            <StatusItem online={snapshot?.collector.state === 'active'} warning={snapshot?.collector.online && snapshot?.collector.state !== 'active'}>
              XR {snapshot?.collector.state === 'active' ? '跟随中' : snapshot?.collector.state === 'paused' ? '已暂停' : '等待'}
            </StatusItem>
            <StatusItem online={!snapshot?.disk.warning} warning={snapshot?.disk.warning} icon={<HardDrive size={14} strokeWidth={1.8} />}>
              {snapshot ? formatBytes(snapshot.disk.free_bytes) : '--'}
            </StatusItem>
          </div>
          <div className="connection-state"><span className={connected ? 'connection-dot connection-dot--online' : 'connection-dot'} />{connected ? '实时连接' : '正在重连'}</div>
        </header>

        <div className={`alert-strip alert-strip--${batch?.error || snapshot?.collector.fault_reason ? 'error' : cameraWarning ? 'warning' : 'info'}`}>
          {phase === 'resetting' ? <PauseCircle size={16} strokeWidth={1.8} /> : <CircleAlert size={16} strokeWidth={1.8} />}
          <span>{alertText ?? '正在读取现场状态。'}</span>
        </div>

        <div className="workspace">
          <main className="camera-grid" aria-label="四路相机实时画面">
            {snapshot ? cameras.map((camera, index) => (
              <CameraTile key={CAMERA_ORDER[index]} cameraKey={CAMERA_ORDER[index]} camera={camera} onExpand={() => camera && setExpanded(camera)} />
            )) : Array.from({ length: 4 }, (_, index) => (
              <section className="camera-tile camera-tile--loading" key={index}><Skeleton.Image active /><Skeleton active paragraph={{ rows: 1 }} title={false} /></section>
            ))}
          </main>

          <aside className="status-rail">
            <section className="episode-panel">
              <div className="panel-heading">
                <strong>当前 Episode</strong>
                <span className={`phase-label phase-label--${phase}`}>{PHASE_LABELS[phase] ?? phase}</span>
              </div>
              <div className="episode-number">{episodeLabel(lastEpisode)}</div>
              <div className="episode-product">
                {activeProduct ?? (batch?.selection_mode === 'flexible' ? '扫码后绑定商品' : '尚未绑定商品')}
                {attempt?.ordinal ? ` · 第 ${attempt.ordinal} / ${attempt.target_episodes ?? batch?.target ?? '-'} 条` : ''}
                {attempt?.attempt_number ? ` · 尝试 ${attempt.attempt_number}` : ''}
              </div>
              <div className={`task-stage task-stage--${phase}`}>{PHASE_DESCRIPTIONS[phase] ?? PHASE_DESCRIPTIONS.unknown}</div>
            </section>

            <section className="progress-panel">
              <div className="panel-heading"><strong>批次进度</strong><span>{progress}%</span></div>
              <div className="progress-value"><strong>{batch?.accepted ?? 0}</strong><span>/ {batch?.target ?? 0}</span></div>
              <Progress percent={progress} showInfo={false} strokeColor="#45a978" trailColor="#28322e" size="small" />
              <div className="progress-stats">
                <span>已接收<strong>{batch?.accepted ?? 0}</strong></span>
                <span>已废弃<strong>{batch?.rejected ?? 0}</strong></span>
                <span>{phase === 'resetting' ? '复原剩余' : '待采集'}<strong>{phase === 'resetting' ? `${Math.ceil(batch?.reset_remaining_seconds ?? 0)} s` : batch?.remaining ?? 0}</strong></span>
              </div>
              {batch?.products && batch.products.length > 1 && (
                <div className="product-progress">
                  {batch.products.map((product) => <span key={product.barcode}>{product.name}<b>{product.accepted}/{product.target}</b></span>)}
                </div>
              )}
            </section>

            <section className="logs-panel">
              <div className="log-section">
                <div className="log-heading"><strong>扫码日志</strong><span>{snapshot?.scanner.online ? '实时' : '等待连接'}</span></div>
                <EventList events={scannerEvents} emptyText="尚无扫码事件" />
              </div>
              <div className="log-section">
                <div className="log-heading"><strong>数采日志</strong><span>当前批次</span></div>
                <EventList events={collectionEvents} emptyText="尚无 Episode 结果" />
              </div>
            </section>
          </aside>
        </div>
      </div>

      <Modal className="camera-modal" open={Boolean(expanded)} footer={null} onCancel={() => setExpanded(null)} width="94vw" centered title={expanded?.title} destroyOnClose>
        {expanded && <img src={`/stream/${expanded.key}`} alt={`${expanded.title}放大实时画面`} />}
      </Modal>
    </ConfigProvider>
  )
}

export default App
