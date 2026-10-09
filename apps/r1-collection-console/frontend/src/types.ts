export type EventLevel = 'success' | 'warning' | 'error' | 'info'

export interface CameraStatus {
  key: string
  title: string
  online: boolean
  low_fps: boolean
  fps: number
  age_ms: number | null
  width: number
  height: number
  sequence: number | null
}

export interface ProductProgress {
  barcode: string
  sku: string
  name: string
  accepted: number
  target: number
}

export interface Attempt {
  episode_id?: number
  barcode?: string | null
  sku?: string | null
  name?: string | null
  ordinal?: number | null
  target_episodes?: number | null
  attempt_number?: number | null
  item_accepted_event_id?: string | null
}

export interface ConsoleEvent {
  id: string
  source: 'scanner' | 'collection'
  type: string
  level: EventLevel
  wall_time_ns: number
  episode_id?: number | null
  barcode?: string | null
  title: string
  detail: string
}

export interface Snapshot {
  server_wall_time_ns: number
  cameras: {
    source_host: string
    error: string | null
    items: Record<string, CameraStatus>
  }
  batch: {
    available: boolean
    task_dir: string
    batch_id: string
    selection_mode: string | null
    status: string
    phase: string
    accepted: number
    target: number
    rejected: number
    remaining: number
    reset_remaining_seconds: number
    active_attempt: Attempt | null
    last_finalized: { episode_id?: number; status?: string } | null
    products: ProductProgress[]
    history: ConsoleEvent[]
    error: string | null
  }
  scanner: {
    available: boolean
    online: boolean
    state: string
    device: string | null
    resolved_device: string | null
    last_scan_at: number | null
    health_age_seconds: number | null
    error_code: string | null
    message: string | null
    log_path: string
  }
  collector: {
    available: boolean
    online: boolean
    state: string
    xr_age_ms: number | null
    fault_reason: string | null
    diagnostics_path: string
    age_seconds: number | null
  }
  disk: {
    path: string
    total_bytes: number
    used_bytes: number
    free_bytes: number
    warning: boolean
  }
  events: ConsoleEvent[]
}
