export interface FileRecord {
  input_path: string
  filename: string
  output_path: string | null
  status: string
  engines_used: string[]
  confidence_summary: Record<string, number | null>
  languages: string[]
  layout_type: string | null
  fallback_fired: boolean
  flagged: boolean
  flag_reason: string | null
  error: string | null
  created_at: string
  updated_at: string
}

export interface ProcessingFile extends FileRecord {
  elapsed_seconds: number
}

export interface LiveStatus {
  status_counts: Record<string, number>
  total: number
  processing: ProcessingFile[]
}

export interface QcCurrent {
  passthrough_count: number
  skip_text_count: number
  ocr_count: number
  mean_confidence: number | null
  engine_win_counts: Record<string, number>
}

export interface RunSummary {
  id: number
  started_at: string
  finished_at: string
  duration_seconds: number
  total_files: number
  done_count: number
  failed_count: number
  flagged_count: number
  passthrough_count: number
  skip_text_count: number
  ocr_count: number
  mean_confidence: number | null
  engine_win_counts: Record<string, number>
  failed_files: { input_path: string; error: string | null }[]
}

const BASE = '/api'

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${path} -> ${res.status}`)
  return res.json() as Promise<T>
}

export const api = {
  liveStatus: () => getJson<LiveStatus>('/status/live'),
  files: (status?: string) =>
    getJson<{ total: number; files: FileRecord[] }>(
      `/files${status ? `?status=${encodeURIComponent(status)}` : ''}`,
    ),
  failures: () => getJson<{ failures: FileRecord[] }>('/failures'),
  qcCurrent: () => getJson<QcCurrent>('/qc/current'),
  historyRuns: () => getJson<{ runs: RunSummary[] }>('/history/runs'),
}
