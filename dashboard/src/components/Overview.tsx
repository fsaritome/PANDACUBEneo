import { api, type LiveStatus, type QcCurrent } from '../api'
import { usePolling } from '../usePolling'
import { StatCard } from './StatCard'

export function Overview() {
  const { data: live, error: liveError } = usePolling<LiveStatus>(api.liveStatus, 3000)
  const { data: qc } = usePolling<QcCurrent>(api.qcCurrent, 5000)

  if (liveError) {
    return <div className="text-red-400">Failed to reach API: {liveError}</div>
  }
  if (!live) {
    return <div className="text-slate-400">Loading…</div>
  }

  const counts = live.status_counts
  const done = counts.done ?? 0
  const failed = counts.failed ?? 0
  const flagged = counts.flagged ?? 0
  const processing = counts.processing ?? 0
  const queued = counts.queued ?? 0
  const total = live.total
  const finished = done + failed + flagged
  const pct = total > 0 ? Math.round((finished / total) * 100) : 0

  return (
    <div className="space-y-6">
      <div>
        <div className="flex justify-between text-sm text-slate-400 mb-1">
          <span>Progress</span>
          <span>
            {finished} / {total} ({pct}%)
          </span>
        </div>
        <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Done" value={done} accent="text-emerald-400" />
        <StatCard label="Processing" value={processing} accent="text-amber-400" />
        <StatCard label="Queued" value={queued} accent="text-slate-300" />
        <StatCard label="Failed" value={failed} accent="text-red-400" />
        {flagged > 0 && <StatCard label="Flagged" value={flagged} accent="text-yellow-400" />}
      </div>

      {qc && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Passthrough (native text)" value={qc.passthrough_count} />
          <StatCard label="Skip-text (no OCR call)" value={qc.skip_text_count} />
          <StatCard label="Full OCR" value={qc.ocr_count} />
          <StatCard
            label="Mean confidence"
            value={qc.mean_confidence != null ? `${qc.mean_confidence.toFixed(2)}%` : '—'}
          />
        </div>
      )}
    </div>
  )
}
