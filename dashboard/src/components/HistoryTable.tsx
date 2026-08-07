import { useState } from 'react'
import { api, type RunSummary } from '../api'
import { usePolling } from '../usePolling'

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}m ${s}s`
}

export function HistoryTable() {
  const { data } = usePolling<{ runs: RunSummary[] }>(api.historyRuns, 15000)
  const [expanded, setExpanded] = useState<number | null>(null)

  if (!data) return <div className="text-slate-400">Loading…</div>
  if (data.runs.length === 0) {
    return <div className="text-slate-400">No completed sweeps recorded yet.</div>
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="text-slate-400 border-b border-slate-700">
            <th className="py-2 pr-4">Run</th>
            <th className="py-2 pr-4">Finished</th>
            <th className="py-2 pr-4">Duration</th>
            <th className="py-2 pr-4">Files</th>
            <th className="py-2 pr-4">Failed</th>
            <th className="py-2 pr-4">Mean conf.</th>
          </tr>
        </thead>
        <tbody>
          {data.runs.map((run) => (
            <>
              <tr
                key={run.id}
                className="border-b border-slate-800 cursor-pointer hover:bg-slate-800/50"
                onClick={() => setExpanded(expanded === run.id ? null : run.id)}
              >
                <td className="py-2 pr-4 font-mono text-slate-200">#{run.id}</td>
                <td className="py-2 pr-4 text-slate-400">
                  {new Date(run.finished_at).toLocaleString()}
                </td>
                <td className="py-2 pr-4">{formatDuration(run.duration_seconds)}</td>
                <td className="py-2 pr-4">{run.total_files}</td>
                <td className={`py-2 pr-4 ${run.failed_count > 0 ? 'text-red-400' : 'text-slate-300'}`}>
                  {run.failed_count}
                </td>
                <td className="py-2 pr-4">
                  {run.mean_confidence != null ? `${run.mean_confidence.toFixed(2)}%` : '—'}
                </td>
              </tr>
              {expanded === run.id && (
                <tr className="bg-slate-800/30">
                  <td colSpan={6} className="py-3 px-4 text-slate-300">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                      <div>Passthrough: {run.passthrough_count}</div>
                      <div>Skip-text: {run.skip_text_count}</div>
                      <div>Full OCR: {run.ocr_count}</div>
                      <div>Flagged: {run.flagged_count}</div>
                    </div>
                    {Object.keys(run.engine_win_counts).length > 0 && (
                      <div className="mt-2 text-xs">
                        Engine wins:{' '}
                        {Object.entries(run.engine_win_counts)
                          .map(([e, n]) => `${e}: ${n}`)
                          .join(', ')}
                      </div>
                    )}
                    {run.failed_files.length > 0 && (
                      <div className="mt-2 text-xs text-red-400">
                        Failed: {run.failed_files.map((f) => f.input_path.split(/[/\\]/).pop()).join(', ')}
                      </div>
                    )}
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  )
}
