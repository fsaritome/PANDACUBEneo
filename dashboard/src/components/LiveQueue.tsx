import { api, type LiveStatus } from '../api'
import { usePolling } from '../usePolling'

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

export function LiveQueue() {
  const { data: live } = usePolling<LiveStatus>(api.liveStatus, 2000)

  if (!live) return <div className="text-slate-400">Loading…</div>

  if (live.processing.length === 0) {
    return <div className="text-slate-400">No files currently processing.</div>
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="text-slate-400 border-b border-slate-700">
            <th className="py-2 pr-4">File</th>
            <th className="py-2 pr-4">Started</th>
            <th className="py-2 pr-4">Elapsed</th>
          </tr>
        </thead>
        <tbody>
          {live.processing.map((f) => (
            <tr key={f.input_path} className="border-b border-slate-800">
              <td className="py-2 pr-4 font-mono text-slate-200">{f.filename}</td>
              <td className="py-2 pr-4 text-slate-400">
                {new Date(f.created_at).toLocaleTimeString()}
              </td>
              <td className="py-2 pr-4 text-amber-400 font-semibold">
                {formatElapsed(f.elapsed_seconds)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
