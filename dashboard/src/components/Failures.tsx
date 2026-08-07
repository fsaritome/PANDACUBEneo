import { api, type FileRecord } from '../api'
import { usePolling } from '../usePolling'

export function Failures() {
  const { data } = usePolling<{ failures: FileRecord[] }>(api.failures, 10000)

  if (!data) return <div className="text-slate-400">Loading…</div>
  if (data.failures.length === 0) {
    return <div className="text-slate-400">No failures. 🎉</div>
  }

  return (
    <div className="space-y-3">
      {data.failures.map((f) => (
        <div key={f.input_path} className="bg-slate-800 border border-red-900/50 rounded-lg p-4">
          <div className="font-mono text-slate-100">{f.filename}</div>
          <div className="text-xs text-slate-400 mt-1">
            failed at {new Date(f.updated_at).toLocaleString()}
          </div>
          {f.error && (
            <pre className="mt-2 text-xs text-red-300 whitespace-pre-wrap break-words">
              {f.error}
            </pre>
          )}
        </div>
      ))}
    </div>
  )
}
