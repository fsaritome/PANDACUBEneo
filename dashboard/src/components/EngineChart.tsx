import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { api, type QcCurrent } from '../api'
import { usePolling } from '../usePolling'

const COLORS: Record<string, string> = {
  tesseract: '#38bdf8',
  paddleocr: '#c084fc',
}

export function EngineChart() {
  const { data: qc } = usePolling<QcCurrent>(api.qcCurrent, 5000)

  if (!qc) return <div className="text-slate-400">Loading…</div>

  const entries = Object.entries(qc.engine_win_counts)
  if (entries.length === 0) {
    return <div className="text-slate-400">No region-level engine wins recorded yet.</div>
  }

  const total = entries.reduce((sum, [, n]) => sum + n, 0)
  const data = entries.map(([engine, n]) => ({
    name: engine,
    value: n,
    pct: total > 0 ? ((n / total) * 100).toFixed(1) : '0',
  }))

  return (
    <div style={{ width: '100%', height: 260 }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            outerRadius={90}
            label={(entry: { name?: string; percent?: number }) =>
              `${entry.name} ${((entry.percent ?? 0) * 100).toFixed(1)}%`
            }
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={COLORS[entry.name] ?? '#94a3b8'} />
            ))}
          </Pie>
          <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
