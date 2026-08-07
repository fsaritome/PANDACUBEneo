interface StatCardProps {
  label: string
  value: string | number
  accent?: string
}

export function StatCard({ label, value, accent }: StatCardProps) {
  return (
    <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
      <div className="text-slate-400 text-sm">{label}</div>
      <div className={`text-2xl font-semibold mt-1 ${accent ?? 'text-slate-100'}`}>{value}</div>
    </div>
  )
}
