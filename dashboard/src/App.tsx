import { useState } from 'react'
import { Overview } from './components/Overview'
import { LiveQueue } from './components/LiveQueue'
import { EngineChart } from './components/EngineChart'
import { HistoryTable } from './components/HistoryTable'
import { Failures } from './components/Failures'

const TABS = ['Overview', 'Live Queue', 'Engine Attribution', 'History', 'Failures'] as const
type Tab = (typeof TABS)[number]

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
      <h2 className="text-lg font-semibold text-slate-100 mb-4">{title}</h2>
      {children}
    </div>
  )
}

function App() {
  const [tab, setTab] = useState<Tab>('Overview')

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-xl font-bold">Patent OCR — Pipeline Dashboard</h1>
      </header>

      <nav className="flex gap-1 px-6 pt-4">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm rounded-t-lg border-b-2 transition-colors ${
              tab === t
                ? 'border-emerald-500 text-emerald-400 bg-slate-900'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            {t}
          </button>
        ))}
      </nav>

      <main className="p-6 space-y-6">
        {tab === 'Overview' && (
          <Section title="Overview">
            <Overview />
          </Section>
        )}
        {tab === 'Live Queue' && (
          <Section title="Currently Processing">
            <LiveQueue />
          </Section>
        )}
        {tab === 'Engine Attribution' && (
          <Section title="Tesseract vs PaddleOCR — region wins (current run)">
            <EngineChart />
          </Section>
        )}
        {tab === 'History' && (
          <Section title="Past sweeps">
            <HistoryTable />
          </Section>
        )}
        {tab === 'Failures' && (
          <Section title="Failed files">
            <Failures />
          </Section>
        )}
      </main>
    </div>
  )
}

export default App
