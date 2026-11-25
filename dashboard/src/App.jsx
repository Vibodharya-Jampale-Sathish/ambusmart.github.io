import { useState } from 'react';
import SimulationView from './SimulationView.jsx';
import LiveView from './LiveView.jsx';

const TABS = [
  ['sim', 'Simulation'],
  ['live', 'Live Microphone'],
];

export default function AmbuSmartDashboard() {
  const [mode, setMode] = useState('sim');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-red-950 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-wrap items-end justify-between gap-4 mb-4">
          <h1 className="text-5xl font-bold">AmbuSmart</h1>
          <div className="flex bg-white/10 rounded-2xl p-1">
            {TABS.map(([key, label]) => (
              <button
                key={key}
                onClick={() => setMode(key)}
                className={`px-5 py-2 rounded-xl font-semibold ${mode === key ? 'bg-white text-black' : 'text-gray-300 hover:text-white'}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {mode === 'sim' ? <SimulationView /> : <LiveView />}
      </div>
    </div>
  );
}
