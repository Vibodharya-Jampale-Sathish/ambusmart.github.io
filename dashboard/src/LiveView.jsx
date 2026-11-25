// AmbuSmart - live microphone mode. The trained MLP scores every 500 ms frame (hop 250 ms);
// the signal follows the acoustic path of the fusion engine:
//   ACTIVATE  when c_a >= threshold on 3 consecutive frames            (IDLE -> PRIORITY GREEN)
//   PASSAGE   when c_a < 0.30 on 3 consecutive frames, min green 5 s    (GREEN -> AMBER 2 s -> RED)
import { useEffect, useRef, useState } from 'react';
import TrafficSignal from './TrafficSignal.jsx';
import { playTestSiren, useMicSiren } from './siren/useMicSiren.js';

const N_ON = 3;
const N_OFF = 3;
const OFF_THR = 0.3;
const MIN_GREEN_MS = 5000;
const AMBER_MS = 2000;
const HISTORY = 40; // 10 s of scores

const stamp = (m) => `${new Date().toLocaleTimeString()} - ${m}`;

function ConfidenceChart({ history, threshold }) {
  const W = 400, H = 120;
  const x = (i) => (i / (HISTORY - 1)) * W;
  const y = (c) => H - c * H;
  const offset = HISTORY - history.length;
  const pts = history.map((c, i) => `${x(i + offset).toFixed(1)},${y(c).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-32" preserveAspectRatio="none">
      <line x1="0" x2={W} y1={y(threshold)} y2={y(threshold)} stroke="#facc15" strokeDasharray="6 4" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      <line x1="0" x2={W} y1={y(OFF_THR)} y2={y(OFF_THR)} stroke="#9ca3af" strokeDasharray="2 4" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      {history.length > 1 && (
        <polyline points={pts} fill="none" stroke="#4ade80" strokeWidth="2.5" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
      )}
    </svg>
  );
}

export default function LiveView() {
  const [threshold, setThreshold] = useState(0.85);
  const [confidence, setConfidence] = useState(0);
  const [history, setHistory] = useState([]);
  const [light, setLight] = useState('RED');
  const [phase, setPhase] = useState('IDLE');
  const [testSiren, setTestSiren] = useState(null);
  const [logs, setLogs] = useState([stamp('Live mode ready - press Start Listening')]);

  const fsm = useRef({ phase: 'IDLE', high: 0, low: 0, greenAt: 0, timer: null });
  const thrRef = useRef(threshold);
  thrRef.current = threshold;
  const stopSiren = useRef(null);

  const addLog = (m) => setLogs((prev) => [stamp(m), ...prev].slice(0, 50));
  const go = (p, l) => { fsm.current.phase = p; setPhase(p); setLight(l); };

  const resetSignal = () => {
    const s = fsm.current;
    clearTimeout(s.timer);
    Object.assign(s, { high: 0, low: 0, timer: null });
    go('IDLE', 'RED');
  };

  const onScore = (c) => {
    setConfidence(c);
    setHistory((h) => [...h, c].slice(-HISTORY));
    const s = fsm.current;
    if (s.phase === 'IDLE') {
      s.high = c >= thrRef.current ? s.high + 1 : 0;
      if (s.high >= N_ON) {
        s.low = 0;
        s.greenAt = Date.now();
        go('GREEN', 'GREEN');
        addLog(`Siren confirmed (c_a >= ${thrRef.current.toFixed(2)} on ${N_ON} frames) - PRIORITY GREEN`);
      }
    } else if (s.phase === 'GREEN') {
      s.low = c < OFF_THR ? s.low + 1 : 0;
      if (s.low >= N_OFF && Date.now() - s.greenAt >= MIN_GREEN_MS) {
        s.high = 0;
        go('AMBER', 'YELLOW');
        addLog('Siren gone - ambulance passed, amber');
        s.timer = setTimeout(() => {
          go('IDLE', 'RED');
          addLog('Traffic system restored to normal cycle');
        }, AMBER_MS);
      }
    }
  };

  const mic = useMicSiren(onScore);

  const startListening = async () => {
    resetSignal();
    setHistory([]);
    addLog('Microphone on - listening for sirens');
    await mic.start();
  };

  const stopListening = () => {
    mic.stop();
    resetSignal();
    setConfidence(0);
    addLog('Microphone off');
  };

  const toggleTestSiren = (kind) => {
    stopSiren.current?.();
    stopSiren.current = null;
    if (testSiren === kind) {
      setTestSiren(null);
      return;
    }
    stopSiren.current = playTestSiren(kind);
    setTestSiren(kind);
  };

  useEffect(() => () => { stopSiren.current?.(); clearTimeout(fsm.current.timer); }, []);
  useEffect(() => { if (mic.error) addLog(`Error: ${mic.error}`); }, [mic.error]);

  const detected = mic.listening && confidence >= threshold;
  const phaseLabel = { IDLE: 'Normal cycle', GREEN: 'Priority green (preemption)', AMBER: 'Clearing - amber' }[phase];

  return (
    <>
      <div className="flex flex-wrap items-center gap-4 mb-8">
        {mic.listening ? (
          <button onClick={stopListening} className="bg-red-600 px-6 py-3 rounded-2xl font-semibold">
            Stop Listening
          </button>
        ) : (
          <button onClick={startListening} className="bg-green-600 px-6 py-3 rounded-2xl font-semibold">
            Start Listening
          </button>
        )}

        <div className="flex items-center gap-2 text-sm text-gray-300">
          <span>Test siren:</span>
          {['wail', 'yelp', 'hilo'].map((k) => (
            <button
              key={k}
              onClick={() => toggleTestSiren(k)}
              className={`px-3 py-2 rounded-xl font-semibold ${testSiren === k ? 'bg-yellow-400 text-black' : 'bg-white/10 hover:bg-white/20'}`}
            >
              {testSiren === k ? 'Stop' : k === 'hilo' ? 'Hi-Lo' : k[0].toUpperCase() + k.slice(1)}
            </button>
          ))}
        </div>

        <label className="flex items-center gap-3 text-sm text-gray-300 ml-auto">
          Detection threshold
          <input
            type="range" min="0.5" max="0.95" step="0.05" value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="accent-yellow-400"
          />
          <span className="font-mono text-white w-10">{threshold.toFixed(2)}</span>
        </label>
      </div>

      {mic.error && (
        <div className="bg-red-500/20 border border-red-500/50 rounded-2xl p-4 mb-6">{mic.error}</div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white/10 rounded-3xl p-6">
          <h2 className="text-xl font-semibold mb-3">Siren Confidence</h2>
          <p className="text-6xl font-bold text-yellow-400">{Math.round(confidence * 100)}%</p>
        </div>

        <div className="bg-white/10 rounded-3xl p-6">
          <h2 className="text-xl font-semibold mb-3">Siren Detection</h2>
          <p className={`text-4xl font-bold ${detected ? 'text-green-400' : 'text-gray-400'}`}>
            {!mic.listening ? 'MIC OFF' : detected ? 'DETECTED' : 'LISTENING'}
          </p>
        </div>

        <div className="bg-white/10 rounded-3xl p-6">
          <h2 className="text-xl font-semibold mb-3">Signal State</h2>
          <p className={`text-3xl font-bold ${phase === 'GREEN' ? 'text-blue-400' : 'text-gray-400'}`}>{phaseLabel}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <TrafficSignal light={light} />

        <div className="flex flex-col gap-6">
          <div className="bg-white/10 rounded-3xl p-6">
            <h2 className="text-xl font-semibold mb-1">Confidence, last 10 s</h2>
            <p className="text-xs text-gray-400 mb-3">
              Yellow line: activation threshold ({N_ON} frames in a row). Grey line: release ({OFF_THR.toFixed(2)}).
            </p>
            <ConfidenceChart history={history} threshold={threshold} />
          </div>

          <div className="bg-white/10 rounded-3xl p-6 flex-1">
            <h2 className="text-xl font-semibold mb-3">Event Log</h2>
            <ul className="text-sm font-mono text-gray-300 space-y-1 max-h-56 overflow-y-auto">
              {logs.map((l, i) => <li key={i}>{l}</li>)}
            </ul>
          </div>
        </div>
      </div>

      <p className="text-xs text-gray-500 mt-6">
        Runs the trained AmbuSmart MLP (128-64) on MFCC features in your browser; no audio leaves this device.
        Play a siren video near the microphone, or use a test siren.
      </p>
    </>
  );
}
