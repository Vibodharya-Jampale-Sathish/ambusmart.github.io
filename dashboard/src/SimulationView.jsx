import { useEffect, useRef, useState } from 'react';
import TrafficSignal from './TrafficSignal.jsx';

export default function SimulationView() {
  const [ambulanceDistance, setAmbulanceDistance] = useState(500);
  const [sirenDetected, setSirenDetected] = useState(false);
  const [gpsValidated, setGpsValidated] = useState(false);
  const [systemActive, setSystemActive] = useState(false);
  const [trafficLight, setTrafficLight] = useState('RED');
  const [logs, setLogs] = useState([
    'System initialized',
    'Waiting for emergency vehicle detection'
  ]);

  const restoreTimer = useRef(null);

  const addLog = (message) => {
    setLogs((prev) => [
      `${new Date().toLocaleTimeString()} - ${message}`,
      ...prev
    ]);
  };

  const clearRestoreTimer = () => {
    clearTimeout(restoreTimer.current);
    restoreTimer.current = null;
  };

  // Clock: move the ambulance 20 m closer every second while the simulation runs.
  useEffect(() => {
    if (!systemActive) return undefined;
    const interval = setInterval(() => {
      setAmbulanceDistance((prev) => Math.max(0, prev - 20));
    }, 1000);
    return () => clearInterval(interval);
  }, [systemActive]);

  // React to the new distance (side effects live here, not inside the state updater).
  useEffect(() => {
    if (!systemActive) return;

    if (ambulanceDistance <= 300 && !sirenDetected) {
      setSirenDetected(true);
      addLog('Acoustic siren detected');
    }

    if (ambulanceDistance <= 250 && !gpsValidated) {
      setGpsValidated(true);
      addLog('GPS proximity validated');
    }

    if (ambulanceDistance <= 0) {
      setSystemActive(false);
      addLog('Ambulance passed intersection');
      setTrafficLight('YELLOW');
      clearRestoreTimer();
      restoreTimer.current = setTimeout(() => {
        setTrafficLight('RED');
        addLog('Traffic system restored to normal');
        restoreTimer.current = null;
      }, 2000);
    } else if (ambulanceDistance <= 250) {
      setTrafficLight('GREEN');
    }
  }, [ambulanceDistance, systemActive, sirenDetected, gpsValidated]);

  useEffect(() => clearRestoreTimer, []);

  const startSimulation = () => {
    clearRestoreTimer();
    setAmbulanceDistance(500);
    setSirenDetected(false);
    setGpsValidated(false);
    setTrafficLight('RED');
    setSystemActive(true);
    setLogs([
      'Simulation started',
      'Monitoring ambulance approach'
    ]);
  };

  const resetSimulation = () => {
    clearRestoreTimer();
    setSystemActive(false);
    setAmbulanceDistance(500);
    setSirenDetected(false);
    setGpsValidated(false);
    setTrafficLight('RED');
    setLogs([
      'System reset completed',
      'Waiting for emergency vehicle detection'
    ]);
  };

  return (
    <>
        <div className="flex gap-4 mb-8">
          <button
            onClick={startSimulation}
            className="bg-green-600 px-6 py-3 rounded-2xl font-semibold"
          >
            Start Simulation
          </button>

          <button
            onClick={resetSimulation}
            className="bg-red-600 px-6 py-3 rounded-2xl font-semibold"
          >
            Reset
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white/10 rounded-3xl p-6">
            <h2 className="text-xl font-semibold mb-3">Ambulance Distance</h2>
            <p className="text-6xl font-bold text-yellow-400">
              {ambulanceDistance}m
            </p>
          </div>

          <div className="bg-white/10 rounded-3xl p-6">
            <h2 className="text-xl font-semibold mb-3">Siren Detection</h2>
            <p className={`text-4xl font-bold ${sirenDetected ? 'text-green-400' : 'text-gray-400'}`}>
              {sirenDetected ? 'DETECTED' : 'WAITING'}
            </p>
          </div>

          <div className="bg-white/10 rounded-3xl p-6">
            <h2 className="text-xl font-semibold mb-3">GPS Validation</h2>
            <p className={`text-4xl font-bold ${gpsValidated ? 'text-blue-400' : 'text-gray-400'}`}>
              {gpsValidated ? 'CONFIRMED' : 'PENDING'}
            </p>
          </div>
        </div>

        <TrafficSignal light={trafficLight} />
    </>
  );
}
