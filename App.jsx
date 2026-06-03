import { useEffect, useState } from 'react';

export default function AmbuSmartDashboard() {
  const [ambulanceDistance, setAmbulanceDistance] = useState(500);
  const [sirenDetected, setSirenDetected] = useState(false);
  const [gpsValidated, setGpsValidated] = useState(false);
  const [systemActive, setSystemActive] = useState(false);
  const [trafficLight, setTrafficLight] = useState('RED');
  const [logs, setLogs] = useState([
    'System initialized',
    'Waiting for emergency vehicle detection'
  ]);

  useEffect(() => {
    let interval;

    if (systemActive) {
      interval = setInterval(() => {
        setAmbulanceDistance((prev) => {
          const updated = prev - 20;

          if (updated <= 300 && !sirenDetected) {
            setSirenDetected(true);
            addLog('Acoustic siren detected');
          }

          if (updated <= 250 && !gpsValidated) {
            setGpsValidated(true);
            addLog('GPS proximity validated');
          }

          if (updated <= 250) {
            setTrafficLight('GREEN');
          }

          if (updated <= 0) {
            addLog('Ambulance passed intersection');
            setTrafficLight('YELLOW');

            setTimeout(() => {
              setTrafficLight('RED');
              addLog('Traffic system restored to normal');
            }, 2000);

            clearInterval(interval);
            return 0;
          }

          return updated;
        });
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [systemActive, sirenDetected, gpsValidated]);

  const addLog = (message) => {
    setLogs((prev) => [
      `${new Date().toLocaleTimeString()} - ${message}`,
      ...prev
    ]);
  };

  const startSimulation = () => {
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
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-black to-red-950 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-5xl font-bold mb-4">AmbuSmart</h1>

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

        <div className="bg-white/10 rounded-3xl p-8">
          <h2 className="text-3xl font-bold mb-8">Traffic Signal</h2>

          <div className="flex flex-col items-center gap-6">
            <div className={`w-28 h-28 rounded-full ${trafficLight === 'RED' ? 'bg-red-500' : 'bg-red-900 opacity-30'}`}></div>

            <div className={`w-28 h-28 rounded-full ${trafficLight === 'YELLOW' ? 'bg-yellow-400' : 'bg-yellow-900 opacity-30'}`}></div>

            <div className={`w-28 h-28 rounded-full ${trafficLight === 'GREEN' ? 'bg-green-500' : 'bg-green-900 opacity-30'}`}></div>
          </div>
        </div>
      </div>
    </div>
  );
}
