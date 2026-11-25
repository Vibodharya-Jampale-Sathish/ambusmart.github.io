export default function TrafficSignal({ light, className = '' }) {
  return (
    <div className={`bg-white/10 rounded-3xl p-8 ${className}`}>
      <h2 className="text-3xl font-bold mb-8">Traffic Signal</h2>

      <div className="flex flex-col items-center gap-6">
        <div className={`w-28 h-28 rounded-full ${light === 'RED' ? 'bg-red-500' : 'bg-red-900 opacity-30'}`}></div>

        <div className={`w-28 h-28 rounded-full ${light === 'YELLOW' ? 'bg-yellow-400' : 'bg-yellow-900 opacity-30'}`}></div>

        <div className={`w-28 h-28 rounded-full ${light === 'GREEN' ? 'bg-green-500' : 'bg-green-900 opacity-30'}`}></div>
      </div>
    </div>
  );
}
