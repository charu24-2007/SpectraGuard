import { useEffect, useState } from 'react';
import PageHeader from '../components/layout/PageHeader';
import { getSignals } from '../services/api';

function groupByEmitter(signals) {
  // Simple grouping: group all BPSK observations together, etc.
  // In a real system this would use RF fingerprinting. Here we group by predicted modulation.
  const groups = {};
  signals.forEach(sig => {
    const key = sig.predicted_modulation || 'Unknown';
    if (!groups[key]) {
      groups[key] = {
        id: `EG-${String(Object.keys(groups).length + 1).padStart(3, '0')}`,
        modulation: key,
        observations: [],
        avgConf: 0,
      };
    }
    groups[key].observations.push(sig);
  });

  // Compute derived stats
  Object.values(groups).forEach(g => {
    g.count = g.observations.length;
    g.firstSeen = g.observations[g.observations.length - 1]?.created_at;
    g.lastSeen = g.observations[0]?.created_at;
    g.avgConf = +(g.observations.reduce((s, o) => s + (o.confidence || 0), 0) / g.count * 100).toFixed(1);
    g.avgSnr = +(g.observations.reduce((s, o) => s + (o.snr_db || 0), 0) / g.count).toFixed(1);
  });

  return Object.values(groups).sort((a, b) => b.count - a.count);
}

export default function Emitters() {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    getSignals().then(r => setSignals(r.data)).finally(() => setLoading(false));
  }, []);

  const groups = groupByEmitter(signals);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Emitter Tracking"
        subtitle="Emitter activity correlation and history."
      />

      {loading ? (
        <div className="text-slate-500 py-8 text-center">Loading observations…</div>
      ) : groups.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-10 text-center">
          <p className="text-slate-500 text-sm">No observations yet.</p>
          <p className="text-slate-400 text-xs mt-1">Run IQ sample analysis to populate observation history.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Emitter Group List */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-slate-800">Emitter Groups</h3>
              <span className="text-xs text-slate-400 bg-slate-100 rounded px-2 py-0.5 font-mono">SIMULATED GROUPING</span>
            </div>
            <ul>
              {groups.map(g => (
                <li
                  key={g.id}
                  onClick={() => setSelected(g)}
                  className={`flex items-center justify-between px-4 py-3 cursor-pointer border-b border-slate-100 last:border-0 hover:bg-slate-50 transition-colors ${selected?.id === g.id ? 'bg-blue-50 border-l-4 border-l-blue-500' : ''}`}
                >
                  <div>
                    <p className="text-sm font-medium text-slate-900">{g.id}</p>
                    <p className="text-xs text-slate-500">{g.modulation}</p>
                  </div>
                  <span className="text-xs bg-slate-100 text-slate-600 rounded px-2 py-0.5 font-mono">
                    {g.count} obs
                  </span>
                </li>
              ))}
            </ul>
          </div>

          {/* Observation Detail */}
          <div className="lg:col-span-2">
            {!selected ? (
              <div className="bg-white border border-slate-200 rounded-xl p-8 text-center text-slate-400 shadow-sm h-full flex flex-col items-center justify-center">
                <p className="text-sm">Select an emitter group to view observations.</p>
              </div>
            ) : (
              <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
                <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-slate-900">{selected.id}</h3>
                    <p className="text-xs text-slate-500">
                      {selected.count} observations · Avg confidence {selected.avgConf}% · Avg SNR {selected.avgSnr} dB
                    </p>
                  </div>
                  <span className="text-xs bg-slate-100 text-slate-600 rounded px-2 py-0.5">{selected.modulation}</span>
                </div>

                {/* Stats row */}
                <div className="grid grid-cols-3 divide-x divide-slate-100 border-b border-slate-100">
                  {[
                    ['Observations', selected.count],
                    ['Avg Confidence', `${selected.avgConf}%`],
                    ['Avg SNR', `${selected.avgSnr} dB`],
                  ].map(([k, v]) => (
                    <div key={k} className="px-4 py-3">
                      <span className="block text-xs text-slate-400 mb-0.5">{k}</span>
                      <span className="text-slate-900 font-semibold tabular-nums">{v}</span>
                    </div>
                  ))}
                </div>

                {/* Timeline */}
                <div className="p-5">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-4">Observation Timeline</h4>
                  <div className="relative border-l border-slate-200 ml-2 space-y-4">
                    {selected.observations.slice(0, 8).map((obs, idx) => (
                      <div key={obs.id} className="relative pl-5">
                        <div className="absolute w-2.5 h-2.5 bg-blue-500 rounded-full -left-[5.5px] top-1.5 ring-2 ring-white" />
                        <div className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs text-slate-400 font-mono">{obs.created_at?.slice(11, 19) || '—'}</span>
                            <span className="font-mono text-xs text-slate-500">{obs.id}</span>
                          </div>
                          <div className="flex gap-4 text-sm">
                            <span className="text-slate-700">
                              <span className="text-xs text-slate-400 mr-1">Mod</span>
                              {obs.predicted_modulation}
                            </span>
                            <span className="text-slate-700">
                              <span className="text-xs text-slate-400 mr-1">SNR</span>
                              {obs.snr_db} dB
                            </span>
                            <span className="text-slate-700">
                              <span className="text-xs text-slate-400 mr-1">Conf</span>
                              {(obs.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <p className="px-5 pb-4 text-xs text-slate-400">
                  Groups are formed by observed modulation. This is application-level correlation, not physical emitter identification.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
