import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import PageHeader from '../components/layout/PageHeader';
import { getDashboard, getActivity, getRandomSample, analyzeSignal } from '../services/api';
import { Activity, CheckCircle, Database, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function KpiCard({ title, value, icon: Icon, color, sub }) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 flex items-start gap-4">
      <div className={`p-3 rounded-full bg-slate-50 ${color} shrink-0`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-1 font-semibold">{title}</p>
        <p className="text-2xl font-bold text-slate-900">{value ?? 'â€”'}</p>
        {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [activity, setActivity] = useState([]);

  const [inputMode, setInputMode] = useState('manual');
  const [sampleIndex, setSampleIndex] = useState('12345');
  const [analyzing, setAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  useEffect(() => {
    Promise.all([getDashboard(), getActivity()])
      .then(([d, a]) => { setStats(d.data); setActivity(a.data); })
      .catch(console.error);
  }, []);

  const handleAnalyze = async () => {
    setAnalyzing(true);
    setErrorMsg('');
    try {
      if (inputMode === 'manual') {
        const idx = parseInt(sampleIndex, 10);
        if (isNaN(idx) || idx < 0) {
          setErrorMsg('Please enter a valid dataset index.');
          setAnalyzing(false);
          return;
        }
        const res = await analyzeSignal(idx);
        navigate(`/signals/${res.data.signal_id}`, { state: { result: res.data } });
      } else {
        const res = await getRandomSample();
        navigate(`/signals/${res.data.signal_id}`, { state: { result: res.data } });
      }
    } catch (e) {
      setErrorMsg('Analysis failed. ' + (e.response?.data?.detail || ''));
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader
        title="RF Spectrum Overview"
        subtitle="Passive spectrum monitoring â€” dataset-backed AI analysis."
      />

      {/* Quick Analyze CTA */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-start gap-8">
          <div className="shrink-0 pt-1">
            <p className="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-1">Quick Analysis</p>
            <p className="text-sm text-slate-700">Enter the sample index</p>
          </div>

          <div className="flex flex-col gap-4 flex-1 max-w-xl">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setInputMode('manual')}
                className={`px-4 py-2 text-xs font-semibold rounded transition-colors border ${
                  inputMode === 'manual'
                    ? 'bg-blue-50 border-blue-200 text-blue-700'
                    : 'bg-white border-slate-200 text-slate-500 hover:bg-slate-50'
                }`}
              >
                Manual Sample Index
              </button>
              <button
                onClick={() => setInputMode('random')}
                className={`px-4 py-2 text-xs font-semibold rounded transition-colors border ${
                  inputMode === 'random'
                    ? 'bg-blue-50 border-blue-200 text-blue-700'
                    : 'bg-white border-slate-200 text-slate-500 hover:bg-slate-50'
                }`}
              >
                Random Sample
              </button>
            </div>

            <div className="flex gap-3">
              {inputMode === 'manual' ? (
                <input
                  type="text"
                  value={sampleIndex}
                  onChange={(e) => setSampleIndex(e.target.value)}
                  placeholder="e.g. 12345"
                  className="flex-1 bg-white border border-slate-300 text-slate-800 px-4 py-2.5 rounded text-sm focus:outline-none focus:border-blue-500 font-mono"
                />
              ) : (
                <div className="flex-1 bg-slate-50 border border-slate-200 text-slate-500 px-4 py-2.5 rounded text-sm italic flex items-center">
                  Random sample selection from dataset.
                </div>
              )}
              <button
                onClick={handleAnalyze}
                disabled={analyzing}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded text-sm font-medium transition-colors shadow-sm shrink-0 disabled:opacity-50"
              >
                {analyzing ? 'Analyzing...' : 'Analyze IQ Sample â†’'}
              </button>
            </div>
            {errorMsg && <p className="text-sm text-red-500">{errorMsg}</p>}
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <KpiCard title="Signals Analyzed" value={stats?.signals_analyzed} icon={Activity} color="text-blue-500" sub="Total" />
        <KpiCard title="Correct Predictions" value={stats?.correct_predictions} icon={CheckCircle} color="text-emerald-500" sub="Verified" />
        <KpiCard title="Avg Confidence" value={stats ? `${stats.average_confidence}%` : 'â€”'} icon={TrendingUp} color="text-indigo-500" sub="Overall" />
        <KpiCard title="Dataset Samples" value={stats?.dataset_samples?.toLocaleString() ?? 0} icon={Database} color="text-slate-600" sub="Available" />
      </div>

      {/* Inference Activity Chart */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <h3 className="text-sm font-semibold text-slate-700 mb-6">Inference Activity</h3>
        <div className="h-64 w-full">
          {activity.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={activity}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="time" tick={{fontSize: 12, fill: '#64748b'}} tickLine={false} axisLine={false} dy={10} />
                <YAxis allowDecimals={false} tick={{fontSize: 12, fill: '#64748b'}} tickLine={false} axisLine={false} dx={-10} />
                <Tooltip
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                  itemStyle={{ color: '#0f172a', fontWeight: 500 }}
                />
                <Line
                  type="monotone"
                  dataKey="count"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ r: 4, strokeWidth: 2, fill: '#fff', stroke: '#3b82f6' }}
                  activeDot={{ r: 6, fill: '#3b82f6' }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-slate-400">
              <Activity className="w-8 h-8 mb-2 opacity-20" />
              <p className="text-sm">No recent activity</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
