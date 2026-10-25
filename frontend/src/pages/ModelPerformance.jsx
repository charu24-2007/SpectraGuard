import { useEffect, useState } from 'react';
import PageHeader from '../components/layout/PageHeader';
import { getModels } from '../services/api';
import { BarChart2, CheckCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function ModelPerformance() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModels()
      .then(res => setModels(res.data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const fmtPct = (v) => {
    const n = parseFloat(v);
    if (isNaN(n)) return '—';
    return n <= 1 ? (n * 100).toFixed(2) + '%' : n.toFixed(2) + '%';
  };

  const chartData = models.map(m => ({
    name: m.Model,
    accuracy: parseFloat(m.Accuracy) <= 1 ? parseFloat(m.Accuracy) * 100 : parseFloat(m.Accuracy),
    f1: parseFloat(m['F1-Score']) <= 1 ? parseFloat(m['F1-Score']) * 100 : parseFloat(m['F1-Score']),
  }));

  const colors = ['#3b82f6', '#8b5cf6', '#10b981'];

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <PageHeader title="Model Performance" subtitle="Evaluation metrics for trained modulation classifiers." />

      {loading ? (
        <div className="text-slate-400 text-sm">Loading model metrics...</div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {models.map((m, i) => (
              <div key={i} className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 bg-blue-50 rounded-full">
                    <BarChart2 className="w-4 h-4 text-blue-600" />
                  </div>
                  <h3 className="font-bold text-slate-900">{m.Model}</h3>
                </div>
                <div className="space-y-3">
                  {[
                    { label: 'Accuracy', val: m.Accuracy },
                    { label: 'Precision', val: m.Precision },
                    { label: 'Recall', val: m.Recall },
                    { label: 'F1-Score', val: m['F1-Score'] },
                  ].map(({ label, val }) => (
                    <div key={label}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-500 font-medium uppercase tracking-wider">{label}</span>
                        <span className="font-bold text-slate-800">{fmtPct(val)}</span>
                      </div>
                      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-blue-500 rounded-full"
                          style={{ width: `${Math.min(100, (parseFloat(val) <= 1 ? parseFloat(val) * 100 : parseFloat(val)))}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
            <h3 className="text-sm font-semibold text-slate-700 mb-6">Accuracy Comparison</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} barSize={40}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 13, fill: '#475569', fontWeight: 600 }} tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fontSize: 12, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                  <Tooltip formatter={(v) => `${v.toFixed(2)}%`} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                  <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                    {chartData.map((_, idx) => <Cell key={idx} fill={colors[idx % colors.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm">
            <p className="text-xs text-slate-400">
              <CheckCircle className="w-3.5 h-3.5 inline mr-1 text-emerald-500" />
              Metrics derived from evaluation on the RadioML 2018.01A test set. ResNet1D is the active inference model.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
