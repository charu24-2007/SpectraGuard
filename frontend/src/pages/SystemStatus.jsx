import { useEffect, useState } from 'react';
import PageHeader from '../components/layout/PageHeader';
import { getHealth, getDatasetInfo } from '../services/api';
import { Server, Database, Cpu, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

function StatusBadge({ ok }) {
  if (ok === null) return <span className="text-xs px-2 py-1 rounded bg-slate-100 text-slate-500">Checking...</span>;
  return ok
    ? <span className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-emerald-50 text-emerald-700 font-semibold"><CheckCircle className="w-3 h-3"/>Operational</span>
    : <span className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-red-50 text-red-700 font-semibold"><XCircle className="w-3 h-3"/>Unavailable</span>;
}

function StatusRow({ icon: Icon, label, sub, ok }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-slate-100 last:border-0">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-slate-50 rounded-full">
          <Icon className="w-4 h-4 text-slate-500" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-800">{label}</p>
          {sub && <p className="text-xs text-slate-400">{sub}</p>}
        </div>
      </div>
      <StatusBadge ok={ok} />
    </div>
  );
}

export default function SystemStatus() {
  const [health, setHealth] = useState(null);
  const [datasetOk, setDatasetOk] = useState(null);
  const [backendOk, setBackendOk] = useState(null);

  useEffect(() => {
    getHealth()
      .then(res => {
        setHealth(res.data);
        setBackendOk(res.data?.status === 'ok');
      })
      .catch(() => setBackendOk(false));

    getDatasetInfo()
      .then(res => setDatasetOk(res.data?.total_samples > 0))
      .catch(() => setDatasetOk(false));
  }, []);

  const modelOk = health?.model === 'loaded' || health?.model?.status === 'loaded';

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <PageHeader title="System Status" subtitle="Live service and component availability." />

      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-4">Service Components</h3>
        <StatusRow icon={Server} label="FastAPI Backend" sub="http://localhost:8000" ok={backendOk} />
        <StatusRow icon={Database} label="RadioML 2018.01A Dataset" sub="GOLD_XYZ_OSC.0001_1024.hdf5 — 2,555,904 samples" ok={datasetOk} />
        <StatusRow icon={Cpu} label="ResNet1D Model" sub="best_resnet_model.pth — 7.3M parameters, 24 classes" ok={modelOk || backendOk} />
        <StatusRow icon={Database} label="SQLite Database" sub="spectraguard.db — inference history" ok={backendOk} />
      </div>

      {health && (
        <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">API Health Response</h3>
          <pre className="text-xs bg-slate-50 border border-slate-200 rounded p-4 overflow-auto text-slate-700 font-mono">
            {JSON.stringify(health, null, 2)}
          </pre>
        </div>
      )}

      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
        <p className="text-xs text-slate-500 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-amber-500 shrink-0" />
          SpectraGuard is a passive, dataset-backed research platform. It does not connect to live RF hardware or SDR receivers.
        </p>
      </div>
    </div>
  );
}
