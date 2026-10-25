import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { getHealth } from '../../services/api';

const PAGE_LABELS = {
  '/signals':      'Signal Analysis',
  '/interference': 'Interference Lab',
};

export default function Topbar() {
  const [time, setTime]       = useState(new Date());
  const [healthOk, setHealth] = useState(null);
  const location              = useLocation();

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const check = () =>
      getHealth()
        .then(r => setHealth(r.data?.status === 'ok'))
        .catch(()  => setHealth(false));
    check();
    const iv = setInterval(check, 15000);
    return () => clearInterval(iv);
  }, []);

  const pageTitle = PAGE_LABELS[location.pathname] ?? 'SpectraGuard';

  const ts = time.toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

  const statusLabel = healthOk === null ? 'Checking…'        : healthOk ? 'Backend Online'  : 'Backend Offline';
  const dotClass    = healthOk === null ? 'bg-slate-500'      : healthOk ? 'bg-emerald-400 animate-pulse' : 'bg-red-500';
  const badgeClass  = healthOk === null ? 'text-slate-400 border-slate-700'
                    : healthOk          ? 'text-emerald-400 border-emerald-800/60 bg-emerald-950/40'
                    :                     'text-red-400 border-red-800/60 bg-red-950/40';

  return (
    <div className="h-12 bg-[#0a1628] border-b border-[#1a3050] flex items-center justify-between px-5 shrink-0">
      {/* Left: page title */}
      <div className="flex items-center gap-3">
        <h1 className="text-sm font-semibold text-slate-200 tracking-wide">{pageTitle}</h1>
        <span className="text-slate-600 hidden sm:block">·</span>
        <span className={`hidden sm:flex items-center gap-1.5 text-[11px] font-medium border rounded-full px-2.5 py-0.5 ${badgeClass}`}>
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${dotClass}`} />
          {statusLabel}
        </span>
      </div>

      {/* Right: clock */}
      <div className="text-xs text-slate-500 font-mono tabular-nums tracking-wider">{ts}</div>
    </div>
  );
}
