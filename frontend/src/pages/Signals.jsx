import { useEffect, useState, useRef, useCallback } from 'react';
import {
  getDashboard, getSignals, getRandomSample, analyzeSignal,
  getSignal, getPrediction, getWaveform, getSpectrum, getSpectrogram,
  getActiveSample, setActiveSample, getModels,
  getSignalPdfUrl, getSignalsCsvUrl,
} from '../services/api';
import {
  Activity, CheckCircle, Database, TrendingUp,
  RefreshCw, Zap, Cpu, ArrowRight, ChevronDown, ChevronUp,
  FileText, Download, Search, ChevronLeft, ChevronRight, X,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts';

const MAX_INDEX = 2_555_903;

/* ─── Dark chart tooltip ─────────────────────────────────── */
const TOOLTIP_STYLE = {
  backgroundColor: '#0d1a2e',
  border: '1px solid #1e3a5f',
  borderRadius: 6,
  fontSize: 11,
  color: '#e2e8f0',
};

/* ─── Spectrogram canvas (dark colourmap) ────────────────── */
function SpectrogramCanvas({ data }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!data?.power?.length) return;
    const power = data.power;
    const nF = power.length, nT = power[0].length;
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = nT; canvas.height = nF;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(nT, nF);
    let mn = Infinity, mx = -Infinity;
    for (let f = 0; f < nF; f++) for (let t = 0; t < nT; t++) {
      if (power[f][t] < mn) mn = power[f][t];
      if (power[f][t] > mx) mx = power[f][t];
    }
    const rng = mx - mn + 1e-9;
    for (let f = 0; f < nF; f++) for (let t = 0; t < nT; t++) {
      const v = (power[f][t] - mn) / rng;
      const px = (f * nT + t) * 4;
      img.data[px]     = v < 0.5 ? Math.round(v * 2 * 6)         : Math.round(6  + (v - 0.5) * 2 * 249);
      img.data[px + 1] = v < 0.5 ? Math.round(v * 2 * 182)       : Math.round(182 + (v - 0.5) * 2 * 73);
      img.data[px + 2] = v < 0.5 ? Math.round(40 + v * 2 * 172)  : Math.round(212 - (v - 0.5) * 2 * 212);
      img.data[px + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data]);

  if (!data?.power?.length)
    return (
      <div className="flex items-center justify-center h-32 rounded border border-[#1a3050] bg-[#0a1220] text-slate-500 text-xs">
        No spectrogram data
      </div>
    );
  return (
    <canvas ref={ref} className="w-full rounded"
      style={{ height: 140, imageRendering: 'pixelated', display: 'block' }} />
  );
}

/* ─── KPI card ───────────────────────────────────────────── */
function KpiCard({ title, value, icon: Icon, accent, sub }) {
  return (
    <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl p-4 flex items-center gap-4">
      <div className={`p-2.5 rounded-lg bg-[#0a1220] shrink-0 ${accent}`}>
        <Icon className="w-4 h-4" />
      </div>
      <div className="min-w-0">
        <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest leading-tight mb-0.5">{title}</p>
        <p className="text-xl font-bold text-slate-100 leading-none tabular-nums">{value ?? '—'}</p>
        {sub && <p className="text-[10px] text-slate-600 mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

/* ─── Inline info row ────────────────────────────────────── */
function InfoRow({ label, value, cls = '' }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-[#1a3050] last:border-0">
      <span className="text-[11px] text-slate-500">{label}</span>
      <span className={`text-[11px] font-semibold text-right ${cls || 'text-slate-200'}`}>{value}</span>
    </div>
  );
}

/* ─── Top-k predictions bar chart ───────────────────────── */
function TopPredictions({ predictions }) {
  if (!predictions?.length) return <p className="text-slate-600 text-xs">No prediction data</p>;
  return (
    <div className="space-y-2">
      {predictions.map((p, i) => {
        const label = p.modulation || p.label || p.class || '?';
        const prob  = typeof p.probability === 'number' ? p.probability : 0;
        return (
          <div key={label} className="flex items-center gap-2">
            <span className={`text-[10px] w-16 text-right font-mono shrink-0 ${i === 0 ? 'text-cyan-300 font-bold' : 'text-slate-500'}`}>
              {label}
            </span>
            <div className="flex-1 bg-[#0a1220] rounded-full h-3 overflow-hidden border border-[#1a3050]">
              <div
                className={`h-full rounded-full ${i === 0 ? 'bg-cyan-500' : 'bg-slate-600'}`}
                style={{ width: `${Math.max(prob, 0.4)}%` }}
              />
            </div>
            <span className={`text-[10px] w-12 font-mono text-right tabular-nums ${i === 0 ? 'text-cyan-300 font-bold' : 'text-slate-500'}`}>
              {prob.toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

/* ─── DSP status badge helpers ───────────────────────────── */
function dspColor(status) {
  if (!status) return { bg: 'bg-slate-900/40', border: 'border-slate-700/40', text: 'text-slate-400' };
  if (status.includes('Possible')) return { bg: 'bg-amber-950/40', border: 'border-amber-600/40', text: 'text-amber-300' };
  if (status.includes('Inconclusive')) return { bg: 'bg-blue-950/40', border: 'border-blue-600/40', text: 'text-blue-300' };
  return { bg: 'bg-emerald-950/40', border: 'border-emerald-600/40', text: 'text-emerald-300' };
}

/* ─── Outcome badge ──────────────────────────────────────── */
function OutcomeBadge({ correct }) {
  return correct
    ? <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950/60 text-emerald-400 border border-emerald-700/40">Match</span>
    : <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950/60 text-rose-400 border border-rose-700/40">Mismatch</span>;
}

/* ─── Result correctness banner ──────────────────────────── */
function CorrectBadge({ isCorrect }) {
  return isCorrect
    ? <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-bold bg-emerald-950/50 text-emerald-400 border border-emerald-700/40">✓ Correct Prediction</span>
    : <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-bold bg-rose-950/50 text-rose-400 border border-rose-700/40">✕ Prediction Mismatch</span>;
}

/* ─── Spinner ────────────────────────────────────────────── */
const Spinner = () => (
  <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="40" strokeDashoffset="10" />
  </svg>
);

/* ─── Normalize inline response ─────────────────────────── */
function normalizeResult(r) {
  const conf = typeof r.confidence === 'number' ? (r.confidence <= 1 ? r.confidence : r.confidence / 100) : 0;
  return {
    id:                      r.signal_id || r.id,
    dataset_index:           r.dataset_index,
    ground_truth_modulation: r.ground_truth_modulation || r.ground_truth?.label,
    predicted_modulation:    r.predicted_modulation    || r.prediction?.label,
    confidence:              conf,
    snr_db:                  r.snr_db,
    model_name:              r.model || r.model_name || 'ResNet1D',
    correct_prediction:      r.correct_prediction ?? (r.result === 'correct'),
    top_predictions:         r.top5 || r.top_predictions || [],
    waveform:                r.waveform  || null,
    spectrum:                r.spectrum  || null,
    spectrogram:             r.spectrogram || null,
    interference:            r.interference || r.dsp_assessment || null,
  };
}

/* ══════════════════════════════════════════════════════════ */
export default function Signals() {
  const [inputMode,       setInputMode]      = useState('manual');
  const [indexInput,      setIndexInput]      = useState('');
  const [selectedModel,   setSelectedModel]   = useState('ResNet1D');
  const [analyzing,       setAnalyzing]       = useState(false);
  const [error,           setError]           = useState('');

  const [result,          setResult]          = useState(null);
  const [waveform,        setWaveform]        = useState(null);
  const [spectrum,        setSpectrum]        = useState(null);
  const [spectrogram,     setSpectrogram]     = useState(null);

  const [stats,           setStats]           = useState(null);
  const [history,         setHistory]         = useState([]);
  const [histLoading,     setHistLoading]     = useState(true);

  // History filtering, search, and pagination
  const [histPage,        setHistPage]        = useState(1);
  const [histTotal,       setHistTotal]       = useState(0);
  const [histPages,       setHistPages]       = useState(1);
  const [searchQuery,     setSearchQuery]     = useState('');
  const [filterModel,     setFilterModel]     = useState('');
  const [filterOutcome,   setFilterOutcome]   = useState('');
  const [filterDsp,       setFilterDsp]       = useState('');

  const [histOpen,        setHistOpen]        = useState(true);
  const [vizOpen,         setVizOpen]         = useState(true);
  const [detailItem,      setDetailItem]      = useState(null);

  const refreshKpis = useCallback(() => getDashboard().then(d => setStats(d.data)).catch(() => {}), []);

  const loadHistory = useCallback((page = histPage) => {
    setHistLoading(true);
    const params = {
      page,
      limit: 15,
      search: searchQuery || undefined,
      model_name: filterModel || undefined,
      outcome: filterOutcome || undefined,
      dsp_status: filterDsp || undefined,
    };
    getSignals(params)
      .then(r => {
        if (r.data?.items) {
          setHistory(r.data.items);
          setHistTotal(r.data.total);
          setHistPages(r.data.pages);
          setHistPage(r.data.page);
        } else if (Array.isArray(r.data)) {
          setHistory(r.data);
          setHistTotal(r.data.length);
          setHistPages(1);
        }
      })
      .catch(() => {})
      .finally(() => setHistLoading(false));
  }, [histPage, searchQuery, filterModel, filterOutcome, filterDsp]);

  useEffect(() => {
    refreshKpis();
    loadHistory(1);
    getModels().catch(() => {});
    getActiveSample().then(res => {
      if (res.data?.active && res.data.dataset_index !== null)
        setIndexInput(String(res.data.dataset_index));
    }).catch(() => {});
  }, [refreshKpis]);

  // Refetch history when search or filter changes
  useEffect(() => {
    loadHistory(1);
  }, [searchQuery, filterModel, filterOutcome, filterDsp]);

  const applyResult = (data) => {
    const norm = normalizeResult(data);
    setResult(norm);
    if (norm.waveform?.i) setWaveform(norm.waveform.i.map((v, i) => ({ idx: i, i: v, q: norm.waveform.q[i] })));
    if (norm.spectrum?.frequency) setSpectrum(norm.spectrum.frequency.map((f, i) => ({
      f: parseFloat(f.toFixed(3)),
      mag: parseFloat((norm.spectrum.magnitude[i] || 0).toFixed(2)),
    })));
    if (norm.spectrogram) setSpectrogram(norm.spectrogram);
    try {
      localStorage.setItem('sg_last_sample', JSON.stringify({ dataset_index: norm.dataset_index, signal_id: norm.id }));
      setActiveSample(norm.dataset_index, norm.id).catch(() => {});
    } catch (_) {}
    refreshKpis();
    loadHistory(1);
  };

  const handleAnalyze = async () => {
    setError(''); setAnalyzing(true);
    try {
      if (inputMode === 'manual') {
        const idx = parseInt(indexInput, 10);
        if (isNaN(idx) || idx < 0 || idx > MAX_INDEX) {
          setError(`Enter a valid index 0 – ${MAX_INDEX.toLocaleString()}`); return;
        }
        applyResult((await analyzeSignal(idx, selectedModel)).data);
      } else {
        const res = await getRandomSample(selectedModel);
        setIndexInput(String(res.data.dataset_index));
        applyResult(res.data);
      }
    } catch (e) {
      setError(e.response?.data?.detail || 'Analysis failed. Verify backend service is running.');
    } finally { setAnalyzing(false); }
  };

  const loadHistRow = async (sig) => {
    try {
      const [sR, pR, wR, spR, sgR] = await Promise.all([
        getSignal(sig.id), getPrediction(sig.id, selectedModel),
        getWaveform(sig.id), getSpectrum(sig.id), getSpectrogram(sig.id),
      ]);
      const s = sR.data;
      const conf = typeof s.confidence === 'number' ? (s.confidence <= 1 ? s.confidence : s.confidence / 100) : 0;
      const norm = {
        ...s,
        confidence: conf,
        top_predictions: pR.data.top_predictions || s.top_predictions || [],
        interference: s.dsp_assessment || null,
      };
      setResult(norm);
      setIndexInput(String(s.dataset_index));
      if (wR.data?.i) setWaveform(wR.data.i.map((v, i) => ({ idx: i, i: v, q: wR.data.q[i] })));
      if (spR.data?.frequency) setSpectrum(spR.data.frequency.map((f, i) => ({
        f: parseFloat(f.toFixed(3)), mag: parseFloat((spR.data.magnitude[i] || 0).toFixed(2)),
      })));
      setSpectrogram(sgR.data);
      try {
        localStorage.setItem('sg_last_sample', JSON.stringify({ dataset_index: s.dataset_index, signal_id: s.id }));
        setActiveSample(s.dataset_index, s.id).catch(() => {});
      } catch (_) {}
    } catch { setError('Failed to retrieve stored signal details.'); }
  };

  const confPct   = result ? (result.confidence <= 1 ? result.confidence * 100 : result.confidence).toFixed(2) : null;
  const isCorrect = result?.correct_prediction === true || result?.correct_prediction === 1;
  const dsp       = result?.interference || null;
  const dspC      = dsp ? dspColor(dsp.status) : null;

  const downloadCsv = () => {
    const params = {
      search: searchQuery || undefined,
      model_name: filterModel || undefined,
      outcome: filterOutcome || undefined,
      dsp_status: filterDsp || undefined,
    };
    window.open(getSignalsCsvUrl(params), '_blank');
  };

  return (
    <div className="space-y-4 max-w-7xl mx-auto">

      {/* ── Analysis Console ──────────────────────────────── */}
      <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl overflow-hidden">
        <div className="px-5 pt-4 pb-3 border-b border-[#1a3050] flex items-center justify-between">
          <div>
            <p className="text-[10px] font-bold text-cyan-500/70 uppercase tracking-widest mb-0.5">Analysis Console</p>
            <h2 className="text-sm font-semibold text-slate-200">Passive RF Modulation Classification &amp; Feature Analysis</h2>
            <p className="text-[11px] text-slate-500 mt-0.5">RadioML 2018.01A · 21.4 GB HDF5 · Ground truth from verified test set</p>
          </div>
        </div>

        {/* Toolbar */}
        <div className="px-5 py-3 flex flex-wrap items-center gap-2.5">
          {/* Mode toggle */}
          <div className="flex rounded-lg border border-[#1a3050] bg-[#0a1220] p-0.5 shrink-0">
            {['manual', 'random'].map(mode => (
              <button key={mode} onClick={() => setInputMode(mode)}
                className={`px-3 py-1.5 text-[11px] font-semibold rounded-md transition-all ${
                  inputMode === mode
                    ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                    : 'text-slate-500 hover:text-slate-300'
                }`}>
                {mode === 'manual' ? 'Manual Index' : <span className="flex items-center gap-1"><Zap className="w-3 h-3 text-amber-400" />Random Sample</span>}
              </button>
            ))}
          </div>

          <div className="h-5 w-px bg-[#1a3050] hidden sm:block shrink-0" />

          {/* Index / label */}
          {inputMode === 'manual' ? (
            <input type="number" min="0" max={MAX_INDEX}
              value={indexInput} onChange={e => setIndexInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleAnalyze()}
              placeholder="Index  0 – 2,555,903"
              className="w-44 bg-[#0a1220] border border-[#1a3050] text-slate-200 placeholder-slate-600 px-3 py-1.5 rounded-lg text-xs focus:outline-none focus:border-cyan-500 font-mono"
            />
          ) : (
            <span className="text-[11px] text-slate-500 italic px-3 py-1.5 bg-[#0a1220] border border-[#1a3050] rounded-lg whitespace-nowrap">
              Alternating Test Set
            </span>
          )}

          <div className="h-5 w-px bg-[#1a3050] hidden sm:block shrink-0" />

          {/* Model selector */}
          <div className="flex items-center gap-2 bg-[#0a1220] border border-[#1a3050] rounded-lg px-3 py-1.5 shrink-0">
            <Cpu className="w-3.5 h-3.5 text-cyan-500/60 shrink-0" />
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}
              className="bg-transparent text-[11px] font-medium text-slate-300 focus:outline-none cursor-pointer">
              <option value="ResNet1D">ResNet1D — 89.99% (benchmark test)</option>
              <option value="SpectraNet">SpectraNet — 88.90% (benchmark test)</option>
              <option value="CNN">CNN — 66.47% (benchmark test)</option>
            </select>
          </div>

          {/* Analyze button */}
          <button onClick={handleAnalyze}
            disabled={analyzing || (inputMode === 'manual' && !indexInput)}
            className="ml-auto flex items-center gap-2 bg-cyan-600 hover:bg-cyan-500 active:bg-cyan-700 text-white px-5 py-1.5 rounded-lg text-xs font-bold transition-all shadow-lg shadow-cyan-900/30 disabled:opacity-40 shrink-0 whitespace-nowrap">
            {analyzing ? <Spinner /> : <Zap className="w-3.5 h-3.5" />}
            {analyzing ? 'Processing…' : 'Analyze Signal'}
          </button>
        </div>

        {error && (
          <div className="mx-5 mb-4 p-3 text-xs text-red-400 bg-red-950/30 border border-red-800/40 rounded-lg">{error}</div>
        )}
      </div>

      {/* ── KPI row ───────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <KpiCard title="Signals Analyzed"    value={stats?.signals_analyzed ?? 0}                           icon={Activity}    accent="text-cyan-400"    sub="Database total" />
        <KpiCard title="Correct Predictions" value={stats?.correct_predictions ?? 0}                        icon={CheckCircle} accent="text-emerald-400" sub="Ground truth matches" />
        <KpiCard title="Avg Confidence"      value={stats ? `${stats.average_confidence}%` : '—'}           icon={TrendingUp}  accent="text-indigo-400"  sub="Softmax verified" />
        <KpiCard title="Dataset Samples"     value={stats?.dataset_samples?.toLocaleString() ?? '2,555,904'} icon={Database}    accent="text-slate-400"   sub="RadioML 2018.01A" />
      </div>

      {/* ── Analysis result ───────────────────────────────── */}
      {result && (
        <div className="space-y-4">
          {/* Header bar with Export PDF button */}
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">
                Sample&nbsp;
                <span className="text-cyan-400 font-mono text-sm">#{result.dataset_index}</span>
              </span>
              <span className="text-slate-600">·</span>
              <span className="text-[11px] text-slate-500 font-mono">{result.model_name || selectedModel}</span>
              <span className="text-slate-600">·</span>
              <span className="text-[10px] text-slate-500 font-mono">ID: {result.id}</span>
            </div>
            <div className="flex items-center gap-2">
              <CorrectBadge isCorrect={isCorrect} />
              <a
                href={getSignalPdfUrl(result.id)}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1.5 px-3 py-1 bg-cyan-950/60 hover:bg-cyan-900/60 border border-cyan-600/50 text-cyan-300 rounded-lg text-xs font-semibold transition-colors shadow-sm"
              >
                <FileText className="w-3.5 h-3.5" />
                Export PDF Report
              </a>
            </div>
          </div>

          {/* Two-column result grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Left — signal metadata + predictions */}
            <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl p-5">
              <p className="text-[10px] font-bold text-cyan-500/70 uppercase tracking-widest mb-3">Signal Information</p>
              <InfoRow label="Sample Index"       value={`#${result.dataset_index}`} />
              <InfoRow label="SNR"                value={`${result.snr_db} dB`} />
              <InfoRow label="Ground Truth"       value={result.ground_truth_modulation}  cls="text-emerald-400 font-bold" />
              <InfoRow label="Model Prediction"   value={result.predicted_modulation}     cls={isCorrect ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'} />
              <InfoRow label="Confidence"         value={`${confPct}%`} />
              <InfoRow label="Model"              value={result.model_name || selectedModel} />
              <InfoRow label="Status"             value={isCorrect ? 'Verified Match' : 'Mismatch'} cls={isCorrect ? 'text-emerald-400' : 'text-rose-400'} />

              <div className="mt-4 pt-4 border-t border-[#1a3050]">
                <p className="text-[10px] font-bold text-cyan-500/70 uppercase tracking-widest mb-2.5">
                  Top-{result.top_predictions?.length || 5} Softmax Predictions
                </p>
                <TopPredictions predictions={result.top_predictions} />
              </div>

              <div className="mt-4 pt-3 border-t border-[#1a3050] grid grid-cols-2 gap-3">
                <div>
                  <p className="text-[10px] text-slate-500 mb-0.5">Ground Truth</p>
                  <p className="text-emerald-400 font-bold text-base">{result.ground_truth_modulation}</p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-slate-500 mb-0.5">Predicted</p>
                  <p className={`font-bold text-base ${isCorrect ? 'text-emerald-400' : 'text-rose-400'}`}>{result.predicted_modulation}</p>
                </div>
              </div>
            </div>

            {/* Right — DSP assessment */}
            <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl p-5 flex flex-col">
              <p className="text-[10px] font-bold text-cyan-500/70 uppercase tracking-widest mb-3">DSP Interference Indicator</p>
              {dsp ? (
                <div className="flex-1 flex flex-col gap-3">
                  {/* Status banner */}
                  <div className={`p-3 rounded-lg border ${dspC.bg} ${dspC.border}`}>
                    <p className={`text-xs font-bold uppercase tracking-wider ${dspC.text}`}>{dsp.status}</p>
                    {dsp.pattern && (
                      <p className="text-[11px] text-slate-400 mt-0.5 font-mono">Pattern: {dsp.pattern}</p>
                    )}
                    <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">{dsp.explanation}</p>
                  </div>

                  {/* Evidence list */}
                  <div>
                    <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Supporting DSP Evidence</p>
                    {dsp.evidence?.length > 0 ? (
                      <ul className="space-y-1">
                        {dsp.evidence.map((ev, i) => (
                          <li key={i} className="flex items-start gap-2 text-[11px] text-slate-400">
                            <span className="text-cyan-500 mt-0.5 shrink-0">›</span>{ev}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-[11px] text-slate-600 italic">No anomalous spectral signatures detected.</p>
                    )}
                  </div>

                  {/* Measured metrics */}
                  {dsp.metrics && (
                    <div className="mt-auto pt-3 border-t border-[#1a3050]">
                      <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Measured DSP Metrics</p>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                        {[
                          ['Power',          dsp.metrics.signal_power,    ' dB'],
                          ['PAPR',           dsp.metrics.papr,            ' dB'],
                          ['Noise Floor',    dsp.metrics.noise_floor,     ' dB'],
                          ['Peak Power',     dsp.metrics.peak_power,      ' dB'],
                          ['Occupancy',      dsp.metrics.occupancy,       ''],
                          ['Flatness (SFM)', dsp.metrics.spectral_flatness,''],
                        ].map(([l, v, u]) => (
                          <div key={l} className="flex justify-between text-[10px]">
                            <span className="text-slate-600">{l}</span>
                            <span className="font-mono text-slate-400">{typeof v === 'number' ? v.toFixed(3) : '—'}{u}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-xs text-slate-600 italic">DSP assessment pending.</p>
              )}

              {/* Interference lab link */}
              <div className="mt-4 pt-3 border-t border-[#1a3050]">
                <a href="/interference"
                  className="flex items-center justify-center gap-2 bg-[#0a1220] hover:bg-cyan-500/10 border border-[#1a3050] hover:border-cyan-500/30 text-cyan-400 py-2 rounded-lg text-[11px] font-semibold transition-all">
                  Open Interference Lab <ArrowRight className="w-3.5 h-3.5" />
                </a>
              </div>
            </div>
          </div>

          {/* ── Visualizations ─────────────────────────────── */}
          <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl overflow-hidden">
            <button
              onClick={() => setVizOpen(v => !v)}
              className="w-full flex items-center justify-between px-5 py-3 hover:bg-white/5 transition-colors">
              <span className="text-[11px] font-bold text-cyan-500/70 uppercase tracking-widest">Signal Visualizations</span>
              {vizOpen ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
            </button>

            {vizOpen && (
              <div className="px-5 pb-5 space-y-4 border-t border-[#1a3050] pt-4">
                {/* Waveform */}
                {waveform?.length > 0 && (
                  <div>
                    <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">I/Q Time-Domain Waveform</p>
                    <p className="text-[10px] text-slate-600 mb-2">1,024 complex samples · raw I and Q components</p>
                    <div className="h-44 bg-[#080f1c] rounded-lg border border-[#1a3050] p-1">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={waveform}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1a3050" vertical={false} />
                          <XAxis dataKey="idx" stroke="#1e3a5f" tick={{ fontSize: 9, fill: '#475569' }} tickLine={false} tickCount={9} />
                          <YAxis stroke="#1e3a5f" tick={{ fontSize: 9, fill: '#475569' }} tickLine={false} />
                          <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v, n) => [Number(v).toFixed(4), n.toUpperCase()]} />
                          <Line type="monotone" dataKey="i" stroke="#06b6d4" dot={false} strokeWidth={1} isAnimationActive={false} name="I" />
                          <Line type="monotone" dataKey="q" stroke="#f43f5e" dot={false} strokeWidth={1} isAnimationActive={false} name="Q" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="flex gap-4 mt-1.5 text-[10px] text-slate-500">
                      <span><span className="inline-block w-3 h-0.5 bg-cyan-500 mr-1.5 align-middle rounded" />In-phase (I)</span>
                      <span><span className="inline-block w-3 h-0.5 bg-rose-500 mr-1.5 align-middle rounded" />Quadrature (Q)</span>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* FFT Spectrum */}
                  {spectrum?.length > 0 && (
                    <div>
                      <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">FFT Power Spectrum</p>
                      <p className="text-[10px] text-slate-600 mb-2">Normalized frequency · Hann-windowed · dB magnitude</p>
                      <div className="h-36 bg-[#080f1c] rounded-lg border border-[#1a3050] p-1">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={spectrum}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1a3050" vertical={false} />
                            <XAxis dataKey="f" stroke="#1e3a5f" tick={{ fontSize: 9, fill: '#475569' }} tickLine={false} />
                            <YAxis stroke="#1e3a5f" tick={{ fontSize: 9, fill: '#475569' }} tickLine={false} unit=" dB" />
                            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => [`${v} dB`]} />
                            <Line type="monotone" dataKey="mag" stroke="#818cf8" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}

                  {/* Spectrogram */}
                  <div>
                    <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">STFT Spectrogram</p>
                    <p className="text-[10px] text-slate-600 mb-2">16 temporal windows · frequency vs time power distribution</p>
                    <div className="bg-[#080f1c] rounded-lg border border-[#1a3050] p-1">
                      <SpectrogramCanvas data={spectrogram} />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── History Section with Filter, Search & Pagination ─── */}
      <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl overflow-hidden">
        <div
          role="button" tabIndex={0}
          onClick={() => setHistOpen(h => !h)}
          onKeyDown={e => e.key === 'Enter' && setHistOpen(h => !h)}
          className="flex items-center justify-between px-5 py-3.5 hover:bg-white/5 transition-colors cursor-pointer select-none"
        >
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-bold text-cyan-500/70 uppercase tracking-widest">
              Inference History Log
            </span>
            {histTotal > 0 && (
              <span className="text-[10px] text-slate-400 bg-[#1a3050] px-2 py-0.5 rounded-full">{histTotal} records</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={e => { e.stopPropagation(); loadHistory(); }}
              title="Refresh"
              className="text-slate-500 hover:text-cyan-400 transition-colors p-1"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
            {histOpen ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
          </div>
        </div>

        {histOpen && (
          <div className="border-t border-[#1a3050]">
            {/* Filter toolbar */}
            <div className="px-5 py-3 bg-[#0a1220] flex flex-wrap items-center gap-2.5 border-b border-[#1a3050]">
              {/* Search */}
              <div className="flex items-center gap-1.5 bg-[#0d1a2e] border border-[#1a3050] rounded-lg px-2.5 py-1 text-xs">
                <Search className="w-3.5 h-3.5 text-slate-500" />
                <input
                  type="text"
                  placeholder="Search index or ID..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="bg-transparent text-slate-200 placeholder-slate-600 focus:outline-none w-36 text-xs font-mono"
                />
                {searchQuery && (
                  <button onClick={() => setSearchQuery('')} className="text-slate-500 hover:text-slate-300">
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>

              {/* Model Filter */}
              <select
                value={filterModel}
                onChange={e => setFilterModel(e.target.value)}
                className="bg-[#0d1a2e] border border-[#1a3050] text-slate-300 text-xs rounded-lg px-2.5 py-1 focus:outline-none cursor-pointer"
              >
                <option value="">All Models</option>
                <option value="ResNet1D">ResNet1D</option>
                <option value="SpectraNet">SpectraNet</option>
                <option value="CNN">CNN</option>
              </select>

              {/* Outcome Filter */}
              <select
                value={filterOutcome}
                onChange={e => setFilterOutcome(e.target.value)}
                className="bg-[#0d1a2e] border border-[#1a3050] text-slate-300 text-xs rounded-lg px-2.5 py-1 focus:outline-none cursor-pointer"
              >
                <option value="">All Outcomes</option>
                <option value="match">Match Only</option>
                <option value="mismatch">Mismatch Only</option>
              </select>

              {/* DSP Filter */}
              <select
                value={filterDsp}
                onChange={e => setFilterDsp(e.target.value)}
                className="bg-[#0d1a2e] border border-[#1a3050] text-slate-300 text-xs rounded-lg px-2.5 py-1 focus:outline-none cursor-pointer"
              >
                <option value="">All DSP States</option>
                <option value="Possible">Possible Interference</option>
                <option value="No strong">No Strong Evidence</option>
                <option value="Inconclusive">Inconclusive</option>
              </select>

              {/* Export CSV Button */}
              <button
                onClick={downloadCsv}
                className="ml-auto flex items-center gap-1.5 px-3 py-1 bg-[#0d1a2e] hover:bg-white/5 border border-[#1a3050] text-slate-300 rounded-lg text-xs font-medium transition-colors"
                title="Download CSV of current filter results"
              >
                <Download className="w-3 h-3 text-cyan-400" />
                Export CSV
              </button>
            </div>

            {/* Table */}
            {histLoading ? (
              <div className="p-8 text-center text-slate-500 text-xs">Loading records…</div>
            ) : history.length === 0 ? (
              <div className="p-10 text-center">
                <p className="text-slate-500 text-sm">No analysis records match current criteria.</p>
                <p className="text-slate-600 text-xs mt-1">Adjust filters or analyze a new signal.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="text-[10px] text-slate-500 uppercase tracking-widest bg-[#0a1220] border-b border-[#1a3050]">
                      {['ID', 'Sample', 'Model', 'Ground Truth', 'Predicted', 'Confidence', 'SNR', 'Outcome', 'DSP Indicator', 'Report', 'Timestamp'].map(h => (
                        <th key={h} className="px-4 py-2.5 font-semibold whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1a3050]">
                    {history.map(s => {
                      const dspStatus = s.dsp_assessment?.status || '—';
                      return (
                        <tr
                          key={s.id}
                          onClick={() => loadHistRow(s)}
                          className="hover:bg-white/5 cursor-pointer transition-colors group"
                        >
                          <td className="px-4 py-3 font-mono text-slate-400 text-[11px] whitespace-nowrap">{s.id}</td>
                          <td className="px-4 py-3 font-mono text-cyan-400 font-bold whitespace-nowrap">#{s.dataset_index}</td>
                          <td className="px-4 py-3 text-slate-500 font-mono text-[11px] whitespace-nowrap">{s.model_name || 'ResNet1D'}</td>
                          <td className="px-4 py-3 font-medium text-slate-300 whitespace-nowrap">{s.ground_truth_modulation}</td>
                          <td className="px-4 py-3 font-semibold text-indigo-300 whitespace-nowrap">{s.predicted_modulation}</td>
                          <td className="px-4 py-3 tabular-nums text-slate-400 whitespace-nowrap">{(s.confidence * 100).toFixed(1)}%</td>
                          <td className="px-4 py-3 tabular-nums text-slate-500 whitespace-nowrap">{s.snr_db} dB</td>
                          <td className="px-4 py-3 whitespace-nowrap"><OutcomeBadge correct={s.correct_prediction} /></td>
                          <td className="px-4 py-3 text-slate-400 text-[11px] whitespace-nowrap max-w-xs truncate">
                            {dspStatus.includes('Possible') ? (
                              <span className="text-amber-400">Possible</span>
                            ) : dspStatus.includes('No strong') ? (
                              <span className="text-emerald-400">No anomaly</span>
                            ) : (
                              <span className="text-slate-500">{dspStatus}</span>
                            )}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap" onClick={e => e.stopPropagation()}>
                            <a
                              href={getSignalPdfUrl(s.id)}
                              target="_blank"
                              rel="noreferrer"
                              className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-mono text-[10px]"
                              title="Download PDF"
                            >
                              <FileText className="w-3 h-3" /> PDF
                            </a>
                          </td>
                          <td className="px-4 py-3 text-slate-600 tabular-nums whitespace-nowrap">{s.created_at?.slice(11, 19) || '—'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Pagination footer */}
            {histPages > 1 && (
              <div className="px-5 py-3 bg-[#0a1220] border-t border-[#1a3050] flex items-center justify-between text-xs text-slate-400">
                <span>Page {histPage} of {histPages} ({histTotal} total records)</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setHistPage(p => Math.max(1, p - 1))}
                    disabled={histPage <= 1}
                    className="p-1 rounded border border-[#1a3050] disabled:opacity-30 hover:bg-white/5"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setHistPage(p => Math.min(histPages, p + 1))}
                    disabled={histPage >= histPages}
                    className="p-1 rounded border border-[#1a3050] disabled:opacity-30 hover:bg-white/5"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
