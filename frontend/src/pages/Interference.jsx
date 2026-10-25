import { useEffect, useState, useRef, useCallback } from 'react';
import {
  simulateInterference, getInterferenceHistory, getActiveSample,
  getSimulationPdfUrl, getSimulationsCsvUrl,
} from '../services/api';
import {
  Info, Zap, Cpu, ArrowLeft, FlaskConical, ChevronDown, ChevronUp,
  FileText, Download, RotateCcw,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip,
} from 'recharts';

/* ─── Dark chart style ───────────────────────────────────── */
const TOOLTIP_STYLE = {
  backgroundColor: '#0d1a2e',
  border: '1px solid #1e3a5f',
  borderRadius: 6,
  fontSize: 11,
  color: '#e2e8f0',
};

/* ─── Dark cyan-tinted spectrogram canvas ────────────────── */
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
      img.data[px]     = v < 0.5 ? Math.round(v * 2 * 6)        : Math.round(6   + (v - 0.5) * 2 * 249);
      img.data[px + 1] = v < 0.5 ? Math.round(v * 2 * 182)      : Math.round(182 + (v - 0.5) * 2 * 73);
      img.data[px + 2] = v < 0.5 ? Math.round(40 + v * 2 * 172) : Math.round(212 - (v - 0.5) * 2 * 212);
      img.data[px + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data]);

  if (!data?.power?.length)
    return (
      <div className="flex items-center justify-center h-32 rounded-lg border border-[#1a3050] bg-[#0a1220] text-slate-500 text-xs">
        No spectrogram data
      </div>
    );
  return (
    <div className="bg-[#080f1c] rounded-lg border border-[#1a3050] overflow-hidden">
      <canvas ref={ref} className="w-full" style={{ height: 130, imageRendering: 'pixelated', display: 'block' }} />
    </div>
  );
}

/* ─── Waveform ───────────────────────────────────────────── */
function WaveformChart({ waveform, iColor = '#06b6d4', qColor = '#f43f5e' }) {
  if (!waveform?.i?.length) return (
    <div className="h-32 flex items-center justify-center bg-[#080f1c] rounded-lg border border-[#1a3050] text-slate-600 text-xs">No data</div>
  );
  const data = waveform.i.map((v, idx) => ({ idx, I: v, Q: waveform.q[idx] }));
  return (
    <div className="h-32 bg-[#080f1c] rounded-lg border border-[#1a3050] p-1">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a3050" vertical={false} />
          <XAxis dataKey="idx" hide />
          <YAxis tick={{ fontSize: 9, fill: '#475569' }} tickLine={false} stroke="#1e3a5f" />
          <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v, n) => [Number(v).toFixed(4), n]} />
          <Line type="monotone" dataKey="I" stroke={iColor} strokeWidth={1} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="Q" stroke={qColor} strokeWidth={1} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ─── Spectrum ───────────────────────────────────────────── */
function SpectrumChart({ spectrum, stroke = '#818cf8' }) {
  if (!spectrum?.frequency?.length) return (
    <div className="h-32 flex items-center justify-center bg-[#080f1c] rounded-lg border border-[#1a3050] text-slate-600 text-xs">No data</div>
  );
  const data = spectrum.frequency.map((f, i) => ({ f: parseFloat(f.toFixed(3)), mag: spectrum.magnitude[i] }));
  return (
    <div className="h-32 bg-[#080f1c] rounded-lg border border-[#1a3050] p-1">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1a3050" vertical={false} />
          <XAxis dataKey="f" tick={{ fontSize: 9, fill: '#475569' }} tickLine={false} stroke="#1e3a5f" />
          <YAxis tick={{ fontSize: 9, fill: '#475569' }} unit=" dB" tickLine={false} stroke="#1e3a5f" />
          <Tooltip contentStyle={TOOLTIP_STYLE} formatter={v => [`${Number(v).toFixed(1)} dB`]} />
          <Line type="monotone" dataKey="mag" stroke={stroke} strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ─── Assessment status helpers ──────────────────────────── */
function statusColors(status) {
  if (!status) return { bg: 'bg-[#0a1628]', border: 'border-[#1a3050]', text: 'text-slate-400', badge: 'bg-slate-800 text-slate-400 border-slate-700' };
  if (status.includes('Possible'))
    return { bg: 'bg-amber-950/40', border: 'border-amber-600/40', text: 'text-amber-300', badge: 'bg-amber-900/50 text-amber-300 border-amber-700/40' };
  if (status.includes('Inconclusive'))
    return { bg: 'bg-blue-950/40',  border: 'border-blue-600/40',  text: 'text-blue-300',  badge: 'bg-blue-900/50 text-blue-300 border-blue-700/40' };
  return     { bg: 'bg-emerald-950/40', border: 'border-emerald-600/40', text: 'text-emerald-300', badge: 'bg-emerald-900/50 text-emerald-300 border-emerald-700/40' };
}

/* ─── Assessment panel ───────────────────────────────────── */
function AssessmentPanel({ number, title, subtitle, assessment, dimmed }) {
  const c = statusColors(assessment?.status);
  return (
    <div className={`border rounded-xl p-4 flex flex-col justify-between ${dimmed ? 'bg-[#0a1628] border-[#1a3050]' : `${c.bg} ${c.border}`}`}>
      <div>
        <div className="flex items-start justify-between gap-2 mb-2">
          <div>
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{number}</span>
            <p className="text-[11px] font-bold text-slate-200 mt-0.5">{title}</p>
            {subtitle && <p className="text-[10px] text-slate-500 mt-0.5">{subtitle}</p>}
          </div>
          {assessment?.status && (
            <span className={`shrink-0 text-[10px] font-bold border px-2 py-0.5 rounded-full ${c.badge}`}>
              {assessment.status}
            </span>
          )}
        </div>
        {assessment?.pattern && (
          <p className={`text-sm font-bold mb-1 ${dimmed ? 'text-slate-300 capitalize' : c.text}`}>
            {assessment.pattern}
          </p>
        )}
        {assessment?.evidence?.length > 0 ? (
          <ul className="space-y-1 mt-2">
            {assessment.evidence.map((ev, i) => (
              <li key={i} className="flex items-start gap-2 text-[11px] text-slate-400">
                <span className="text-cyan-500 mt-0.5 shrink-0">›</span>{ev}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[11px] text-slate-500 italic mt-1">
            {dimmed ? 'Controlled parameters only — not an independent detector output.' : 'No anomalous signatures detected.'}
          </p>
        )}
      </div>
      {assessment?.explanation && !dimmed && (
        <p className="text-[10px] text-slate-500 mt-3 leading-relaxed border-t border-white/5 pt-2">
          {assessment.explanation}
        </p>
      )}
    </div>
  );
}

/* ─── Metric diff row ────────────────────────────────────── */
function MetricRow({ label, orig, interf, unit = '' }) {
  const oV = typeof orig   === 'number' ? orig   : 0;
  const iV = typeof interf === 'number' ? interf : 0;
  const d  = iV - oV;
  return (
    <div className="flex items-center justify-between py-2 border-b border-[#1a3050] last:border-0 text-[11px]">
      <span className="text-slate-400">{label}</span>
      <div className="flex gap-4 font-mono tabular-nums">
        <span className="w-20 text-right text-slate-500">{oV.toFixed(3)}{unit}</span>
        <span className="w-20 text-right text-slate-200 font-semibold">{iV.toFixed(3)}{unit}</span>
        <span className={`w-20 text-right font-bold ${d > 0.01 ? 'text-rose-400' : d < -0.01 ? 'text-cyan-400' : 'text-slate-500'}`}>
          {d > 0 ? '+' : ''}{d.toFixed(3)}{unit}
        </span>
      </div>
    </div>
  );
}

/* ─── Modulation badge ───────────────────────────────────── */
function ModBadge({ label, mod, conf, isCorrect }) {
  return (
    <div className="bg-[#0a1220] border border-[#1a3050] rounded-xl p-4">
      <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1.5">{label}</p>
      <p className="text-lg font-bold text-slate-100">{mod || 'Unknown'}</p>
      <p className="text-[11px] text-slate-500 mt-0.5 font-mono">{(Number(conf) * 100).toFixed(2)}% confidence</p>
      {isCorrect !== undefined && (
        <span className={`inline-block mt-2 text-[10px] px-2 py-0.5 rounded border font-bold ${
          isCorrect ? 'bg-emerald-950/50 text-emerald-400 border-emerald-700/40'
                    : 'bg-rose-950/50 text-rose-400 border-rose-700/40'
        }`}>
          {isCorrect ? '✓ Matches Ground Truth' : '✕ Prediction Mismatch'}
        </span>
      )}
    </div>
  );
}

/* ─── Spinner ────────────────────────────────────────────── */
const Spinner = () => (
  <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="40" strokeDashoffset="10" />
  </svg>
);

/* ══════════════════════════════════════════════════════════ */
export default function Interference() {
  const [activeSample,    setActiveSampleState] = useState(null);
  const [selectedModel,   setSelectedModel]     = useState('ResNet1D');
  const [patternChoice,   setPatternChoice]     = useState(''); // '' = Auto-Random
  const [strengthChoice,  setStrengthChoice]    = useState('1.2');
  const [seedInput,       setSeedInput]         = useState('');
  const [history,         setHistory]           = useState([]);
  const [loading,         setLoading]           = useState(false);
  const [error,           setError]             = useState('');
  const [result,          setResult]            = useState(null);
  const [histOpen,        setHistOpen]          = useState(true);

  const loadHistory = useCallback(async () => {
    try {
      const r = await getInterferenceHistory({ limit: 25, page: 1 });
      setHistory(r.data?.items || r.data || []);
    } catch {}
  }, []);

  useEffect(() => {
    loadHistory();
    try {
      const s = JSON.parse(localStorage.getItem('sg_last_sample'));
      if (s?.dataset_index !== undefined) setActiveSampleState(s);
    } catch {}
    getActiveSample().then(res => {
      if (res.data?.active && res.data.dataset_index !== null)
        setActiveSampleState({ dataset_index: res.data.dataset_index, signal_id: res.data.signal_id });
    }).catch(() => {});
  }, [loadHistory]);

  const handleAnalyze = async () => {
    if (!activeSample) { setError('No active sample. Analyze a signal in Signal Analysis first.'); return; }
    setLoading(true); setError('');
    try {
      const payload = {
        dataset_index: activeSample.dataset_index,
        model_name: selectedModel,
        interference_type: patternChoice || undefined,
        strength: strengthChoice ? parseFloat(strengthChoice) : undefined,
        seed: seedInput ? parseInt(seedInput, 10) : undefined,
      };
      const res = await simulateInterference(payload);
      setResult(res.data);
      loadHistory();
    } catch (e) {
      setError(e.response?.data?.detail || 'Interference simulation failed.');
    } finally { setLoading(false); }
  };

  const modChanged  = result && result.original?.modulation?.predicted !== result.interfered?.modulation?.predicted;
  const origCorrect = result && result.ground_truth !== 'Unknown' ? result.original.modulation.predicted === result.ground_truth : undefined;
  const infCorrect  = result && result.ground_truth !== 'Unknown' ? result.interfered.modulation.predicted === result.ground_truth : undefined;

  const downloadHistoryCsv = () => {
    window.open(getSimulationsCsvUrl(), '_blank');
  };

  return (
    <div className="space-y-4 max-w-7xl mx-auto">

      {/* ── Header control bar ────────────────────────────── */}
      <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl overflow-hidden">
        <div className="px-5 pt-4 pb-3 border-b border-[#1a3050] flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <FlaskConical className="w-4 h-4 text-cyan-400 shrink-0" />
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold text-slate-200">Interference Lab</h2>
                <span className="text-[9px] font-bold tracking-widest uppercase px-2 py-0.5 rounded-full bg-amber-500/15 text-amber-400 border border-amber-600/30">
                  SIMULATION
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Controlled synthetic interference overlay · independent DSP detector evaluation
              </p>
            </div>
          </div>
        </div>

        {/* Toolbar */}
        <div className="px-5 py-3 flex flex-wrap items-center gap-2.5">
          {/* Source sample reference */}
          <div className="flex items-center gap-2 bg-[#0a1220] border border-[#1a3050] rounded-lg px-3 py-1.5 shrink-0">
            {activeSample ? (
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-slate-500">Source:</span>
                <span className="font-mono text-cyan-400 font-bold text-xs">#{activeSample.dataset_index}</span>
                {result && result.ground_truth !== 'Unknown' && (
                  <span className="text-[11px] text-slate-400 hidden sm:inline">
                    · GT: <b>{result.ground_truth}</b> · {result.snr_db} dB
                  </span>
                )}
              </div>
            ) : (
              <span className="text-xs text-amber-400 flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5" /> No active sample
              </span>
            )}
          </div>

          {/* Model selector */}
          <div className="flex items-center gap-1.5 bg-[#0a1220] border border-[#1a3050] rounded-lg px-2.5 py-1.5 shrink-0">
            <Cpu className="w-3.5 h-3.5 text-cyan-500/60 shrink-0" />
            <select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              className="bg-transparent text-[11px] font-medium text-slate-300 focus:outline-none cursor-pointer"
            >
              <option value="ResNet1D">ResNet1D</option>
              <option value="SpectraNet">SpectraNet</option>
              <option value="CNN">CNN</option>
            </select>
          </div>

          {/* Pattern archetype selector */}
          <select
            value={patternChoice}
            onChange={e => setPatternChoice(e.target.value)}
            className="bg-[#0a1220] border border-[#1a3050] text-slate-300 text-[11px] rounded-lg px-2.5 py-1.5 focus:outline-none cursor-pointer shrink-0"
          >
            <option value="">Pattern: Auto-Random</option>
            <option value="tone-like">Tone-like (CW Carrier)</option>
            <option value="sweep-like">Sweep-like (Chirp)</option>
            <option value="pulsed">Pulsed (Burst)</option>
            <option value="broadband-noise-like">Broadband Noise (Barrage)</option>
          </select>

          {/* Strength selector */}
          <select
            value={strengthChoice}
            onChange={e => setStrengthChoice(e.target.value)}
            className="bg-[#0a1220] border border-[#1a3050] text-slate-300 text-[11px] rounded-lg px-2.5 py-1.5 focus:outline-none cursor-pointer shrink-0"
          >
            <option value="0.8">Strength: 0.8x</option>
            <option value="1.0">Strength: 1.0x</option>
            <option value="1.2">Strength: 1.2x (Default)</option>
            <option value="1.5">Strength: 1.5x</option>
          </select>

          {/* Optional Seed */}
          <input
            type="number"
            placeholder="Seed (opt)"
            value={seedInput}
            onChange={e => setSeedInput(e.target.value)}
            className="w-24 bg-[#0a1220] border border-[#1a3050] text-slate-200 placeholder-slate-600 px-2 py-1.5 rounded-lg text-xs font-mono focus:outline-none"
            title="Optional integer seed for deterministic reproducibility"
          />

          {/* Analyze button */}
          <button
            onClick={handleAnalyze}
            disabled={loading || !activeSample}
            className="ml-auto flex items-center gap-2 bg-cyan-600 hover:bg-cyan-500 active:bg-cyan-700 text-white px-5 py-1.5 rounded-lg text-xs font-bold transition-all shadow-lg shadow-cyan-900/30 disabled:opacity-40 shrink-0 whitespace-nowrap"
          >
            {loading ? <Spinner /> : <Zap className="w-3.5 h-3.5" />}
            {loading ? 'Simulating…' : 'Analyze Interference'}
          </button>
        </div>

        {error && (
          <div className="mx-5 mb-4 p-3 text-xs text-red-400 bg-red-950/30 border border-red-800/40 rounded-lg">{error}</div>
        )}
      </div>

      {/* ── Results ───────────────────────────────────────── */}
      {result && (
        <div className="space-y-4">
          {/* Header bar with Export PDF button */}
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">
                Simulation Record&nbsp;
                <span className="text-cyan-400 font-mono text-sm">{result.simulation_id}</span>
              </span>
              <span className="text-slate-600">·</span>
              <span className="text-[11px] text-slate-500 font-mono">Source #{result.dataset_index}</span>
            </div>
            <a
              href={getSimulationPdfUrl(result.simulation_id)}
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-1.5 px-3 py-1 bg-cyan-950/60 hover:bg-cyan-900/60 border border-cyan-600/50 text-cyan-300 rounded-lg text-xs font-semibold transition-colors shadow-sm"
            >
              <FileText className="w-3.5 h-3.5" />
              Export Simulation PDF Report
            </a>
          </div>

          {/* Three assessment panels */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            <AssessmentPanel
              number="01"
              title="Original Signal Assessment"
              subtitle="DSP detector — baseline sample"
              assessment={result.original?.assessment}
            />
            <AssessmentPanel
              number="02"
              title="Injected Test Pattern"
              subtitle="Simulation metadata only — not a detector output"
              dimmed
              assessment={{
                status:  null,
                pattern: result.injection?.type?.replace(/-/g, ' '),
                evidence: [
                  `Amplitude factor: ${result.injection?.strength}x`,
                  `Random seed: ${result.injection?.seed ?? 'Randomized'}`,
                  ...(result.injection?.params
                    ? Object.entries(result.injection.params).map(([k, v]) => `${k.replace('_', ' ')}: ${v}`)
                    : []),
                ],
              }}
            />
            <AssessmentPanel
              number="03"
              title="Derived Signal Assessment"
              subtitle="DSP detector — modified signal"
              assessment={result.interfered?.assessment}
            />
          </div>

          {/* Verification note */}
          {result.original?.assessment?.status !== result.interfered?.assessment?.status && (
            <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl p-4 text-[11px] text-slate-400 leading-relaxed flex items-start gap-2">
              <Info className="w-4 h-4 text-cyan-500 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-slate-300">Assessment differs between original and derived signal. </span>
                This difference is expected: the derived signal has been modified by synthetic software overlay in memory.
                The detector independently evaluated the modified waveform. The injected test pattern is simulation metadata only,
                not an independently detected result.
              </div>
            </div>
          )}

          {/* Modulation robustness */}
          <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-[10px] font-bold text-cyan-500/70 uppercase tracking-widest">
                  Modulation Classification Robustness — {result.model || selectedModel}
                </p>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  A classification shift is an observation of model sensitivity, not proof of interference.
                </p>
              </div>
              {result.ground_truth !== 'Unknown' && (
                <span className="text-[10px] font-bold bg-emerald-950/50 border border-emerald-700/40 text-emerald-400 px-2.5 py-1 rounded-lg shrink-0">
                  GT: {result.ground_truth}
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <ModBadge label="Original Signal (Baseline)"           mod={result.original.modulation.predicted}   conf={result.original.modulation.confidence}   isCorrect={origCorrect} />
              <ModBadge label="Derived Signal (With Interference)"   mod={result.interfered.modulation.predicted} conf={result.interfered.modulation.confidence} isCorrect={infCorrect}  />
            </div>

            {modChanged ? (
              <div className="mt-3 p-3 text-[11px] text-amber-300 bg-amber-950/30 border border-amber-700/30 rounded-lg leading-relaxed">
                <strong>Classification shift detected:</strong> Prediction changed from <strong>{result.original.modulation.predicted}</strong> → <strong>{result.interfered.modulation.predicted}</strong>.
                The synthetic interference masked or distorted constellation features visible to the classifier.
              </div>
            ) : (
              <p className="mt-3 text-[11px] text-slate-500 italic">
                Classification stable — both signals classified as <strong className="text-slate-400">{result.original.modulation.predicted}</strong>.
              </p>
            )}
          </div>

          {/* Side-by-side visualizations */}
          <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl p-5">
            <p className="text-[10px] font-bold text-cyan-500/70 uppercase tracking-widest mb-4">Signal Visualizations — Original vs Derived</p>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Waveforms */}
              <div>
                <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-widest mb-1.5">Original I/Q Waveform</p>
                <WaveformChart waveform={result.original.waveform} iColor="#06b6d4" qColor="#6366f1" />
              </div>
              <div>
                <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-widest mb-1.5">Derived I/Q Waveform</p>
                <WaveformChart waveform={result.interfered.waveform} iColor="#f59e0b" qColor="#f43f5e" />
              </div>

              {/* Spectra */}
              <div>
                <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-widest mb-1.5">Original Spectrum (FFT)</p>
                <SpectrumChart spectrum={result.original.spectrum} stroke="#818cf8" />
              </div>
              <div>
                <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-widest mb-1.5">Derived Spectrum (FFT)</p>
                <SpectrumChart spectrum={result.interfered.spectrum} stroke="#f43f5e" />
              </div>

              {/* Spectrograms */}
              <div>
                <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-widest mb-1.5">Original STFT Spectrogram</p>
                <SpectrogramCanvas data={result.original.spectrogram} />
              </div>
              <div>
                <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-widest mb-1.5">Derived STFT Spectrogram</p>
                <SpectrogramCanvas data={result.interfered.spectrogram} />
              </div>
            </div>
          </div>

          {/* DSP Feature differential table */}
          <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-[10px] font-bold text-cyan-500/70 uppercase tracking-widest">DSP Feature Differential</p>
                <p className="text-[11px] text-slate-500 mt-0.5">Measured values — original baseline vs derived signal</p>
              </div>
              <div className="hidden sm:flex gap-4 text-[10px] font-bold text-slate-500 uppercase tracking-widest font-mono">
                <span className="w-20 text-right">Original</span>
                <span className="w-20 text-right text-slate-300">Derived</span>
                <span className="w-20 text-right">Δ Delta</span>
              </div>
            </div>
            <MetricRow label="Mean Signal Power"           orig={result.original.metrics.signal_power}      interf={result.interfered.metrics.signal_power}      unit=" dB" />
            <MetricRow label="Peak-to-Avg Power (PAPR)"   orig={result.original.metrics.papr}              interf={result.interfered.metrics.papr}              unit=" dB" />
            <MetricRow label="Spectral Noise Floor"        orig={result.original.metrics.noise_floor}       interf={result.interfered.metrics.noise_floor}       unit=" dB" />
            <MetricRow label="Peak Spectral Power"         orig={result.original.metrics.peak_power}        interf={result.interfered.metrics.peak_power}        unit=" dB" />
            <MetricRow label="Peak-to-Noise Ratio"         orig={result.original.metrics.peak_to_noise}     interf={result.interfered.metrics.peak_to_noise}     unit=" dB" />
            <MetricRow label="Spectral Occupancy"          orig={result.original.metrics.occupancy}         interf={result.interfered.metrics.occupancy} />
            <MetricRow label="Spectral Flatness (SFM)"     orig={result.original.metrics.spectral_flatness} interf={result.interfered.metrics.spectral_flatness} />
            <MetricRow label="Temporal Energy Variance"    orig={result.original.metrics.energy_variance}   interf={result.interfered.metrics.energy_variance} />
            <MetricRow label="STFT Peak Freq Variance"     orig={result.original.metrics.freq_variance}     interf={result.interfered.metrics.freq_variance} />
          </div>
        </div>
      )}

      {/* ── History table ─────────────────────────────────── */}
      <div className="bg-[#0d1a2e] border border-[#1a3050] rounded-xl overflow-hidden">
        <div
          role="button" tabIndex={0}
          onClick={() => setHistOpen(h => !h)}
          onKeyDown={e => e.key === 'Enter' && setHistOpen(h => !h)}
          className="flex items-center justify-between px-5 py-3.5 hover:bg-white/5 transition-colors cursor-pointer select-none"
        >
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-bold text-cyan-500/70 uppercase tracking-widest">Simulation History</span>
            {history.length > 0 && (
              <span className="text-[10px] text-slate-500 bg-[#1a3050] px-2 py-0.5 rounded-full">{history.length}</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={e => { e.stopPropagation(); loadHistory(); }}
              className="text-slate-500 hover:text-cyan-400 p-1"
              title="Refresh"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
            {histOpen ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
          </div>
        </div>

        {histOpen && (
          <div className="border-t border-[#1a3050]">
            <div className="px-5 py-2.5 bg-[#0a1220] flex items-center justify-between border-b border-[#1a3050]">
              <span className="text-xs text-slate-500">Persisted simulation runs with reproducibility parameters</span>
              <button
                onClick={downloadHistoryCsv}
                className="flex items-center gap-1.5 px-3 py-1 bg-[#0d1a2e] hover:bg-white/5 border border-[#1a3050] text-slate-300 rounded-lg text-xs font-medium transition-colors"
              >
                <Download className="w-3 h-3 text-cyan-400" />
                Export Simulations CSV
              </button>
            </div>

            {history.length === 0 ? (
              <div className="p-8 text-center text-slate-500 text-xs">No simulation records logged yet.</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-[11px]">
                  <thead>
                    <tr className="text-[10px] text-slate-500 uppercase tracking-widest bg-[#0a1220] border-b border-[#1a3050]">
                      {['ID', 'Source Sample', 'Injected Pattern', 'Strength', 'Seed', 'Detected Signature', 'Assessment', 'Report', 'Timestamp'].map(h => (
                        <th key={h} className="px-4 py-2.5 font-semibold whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1a3050]">
                    {history.map((h, i) => {
                      const c =
                        h.status?.includes('Possible')
                          ? 'bg-amber-900/40 text-amber-300 border-amber-700/30'
                          : h.status?.includes('Inconclusive')
                          ? 'bg-blue-900/40 text-blue-300 border-blue-700/30'
                          : 'bg-emerald-900/40 text-emerald-300 border-emerald-700/30';
                      return (
                        <tr key={i} className="hover:bg-white/5 transition-colors">
                          <td className="px-4 py-3 font-mono text-slate-400 text-[10px] whitespace-nowrap">{h.id}</td>
                          <td className="px-4 py-3 font-mono text-cyan-400 font-bold whitespace-nowrap">#{h.dataset_index}</td>
                          <td className="px-4 py-3 text-slate-400 capitalize whitespace-nowrap">{h.interference_type?.replace(/-/g, ' ')}</td>
                          <td className="px-4 py-3 text-slate-400 font-mono whitespace-nowrap">{h.strength}×</td>
                          <td className="px-4 py-3 text-slate-500 font-mono whitespace-nowrap">{h.seed ?? '—'}</td>
                          <td className="px-4 py-3 text-slate-300 font-medium whitespace-nowrap">{h.detected_pattern}</td>
                          <td className="px-4 py-3 whitespace-nowrap">
                            <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${c}`}>{h.status}</span>
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap">
                            <a
                              href={getSimulationPdfUrl(h.id)}
                              target="_blank"
                              rel="noreferrer"
                              className="text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-mono text-[10px]"
                            >
                              <FileText className="w-3 h-3" /> PDF
                            </a>
                          </td>
                          <td className="px-4 py-3 text-slate-600 tabular-nums whitespace-nowrap">{h.created_at?.slice(11, 19) || '—'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
