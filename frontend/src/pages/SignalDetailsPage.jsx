import { useEffect, useState, useRef } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import PageHeader from '../components/layout/PageHeader';
import { getSignal, getPrediction, getWaveform, getSpectrum, getSpectrogram } from '../services/api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

// ── Spectrogram rendered on HTML Canvas ──────────────────────────────────────
function SpectrogramCanvas({ data }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!data?.power || data.power.length === 0) return;
    const power = data.power;  // [freq_bins][time_steps]
    const nFreq = power.length;
    const nTime = power[0].length;
    const canvas = canvasRef.current;
    canvas.width = nTime;
    canvas.height = nFreq;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(nTime, nFreq);

    let minP = Infinity, maxP = -Infinity;
    for (let f = 0; f < nFreq; f++)
      for (let t = 0; t < nTime; t++) {
        if (power[f][t] < minP) minP = power[f][t];
        if (power[f][t] > maxP) maxP = power[f][t];
      }

    const range = maxP - minP + 1e-9;
    for (let f = 0; f < nFreq; f++) {
      for (let t = 0; t < nTime; t++) {
        const val = (power[f][t] - minP) / range;
        const px = (f * nTime + t) * 4;
        // Viridis-like colormap
        imgData.data[px]     = Math.round(val < 0.5 ? val * 2 * 68   : 68 + (val - 0.5) * 2 * 187);
        imgData.data[px + 1] = Math.round(val < 0.5 ? val * 2 * 168  : 168 + (val - 0.5) * 2 * 87);
        imgData.data[px + 2] = Math.round(val < 0.5 ? 130 - val * 2 * 50 : 80 - (val - 0.5) * 2 * 40);
        imgData.data[px + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }, [data]);

  if (!data?.power || data.power.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 bg-slate-50 rounded text-slate-400 text-sm">
        No spectrogram data
      </div>
    );
  }
  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '180px', imageRendering: 'pixelated' }}
      className="rounded border border-slate-200"
    />
  );
}

// ── InfoRow ───────────────────────────────────────────────────────────────────
function InfoRow({ label, value, className = '' }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-slate-100 last:border-0">
      <span className="text-slate-500 text-sm">{label}</span>
      <span className={`text-sm font-medium text-slate-900 text-right ${className}`}>{value}</span>
    </div>
  );
}

// ── Top-5 Predictions bar ─────────────────────────────────────────────────────
function TopPredictions({ predictions }) {
  if (!predictions || predictions.length === 0) return (
    <p className="text-slate-400 text-xs">No prediction data</p>
  );
  return (
    <div className="space-y-2.5">
      {predictions.map((p, i) => {
        const label = p.modulation || p.label || p.class || '?';
        const prob = typeof p.probability === 'number' ? p.probability : 0;
        return (
          <div key={label} className="flex items-center gap-3">
            <span className={`text-xs w-16 text-right font-mono shrink-0 ${i === 0 ? 'text-blue-700 font-semibold' : 'text-slate-500'}`}>
              {label}
            </span>
            <div className="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
              <div
                className={`h-full rounded-full ${i === 0 ? 'bg-blue-500' : 'bg-slate-300'}`}
                style={{ width: `${Math.max(prob, 0.3)}%` }}
              />
            </div>
            <span className={`text-xs w-16 font-mono tabular-nums ${i === 0 ? 'text-blue-700 font-semibold' : 'text-slate-500'}`}>
              {prob.toFixed(2)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Result Badge ──────────────────────────────────────────────────────────────
function ResultBadge({ isCorrect }) {
  if (isCorrect === true || isCorrect === 1) {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold bg-emerald-100 text-emerald-800 border border-emerald-300">
        ✓ Correct
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold bg-red-100 text-red-800 border border-red-300">
      ✕ Prediction Mismatch
    </span>
  );
}

// ── Normalize inline result to a canonical shape ──────────────────────────────
function normalizeInlineResult(r) {
  // confidence stored as 0–1 from backend
  const conf = typeof r.confidence === 'number'
    ? (r.confidence <= 1 ? r.confidence : r.confidence / 100)
    : 0;

  return {
    id: r.signal_id,
    dataset_index: r.dataset_index,
    ground_truth_modulation: r.ground_truth_modulation || r.ground_truth?.label,
    predicted_modulation: r.predicted_modulation || r.prediction?.label,
    confidence: conf,
    snr_db: r.snr_db,
    model_name: r.model || 'ResNet1D',
    correct_prediction: r.correct_prediction ?? (r.result === 'correct'),
    top_predictions: r.top5 || r.top_predictions || [],
    waveform: r.waveform || null,
    spectrum: r.spectrum || null,
    spectrogram: r.spectrogram || null,
  };
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function SignalDetailsPage() {
  const { signalId } = useParams();
  const { state } = useLocation();

  const [signal, setSignal] = useState(null);
  const [waveform, setWaveform] = useState(null);
  const [spectrum, setSpectrum] = useState(null);
  const [spectrogram, setSpectrogram] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const inlineResult = state?.result;

    if (inlineResult) {
      // Data passed directly from analyze/random endpoint — no extra fetch needed
      const norm = normalizeInlineResult(inlineResult);
      setSignal(norm);

      if (norm.waveform?.i) {
        setWaveform(norm.waveform.i.map((val, idx) => ({ idx, i: val, q: norm.waveform.q[idx] })));
      }
      if (norm.spectrum?.frequency) {
        setSpectrum(
          norm.spectrum.frequency.map((f, idx) => ({
            f: parseFloat(f.toFixed(4)),
            mag: parseFloat((norm.spectrum.magnitude[idx] || 0).toFixed(2)),
          }))
        );
      }
      if (norm.spectrogram) {
        setSpectrogram(norm.spectrogram);
      }
      setLoading(false);
    } else {
      // Fetch from API (e.g. navigated from history table)
      Promise.all([
        getSignal(signalId),
        getPrediction(signalId),
        getWaveform(signalId),
        getSpectrum(signalId),
        getSpectrogram(signalId),
      ])
        .then(([sigRes, predRes, wfRes, spRes, sgRes]) => {
          const s = sigRes.data;
          setSignal({
            id: s.id,
            dataset_index: s.dataset_index,
            ground_truth_modulation: s.ground_truth_modulation,
            predicted_modulation: s.predicted_modulation,
            confidence: s.confidence,          // stored as 0–1 in SQLite
            snr_db: s.snr_db,
            model_name: s.model_name || 'ResNet1D',
            correct_prediction: s.correct_prediction,
            top_predictions: predRes.data.top_predictions || [],
          });
          const { i, q } = wfRes.data;
          setWaveform(i.map((val, idx) => ({ idx, i: val, q: q[idx] })));
          const { frequency, magnitude } = spRes.data;
          setSpectrum(
            frequency.map((f, idx) => ({
              f: parseFloat(f.toFixed(4)),
              mag: parseFloat((magnitude[idx] || 0).toFixed(2)),
            }))
          );
          setSpectrogram(sgRes.data);
        })
        .catch(err => setError(err.message || 'Failed to load signal.'))
        .finally(() => setLoading(false));
    }
  }, [signalId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-slate-500 text-center">
          <svg className="animate-spin w-8 h-8 mx-auto mb-3 text-blue-500" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="40" strokeDashoffset="10" />
          </svg>
          <p className="text-sm">Running ResNet1D inference…</p>
        </div>
      </div>
    );
  }

  if (error) return <div className="p-8 text-red-600 bg-red-50 rounded-lg">{error}</div>;
  if (!signal) return <div className="p-8 text-slate-500">Signal not found.</div>;

  const confPct = typeof signal.confidence === 'number'
    ? (signal.confidence <= 1 ? signal.confidence * 100 : signal.confidence).toFixed(2)
    : '—';

  const isCorrect = signal.correct_prediction === true || signal.correct_prediction === 1;

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-4">
        <PageHeader
          title={`Dataset Sample #${signal.dataset_index}`}
          subtitle={`Signal ID: ${signal.id} · RadioML 2018.01A · ResNet1D`}
        />
        <div className="shrink-0 mt-1">
          <ResultBadge isCorrect={isCorrect} />
        </div>
      </div>

      {/* Signal Information + Top-5 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

        {/* Signal Information */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Signal Information</h3>
          <InfoRow label="Sample Index"   value={signal.dataset_index} />
          <InfoRow label="SNR"            value={`${signal.snr_db} dB`} />
          <InfoRow label="Ground Truth"   value={signal.ground_truth_modulation}
                   className="text-emerald-700 font-bold" />
          <InfoRow label="Prediction"     value={signal.predicted_modulation}
                   className={isCorrect ? 'text-emerald-700 font-bold' : 'text-red-700 font-bold'} />
          <InfoRow label="Confidence"     value={`${confPct}%`} />
          <InfoRow
            label="Result"
            value={isCorrect ? '✓ Correct' : '✕ Prediction Mismatch'}
            className={isCorrect ? 'text-emerald-700 font-bold' : 'text-red-700 font-bold'}
          />
          <InfoRow label="IQ Samples"     value="1024" />
          <InfoRow label="Representation" value="I / Q" />
          <InfoRow label="Model"          value={signal.model_name || 'ResNet1D'} />
        </div>

        {/* Top-5 Predictions */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Top-5 Predictions</h3>
          <p className="text-xs text-slate-400 mb-4">
            Softmax probabilities from ResNet1D. Actual model output — not hardcoded.
          </p>
          <TopPredictions predictions={signal.top_predictions} />

          <div className="mt-5 pt-4 border-t border-slate-100 grid grid-cols-2 gap-4">
            <div>
              <span className="block text-xs text-slate-400 mb-1">Ground Truth</span>
              <span className="text-emerald-700 font-bold text-xl">{signal.ground_truth_modulation}</span>
            </div>
            <div className="text-right">
              <span className="block text-xs text-slate-400 mb-1">Prediction</span>
              <span className={`font-bold text-xl ${isCorrect ? 'text-emerald-700' : 'text-red-700'}`}>
                {signal.predicted_modulation}
              </span>
            </div>
          </div>

          <p className="text-xs text-slate-400 mt-4 italic">
            A high confidence does not guarantee a correct prediction.
            Confidence is the softmax probability for the predicted class.
          </p>
        </div>
      </div>

      {/* I/Q Waveform */}
      {waveform && waveform.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">I/Q Waveform</h3>
          <p className="text-xs text-slate-400 mb-4">
            Raw I and Q channel values from HDF5 sample #{signal.dataset_index}. 1024 complex samples.
          </p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={waveform}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis
                  dataKey="idx"
                  stroke="#94a3b8"
                  fontSize={10}
                  label={{ value: 'Sample (0–1023)', position: 'insideBottom', offset: -2, fill: '#94a3b8', fontSize: 10 }}
                />
                <YAxis stroke="#94a3b8" fontSize={10} label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', borderColor: '#e2e8f0', fontSize: 11 }}
                  formatter={(v, name) => [v.toFixed(4), name.toUpperCase()]}
                />
                <Line type="monotone" dataKey="i" stroke="#3b82f6" dot={false} strokeWidth={1} isAnimationActive={false} name="I" />
                <Line type="monotone" dataKey="q" stroke="#ef4444" dot={false} strokeWidth={1} isAnimationActive={false} name="Q" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-5 mt-2 text-xs text-slate-500">
            <span><span className="inline-block w-5 h-1 bg-blue-500 mr-1.5 align-middle rounded" />I channel</span>
            <span><span className="inline-block w-5 h-1 bg-red-500 mr-1.5 align-middle rounded" />Q channel</span>
          </div>
        </div>
      )}

      {/* Frequency Spectrum */}
      {spectrum && spectrum.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Frequency Spectrum</h3>
          <p className="text-xs text-slate-400 mb-4">
            FFT magnitude (dB). Normalized frequency axis — RadioML 2018.01A does not provide physical carrier frequency.
          </p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={spectrum}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="f" stroke="#94a3b8" fontSize={10} label={{ value: 'Normalized Frequency', position: 'insideBottom', offset: -2, fill: '#94a3b8', fontSize: 10 }} />
                <YAxis stroke="#94a3b8" fontSize={10} label={{ value: 'Magnitude (dB)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#fff', borderColor: '#e2e8f0', fontSize: 11 }}
                  formatter={(v) => [`${v} dB`]}
                />
                <Line type="monotone" dataKey="mag" stroke="#8b5cf6" dot={false} strokeWidth={1} isAnimationActive={false} name="Magnitude" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Spectrogram */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">IQ Spectrogram</h3>
        <p className="text-xs text-slate-400 mb-4">
          STFT time-frequency representation. X: time windows · Y: normalized frequency · Color: power (dB).
        </p>
        <SpectrogramCanvas data={spectrogram} />
        <div className="flex justify-between text-xs text-slate-400 mt-2">
          <span>← Time →</span>
          <span>Normalized frequency (vertical axis)</span>
        </div>
      </div>

      {/* Disclaimer */}
      <p className="text-xs text-slate-400 text-center pb-2">
        Modulation classification only. Does not establish emitter identity, intent, or threat status.
      </p>
    </div>
  );
}
