"""
InterferenceService: Signal Processing, Feature Extraction, and Rule-Based Interference Assessment.
Operates on raw I/Q signal data from RadioML 2018.01A.
Never modifies original dataset records.
"""
import numpy as np
import uuid
import random as pyrandom
import json
from typing import Dict, Any, Tuple, Optional, List
from backend.database.database import get_db


# Standardized assessment states
STATE_NO_EVIDENCE = "No strong interference-like evidence"
STATE_POSSIBLE = "Possible interference-like activity"
STATE_INCONCLUSIVE = "Inconclusive"

# Feature metadata with units and physical/mathematical definitions
FEATURE_METADATA = {
    "signal_power": {
        "unit": "dB",
        "label": "Mean Signal Power",
        "description": "Average total power in dB relative to unit impedance (10*log10(mean(|x|^2)))",
    },
    "energy": {
        "unit": "",
        "label": "Signal Energy",
        "description": "Integrated sum of instantaneous power over 1,024 complex samples",
    },
    "papr": {
        "unit": "dB",
        "label": "Peak-to-Average Power Ratio",
        "description": "PAPR = 10*log10(max(|x|^2) / mean(|x|^2)), with zero-power guard",
    },
    "noise_floor": {
        "unit": "dB",
        "label": "Spectral Noise Floor",
        "description": "Estimated median power spectral density across all FFT bins",
    },
    "peak_power": {
        "unit": "dB",
        "label": "Peak Spectral Power",
        "description": "Maximum power spectral density bin in Hann-windowed FFT",
    },
    "peak_to_noise": {
        "unit": "dB",
        "label": "Peak-to-Noise Ratio",
        "description": "Spectral prominence: difference between peak bin and noise floor (dB)",
    },
    "occupancy": {
        "unit": "",
        "label": "Spectral Occupancy",
        "description": "Fraction of spectral bins exceeding noise floor + 10 dB threshold",
    },
    "spectral_flatness": {
        "unit": "",
        "label": "Spectral Flatness (SFM)",
        "description": "Wiener entropy: geometric mean / arithmetic mean of power spectrum (0=tonal, 1=white noise)",
    },
    "energy_variance": {
        "unit": "",
        "label": "Temporal Energy Variance",
        "description": "Normalized variance of short-window power across 16 temporal observation slices",
    },
    "freq_variance": {
        "unit": "bins²",
        "label": "STFT Peak Frequency Variance",
        "description": "Variance of dominant frequency bin indices across successive observation windows",
    },
}


class InterferenceService:
    """
    Modular DSP feature extractor and heuristic anomaly/interference detector.
    Provides mathematically defensible feature extraction from normalized complex signals.
    """
    DETECTOR_VERSION = "2.1.0-dsp"
    LIMITATIONS_NOTE = (
        "Analysis performed using calibrated signal-processing heuristics and differential energy metrics. "
        "RadioML 2018.01A does not contain operational interference ground-truth labels; "
        "results represent anomaly indicators and should not be construed as confirmed electronic warfare attacks."
    )

    SUPPORTED_PATTERNS = ["tone-like", "sweep-like", "pulsed", "broadband-noise-like"]

    # ── Simulation & Pattern Synthesis ─────────────────────────────────────────

    def random_injection(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Randomly select one of four synthetic interference patterns and generate valid,
        physically bounded parameters.
        Returns the chosen pattern type, strength factor, and pattern-specific parameters.
        """
        rng = pyrandom.Random(seed) if seed is not None else pyrandom

        itype = rng.choice(self.SUPPORTED_PATTERNS)
        strength = round(rng.uniform(0.6, 1.8), 2)

        params: Dict[str, Any] = {}
        if itype == "tone-like":
            # Normalized frequency in [-0.45, 0.45] cycles/sample (avoiding Nyquist edge)
            params["frequency"] = round(rng.uniform(-0.45, 0.45), 3)
        elif itype == "sweep-like":
            # Linear chirp spanning a frequency interval across the sample
            f_start = round(rng.uniform(-0.45, -0.1), 3)
            f_end = round(rng.uniform(0.1, 0.45), 3)
            params["f_start"] = f_start
            params["f_end"] = f_end
        elif itype == "pulsed":
            # Duty cycle parameters as fraction of window length
            params["duration"] = round(rng.uniform(0.04, 0.25), 3)
            params["interval"] = round(rng.uniform(0.15, 0.5), 3)
        elif itype == "broadband-noise-like":
            # Fractional spectral bandwidth in [0.4, 1.0]
            params["bandwidth"] = round(rng.uniform(0.4, 1.0), 2)

        return {
            "type": itype,
            "strength": strength,
            "params": params,
            "seed": seed,
        }

    def generate_interference(
        self,
        iq: np.ndarray,
        interference_type: str,
        strength: float,
        params: dict,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes an interference waveform in memory and superimposes it onto the clean I/Q array.
        The original input ndarray is NEVER modified (explicit copy).
        Returns:
            interfered_iq: np.ndarray shape (N, 2), float32
            interf_waveform: np.ndarray shape (N,), complex128
        """
        # Isolated random state for reproducibility without polluting global RNG
        np_rng = np.random.RandomState(seed) if seed is not None else np.random

        N = len(iq)
        t = np.arange(N, dtype=np.float64)
        clean_i = np.array(iq[:, 0], dtype=np.float64, copy=True)
        clean_q = np.array(iq[:, 1], dtype=np.float64, copy=True)
        clean = clean_i + 1j * clean_q

        clean_power = float(np.mean(np.abs(clean) ** 2))
        if clean_power <= 0 or not np.isfinite(clean_power):
            clean_power = 1.0

        # Interference amplitude scaled relative to clean signal RMS power
        A = np.sqrt(clean_power) * float(strength) * 4.5
        interf = np.zeros(N, dtype=np.complex128)

        if interference_type == "tone-like":
            freq = float(params.get("frequency", 0.25))
            interf = A * np.exp(1j * 2 * np.pi * freq * t)

        elif interference_type == "sweep-like":
            f_start = float(params.get("f_start", -0.4))
            f_end = float(params.get("f_end", 0.4))
            # Phase for linear frequency sweep
            phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) / N * (t ** 2))
            interf = A * np.exp(1j * phase)

        elif interference_type == "pulsed":
            dur = max(1, int(float(params.get("duration", 0.1)) * N))
            period = max(1, int(float(params.get("interval", 0.3)) * N))
            env = np.zeros(N, dtype=np.float64)
            for start in range(0, N, period):
                env[start : min(start + dur, N)] = 1.0
            noise = (np_rng.randn(N) + 1j * np_rng.randn(N)) / np.sqrt(2.0)
            interf = A * env * noise

        elif interference_type == "broadband-noise-like":
            bw = float(params.get("bandwidth", 1.0))
            noise = (np_rng.randn(N) + 1j * np_rng.randn(N)) / np.sqrt(2.0)
            if bw < 1.0:
                freqs = np.fft.fftfreq(N)
                mask = np.abs(freqs) <= (bw / 2.0)
                noise = np.fft.ifft(np.fft.fft(noise) * mask)
            interf = A * noise

        interfered = clean + interf
        out = np.zeros_like(iq, dtype=np.float32)
        out[:, 0] = np.real(interfered).astype(np.float32)
        out[:, 1] = np.imag(interfered).astype(np.float32)

        return out, interf

    # ── DSP Feature Extraction ────────────────────────────────────────────────

    def extract_features(self, iq: np.ndarray) -> Dict[str, Any]:
        """
        Computes deterministic, numerically stable DSP features from a complex I/Q signal.
        Normalized frequency units (cycles/sample) are used because physical sampling rate
        is not specified in RadioML 2018.01A metadata.
        """
        I = iq[:, 0].astype(np.float64)
        Q = iq[:, 1].astype(np.float64)
        sig = I + 1j * Q
        N = len(sig)

        # 1. Power & Energy Statistics
        inst_power = np.abs(sig) ** 2
        pwr_linear = float(np.mean(inst_power))
        sig_power_db = float(10.0 * np.log10(pwr_linear + 1e-12))
        total_energy = float(pwr_linear * N)

        peak_inst_pwr = float(np.max(inst_power)) if N > 0 else 0.0
        if peak_inst_pwr <= 0.0 or pwr_linear <= 0.0:
            papr_db = 0.0
        else:
            papr_db = float(10.0 * np.log10(peak_inst_pwr / (pwr_linear + 1e-12)))

        # 2. Windowed Frequency Spectrum (Hann window for reduced spectral leakage)
        window = np.hanning(N)
        win_sig = sig * window
        win_norm = np.sum(window ** 2) + 1e-12

        fft_vals = np.fft.fftshift(np.fft.fft(win_sig))
        power_spectrum = (np.abs(fft_vals) ** 2) / win_norm
        mag_db = 10.0 * np.log10(power_spectrum + 1e-12)

        noise_floor_db = float(np.median(mag_db))
        peak_pwr_db = float(np.max(mag_db))
        peak_to_noise_db = float(peak_pwr_db - noise_floor_db)

        # Spectral Flatness Measure (SFM): Geometric Mean / Arithmetic Mean of power spectrum
        log_spec = np.log(power_spectrum + 1e-12)
        geom_mean = np.exp(np.mean(log_spec))
        arith_mean = np.mean(power_spectrum) + 1e-12
        spectral_flatness = float(geom_mean / arith_mean)

        # Spectral Occupancy: fraction of bins exceeding noise floor + 10 dB
        occupancy = float(np.sum(mag_db > (noise_floor_db + 10.0)) / N)

        # 3. Time-Frequency Window Analysis (STFT across 16 windows)
        num_windows = 16
        w_size = N // num_windows
        peak_freq_indices: List[int] = []
        window_energies: List[float] = []

        for i in range(num_windows):
            w = sig[i * w_size : (i + 1) * w_size]
            w_pwr = float(np.mean(np.abs(w) ** 2))
            window_energies.append(w_pwr)
            w_fft = np.abs(np.fft.fft(w))
            peak_freq_indices.append(int(np.argmax(w_fft)))

        mean_w_energy = float(np.mean(window_energies)) + 1e-12
        energy_variance = float(np.var(window_energies) / (mean_w_energy ** 2))
        freq_variance = float(np.var(peak_freq_indices))

        return {
            "signal_power": sig_power_db,
            "energy": total_energy,
            "papr": papr_db,
            "noise_floor": noise_floor_db,
            "peak_power": peak_pwr_db,
            "peak_to_noise": peak_to_noise_db,
            "occupancy": occupancy,
            "spectral_flatness": spectral_flatness,
            "energy_variance": energy_variance,
            "freq_variance": freq_variance,
        }

    # ── Explainable Rule-Based Assessment ──────────────────────────────────────

    def analyze_signal(self, iq: np.ndarray, original_iq: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluates a signal using DSP features and explainable detection rules.
        When original_iq is provided, differential analysis is performed against the baseline.
        Does NOT rely on simulation metadata or injected type.
        """
        metrics = self.extract_features(iq)

        power_delta = 0.0
        occupancy_delta = 0.0
        freq_var_delta = 0.0
        energy_var_delta = 0.0

        if original_iq is not None:
            base_metrics = self.extract_features(original_iq)
            power_delta = metrics["signal_power"] - base_metrics["signal_power"]
            occupancy_delta = metrics["occupancy"] - base_metrics["occupancy"]
            freq_var_delta = metrics["freq_variance"] - base_metrics["freq_variance"]
            energy_var_delta = metrics["energy_variance"] - base_metrics["energy_variance"]

        peak_to_noise = metrics["peak_to_noise"]
        freq_var = metrics["freq_variance"]
        energy_var = metrics["energy_variance"]
        occupancy = metrics["occupancy"]
        use_diff = original_iq is not None

        # Default assessment: No strong interference-like evidence
        status = STATE_NO_EVIDENCE
        pattern = "No obvious anomaly"
        evidence: List[str] = []
        explanation = (
            "The signal exhibits spectral and temporal feature distributions consistent with standard "
            "digitally modulated signals in noise, with no dominant narrowband peaks, chirps, or burst anomalies."
        )

        # Rule 1: Tone-like (Narrowband stationary peak)
        if peak_to_noise > 22.0 and freq_var < 8.0:
            if not use_diff or power_delta > 1.8:
                status = STATE_POSSIBLE
                pattern = "Tone-like"
                evidence = [
                    f"Strong narrowband spectral peak ({peak_to_noise:.1f} dB above noise floor)",
                    f"Stationary frequency peak across time windows (variance={freq_var:.2f})",
                    "Continuous temporal energy profile",
                ]
                explanation = (
                    "Observed a high peak-to-noise ratio accompanied by temporal frequency stability across "
                    "successive STFT windows, characteristic of a continuous-wave (CW) or tone carrier."
                )

        # Rule 2: Sweep-like (Chirp / moving spectral peak)
        elif peak_to_noise > 12.0 and freq_var > 15.0:
            if not use_diff or (power_delta > 1.0 and freq_var_delta > 8.0):
                status = STATE_POSSIBLE
                pattern = "Sweep-like"
                evidence = [
                    f"Migrating spectral peak across successive STFT windows (freq variance={freq_var:.1f})",
                    f"Moderate-to-high spectral prominence ({peak_to_noise:.1f} dB)",
                    "Dynamic spectral center of gravity",
                ]
                explanation = (
                    "The dominant spectral bin shifts systematically across temporal analysis windows, "
                    "consistent with linear chirp or swept-frequency interference."
                )

        # Rule 3: Pulsed (Burst-like temporal energy fluctuation)
        elif energy_var > 0.8:
            if not use_diff or energy_var_delta > 0.4:
                status = STATE_POSSIBLE
                pattern = "Pulsed"
                evidence = [
                    f"High normalized temporal energy variance (variance={energy_var:.2f})",
                    "Intermittent power surges detected across short-time observation intervals",
                ]
                explanation = (
                    "Short-window energy calculations reveal repetitive, high-amplitude bursts separated "
                    "by low-energy intervals, characteristic of pulsed signals."
                )

        # Rule 4: Broadband-noise-like (Wideband raised floor & high occupancy without single peak)
        elif occupancy > 0.35 and peak_to_noise < 18.0:
            if not use_diff or occupancy_delta > 0.08:
                status = STATE_POSSIBLE
                pattern = "Broadband-noise-like"
                evidence = [
                    f"High spectral occupancy ({occupancy:.1%} of spectrum exceeds noise threshold)",
                    f"Elevated noise floor without dominant single-carrier peak (P/N={peak_to_noise:.1f} dB)",
                    f"Spectral flatness score={metrics['spectral_flatness']:.3f}",
                ]
                explanation = (
                    "Spectral power is broadly distributed across frequency bins with no single dominant carrier, "
                    "indicating wideband noise or barrage interference."
                )

        # Rule 5: Anomalous power elevation without classified signature
        elif use_diff and power_delta > 5.0:
            status = STATE_POSSIBLE
            pattern = "Anomalous power increase"
            evidence = [f"Total signal power increased by {power_delta:.1f} dB relative to baseline"]
            explanation = "Substantial broad power increase observed; specific modulation pattern unclassified."

        # Rule 6: Inconclusive (borderline metrics)
        elif use_diff and (1.2 < power_delta <= 5.0 or 0.04 < occupancy_delta <= 0.08):
            status = STATE_INCONCLUSIVE
            pattern = "Inconclusive signature"
            evidence = [
                f"Mild power change (delta={power_delta:.1f} dB)",
                f"Mild occupancy change (delta={occupancy_delta:.1%})",
            ]
            explanation = (
                "Mild spectral or energy deviations observed, but evidence is insufficient or ambiguous "
                "to reliably confirm or categorize interference."
            )

        return {
            "status": status,
            "pattern": pattern,
            "evidence": evidence,
            "explanation": explanation,
            "metrics": metrics,
            "detector_version": self.DETECTOR_VERSION,
            "limitations_note": self.LIMITATIONS_NOTE,
        }

    # ── Database Persistence ──────────────────────────────────────────────────

    def save_simulation(
        self,
        dataset_index: int,
        interference_type: str,
        strength: float,
        detected_pattern: str,
        status: str,
        params: Optional[dict] = None,
        seed: Optional[int] = None,
        original_assessment: Optional[dict] = None,
        interfered_assessment: Optional[dict] = None,
        orig_pred: Optional[str] = None,
        interf_pred: Optional[str] = None,
        orig_conf: Optional[float] = None,
        interf_conf: Optional[float] = None,
        model_name: str = "ResNet1D",
    ) -> str:
        sim_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
        conn = get_db()
        try:
            conn.cursor().execute(
                """
                INSERT INTO interference_simulations
                  (id, dataset_index, interference_type, strength, detected_pattern, status,
                   params, seed, original_assessment, interfered_assessment,
                   orig_pred, interf_pred, orig_conf, interf_conf, model_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sim_id,
                    dataset_index,
                    interference_type,
                    float(strength),
                    detected_pattern,
                    status,
                    json.dumps(params or {}),
                    seed,
                    json.dumps(original_assessment or {}),
                    json.dumps(interfered_assessment or {}),
                    orig_pred or "",
                    interf_pred or "",
                    float(orig_conf) if orig_conf is not None else 0.0,
                    float(interf_conf) if interf_conf is not None else 0.0,
                    model_name,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return sim_id

    def get_simulation(self, sim_id: str) -> Optional[Dict[str, Any]]:
        conn = get_db()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM interference_simulations WHERE id = ?", (sim_id,))
            row = cur.fetchone()
            if not row:
                return None
            d = dict(row)
            if d.get("params") and isinstance(d["params"], str):
                try: d["params"] = json.loads(d["params"])
                except Exception: pass
            if d.get("original_assessment") and isinstance(d["original_assessment"], str):
                try: d["original_assessment"] = json.loads(d["original_assessment"])
                except Exception: pass
            if d.get("interfered_assessment") and isinstance(d["interfered_assessment"], str):
                try: d["interfered_assessment"] = json.loads(d["interfered_assessment"])
                except Exception: pass
            return d
        finally:
            conn.close()

    def get_history(
        self,
        limit: int = 30,
        page: int = 1,
        interference_type: Optional[str] = None,
        status: Optional[str] = None,
        dataset_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        conn = get_db()
        try:
            cur = conn.cursor()
            where_clauses = []
            params: List[Any] = []

            if interference_type:
                where_clauses.append("interference_type = ?")
                params.append(interference_type)
            if status:
                where_clauses.append("status LIKE ?")
                params.append(f"%{status}%")
            if dataset_index is not None:
                where_clauses.append("dataset_index = ?")
                params.append(dataset_index)

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            # Count total
            cur.execute(f"SELECT COUNT(*) FROM interference_simulations {where_sql}", tuple(params))
            total = cur.fetchone()[0]

            offset = max(0, (page - 1) * limit)
            query_params = list(params) + [limit, offset]

            cur.execute(
                f"""
                SELECT id, dataset_index, interference_type, strength, detected_pattern, status,
                       params, seed, orig_pred, interf_pred, orig_conf, interf_conf, model_name, created_at
                FROM interference_simulations
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                tuple(query_params),
            )
            rows = cur.fetchall()
            items = []
            for r in rows:
                item = dict(r)
                if item.get("params") and isinstance(item["params"], str):
                    try: item["params"] = json.loads(item["params"])
                    except Exception: pass
                items.append(item)

            pages = max(1, (total + limit - 1) // limit)
            return {
                "items": items,
                "total": total,
                "page": page,
                "limit": limit,
                "pages": pages,
            }
        finally:
            conn.close()


interference_service = InterferenceService()
