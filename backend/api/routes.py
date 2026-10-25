"""
SpectraGuard API Routes.
End-to-end dataset-backed RF signal analysis, multi-model modulation classification,
DSP feature extraction, explainable interference assessment, and verified reports.
"""
import io
import os
import uuid
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from backend.database.database import get_db
from backend.services.model_service import model_service
from backend.services.dataset_service import dataset_service
from backend.services.signal_processing import compute_spectrum, compute_spectrogram
from backend.services.interference_service import interference_service
from backend.services.analysis_session import session_service
from backend.services.report_service import report_service
from backend.api.schemas import (
    AnalyzeRequest,
    AutoSimulateRequest,
    ActiveSampleSetRequest,
)

logger = logging.getLogger("spectraguard.api")
router = APIRouter()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STATE_FILE = os.path.join(PROJECT_ROOT, "backend", "random_sequence_state.json")


# ─── Internal Pipeline Helpers ────────────────────────────────────────────────

def _run_inference(idx: int, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes full pipeline for one HDF5 sample index:
      HDF5[idx] -> I/Q + GT + SNR -> Model Inference -> Spectrum -> Spectrogram -> DSP Assessment.
    Does NOT write to DB.
    """
    sample = dataset_service.get_sample(idx)
    iq = sample["iq"]                      # (1024, 2) float32
    ground_truth = sample["ground_truth"]  # e.g. "QPSK"
    snr_db = sample["snr_db"]             # e.g. 18.0
    gt_class_id = sample["gt_class_id"]   # 0–23

    # Multi-model prediction
    pred_res = model_service.predict(iq, model_name=model_name)
    predicted_class = pred_res["predicted_class"]
    pred_class_id = pred_res["predicted_class_id"]
    confidence = float(pred_res["confidence"])
    top_predictions = pred_res["top_predictions"]
    active_model = pred_res["model_name"]

    # Frequency-domain representations
    spectrum = compute_spectrum(iq)
    spectrogram_data = compute_spectrogram(iq)

    # Independent DSP feature extraction and rule-based anomaly detection
    dsp_assessment = interference_service.analyze_signal(iq)

    is_correct = bool(predicted_class == ground_truth)

    return {
        "iq": iq,
        "ground_truth": ground_truth,
        "gt_class_id": gt_class_id,
        "snr_db": snr_db,
        "predicted_class": predicted_class,
        "pred_class_id": pred_class_id,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "model_name": active_model,
        "spectrum": spectrum,
        "spectrogram_data": spectrogram_data,
        "interference_assessment": dsp_assessment,
        "is_correct": is_correct,
        "is_test_sample": sample.get("is_test_sample", False),
    }


def _save_to_db(sig_id: str, idx: int, inf: Dict[str, Any], sim_id: Optional[str] = None):
    """Persists inference record into SQLite signals table with parameterized queries."""
    conn = get_db()
    try:
        cur = conn.cursor()
        dsp_json = json.dumps(inf.get("interference_assessment", {}))
        top_json = json.dumps(inf.get("top_predictions", []))
        cur.execute(
            """
            INSERT INTO signals
              (id, dataset_index, ground_truth_modulation, predicted_modulation,
               confidence, snr_db, model_name, data_source, correct_prediction,
               dsp_assessment, top_predictions, simulation_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sig_id,
                idx,
                inf["ground_truth"],
                inf["predicted_class"],
                float(inf["confidence"]),
                float(inf["snr_db"]),
                inf.get("model_name", "ResNet1D"),
                "RadioML 2018.01A",
                int(inf["is_correct"]),
                dsp_json,
                top_json,
                sim_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _build_response(sig_id: str, idx: int, inf: Dict[str, Any]) -> Dict[str, Any]:
    """Constructs the standard standardized response payload."""
    iq = inf["iq"]
    conf_pct = round(inf["confidence"] * 100, 2)
    return {
        "signal_id": sig_id,
        "dataset_index": idx,
        "is_test_sample": inf.get("is_test_sample", True),
        "ground_truth_modulation": inf["ground_truth"],
        "ground_truth": {
            "class_id": inf["gt_class_id"],
            "label": inf["ground_truth"],
        },
        "snr_db": inf["snr_db"],
        "predicted_modulation": inf["predicted_class"],
        "prediction": {
            "class_id": inf["pred_class_id"],
            "label": inf["predicted_class"],
            "confidence": inf["confidence"],
            "confidence_pct": conf_pct,
        },
        "confidence": inf["confidence"],
        "confidence_pct": conf_pct,
        "correct_prediction": inf["is_correct"],
        "result": "correct" if inf["is_correct"] else "incorrect",
        "model": inf.get("model_name", "ResNet1D"),
        "data_source": "RadioML 2018.01A",
        "iq_samples": 1024,
        "representation": "I/Q",
        "top_predictions": inf["top_predictions"],
        "top5": inf["top_predictions"],
        "waveform": {
            "i": iq[:, 0].tolist(),
            "q": iq[:, 1].tolist(),
        },
        "spectrum": inf["spectrum"],
        "spectrogram": inf["spectrogram_data"],
        "interference": {
            "status": inf["interference_assessment"]["status"],
            "pattern": inf["interference_assessment"]["pattern"],
            "evidence": inf["interference_assessment"]["evidence"],
            "explanation": inf["interference_assessment"]["explanation"],
            "metrics": inf["interference_assessment"]["metrics"],
            "detector_version": inf["interference_assessment"].get("detector_version", "2.1.0-dsp"),
            "limitations_note": inf["interference_assessment"].get("limitations_note", ""),
        },
    }


# ─── System & Metadata Endpoints ──────────────────────────────────────────────

@router.get("/health", tags=["System"], summary="Health check and model load status")
def health():
    return {
        "status": "ok",
        "service": "SpectraGuard API",
        "model": model_service.status,
    }


@router.get("/dataset/info", tags=["Dataset"], summary="RadioML 2018.01A metadata")
def dataset_info():
    try:
        return dataset_service.info()
    except Exception as e:
        logger.error(f"Failed to fetch dataset info: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@router.get("/models", tags=["Models"], summary="Model catalog and measured evaluation metrics")
def get_models():
    """Returns the three evaluated models with actual measured test metrics from training."""
    return model_service.list_models()


# ─── Dashboard Aggregate Metrics ──────────────────────────────────────────────

@router.get("/dashboard", tags=["Dashboard"], summary="Aggregate system metrics")
def get_dashboard():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM signals")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM signals WHERE correct_prediction=1")
        correct = cur.fetchone()[0]

        cur.execute("SELECT AVG(confidence) FROM signals")
        avg_conf_row = cur.fetchone()[0]
        avg_conf = round((avg_conf_row or 0) * 100, 1)
    finally:
        conn.close()

    try:
        ds_info = dataset_service.info()
        total_samples = ds_info["samples"]
        test_samples = ds_info.get("test_samples", 46800)
    except Exception:
        total_samples = 2_555_904
        test_samples = 46_800

    return {
        "signals_analyzed": total,
        "correct_predictions": correct,
        "average_confidence": avg_conf,
        "dataset_samples": total_samples,
        "test_samples": test_samples,
    }


@router.get("/dashboard/activity", tags=["Dashboard"])
def get_activity():
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT strftime('%H:%M', created_at) as bucket, COUNT(*) as count
            FROM signals
            GROUP BY strftime('%Y-%m-%d %H', created_at), (strftime('%M', created_at) / 15)
            ORDER BY created_at DESC
            LIMIT 20
        """)
        rows = cur.fetchall()
        return [{"time": r[0], "count": r[1]} for r in reversed(rows)]
    finally:
        conn.close()


# ─── Signal History & Artifacts ───────────────────────────────────────────────

@router.get("/signals", tags=["Signals"], summary="Recent inference history with filtering and pagination")
def get_signals(
    limit: int = Query(50, ge=1, le=500),
    page: int = Query(1, ge=1),
    search: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    dsp_status: Optional[str] = Query(None),
    sort_by: str = Query("created_at"),
    sort_dir: str = Query("desc"),
):
    """
    Returns paginated historical inference records with optional search and filters.
    Preserves all stored metadata, ground truth, prediction, and DSP findings.
    """
    conn = get_db()
    try:
        cur = conn.cursor()
        where_clauses = []
        params: List[Any] = []

        if search:
            search_clean = search.strip()
            if search_clean.startswith("#"):
                search_clean = search_clean[1:]
            if search_clean.isdigit():
                where_clauses.append("(dataset_index = ? OR id LIKE ?)")
                params.extend([int(search_clean), f"%{search_clean}%"])
            else:
                where_clauses.append("(id LIKE ? OR ground_truth_modulation LIKE ? OR predicted_modulation LIKE ?)")
                params.extend([f"%{search_clean}%", f"%{search_clean}%", f"%{search_clean}%"])

        if model_name:
            where_clauses.append("model_name = ?")
            params.append(model_name)

        if outcome == "match":
            where_clauses.append("correct_prediction = 1")
        elif outcome == "mismatch":
            where_clauses.append("correct_prediction = 0")

        if dsp_status:
            where_clauses.append("dsp_assessment LIKE ?")
            params.append(f"%{dsp_status}%")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Total count
        cur.execute(f"SELECT COUNT(*) FROM signals {where_sql}", tuple(params))
        total = cur.fetchone()[0]

        # Valid sort columns
        allowed_sort = {"created_at", "confidence", "snr_db", "dataset_index", "model_name"}
        s_col = sort_by if (isinstance(sort_by, str) and sort_by in allowed_sort) else "created_at"
        s_dir = "ASC" if (isinstance(sort_dir, str) and sort_dir.lower() == "asc") else "DESC"

        offset = max(0, (page - 1) * limit)
        query_params = list(params) + [limit, offset]

        cur.execute(
            f"""
            SELECT id, dataset_index, ground_truth_modulation, predicted_modulation,
                   confidence, snr_db, model_name, data_source, correct_prediction,
                   dsp_assessment, top_predictions, simulation_id, created_at
            FROM signals
            {where_sql}
            ORDER BY {s_col} {s_dir}
            LIMIT ? OFFSET ?
            """,
            tuple(query_params),
        )
        rows = cur.fetchall()
        items = []
        for r in rows:
            item = dict(r)
            if item.get("dsp_assessment") and isinstance(item["dsp_assessment"], str):
                try: item["dsp_assessment"] = json.loads(item["dsp_assessment"])
                except Exception: pass
            if item.get("top_predictions") and isinstance(item["top_predictions"], str):
                try: item["top_predictions"] = json.loads(item["top_predictions"])
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


@router.get("/signals/{signal_id}", tags=["Signals"], summary="Fetch single signal details")
def get_signal(signal_id: str):
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM signals WHERE id=?", (signal_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Signal '{signal_id}' not found")
        d = dict(row)
        if d.get("dsp_assessment") and isinstance(d["dsp_assessment"], str):
            try: d["dsp_assessment"] = json.loads(d["dsp_assessment"])
            except Exception: pass
        if d.get("top_predictions") and isinstance(d["top_predictions"], str):
            try: d["top_predictions"] = json.loads(d["top_predictions"])
            except Exception: pass
        return d
    finally:
        conn.close()


@router.get("/signals/{signal_id}/prediction", tags=["Signals"])
def get_prediction(signal_id: str, model_name: Optional[str] = Query(None)):
    sig = get_signal(signal_id)
    sample = dataset_service.get_sample(sig["dataset_index"])
    iq = sample["iq"]
    chosen_model = model_name or sig.get("model_name", "ResNet1D")
    result = model_service.predict(iq, model_name=chosen_model)
    return {
        "model": result["model_name"],
        "predicted_modulation": result["predicted_class"],
        "confidence": round(result["confidence"] * 100, 2),
        "snr_db": sig["snr_db"],
        "dataset": "RadioML 2018.01A",
        "num_classes": 24,
        "top_predictions": result["top_predictions"],
    }


@router.get("/signals/{signal_id}/waveform", tags=["Signals"])
def get_waveform(signal_id: str):
    sig = get_signal(signal_id)
    sample = dataset_service.get_sample(sig["dataset_index"])
    iq = sample["iq"]
    return {"i": iq[:, 0].tolist(), "q": iq[:, 1].tolist()}


@router.get("/signals/{signal_id}/spectrum", tags=["Signals"])
def get_spectrum(signal_id: str):
    sig = get_signal(signal_id)
    sample = dataset_service.get_sample(sig["dataset_index"])
    return compute_spectrum(sample["iq"])


@router.get("/signals/{signal_id}/spectrogram", tags=["Signals"])
def get_spectrogram(signal_id: str):
    sig = get_signal(signal_id)
    sample = dataset_service.get_sample(sig["dataset_index"])
    return compute_spectrogram(sample["iq"])


# ─── Core Analysis & Sequence Sample Endpoints ────────────────────────────────

def _get_sequence_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"next": "correct", "correct_pool": [], "incorrect_pool": []}


def _save_sequence_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning(f"Could not save sequence state: {e}")


@router.get("/sample/random", tags=["Inference"], summary="Controlled alternating test-set sample")
def get_random_sample(model_name: Optional[str] = Query(None)):
    """
    Selects a test-set sample alternating between verified correct and incorrect predictions.
    Saves the analyzed sample to the active session and history.
    """
    if len(dataset_service.test_indices) == 0:
        dataset_service._open()
    if len(dataset_service.test_indices) == 0:
        raise HTTPException(status_code=503, detail="Test set indices unavailable.")

    active_model = model_name or "ResNet1D"
    state = _get_sequence_state()
    target_result = state.get("next", "correct")
    pool_key = f"{target_result}_pool"

    found_idx = None
    inf = None

    # Check pool cache first
    while state.get(pool_key) and found_idx is None:
        idx = state[pool_key].pop(0)
        try:
            res = _run_inference(idx, model_name=active_model)
            if (target_result == "correct" and res["is_correct"]) or (target_result == "incorrect" and not res["is_correct"]):
                found_idx = idx
                inf = res
        except Exception:
            pass

    # Search dynamically if not found in cache
    if found_idx is None:
        test_pool = dataset_service.test_indices.tolist()
        import random
        for _ in range(250):
            idx = int(random.choice(test_pool))
            try:
                res = _run_inference(idx, model_name=active_model)
            except Exception:
                continue

            if (target_result == "correct" and res["is_correct"]) or (target_result == "incorrect" and not res["is_correct"]):
                found_idx = idx
                inf = res
                break
            else:
                other_key = "incorrect_pool" if target_result == "correct" else "correct_pool"
                if len(state.get(other_key, [])) < 30:
                    state.setdefault(other_key, []).append(idx)

    if found_idx is None:
        import random
        found_idx = int(random.choice(dataset_service.test_indices))
        inf = _run_inference(found_idx, model_name=active_model)

    # Alternate next target
    state["next"] = "incorrect" if target_result == "correct" else "correct"
    _save_sequence_state(state)

    sig_id = f"SG-{uuid.uuid4().hex[:6].upper()}"
    try:
        _save_to_db(sig_id, found_idx, inf)
    except Exception as e:
        logger.error(f"Failed to save random sample inference to DB: {e}")

    session_service.set_active_sample(found_idx, sig_id, {"ground_truth": inf["ground_truth"], "snr_db": inf["snr_db"]})
    return _build_response(sig_id, found_idx, inf)


@router.post("/signals/analyze", tags=["Inference"], summary="Analyze specific dataset sample")
def analyze_signal(req: AnalyzeRequest):
    """
    Performs deterministic modulation classification and independent DSP feature extraction
    on a selected RadioML 2018.01A HDF5 index using the specified model checkpoint.
    """
    idx = req.dataset_index
    try:
        sample = dataset_service.get_sample(idx)
    except IndexError as e:
        raise HTTPException(status_code=400, detail=f"Index {idx} out of range [0, {dataset_service.TOTAL_SAMPLES - 1}]")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read HDF5 sample: {e}")

    active_model = req.model_name or "ResNet1D"
    inf = _run_inference(idx, model_name=active_model)
    sig_id = f"SG-{uuid.uuid4().hex[:6].upper()}"

    try:
        _save_to_db(sig_id, idx, inf)
    except Exception as e:
        logger.error(f"Failed to persist signal analysis record: {e}")

    session_service.set_active_sample(idx, sig_id, {"ground_truth": inf["ground_truth"], "snr_db": inf["snr_db"]})
    return _build_response(sig_id, idx, inf)


@router.get("/sample/active", tags=["Session"], summary="Get current active sample reference")
def get_active_sample():
    sample = session_service.get_active_sample()
    if sample:
        return {"active": True, **sample}
    return {"active": False, "dataset_index": None, "signal_id": None}


@router.post("/sample/active", tags=["Session"], summary="Set active sample reference")
def set_active_sample(req: ActiveSampleSetRequest):
    try:
        sample = dataset_service.get_sample(req.dataset_index)
        meta = {"ground_truth": sample["ground_truth"], "snr_db": sample["snr_db"]}
    except Exception:
        meta = {}
    session_service.set_active_sample(req.dataset_index, req.signal_id, meta)
    return {"status": "ok", "dataset_index": req.dataset_index, "signal_id": req.signal_id}


@router.get("/sample/search", tags=["Dataset"], summary="Search dataset by modulation and SNR")
def search_dataset(
    modulation: Optional[str] = Query(None),
    min_snr: Optional[float] = Query(None),
    model_name: Optional[str] = Query(None),
):
    try:
        idx = dataset_service.search_sample(modulation, min_snr)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    active_model = model_name or "ResNet1D"
    inf = _run_inference(idx, model_name=active_model)
    sig_id = f"SG-{uuid.uuid4().hex[:6].upper()}"
    try:
        _save_to_db(sig_id, idx, inf)
    except Exception:
        pass

    session_service.set_active_sample(idx, sig_id, {"ground_truth": inf["ground_truth"], "snr_db": inf["snr_db"]})
    return _build_response(sig_id, idx, inf)


# ─── Interference Analysis ────────────────────────────────────────────────────

@router.post("/interference/simulate", tags=["Interference"], summary="Simulated interference analysis")
def simulate_interference(req: AutoSimulateRequest):
    """
    Executes controlled simulated interference workflow on the exact source sample:
      1. Resolves sample index (explicitly passed or from active backend session).
      2. Synthesizes an interference overlay in memory (original HDF5 is never modified).
      3. Performs DSP feature extraction on both original and derived signals.
      4. Runs independent rule-based detection on both signals.
      5. Runs modulation classification on both signals using the specified model.
      6. Returns comprehensive side-by-side comparison payload and persists simulation run.
    """
    idx = req.dataset_index
    if idx is None:
        active = session_service.get_active_sample()
        if active and active.get("dataset_index") is not None:
            idx = active["dataset_index"]
        else:
            raise HTTPException(
                status_code=400,
                detail="No dataset index specified and no active sample selected. Select a sample in Signal Analysis first."
            )

    try:
        sample = dataset_service.get_sample(idx)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid sample index {idx}: {e}")

    original_iq = sample["iq"]
    ground_truth = sample.get("ground_truth", "Unknown")
    snr_db = float(sample.get("snr_db", 0.0))
    chosen_model = req.model_name or "ResNet1D"

    # Pattern selection: explicit or auto-random
    if req.interference_type and req.interference_type in interference_service.SUPPORTED_PATTERNS:
        inj_type = req.interference_type
        inj_strength = float(req.strength) if req.strength is not None else 1.2
        inj_params = req.params or {}
        # Fill sensible defaults if params omitted
        if inj_type == "tone-like" and "frequency" not in inj_params:
            inj_params["frequency"] = 0.25
        elif inj_type == "sweep-like" and "f_start" not in inj_params:
            inj_params["f_start"] = -0.35
            inj_params["f_end"] = 0.35
        elif inj_type == "pulsed" and "duration" not in inj_params:
            inj_params["duration"] = 0.1
            inj_params["interval"] = 0.3
        elif inj_type == "broadband-noise-like" and "bandwidth" not in inj_params:
            inj_params["bandwidth"] = 0.8
    else:
        injection = interference_service.random_injection(seed=req.seed)
        inj_type = injection["type"]
        inj_strength = injection["strength"]
        inj_params = injection["params"]

    # Synthesize in memory (original_iq is strictly copied, never mutated)
    interfered_iq, _ = interference_service.generate_interference(
        original_iq, inj_type, inj_strength, inj_params, seed=req.seed
    )

    # DSP analysis on original baseline and derived signal
    original_analysis = interference_service.analyze_signal(original_iq)
    interfered_analysis = interference_service.analyze_signal(interfered_iq, original_iq=original_iq)

    # Modulation classification for before / after comparison
    try:
        orig_pred = model_service.predict(original_iq, model_name=chosen_model)
        interf_pred = model_service.predict(interfered_iq, model_name=chosen_model)
    except Exception as e:
        logger.error(f"Prediction failure during simulation: {e}")
        orig_pred = {"predicted_class": "Unknown", "confidence": 0.0, "top_predictions": []}
        interf_pred = {"predicted_class": "Unknown", "confidence": 0.0, "top_predictions": []}

    original_spectrum = compute_spectrum(original_iq)
    original_spectrogram = compute_spectrogram(original_iq)
    interfered_spectrum = compute_spectrum(interfered_iq)
    interfered_spectrogram = compute_spectrogram(interfered_iq)

    orig_pred_class = orig_pred.get("predicted_class", "Unknown")
    interf_pred_class = interf_pred.get("predicted_class", "Unknown")
    orig_conf = float(orig_pred.get("confidence", 0.0))
    interf_conf = float(interf_pred.get("confidence", 0.0))

    # Save to history with rich provenance metadata
    sim_id = interference_service.save_simulation(
        dataset_index=idx,
        interference_type=inj_type,
        strength=inj_strength,
        detected_pattern=interfered_analysis["pattern"],
        status=interfered_analysis["status"],
        params=inj_params,
        seed=req.seed,
        original_assessment=original_analysis,
        interfered_assessment=interfered_analysis,
        orig_pred=orig_pred_class,
        interf_pred=interf_pred_class,
        orig_conf=orig_conf,
        interf_conf=interf_conf,
        model_name=chosen_model,
    )

    return {
        "simulation_id": sim_id,
        "dataset_index": idx,
        "ground_truth": ground_truth,
        "snr_db": snr_db,
        "model": chosen_model,
        "injection": {
            "type": inj_type,
            "strength": inj_strength,
            "params": inj_params,
            "seed": req.seed,
        },
        "original": {
            "waveform": {"i": original_iq[:, 0].tolist(), "q": original_iq[:, 1].tolist()},
            "spectrum": original_spectrum,
            "spectrogram": original_spectrogram,
            "metrics": original_analysis["metrics"],
            "modulation": {
                "predicted": orig_pred_class,
                "confidence": round(orig_conf, 4),
                "top5": orig_pred.get("top_predictions", []),
            },
            "assessment": {
                "status": original_analysis["status"],
                "pattern": original_analysis["pattern"],
                "evidence": original_analysis["evidence"],
                "explanation": original_analysis["explanation"],
            },
        },
        "interfered": {
            "waveform": {"i": interfered_iq[:, 0].tolist(), "q": interfered_iq[:, 1].tolist()},
            "spectrum": interfered_spectrum,
            "spectrogram": interfered_spectrogram,
            "metrics": interfered_analysis["metrics"],
            "modulation": {
                "predicted": interf_pred_class,
                "confidence": round(interf_conf, 4),
                "top5": interf_pred.get("top_predictions", []),
            },
            "assessment": {
                "status": interfered_analysis["status"],
                "pattern": interfered_analysis["pattern"],
                "evidence": interfered_analysis["evidence"],
                "explanation": interfered_analysis["explanation"],
            },
        },
        "analysis": {
            "status": interfered_analysis["status"],
            "pattern": interfered_analysis["pattern"],
            "evidence": interfered_analysis["evidence"],
            "explanation": interfered_analysis["explanation"],
        },
    }


@router.get("/interference/history", tags=["Interference"], summary="Paginated simulation history")
def get_interference_history(
    limit: int = Query(30, ge=1, le=100),
    page: int = Query(1, ge=1),
    interference_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    dataset_index: Optional[int] = Query(None),
):
    return interference_service.get_history(
        limit=limit,
        page=page,
        interference_type=interference_type,
        status=status,
        dataset_index=dataset_index,
    )


@router.get("/interference/simulations/{sim_id}", tags=["Interference"], summary="Fetch single simulation details")
def get_simulation_detail(sim_id: str):
    sim = interference_service.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found")
    return sim


# ─── Report & Export Endpoints (Phase 4) ──────────────────────────────────────

@router.get("/reports/signal/{signal_id}/pdf", tags=["Reports"], summary="Export Signal Analysis PDF Report")
def export_signal_pdf(signal_id: str):
    """
    Renders an official, verifiable Signal Analysis Report PDF using ReportLab.
    Includes classification findings, top predictions, DSP metrics, and verification QR code.
    """
    sig = get_signal(signal_id)
    # If dsp_assessment was not stored in older DB rows, compute it on the fly
    if not sig.get("dsp_assessment"):
        try:
            sample = dataset_service.get_sample(sig["dataset_index"])
            sig["dsp_assessment"] = interference_service.analyze_signal(sample["iq"])
        except Exception:
            sig["dsp_assessment"] = {}

    buf = io.BytesIO()
    try:
        meta = report_service.generate_signal_pdf(sig, buf)
    except Exception as e:
        logger.error(f"Failed to generate signal PDF report: {e}")
        raise HTTPException(status_code=500, detail=f"Report rendering failed: {e}")

    pdf_bytes = buf.getvalue()
    buf.close()

    filename = f"SpectraGuard_Report_{signal_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-SpectraGuard-Report-ID": signal_id,
            "X-SpectraGuard-Ref-Code": meta["verification_code"],
        },
    )


@router.get("/reports/simulation/{sim_id}/pdf", tags=["Reports"], summary="Export Simulation Lab PDF Report")
def export_simulation_pdf(sim_id: str):
    """
    Renders an official Simulation Lab Analysis Report PDF using ReportLab.
    Includes tripartite assessment, classification robustness, and reproducibility parameters.
    """
    sim = interference_service.get_simulation(sim_id)
    if not sim:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found")

    buf = io.BytesIO()
    try:
        meta = report_service.generate_simulation_pdf(sim, buf)
    except Exception as e:
        logger.error(f"Failed to generate simulation PDF report: {e}")
        raise HTTPException(status_code=500, detail=f"Report rendering failed: {e}")

    pdf_bytes = buf.getvalue()
    buf.close()

    filename = f"SpectraGuard_Simulation_{sim_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-SpectraGuard-Simulation-ID": sim_id,
            "X-SpectraGuard-Ref-Code": meta["verification_code"],
        },
    )


@router.get("/reports/signals/csv", tags=["Reports"], summary="Export Signals History CSV")
def export_signals_csv(
    search: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    dsp_status: Optional[str] = Query(None),
):
    """Streams an RFC-4180 compliant CSV export containing all historical signals matching filters."""
    # Retrieve up to 2000 records matching current filter
    res = get_signals(
        limit=2000,
        page=1,
        search=search,
        model_name=model_name,
        outcome=outcome,
        dsp_status=dsp_status,
        sort_by="created_at",
        sort_dir="desc",
    )
    items = res["items"]
    csv_text = report_service.generate_signals_csv(items)

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="SpectraGuard_Signals_History.csv"',
        },
    )


@router.get("/reports/simulations/csv", tags=["Reports"], summary="Export Simulations History CSV")
def export_simulations_csv(
    interference_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """Streams an RFC-4180 compliant CSV export containing all historical simulation runs."""
    res = interference_service.get_history(limit=2000, page=1, interference_type=interference_type, status=status)
    items = res["items"]
    csv_text = report_service.generate_simulations_csv(items)

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="SpectraGuard_Simulations_History.csv"',
        },
    )


@router.get("/reports/verify/{ref_code}", tags=["Reports"], summary="Verify Report Authenticity")
def verify_report_authenticity(ref_code: str):
    """
    Validates report reference codes (e.g. REF-SG-xxxxxx or REF-SIM-xxxxxx)
    against the local database to verify authenticity and report provenance.
    """
    clean_code = ref_code.strip()
    # Check signals
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, dataset_index, model_name, ground_truth_modulation, predicted_modulation, confidence, created_at FROM signals")
        for row in cur.fetchall():
            d = dict(row)
            if d["id"] in clean_code or clean_code in f"REF-{d['id']}":
                return {
                    "valid": True,
                    "type": "Signal Analysis Report",
                    "id": d["id"],
                    "dataset_index": d["dataset_index"],
                    "model": d["model_name"],
                    "ground_truth": d["ground_truth_modulation"],
                    "prediction": d["predicted_modulation"],
                    "confidence": d["confidence"],
                    "created_at": d["created_at"],
                    "status": "Verified Authentic",
                }

        cur.execute("SELECT id, dataset_index, interference_type, strength, status, created_at FROM interference_simulations")
        for row in cur.fetchall():
            d = dict(row)
            if d["id"] in clean_code or clean_code in f"REF-{d['id']}":
                return {
                    "valid": True,
                    "type": "Simulation Lab Report",
                    "id": d["id"],
                    "dataset_index": d["dataset_index"],
                    "interference_type": d["interference_type"],
                    "strength": d["strength"],
                    "status": d["status"],
                    "created_at": d["created_at"],
                    "status_label": "Verified Authentic",
                }

        return {
            "valid": False,
            "ref_code": ref_code,
            "status": "Reference code not found in SpectraGuard registry",
        }
    finally:
        conn.close()
