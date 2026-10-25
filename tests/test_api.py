import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.dataset_service import TOTAL_SAMPLES


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model"]["loaded"] is True
    assert "ResNet1D" in data["model"]["active_models"]


def test_dataset_info_endpoint(client):
    res = client.get("/api/dataset/info")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset"] == "RadioML 2018.01A"
    assert data["samples"] == TOTAL_SAMPLES


def test_models_catalog_endpoint(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 3
    model_names = [m["name"] for m in data]
    assert "ResNet1D" in model_names
    assert "SpectraNet" in model_names
    assert "CNN" in model_names


def test_dashboard_endpoint(client):
    res = client.get("/api/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert "signals_analyzed" in data
    assert "correct_predictions" in data
    assert "average_confidence" in data


def test_analyze_signal_valid_and_invalid(client):
    # Valid analysis with default model
    res = client.post("/api/signals/analyze", json={"dataset_index": 50, "model_name": "ResNet1D"})
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_index"] == 50
    assert data["model"] == "ResNet1D"
    assert "waveform" in data
    assert "spectrum" in data
    assert "spectrogram" in data
    assert "interference" in data
    assert len(data["top_predictions"]) == 5

    # Valid analysis with SpectraNet
    res_sp = client.post("/api/signals/analyze", json={"dataset_index": 50, "model_name": "SpectraNet"})
    assert res_sp.status_code == 200
    assert res_sp.json()["model"] == "SpectraNet"

    # Invalid negative index
    res_bad = client.post("/api/signals/analyze", json={"dataset_index": -1})
    assert res_bad.status_code == 422  # Pydantic validation error

    # Invalid out of bounds index
    res_oob = client.post("/api/signals/analyze", json={"dataset_index": TOTAL_SAMPLES + 10})
    assert res_oob.status_code == 422


def test_random_sample_alternating(client):
    res1 = client.get("/api/sample/random")
    assert res1.status_code == 200
    d1 = res1.json()

    res2 = client.get("/api/sample/random")
    assert res2.status_code == 200
    d2 = res2.json()

    # Should alternate outcome
    assert d1["result"] != d2["result"]


def test_active_sample_session(client):
    # Set active sample
    res_set = client.post("/api/sample/active", json={"dataset_index": 1234})
    assert res_set.status_code == 200

    # Get active sample
    res_get = client.get("/api/sample/active")
    assert res_get.status_code == 200
    assert res_get.json()["dataset_index"] == 1234


def test_interference_simulate(client):
    res = client.post("/api/interference/simulate", json={"dataset_index": 42, "model_name": "ResNet1D", "seed": 42})
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_index"] == 42
    assert "simulation_id" in data
    assert "injection" in data
    assert "original" in data
    assert "interfered" in data
    assert "metrics" in data["original"]
    assert "metrics" in data["interfered"]
    assert "assessment" in data["original"]
    assert "assessment" in data["interfered"]

    # Test single simulation detail retrieval
    sim_id = data["simulation_id"]
    res_detail = client.get(f"/api/interference/simulations/{sim_id}")
    assert res_detail.status_code == 200
    assert res_detail.json()["id"] == sim_id


def test_signals_pagination_and_filters(client):
    # Ensure at least one record exists
    client.post("/api/signals/analyze", json={"dataset_index": 100, "model_name": "ResNet1D"})

    res = client.get("/api/signals?limit=10&page=1")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "pages" in data
    assert data["page"] == 1
    assert len(data["items"]) <= 10


def test_pdf_and_csv_reports(client):
    # Analyze a sample first to have a valid record
    res_ana = client.post("/api/signals/analyze", json={"dataset_index": 200, "model_name": "ResNet1D"})
    assert res_ana.status_code == 200
    sig_id = res_ana.json()["signal_id"]

    # 1. Test Signal PDF export
    res_pdf = client.get(f"/api/reports/signal/{sig_id}/pdf")
    assert res_pdf.status_code == 200
    assert res_pdf.headers["content-type"] == "application/pdf"
    assert len(res_pdf.content) > 1000

    # 2. Test Signals CSV export
    res_csv = client.get("/api/reports/signals/csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    assert b"dataset_index" in res_csv.content
    assert b"predicted_modulation" in res_csv.content

    # 3. Simulate and test Simulation PDF export
    res_sim = client.post("/api/interference/simulate", json={"dataset_index": 200, "model_name": "ResNet1D", "seed": 99})
    assert res_sim.status_code == 200
    sim_id = res_sim.json()["simulation_id"]

    res_sim_pdf = client.get(f"/api/reports/simulation/{sim_id}/pdf")
    assert res_sim_pdf.status_code == 200
    assert res_sim_pdf.headers["content-type"] == "application/pdf"
    assert len(res_sim_pdf.content) > 1000

    # 4. Test Simulations CSV export
    res_sim_csv = client.get("/api/reports/simulations/csv")
    assert res_sim_csv.status_code == 200
    assert b"simulation_id" in res_sim_csv.content or b"interference_type" in res_sim_csv.content

    # 5. Test Report Verification Endpoint
    res_ver = client.get(f"/api/reports/verify/{sig_id}")
    assert res_ver.status_code == 200
    assert res_ver.json()["valid"] is True
    assert res_ver.json()["id"] == sig_id


def test_duplicate_analyses_have_distinct_ids(client):
    """Verifies that analyzing the exact same sample multiple times generates distinct unique analysis IDs."""
    res1 = client.post("/api/signals/analyze", json={"dataset_index": 333, "model_name": "ResNet1D"})
    res2 = client.post("/api/signals/analyze", json={"dataset_index": 333, "model_name": "ResNet1D"})
    assert res1.status_code == 200
    assert res2.status_code == 200
    id1 = res1.json()["signal_id"]
    id2 = res2.json()["signal_id"]
    assert id1 != id2
    assert id1.startswith("SG-")
    assert id2.startswith("SG-")


def test_missing_and_invalid_ids_return_errors(client):
    """Verifies graceful 404 responses for nonexistent analysis or simulation IDs."""
    res_sig = client.get("/api/signals/SG-NONEXISTENT")
    assert res_sig.status_code == 404

    res_pdf = client.get("/api/reports/signal/SG-NONEXISTENT/pdf")
    assert res_pdf.status_code == 404

    res_sim = client.get("/api/interference/simulations/SIM-NONEXISTENT")
    assert res_sim.status_code == 404

    res_sim_pdf = client.get("/api/reports/simulation/SIM-NONEXISTENT/pdf")
    assert res_sim_pdf.status_code == 404
