"""
rf_service.py — real RF data helpers backed by SQLite and HDF5.
"""
import csv
import os
from backend.database.database import get_db

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class RFService:
    def get_dashboard_stats(self):
        """Read actual inference counts from SQLite."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM signals")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM signals WHERE correct_prediction=1")
        correct = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM signals")
        row = cursor.fetchone()[0]
        avg_conf = round((row or 0) * 100, 1)

        conn.close()

        return {
            "signals_analyzed": total,
            "correct_predictions": correct,
            "average_confidence": avg_conf,
            "dataset_samples": 2_555_904,
        }

    def get_signals(self):
        """Return last 50 inference records from SQLite."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM signals ORDER BY created_at DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_signal_detail(self, signal_id: str):
        """Return one inference record or None."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM signals WHERE id=?", (signal_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_models(self):
        """Return model evaluation results from CSV. No dummy fallback."""
        csv_path = os.path.join(
            PROJECT_ROOT, "outputs", "results", "model_comparison_results.csv"
        )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"model_comparison_results.csv not found at {csv_path}. "
                "Run the training notebook to generate it."
            )
        models = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                models.append(row)
        return models


rf_service = RFService()
