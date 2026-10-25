"""
SpectraGuard SQLite Database Manager.
Handles persistent analysis and simulation history with indexed queries and safe migrations.
"""
import sqlite3
import os
import logging

logger = logging.getLogger("spectraguard.database")

DB_PATH = os.environ.get(
    "SPECTRAGUARD_DB_PATH",
    os.path.join(os.path.dirname(__file__), "spectraguard.db")
)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(cursor, table: str, col_name: str, col_type: str):
    """Safely adds a column to an existing table if it does not already exist."""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if col_name not in columns:
        logger.info(f"Adding column '{col_name}' ({col_type}) to table '{table}'")
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")


def init_db():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = get_db()
    cursor = conn.cursor()

    # Core signals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id TEXT PRIMARY KEY,
            dataset_index INTEGER NOT NULL,
            ground_truth_modulation TEXT,
            predicted_modulation TEXT,
            confidence REAL,
            snr_db REAL,
            model_name TEXT DEFAULT 'ResNet1D',
            data_source TEXT DEFAULT 'RadioML 2018.01A',
            correct_prediction INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Core simulations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interference_simulations (
            id TEXT PRIMARY KEY,
            dataset_index INTEGER NOT NULL,
            interference_type TEXT,
            strength REAL,
            detected_pattern TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Safe column migrations for Phase 3 & 4 data completeness
    _ensure_column(cursor, "signals", "dsp_assessment", "TEXT")
    _ensure_column(cursor, "signals", "top_predictions", "TEXT")
    _ensure_column(cursor, "signals", "simulation_id", "TEXT")

    _ensure_column(cursor, "interference_simulations", "params", "TEXT")
    _ensure_column(cursor, "interference_simulations", "seed", "INTEGER")
    _ensure_column(cursor, "interference_simulations", "original_assessment", "TEXT")
    _ensure_column(cursor, "interference_simulations", "interfered_assessment", "TEXT")
    _ensure_column(cursor, "interference_simulations", "orig_pred", "TEXT")
    _ensure_column(cursor, "interference_simulations", "interf_pred", "TEXT")
    _ensure_column(cursor, "interference_simulations", "orig_conf", "REAL")
    _ensure_column(cursor, "interference_simulations", "interf_conf", "REAL")
    _ensure_column(cursor, "interference_simulations", "model_name", "TEXT DEFAULT 'ResNet1D'")

    # Performance indexes for historical queries and filtering
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_dataset_index ON signals (dataset_index)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals (created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_model_name ON signals (model_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_correct ON signals (correct_prediction)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_predicted ON signals (predicted_modulation)")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_dataset_index ON interference_simulations (dataset_index)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_created_at ON interference_simulations (created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_status ON interference_simulations (status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_type ON interference_simulations (interference_type)")

    conn.commit()
    conn.close()
