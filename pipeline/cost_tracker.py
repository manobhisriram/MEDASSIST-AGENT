import sqlite3
from datetime import datetime

DB_PATH = "medassist_runs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT,
            input_text TEXT,
            severity TEXT,
            total_tokens INTEGER,
            model_used TEXT,
            success INTEGER
        )
    """)
    conn.commit()
    conn.close()

def log_run(run_id: str, input_text: str, severity: str, total_tokens: int, model: str, success: bool):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO agent_runs (run_id, timestamp, input_text, severity, total_tokens, model_used, success)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (run_id, datetime.now().isoformat(), input_text[:500], severity, total_tokens, model, int(success)))
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), AVG(total_tokens) FROM agent_runs WHERE success=1")
    result = cursor.fetchone()
    conn.close()
    return {"total_runs": result[0], "avg_tokens": result[1] or 0}

init_db()