import sqlite3
import json
import datetime
import os

class PersistenceAgent:
    """
    Saves simulation telemetry to a local SQLite database.
    """
    def __init__(self, db_path="analytics/telemetry.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metrics TEXT,
                telemetry_json TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_run(self, metrics, telemetry):
        """
        Saves a completed run.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO simulation_runs (timestamp, metrics, telemetry_json) VALUES (?, ?, ?)",
            (timestamp, json.dumps(metrics), json.dumps(telemetry))
        )
        conn.commit()
        conn.close()
        print(f"Run saved to database at {timestamp}")
