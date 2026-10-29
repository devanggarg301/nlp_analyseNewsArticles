import json
import sqlite3
import threading
from datetime import datetime, timezone
from uuid import uuid4
from .config import RUNTIME

LOCK = threading.RLock()
DB = RUNTIME / "prism.sqlite3"

def connect():
    db = sqlite3.connect(DB, timeout=20)
    db.execute("PRAGMA journal_mode=WAL")
    return db

def init():
    with connect() as db:
        db.execute("CREATE TABLE IF NOT EXISTS workspaces (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, workspace_id TEXT, data TEXT NOT NULL)")
    # A job cannot resume its in-memory executor after a server restart.
    with connect() as db:
        for jid, raw in db.execute("SELECT id,data FROM jobs").fetchall():
            job = json.loads(raw)
            if job["status"] in ("queued", "running"):
                job.update(status="failed", error="Server restarted. Run the analysis again.")
                db.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(job), jid))

def now():
    return datetime.now(timezone.utc).isoformat()

def save(ws):
    ws["updated_at"] = now()
    with LOCK, connect() as db:
        db.execute("INSERT OR REPLACE INTO workspaces VALUES (?,?)", (ws["id"], json.dumps(ws, ensure_ascii=False)))
    return ws

def get(wid):
    with connect() as db:
        row = db.execute("SELECT data FROM workspaces WHERE id=?", (wid,)).fetchone()
    return json.loads(row[0]) if row else None

def listing():
    with connect() as db:
        rows = db.execute("SELECT data FROM workspaces").fetchall()
    records = [json.loads(r[0]) for r in rows]
    return sorted([{"id": w["id"], "title": w["title"], "article_count": len(w["articles"]), "updated_at": w["updated_at"], "has_analysis": bool(w.get("analysis"))} for w in records], key=lambda w:w["updated_at"], reverse=True)

def create(title):
    return save({"id": uuid4().hex, "title": title, "articles": [], "analysis": None, "chat": [], "created_at": now()})

def job_save(job):
    with LOCK, connect() as db:
        db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?)", (job["id"], job["workspace_id"], json.dumps(job)))
    return job

def job_get(jid):
    with connect() as db:
        row = db.execute("SELECT data FROM jobs WHERE id=?", (jid,)).fetchone()
    return json.loads(row[0]) if row else None

