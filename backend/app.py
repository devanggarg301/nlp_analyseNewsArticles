import csv
import io
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from uuid import uuid4
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from . import store, models, llm
from .analysis import analyze, duplicate, event_warnings, merge_entities
from .config import ROOT
from .demo import DEMO
from .fetcher import extract_url
from .schemas import WorkspaceInput, ArticleInput, URLInput, AnalyzeInput, ChatInput, MergeInput

EXECUTOR = ThreadPoolExecutor(max_workers=2)
ACTIVE = set()
ACTIVE_LOCK = threading.RLock()

@asynccontextmanager
async def lifespan(app):
    store.init()
    yield

app = FastAPI(title="Perspective Prism",version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173","http://localhost:5173","http://127.0.0.1:8000"],allow_methods=["GET","POST","PATCH","DELETE"],allow_headers=["Content-Type"])

def workspace(wid):
    ws = store.get(wid)
    if not ws:
        raise HTTPException(404,"Workspace not found.")
    return ws

def editable(wid):
    ws = workspace(wid)
    with ACTIVE_LOCK:
        if wid in ACTIVE:
            raise HTTPException(409,"Analysis is running. Wait before editing articles.")
    return ws

@app.get("/api/health")
def health():
    return {"status":"ok","llm_configured":llm.configured(),"models":models.status()}

@app.get("/api/workspaces")
def workspaces():
    return store.listing()

@app.post("/api/workspaces")
def create_workspace(body:WorkspaceInput):
    return store.create(body.title)

@app.get("/api/workspaces/{wid}")
def read_workspace(wid:str):
    ws = workspace(wid)
    # Notes describe the saved analysis, not necessarily today's provider configuration.
    # Keep stored provenance intact and make the distinction explicit in the response.
    if ws.get("analysis") and llm.configured():
        ws["analysis"]["warnings"] = [
            "This analysis was created before API setup. Gemini is configured now; rerun the comparison to update entity sentiment. Chat uses the current API settings."
            if note.startswith("No API key configured:") else note
            for note in ws["analysis"].get("warnings", [])
        ]
    return ws

@app.get("/api/examples")
def examples():
    return [{"id":"riverton","title":DEMO["title"],"description":DEMO["description"],"articles":3}]

@app.post("/api/examples/riverton")
def load_demo():
    ws = store.create(DEMO["title"])
    ws["articles"] = [{"id":uuid4().hex,**a,"url":None} for a in DEMO["articles"]]
    ws["demo"] = True
    return store.save(ws)

@app.post("/api/extract")
def extract(body:URLInput):
    try:
        return extract_url(body.url)
    except Exception as exc:
        if isinstance(exc,ValueError):
            message = str(exc)
        else:
            message = "The article could not be fetched. Check your connection or paste its text instead."
        raise HTTPException(422,message)

@app.post("/api/preview-upload")
async def preview_upload(file:UploadFile = File(...)):
    from .documents import MAX_UPLOAD, extract_document
    data = await file.read(MAX_UPLOAD+1)
    if len(data)>MAX_UPLOAD:
        raise HTTPException(422,"Upload must be smaller than 10 MB.")
    if (file.filename or "").lower().endswith((".pdf",".docx",".doc")):
        try:
            return {"articles":[extract_document(file.filename,data)]}
        except ValueError as exc:
            raise HTTPException(422,str(exc))
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(422,"Text and CSV files must use UTF-8 encoding.")
    if (file.filename or "").lower().endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
        if not 1<=len(rows)<=5:
            raise HTTPException(422,"CSV must contain between one and five articles.")
        if not all("text" in r and "title" in r and "publisher" in r for r in rows):
            raise HTTPException(422,"CSV columns required: title,publisher,text. Optional: url,published_at.")
        drafts = [{"title":r["title"],"publisher":r["publisher"],"text":r["text"],"url":r.get("url") or None,"published_at":r.get("published_at") or None,"source_type":"upload"} for r in rows]
    elif (file.filename or "").lower().endswith(".txt"):
        drafts = [{"title":(file.filename or "Article").rsplit(".",1)[0],"publisher":"Uploaded article","text":text,"source_type":"upload"}]
    else:
        raise HTTPException(422,"Supported formats are PDF, Word (.docx), TXT and CSV.")
    return {"articles":drafts}

@app.post("/api/workspaces/{wid}/articles")
def add_article(wid:str,body:ArticleInput):
    with store.LOCK:
        ws = editable(wid)
        if len(ws["articles"])>=5:
            raise HTTPException(422,"A workspace supports at most five articles.")
        if duplicate(body.text,ws["articles"]):
            raise HTTPException(409,"This article text is already in the workspace.")
        article = {"id":uuid4().hex,**body.model_dump()}
        ws["articles"].append(article)
        ws.update(analysis=None,chat=[])
        return store.save(ws)

@app.patch("/api/workspaces/{wid}/articles/{aid}")
def edit_article(wid:str,aid:str,body:ArticleInput):
    with store.LOCK:
        ws = editable(wid)
        if not any(a["id"]==aid for a in ws["articles"]):
            raise HTTPException(404,"Article not found.")
        if duplicate(body.text,ws["articles"],aid):
            raise HTTPException(409,"This article text is already in the workspace.")
        ws["articles"] = [{"id":aid,**body.model_dump()} if a["id"]==aid else a for a in ws["articles"]]
        ws.update(analysis=None,chat=[])
        return store.save(ws)

@app.delete("/api/workspaces/{wid}/articles/{aid}")
def remove_article(wid:str,aid:str):
    with store.LOCK:
        ws = editable(wid)
        ws["articles"] = [a for a in ws["articles"] if a["id"]!=aid]
        ws.update(analysis=None,chat=[])
        return store.save(ws)

def run_job(jid,wid,articles):
    def progress(stage,percent):
        job = store.job_get(jid)
        job.update(status="running",stage=stage,progress=percent)
        store.job_save(job)
    try:
        result = analyze(articles,progress,workspace(wid).get("entity_merges",[]))
        with store.LOCK:
            ws = workspace(wid)
            previous_key = (ws.get("analysis") or {}).get("retrieval", {}).get("key")
            # Re-enriching unchanged evidence should not erase the user's conversations.
            history = ws.get("chat", []) if previous_key == result["retrieval"]["key"] else []
            ws.update(analysis=result,chat=history)
            store.save(ws)
        job = store.job_get(jid)
        job.update(status="complete",stage="Comparison ready",progress=100)
        store.job_save(job)
    except Exception:
        job = store.job_get(jid)
        job.update(status="failed",error="Analysis could not complete. Check the server terminal and model setup, then retry.")
        store.job_save(job)
        import logging
        logging.exception("Analysis failed")
    finally:
        with ACTIVE_LOCK:
            ACTIVE.discard(wid)

@app.post("/api/workspaces/{wid}/analyze")
def start_analysis(wid:str,body:AnalyzeInput):
    with store.LOCK, ACTIVE_LOCK:
        ws = editable(wid)
        if not 2<=len(ws["articles"])<=5:
            raise HTTPException(422,"Add between two and five articles.")
        warnings = event_warnings(ws["articles"])
        if warnings and not body.confirm_unrelated:
            return {"needs_confirmation":True,"warnings":warnings}
        jid = uuid4().hex
        job = store.job_save({"id":jid,"workspace_id":wid,"status":"queued","stage":"Waiting to start","progress":0})
        ACTIVE.add(wid)
        EXECUTOR.submit(run_job,jid,wid,ws["articles"])
        return {"job":job,"warnings":warnings}

@app.get("/api/jobs/{jid}")
def read_job(jid:str):
    job = store.job_get(jid)
    if not job:
        raise HTTPException(404,"Job not found.")
    return job

@app.post("/api/workspaces/{wid}/entities/merge")
def merge(wid:str,body:MergeInput):
    with store.LOCK:
        ws = editable(wid)
        if not ws["analysis"]:
            raise HTTPException(409,"Analyze the articles first.")
        try:
            ws["analysis"]["entities"] = merge_entities(ws["analysis"]["entities"],body.names,body.canonical)
            ws.setdefault("entity_merges",[]).append(body.model_dump())
            ws["analysis"]["counts"]["entities"] = len(ws["analysis"]["entities"])
            return store.save(ws)
        except ValueError as exc:
            raise HTTPException(422,str(exc))

@app.post("/api/workspaces/{wid}/chat")
def chat(wid:str,body:ChatInput):
    ws = editable(wid)
    if not ws["analysis"]:
        raise HTTPException(409,"Analyze the articles before asking questions.")
    from .retrieval import answer
    result = answer(body.question,ws["analysis"],ws["articles"])
    with store.LOCK:
        current = workspace(wid)
        # Do not save a response against changed evidence.
        if not current["analysis"] or current["analysis"]["retrieval"]["key"] != ws["analysis"]["retrieval"]["key"]:
            raise HTTPException(409,"Articles changed. Ask again after analysis.")
        current["chat"].append({"id":uuid4().hex,"question":body.question,**result})
        store.save(current)
    return result

def markdown(ws):
    result = ws["analysis"]
    lines = ["# "+ws["title"],"","Perspective Prism — evidence-based coverage comparison","","> Model interpretations are not truth verification. Missing coverage does not establish deliberate omission.",""]
    if ws.get("demo"):
        lines += ["**Fictional teaching example. All publishers and reporting are invented.**",""]
    for a in ws["articles"]:
        lines += [f'## {a["publisher"]}: {a["title"]}',f'Date: {a.get("published_at") or "Not supplied"}',f'Source: {a.get("url") or a["source_type"]}',""]
    if not result:
        return "\n".join(lines+["Analysis has not been run."])
    lookup = {p["id"]:p for p in result["passages"]}
    lines += ["## Model status","",json.dumps(result["model_status"],ensure_ascii=False,indent=2),"","## Loaded-language findings",""]
    for row in result["loaded"]:
        if row["label"]=="potentially_loaded":
            p = lookup[row["passage_id"]]
            lines += [f'[{p["id"]}] {p["text"]}',f'Method: {row["method"]}; score: {row["score"] if row["score"] is not None else "unscored"}',""]
    lines += ["## Entity portrayal",""]
    for e in result["entities"]:
        lines += ["### "+e["name"]]
        for v in e["portrayals"]:
            lines += [f'- {v["article_id"]}: {v["label"]}. {v["explanation"]} Evidence: '+", ".join(v["evidence"])]
    lines += ["","## Aligned passages",""]
    for c in result["claims"]:
        lines += [f'- {c["label"]}: [{c["left"]}] ↔ [{c["right"]}]. {c["reason"]}']
    lines += ["","## Coverage",""]
    for row in result["coverage"]:
        lines += [row["topic"]]
        for aid,c in row["cells"].items():
            lines += [f'- {aid}: {"Found" if c["found"] else "Not found"} '+", ".join(c["evidence"])]
    lines += ["","## Evidence index",""]
    for p in result["passages"]:
        lines += [f'### {p["id"]}',f'Article: {p["article_id"]} · original offsets {p["start"]}–{p["end"]}',p["text"],""]
    return "\n".join(lines)

@app.get("/api/workspaces/{wid}/export")
def export(wid:str,format:str="json"):
    ws = workspace(wid)
    if format not in ("json","markdown"):
        raise HTTPException(422,"Export format must be json or markdown.")
    if format=="json":
        return Response(json.dumps(ws,ensure_ascii=False,indent=2),media_type="application/json",headers={"Content-Disposition":'attachment; filename="perspective-prism.json"'})
    return Response(markdown(ws),media_type="text/markdown",headers={"Content-Disposition":'attachment; filename="perspective-prism.md"'})

@app.get("/api/evaluation")
def evaluation():
    return {"models":models.status(),"annotation_status":"Human evaluation not completed. Templates and annotation guidelines are included in evaluation/.","limitations":["Model scores are not calibrated certainty.","Bias labels are context-dependent.","Pretrained model performance on these articles has not been established."]}

DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets",StaticFiles(directory=DIST/"assets"),name="assets")
    @app.get("/")
    def index():
        return FileResponse(DIST/"index.html")

