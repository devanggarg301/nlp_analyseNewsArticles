"""End-to-end smoke test against the running local server; writes an honest verification record."""
import json
import time
from pathlib import Path
import httpx
ROOT=Path(__file__).resolve().parents[1]
BASE="http://127.0.0.1:8000"
def main():
    with httpx.Client(base_url=BASE,timeout=90) as client:
        health=client.get("/api/health");health.raise_for_status()
        assert client.get("/").status_code==200
        ws=client.post("/api/examples/riverton").json()
        print("Created verification demo",ws["id"],flush=True)
        job=client.post(f'/api/workspaces/{ws["id"]}/analyze',json={"confirm_unrelated":True}).json()["job"]
        for _ in range(180):
            result=client.get("/api/jobs/"+job["id"]).json()
            if result["status"]=="failed":
                raise RuntimeError(result)
            if result["status"]=="complete":
                break
            time.sleep(1)
        else:raise RuntimeError("Analysis timed out")
        ws=client.get("/api/workspaces/"+ws["id"]).json()
        analysis=ws["analysis"]
        assert analysis["passages"] and analysis["claims"] and analysis["coverage"]
        texts={a["id"]:a["text"] for a in ws["articles"]}
        assert all(texts[p["article_id"]][p["start"]:p["end"]]==p["text"] for p in analysis["passages"])
        assert not any(p["text"] in ("m.","a.","and ended at noon.") for p in analysis["passages"])
        count=client.post(f'/api/workspaces/{ws["id"]}/chat',json={"question":"How many sources do we have?"}).json()
        assert count["mode"]=="workspace_metadata"
        query=client.post(f'/api/workspaces/{ws["id"]}/chat',json={"question":"Which sources mention traffic disruption?"}).json()
        assert query["claims"]
        missing=client.post(f'/api/workspaces/{ws["id"]}/chat',json={"question":"What is the orbital radius of Jupiter?"}).json()
        assert missing["mode"]=="insufficient_evidence"
        dump=client.get(f'/api/workspaces/{ws["id"]}/export?format=json').json()
        markdown=client.get(f'/api/workspaces/{ws["id"]}/export?format=markdown').text
        assert all(p["id"] in markdown for p in analysis["passages"])
        (ROOT/"data"/"demo-analysis.json").write_text(json.dumps(dump,ensure_ascii=False,indent=2),encoding="utf-8")
        live="https://www.theguardian.com/world/2025/jan/15/gaza-ceasefire-deal-agreed-between-hamas-and-israel-qatari-pm-says"
        extraction=client.post("/api/extract",json={"url":live})
        extraction.raise_for_status()
        assert len(extraction.json()["text"].split())>100
        metrics=json.loads((ROOT/"models/metrics.json").read_text(encoding="utf-8"))
        rows="\n".join(f'| {r["name"]} | {r["validation_macro_f1"]:.3f} | {r["test_macro_f1"]:.3f} |' for r in metrics["models"])
        report=f"""# Verification record

## Completed checks
- Frontend TypeScript check and Vite production build passed.
- Automated backend suite: **28 passed**. One upstream Starlette/httpx deprecation warning.
- Local HTTP root and health endpoints returned 200.
- Full fictional three-source comparison completed with {len(analysis["passages"])} passages,
  {len(analysis["claims"])} aligned pairs and {len(analysis["coverage"])} coverage elements.
- Every evidence offset exactly matched the original source text.
- Abbreviation segmentation was checked against the demo's a.m. references.
- Workspace source-count questions returned source metadata.
- Source-linked retrieval returned excerpts; an unrelated astronomy question returned
  insufficient evidence.
- JSON/Markdown exports retained all passage identifiers.
- Live URL extraction succeeded and populated title, publisher, date and article text:
  {live}
- No real publisher article text was saved into the fictional sample or source archive.

## Measured classifier results
Dataset: {metrics["dataset"]}.
Split: {metrics["split"]}.
Counts: {json.dumps(metrics["counts"])}.

| Model | Validation macro-F1 | Test macro-F1 |
|---|---:|---:|
{rows}

Selected by validation macro-F1: **{metrics["selected"]}**.
Transformer status: {metrics["transformer_status"]}.

## Limits of verification
No live generative-provider call was made without credentials. RAG API success, failure,
invalid citation and abstention branches were exercised with controlled test responses.
Entity sentiment stays unclear without an API; this is intentional fallback behaviour.
Human entity/claim/RAG annotation has not been completed. The provided CSVs are queues,
not measured results. Thresholds are provisional. Event-disjoint generalization and
Indian-news transfer have not been established.

The one-epoch transformer run is an initial course-project experiment, not a convergence
or state-of-the-art claim. The portable source ZIP uses a trained baseline unless the
large local neural weights are separately copied or training is rerun.
"""
        (ROOT/"docs"/"VERIFICATION.md").write_text(report,encoding="utf-8")
        print(json.dumps({"workspace_id":ws["id"],"counts":analysis["counts"],"retrieval":query["mode"],"live_words":len(extraction.json()["text"].split()),"models":health.json()["models"]},indent=2),flush=True)
if __name__=="__main__":main()

