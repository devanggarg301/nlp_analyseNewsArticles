"""Build a portable source archive, excluding credentials, runtime data and downloaded corpora."""
from pathlib import Path
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
archive=ROOT.parent/"Perspective-Prism-source.zip"
excluded={".git",".venv","node_modules","__pycache__",".pytest_cache",".cache","runtime","roberta-checkpoints","raw","processed"}
# Neural weights stay in the ready-to-run local project; recreate them with the supplied scripts.
large_models={"embeddings","nli","ner","roberta","cache"}
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as output:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel=path.relative_to(ROOT)
        if any(part in excluded for part in rel.parts) or path.name==".env" or path.suffix==".tsbuildinfo":
            continue
        if rel.parts[0]=="models" and any(part in large_models for part in rel.parts):
            continue
        if str(rel).replace("\\","/")=="models/selected.json":
            selected=json.loads(path.read_text(encoding="utf-8"))
            if selected["kind"]=="transformer":
                metrics=json.loads((ROOT/"models/metrics.json").read_text(encoding="utf-8"))
                baselines=[m for m in metrics["models"] if m["name"].startswith("TF-IDF")]
                best=max(baselines,key=lambda m:m["validation_macro_f1"])
                selected={"name":best["name"],"kind":"sklearn","path":"logistic.joblib" if "Logistic" in best["name"] else "svm.joblib","validation_macro_f1":best["validation_macro_f1"],"package_note":"Portable source uses the best baseline. Large fine-tuned neural weights remain in the original local project."}
                output.writestr(str(Path("perspective-prism")/rel),json.dumps(selected,indent=2))
                continue
        output.write(path,Path("perspective-prism")/rel)
print(archive)

