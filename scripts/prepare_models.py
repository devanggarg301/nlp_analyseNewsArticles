"""Download optional public pretrained components. Run explicitly, never from an API request."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.config import MODELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--components",nargs="+",default=["embeddings","nli","ner"],choices=["embeddings","nli","ner"])
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    for component in args.components:
        print("Preparing",component,flush=True)
        if component=="embeddings":
            snapshot_download("sentence-transformers/all-MiniLM-L6-v2",local_dir=MODELS/"embeddings",ignore_patterns=["onnx/*","openvino/*","*.h5","rust_model.ot","pytorch_model.bin"])
        elif component=="nli":
            snapshot_download("cross-encoder/nli-distilroberta-base",local_dir=MODELS/"nli",ignore_patterns=["onnx/*","openvino/*","*.h5","rust_model.ot"])
        elif component=="ner":
            import httpx
            import spacy
            import tempfile
            import zipfile
            url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
            data = httpx.get(url,follow_redirects=True,timeout=120)
            data.raise_for_status()
            # The public wheel is a ZIP; copy only the known model directory.
            import io
            with zipfile.ZipFile(io.BytesIO(data.content)) as archive:
                prefix = "en_core_web_sm/en_core_web_sm-3.8.0/"
                for member in archive.namelist():
                    if member.startswith(prefix) and not member.endswith("/"):
                        relative = Path(member[len(prefix):])
                        if ".." in relative.parts:
                            continue
                        target = MODELS/"ner"/relative
                        target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes(archive.read(member))
        print("Ready:",component,flush=True)

if __name__=="__main__":
    main()

