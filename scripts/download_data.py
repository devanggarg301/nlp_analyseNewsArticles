"""Download BABE from its public research repository; preserve provenance."""
import json
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"data"/"raw"

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    base = "https://raw.githubusercontent.com/Media-Bias-Group/MBIB/main/datasets/010_BABE/"
    for remote,local in [("raw/final_labels_SG2.xlsx","babe.xlsx"),("README.md","BABE-README.md")]:
        response = httpx.get(base+remote,follow_redirects=True,timeout=120)
        response.raise_for_status()
        (DEST/local).write_bytes(response.content)
        print("Saved",local,flush=True)
    (DEST/"provenance.json").write_text(json.dumps({"source":base,"paper":"https://aclanthology.org/2021.findings-emnlp.101/","note":"Review source dataset terms before redistribution. Downloaded research data are not bundled in the source archive."},indent=2),encoding="utf-8")

if __name__=="__main__":
    main()

