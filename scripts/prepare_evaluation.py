"""Create unlabelled evaluation queues. Never invent human annotations."""
import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.analysis import split_passages
from backend.demo import DEMO

OUT=ROOT/"evaluation"
FIELDS={
    "entity":["example_id","event_id","article_id","passage_id","text","target","annotator_a_label","annotator_b_label","adjudicated_label","speaker_a","speaker_b","adjudicated_speaker","notes"],
    "claims":["example_id","event_id","left_text","right_text","annotator_a_label","annotator_b_label","adjudicated_label","notes"],
    "rag":["example_id","event_id","question","answerable","gold_evidence_ids","generated_answer","generated_evidence_ids","annotator_a_supported","annotator_b_supported","adjudicated_supported","notes"],
}
def main():
    OUT.mkdir(exist_ok=True)
    for name,fields in FIELDS.items():
        path=OUT/(name+"_annotations.csv")
        if path.exists():
            continue
        count={"entity":100,"claims":100,"rag":30}[name]
        with path.open("w",encoding="utf-8",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=fields)
            writer.writeheader()
            writer.writerows({"example_id":f"{name}-{i+1:03d}"} for i in range(count))
        print("Created unlabelled queue:",path.name)
    # Separate tiny fictional fixtures for annotation practice, never held-out evidence.
    demo=ROOT/"data"/"demo.json"
    demo.parent.mkdir(exist_ok=True)
    demo.write_text(json.dumps(DEMO,ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":
    main()

