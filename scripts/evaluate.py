"""Score adjudicated entity/claim labels and evidence retrieval."""
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("csv",type=Path)
    parser.add_argument("--prediction-column",default="prediction")
    parser.add_argument("--gold-column",default="adjudicated_label")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    data=pd.read_csv(args.csv)
    if args.prediction_column not in data or args.gold_column not in data:
        raise SystemExit("Add adjudicated gold labels and a prediction column before scoring.")
    rows=data.dropna(subset=[args.prediction_column,args.gold_column])
    if rows.empty:
        raise SystemExit("No adjudicated predictions available. Human evaluation has not been completed.")
    result={"examples":len(rows),"classification":classification_report(rows[args.gold_column],rows[args.prediction_column],output_dict=True,zero_division=0)}
    if "annotator_a_label" in data and "annotator_b_label" in data:
        both=data.dropna(subset=["annotator_a_label","annotator_b_label"])
        result["annotator_agreement_kappa"]=float(cohen_kappa_score(both["annotator_a_label"],both["annotator_b_label"])) if len(both) else None
    print(json.dumps(result,indent=2))
    if args.output:
        args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()

