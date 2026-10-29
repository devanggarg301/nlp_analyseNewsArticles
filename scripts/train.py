"""Reproducible BABE experiments. Select on validation macro-F1, never test scores."""
import argparse
import json
import random
import re
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.config import MODELS

def normalize(text):
    return " ".join(re.findall(r"\w+",str(text).lower()))

def prepare(path):
    raw = pd.read_excel(path) if path.suffix==".xlsx" else pd.read_csv(path)
    columns = {str(c).lower().strip():c for c in raw.columns}
    text_col = columns.get("text",columns.get("sentence"))
    label_col = columns.get("label_bias",columns.get("label"))
    if text_col is None or label_col is None:
        raise ValueError("Dataset requires text/sentence and label_bias/label columns. Found: "+str(list(raw.columns)))
    df = raw.dropna(subset=[text_col,label_col]).copy()
    df["text"] = df[text_col].astype(str)
    mapping = {"biased":1,"non-biased":0,"non biased":0,"0":0,"1":1}
    df["label"] = df[label_col].astype(str).str.lower().str.strip().map(mapping)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["norm"] = df["text"].map(normalize)
    conflicts = df.groupby("norm")["label"].nunique()
    df = df[~df["norm"].isin(conflicts[conflicts>1].index)].drop_duplicates("norm").reset_index(drop=True)
    if len(df)<100:
        raise ValueError("At least 100 valid labelled examples are required.")
    # Union source-document groups and near-duplicate text clusters before splitting.
    parents = list(range(len(df)))
    def find(i):
        while parents[i]!=i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i,j):
        parents[find(j)] = find(i)
    group_col = next((columns[k] for k in ("event_id","article_id","news_link","url") if k in columns),None)
    if group_col:
        for _,rows in df.groupby(group_col):
            ids = list(rows.index)
            for i in ids[1:]:
                union(ids[0],i)
    vector = TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2)
    features = vector.fit_transform(df["norm"])
    similarities = (features @ features.T).tocoo()
    for i,j,score in zip(similarities.row,similarities.col,similarities.data):
        if i<j and score>=0.92:
            union(int(i),int(j))
    groups = np.array([find(i) for i in range(len(df))])
    splitter = GroupShuffleSplit(n_splits=1,test_size=.15,random_state=42)
    trainval,test = next(splitter.split(df,df["label"],groups))
    splitter2 = GroupShuffleSplit(n_splits=1,test_size=.1764705882,random_state=42)
    tr,va = next(splitter2.split(df.iloc[trainval],df.iloc[trainval]["label"],groups[trainval]))
    train, val = trainval[tr],trainval[va]
    for ids in (train,val,test):
        if df.iloc[ids]["label"].nunique()!=2:
            raise ValueError("Split lacks both labels; inspect dataset grouping.")
    assert not(set(groups[train]) & set(groups[val]) or set(groups[train]) & set(groups[test]) or set(groups[val]) & set(groups[test]))
    split = "Source-document and near-duplicate group split; event identifiers unavailable" if group_col else "Near-duplicate group split; document/event identifiers unavailable"
    if group_col is not None and group_col == columns.get("event_id"):
        split = "Event and near-duplicate group split"
    return df, train,val,test, split

def row_metrics(name,y_val,p_val,y_test,p_test):
    return {"name":name,"validation_macro_f1":float(f1_score(y_val,p_val,average="macro")),"test_macro_f1":float(f1_score(y_test,p_test,average="macro")),"test_precision":float(precision_score(y_test,p_test,average="macro",zero_division=0)),"test_recall":float(recall_score(y_test,p_test,average="macro",zero_division=0)),"confusion_matrix":confusion_matrix(y_test,p_test,labels=[0,1]).tolist()}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=ROOT/"data/raw/babe.xlsx")
    parser.add_argument("--transformer",action="store_true")
    parser.add_argument("--epochs",type=int,default=3)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--max-length",type=int,default=192)
    args=parser.parse_args()
    random.seed(42);np.random.seed(42)
    df,train,val,test,split = prepare(args.data)
    MODELS.mkdir(exist_ok=True)
    split_dir=ROOT/"data"/"processed"
    split_dir.mkdir(parents=True,exist_ok=True)
    for name,ids in [("train",train),("validation",val),("test",test)]:
        # Do not include raw dataset in the distribution archive.
        df.iloc[ids][["text","label"]].to_csv(split_dir/(name+".csv"),index=False)
    reports=[];candidates=[]
    for name,estimator,filename in [("TF-IDF + Logistic Regression",LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42),"logistic.joblib"),("TF-IDF + Linear SVM",LinearSVC(class_weight="balanced",random_state=42),"svm.joblib")]:
        pipeline=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=50000)),("classifier",estimator)])
        pipeline.fit(df.iloc[train]["text"],df.iloc[train]["label"])
        pv=pipeline.predict(df.iloc[val]["text"])
        # Selection score is computed before test evaluation.
        val_score=float(f1_score(df.iloc[val]["label"],pv,average="macro"))
        joblib.dump(pipeline,MODELS/filename)
        pt=pipeline.predict(df.iloc[test]["text"])
        reports.append(row_metrics(name,df.iloc[val]["label"],pv,df.iloc[test]["label"],pt))
        candidates.append({"name":name,"kind":"sklearn","path":filename,"validation_macro_f1":val_score})
        print(name, json.dumps(reports[-1]),flush=True)
    report={"dataset":"BABE SG2 expert sentence labels","seed":42,"split":split,"counts":{"train":len(train),"validation":len(val),"test":len(test)},"models":reports,"selected":max(candidates,key=lambda c:c["validation_macro_f1"])["name"],"transformer_status":"Not trained in this run","training_parameters":{"roberta_epochs":args.epochs if args.transformer else 0,"max_length":args.max_length,"batch_size":args.batch_size,"gradient_accumulation_steps":4,"learning_rate":2e-5,"near_duplicate_cosine":0.92}}
    def save_report():
        selected=max(candidates,key=lambda c:c["validation_macro_f1"])
        report["selected"]=selected["name"]
        (MODELS/"selected.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
        (MODELS/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    save_report()
    if args.transformer:
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding,set_seed
        torch.set_num_threads(4)
        set_seed(42)
        tokenizer=AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        model=AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base",num_labels=2,id2label={0:"non-biased",1:"biased"},label2id={"non-biased":0,"biased":1})
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, ids):
                self.rows=df.iloc[ids].reset_index(drop=True)
            def __len__(self):return len(self.rows)
            def __getitem__(self,index):
                item=tokenizer(self.rows.iloc[index]["text"],truncation=True,max_length=args.max_length)
                item["labels"]=int(self.rows.iloc[index]["label"])
                return item
        training_args=TrainingArguments(output_dir=str(MODELS/"roberta-checkpoints"),num_train_epochs=args.epochs,per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=args.batch_size*2,gradient_accumulation_steps=4,learning_rate=2e-5,weight_decay=.01,eval_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="macro_f1",save_total_limit=1,fp16=torch.cuda.is_available(),report_to="none",seed=42,dataloader_pin_memory=torch.cuda.is_available())
        trainer=Trainer(model=model,args=training_args,train_dataset=TextDataset(train),eval_dataset=TextDataset(val),data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=lambda result:{"macro_f1":f1_score(result.label_ids,np.argmax(result.predictions,axis=-1),average="macro")})
        print("Training RoBERTa on", "CUDA" if torch.cuda.is_available() else "CPU",flush=True)
        trainer.train()
        pv=np.argmax(trainer.predict(TextDataset(val)).predictions,axis=-1)
        score=float(f1_score(df.iloc[val]["label"],pv,average="macro"))
        trainer.save_model(str(MODELS/"roberta"));tokenizer.save_pretrained(MODELS/"roberta")
        pt=np.argmax(trainer.predict(TextDataset(test)).predictions,axis=-1)
        reports.append(row_metrics("Fine-tuned RoBERTa-base",df.iloc[val]["label"],pv,df.iloc[test]["label"],pt))
        candidates.append({"name":"Fine-tuned RoBERTa-base","kind":"transformer","path":"roberta","validation_macro_f1":score})
        report["transformer_status"]=f"Fine-tuned for {args.epochs} epochs; best validation checkpoint selected"
        save_report()
    print("Selected:",report["selected"],flush=True)

if __name__=="__main__":
    main()

