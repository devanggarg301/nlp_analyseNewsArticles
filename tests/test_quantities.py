import pytest
from backend.quantities import extract_counts,numerical_pairs
from backend import models
from backend.analysis import align_claims

def pair(a,b):
    return numerical_pairs([{"id":"a:p1","article_id":"a","text":a},{"id":"b:p1","article_id":"b","text":b}])

@pytest.mark.parametrize("left,right",[
    ("Exactly five people had died by 6 p.m.","Exactly three people had died by 6 p.m."),
    ("Exactly 5 people had died by 18:00.","Exactly three people had died by 6 p.m."),
    ("Five people were killed in the cyclone.","The cyclone killed three people."),
    ("At most three people died.","Exactly five people died."),
    ("Five people were injured.","Three people were injured."),
    ("The death toll was five.","The death toll was three."),
])
def test_disjoint_counts_flagged(left,right):
    result=pair(left,right)
    assert result and result[0]["label"]=="potential_contradiction"

@pytest.mark.parametrize("left,right,label",[
    ("At least three people died.","Five people died.","supporting"),
    ("At least three people died.","At least five people died.","supporting"),
    ("More than three people died.","Five people died.","supporting"),
    ("Exactly three people had died by 12 p.m.","Exactly five people had died by 6 p.m.","unresolved"),
    ("Five people died in District A.","Three people died in District B.","unresolved"),
    ("Five people died on Monday.","Three people died on Tuesday.","unresolved"),
    ("About five people died.","Exactly three people died.","unresolved"),
    ("Three children died.","Five people died.","unresolved"),
    ("Three people died in Cyclone Alpha.","Five people died in Cyclone Beta.","unresolved"),
])
def test_bounds_and_scope(left,right,label):
    assert pair(left,right)[0]["label"]==label

def test_words_and_commas():
    assert extract_counts("One hundred and five people died.")[0]["value"]==105
    assert extract_counts("1,400 people died.")[0]["value"]==1400

def test_numeric_claims_bypass_retrieval_cutoff_and_nli(monkeypatch):
    import numpy as np
    monkeypatch.setattr(models,"ner",lambda:None)
    monkeypatch.setattr(models,"encoder",lambda:object())
    monkeypatch.setattr(models,"similarities",lambda texts:(np.eye(2),"fixture"))
    monkeypatch.setattr(models,"nli",lambda:lambda *a,**k:pytest.fail("NLI should not override the numerical comparison"))
    passages=[{"id":"a:p1","article_id":"a","text":"Exactly five people had died by 6 p.m."},{"id":"b:p1","article_id":"b","text":"Exactly three people had died by 6 p.m."}]
    result=align_claims(passages)
    assert result[0]["label"]=="potential_contradiction"
    assert result[0]["similarity"] is None

def test_questions_and_negated_counts_are_not_assertions():
    assert extract_counts("Did five people die?")==[]
    assert extract_counts("Officials denied that five people died.")==[]

