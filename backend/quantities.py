"""Conservative comparison of count claims; independent of embeddings and NLI."""
import re
import math

WORDS={"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
NUM=r"(?:\d[\d,]*|(?:"+"|".join(WORDS)+r")(?:[ -](?:"+"|".join(WORDS)+r"|hundred|thousand|million|and))*)"
QUAL=r"(?:exactly|at least|at most|more than|fewer than|less than|no more than|no fewer than|up to|over|under|approximately|about|around|nearly)"
PERSON=r"(?:people|persons|residents|civilians|women|men|children|workers|soldiers)"
METRIC=r"(?:died|killed|dead|deaths|fatalities|injured|injuries)"
PATTERNS=[
    re.compile(r"\b(?P<qual>"+QUAL+r")?\s*(?P<num>"+NUM+r")\s+(?:(?P<population>"+PERSON+r")\s+)?(?:(?:had|have|has|been|were|was|are|is|reportedly|confirmed|found)\s+){0,4}(?P<metric>"+METRIC+r")\b",re.I),
    re.compile(r"\b(?P<metric>killing|killed|injuring)\s+(?P<qual>"+QUAL+r")?\s*(?P<num>"+NUM+r")\s+(?P<population>"+PERSON+r")\b",re.I),
    re.compile(r"\b(?P<metric>death toll|fatalities|injuries)(?:\s+(?:was|is|of|stood at|rose to|reached))?\s+(?P<qual>"+QUAL+r")?\s*(?P<num>"+NUM+r")\b",re.I),
]

def number(text):
    if re.fullmatch(r"\d[\d,]*",text):return int(text.replace(",",""))
    total=current=0
    for token in re.split(r"[ -]+",text.lower()):
        if token in WORDS:current+=WORDS[token]
        elif token=="hundred":current=max(current,1)*100
        elif token in ("thousand","million"):
            total+=max(current,1)*(1000 if token=="thousand" else 1000000);current=0
    return total+current

def times(text):
    values=set()
    pattern=r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?"
    for m in re.finditer(pattern,text,re.I):
        hour=int(m[1])%12+(12 if m[3].lower()=="p" else 0)
        values.add(f"{hour:02d}:{int(m[2] or 0):02d}")
    for m in re.finditer(r"\b([01]?\d|2[0-3]):([0-5]\d)\b",text):
        # Do not also interpret a 12-hour time as an unqualified 24-hour time.
        if not re.match(r"\s*[ap]\.?\s*m",text[m.end():],re.I):
            values.add(f"{int(m[1]):02d}:{m[2]}")
    return values

def extract_counts(text):
    result=[]
    for pattern in PATTERNS:
        for m in pattern.finditer(text):
            # 'No ... died', a hypothetical, or a question is not an asserted toll.
            prefix=text[max(0,m.start()-30):m.start()].lower().rstrip()
            if re.search(r"\b(?:not|denied|if|whether)(?:\s+that)?$",prefix) or text.strip().endswith("?"):
                continue
            value=number(m["num"].lower())
            qualifier=(m["qual"] or "").lower()
            population=(m.groupdict().get("population") or "people").lower()
            if population in ("persons","residents","civilians"):population="people"
            metric="injuries" if m["metric"].lower() in ("injured","injuring","injuries") else "deaths"
            low=high=value
            approximate=qualifier in ("approximately","about","around","nearly")
            if qualifier in ("at least","no fewer than"):high=math.inf
            elif qualifier in ("more than","over"):low,high=value+1,math.inf
            elif qualifier in ("at most","no more than","up to"):low=0
            elif qualifier in ("fewer than","less than","under"):low,high=0,max(0,value-1)
            if not any(r["start"]<=m.start()<r["end"] for r in result):
                result.append({"metric":metric,"value":value,"qualifier":qualifier or "reported count","population":population,"low":low,"high":high,"approximate":approximate,"start":m.start(),"end":m.end()})
    return result

def context(text,ner=None):
    days=set(re.findall(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|yesterday|today|tomorrow)\b",text.lower()))
    dates=set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b",text.lower()))
    places=set()
    if ner:
        places={e.text.casefold() for e in ner(text).ents if e.label_ in ("GPE","LOC")}
    places.update(m.casefold() for m in re.findall(r"\b[Ii]n\s+((?:(?:District|State|City|Province)\s+)?[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)",text))
    # Named storms prevent comparing two explicitly different disasters.
    events=set(re.findall(r"\b(?:cyclone|hurricane|storm)\s+([A-Z][a-z]+)\b",text))
    return {"times":times(text),"days":days,"dates":dates,"places":places,"events":events}

def compare_counts(left,right,left_context,right_context):
    if left["metric"]!=right["metric"]:
        return None
    if left["population"]!=right["population"]:
        return {"label":"unresolved","reason":"These counts describe different populations or subgroups; they cannot be treated as conflicting totals.","context_warning":True}
    a,b=left_context,right_context
    for field in ("events","places"):
        if a[field] and b[field] and not a[field].intersection(b[field]):
            return {"label":"unresolved","reason":f"Different reported {left['metric']} counts, but explicit {field} differ. These may describe different scopes; not a confirmed contradiction.","context_warning":True}
    for field in ("times","days","dates"):
        if a[field]!=b[field] and (a[field] or b[field]):
            return {"label":"unresolved","reason":f"Reported {left['metric']} counts differ in time context or one has no as-of time. Check whether this is an update before calling it a contradiction.","context_warning":True}
    if left["approximate"] or right["approximate"]:
        return {"label":"unresolved","reason":"At least one count is approximate; the difference needs review rather than an exact-count contradiction.","context_warning":True}
    disjoint=max(left["low"],right["low"])>min(left["high"],right["high"])
    if disjoint:
        return {"label":"potential_contradiction","reason":f"Non-overlapping {left['metric']} counts: {left['qualifier']} {left['value']} versus {right['qualifier']} {right['value']}. On the supplied time/place context they cannot both describe the same total. Verify that the event and counting scope are the same.","context_warning":False}
    return {"label":"supporting","reason":f"The numerical claims are compatible: {left['qualifier']} {left['value']} and {right['qualifier']} {right['value']}. Bounds overlap; this does not independently verify either report.","context_warning":False}

def numerical_pairs(passages,ner=None):
    candidates=[]
    rows=[]
    for p in passages:
        claims=extract_counts(p["text"])
        if claims:rows.append((p,claims,context(p["text"],ner)))
    for i,(p,left,pc) in enumerate(rows):
        for q,right,qc in rows[i+1:]:
            if p["article_id"]==q["article_id"]:continue
            # Avoid equating an entire article's several counts to one another.
            if len(left)!=1 or len(right)!=1:continue
            result=compare_counts(left[0],right[0],pc,qc)
            if result is None:continue
            candidates.append({"left":p["id"],"right":q["id"],**result,"method":"Numerical count and scope comparison","directional":None,"similarity":None})
    order={"potential_contradiction":0,"unresolved":1,"supporting":2}
    candidates.sort(key=lambda r:order[r["label"]])
    return candidates[:48]

