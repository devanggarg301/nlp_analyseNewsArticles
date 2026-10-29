"""Publication dates from explicit source labels, never file creation/event dates."""
import json
import re
from datetime import date

MONTHS = {name: index for index, name in enumerate((
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"), 1)}
MONTHS.update({name[:3]: number for name, number in list(MONTHS.items())})
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
DAY = r"\d{1,2}(?:st|nd|rd|th)?"
DATE = rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|{MONTH}\s+{DAY},?\s+\d{{4}}|{DAY}\s+{MONTH},?\s+\d{{4}})"


def calendar_date(value):
    """Normalize an explicit calendar date to the date input's ISO format."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:$|[T\s])", value)
    try:
        if iso:
            return date(*map(int, iso.groups())).isoformat()
        match = re.match(rf"^({MONTH})\s+({DAY}),?\s+(\d{{4}})\b", value, re.I)
        if match:
            month, day, year = match.groups()
        else:
            match = re.match(rf"^({DAY})\s+({MONTH}),?\s+(\d{{4}})\b", value, re.I)
            if not match:
                return None
            day, month, year = match.groups()
        return date(int(year), MONTHS[month.lower()], int(re.sub(r"\D", "", day))).isoformat()
    except ValueError:
        return None


def publication_from_text(text):
    # Only publication labels near the article header. Updated dates and dates
    # mentioned in a story are deliberately not substituted for publication.
    pattern = rf"\b(?:Published(?:\s*On)?|Publication\s+date)\s*[:\-]?\s*({DATE})"
    for match in re.finditer(pattern, text[:5000], re.I):
        prefix = text[max(0, match.start()-20):match.start()]
        if re.search(r"\b(?:was|were|been|originally)\s*$", prefix, re.I):
            continue
        result = calendar_date(match.group(1))
        if result:
            return result
    return None


def publication_from_html(soup):
    for name in ("article:published_time", "datePublished", "pubdate",
                 "parsely-pub-date", "DC.date.issued", "dcterms.issued", "date"):
        for tag in soup.find_all("meta"):
            if (tag.get("property") or tag.get("name") or "").lower() == name.lower():
                result = calendar_date(tag.get("content"))
                if result:
                    return result

    def structured(node):
        if isinstance(node, list):
            for child in node:
                result = structured(child)
                if result:
                    return result
        elif isinstance(node, dict):
            types = node.get("@type", [])
            types = types if isinstance(types, list) else [types]
            if any(isinstance(t, str) and t.rsplit("/", 1)[-1].lower() in
                   {"article", "newsarticle", "reportagenewsarticle", "blogposting"} for t in types):
                result = calendar_date(node.get("datePublished"))
                if result:
                    return result
            for child in node.values():
                if isinstance(child, (dict, list)):
                    result = structured(child)
                    if result:
                        return result
        return None

    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            result = structured(json.loads(tag.get_text()))
            if result:
                return result
        except (ValueError, TypeError, RecursionError):
            continue
    for tag in soup.select('time[itemprop="datePublished"], time.published, time[pubdate]'):
        result = calendar_date(tag.get("datetime") or tag.get_text(" ", strip=True))
        if result:
            return result
    return publication_from_text(soup.get_text(" ", strip=True))
