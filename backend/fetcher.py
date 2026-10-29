"""Public HTTP(S) extraction with validated redirects and DNS-pinned connections."""
import http.client
import ipaddress
import socket
import ssl
from urllib.parse import urlsplit, urljoin
from bs4 import BeautifulSoup
import trafilatura

MAX_BYTES = 2_000_000

def public_target(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a public http:// or https:// article URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in (80,443):
        raise ValueError("Only standard web ports 80 and 443 are supported.")
    try:
        addresses = sorted({r[4][0] for r in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)})
    except socket.gaierror as exc:
        raise ValueError("The article hostname could not be resolved. Check the URL or paste the article text.") from exc
    if not addresses or any(not ipaddress.ip_address(a).is_global for a in addresses):
        raise ValueError("Only public internet addresses are allowed.")
    return parsed, port, addresses[0]

def request_page(url):
    parsed, port, address = public_target(url)
    # Connect to the validated IP, retaining original hostname for TLS/SNI and Host.
    conn = http.client.HTTPConnection(parsed.hostname, port, timeout=15)
    raw = socket.create_connection((address, port), timeout=15)
    if parsed.scheme == "https":
        try:
            raw = ssl.create_default_context().wrap_socket(raw, server_hostname=parsed.hostname)
        except Exception:
            raw.close()
            raise
    conn.sock = raw
    try:
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        conn.request("GET", path, headers={"Host": parsed.netloc, "User-Agent": "PerspectivePrism/1.0 (academic article analysis)", "Accept": "text/html,text/plain", "Accept-Encoding": "identity"})
        response = conn.getresponse()
        headers = dict((k.lower(),v) for k,v in response.getheaders())
        if response.status in (301,302,303,307,308):
            return response.status, headers, b""
        if response.status != 200:
            raise ValueError(f"Publisher returned HTTP {response.status}. Paste or upload the article instead.")
        if not any(t in headers.get("content-type","") for t in ("text/html","text/plain","application/xhtml")):
            raise ValueError("This URL is not an HTML or text article.")
        if int(headers.get("content-length","0")) > MAX_BYTES:
            raise ValueError("Article page is too large (2 MB limit).")
        body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("Article page is too large (2 MB limit).")
        return response.status, headers, body
    finally:
        conn.close()

def extract_url(url):
    current = url.strip()
    for _ in range(5):
        status, headers, body = request_page(current)
        if status in (301,302,303,307,308):
            if not headers.get("location"):
                raise ValueError("Publisher returned an invalid redirect.")
            current = urljoin(current, headers["location"])
            continue
        html = body.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
        if len(text.split()) < 40:
            raise ValueError("Could not extract sufficient article text. The page may require login or JavaScript. Paste its text instead.")
        def meta(*names):
            for name in names:
                element = soup.find("meta", attrs={"property":name}) or soup.find("meta", attrs={"name":name})
                if element and element.get("content"):
                    return element["content"]
            return None
        title = meta("og:title", "twitter:title") or (soup.title.get_text() if soup.title else "Untitled article")
        return {"title":title.strip(), "publisher":meta("og:site_name") or urlsplit(current).hostname, "published_at":meta("article:published_time","datePublished","date"), "url":current, "text":text, "source_type":"url", "warnings":["Review extracted text before adding it."] + (["Article exceeds 3,000 words. Shorten it in the preview."] if len(text.split())>3000 else [])}
    raise ValueError("Too many redirects. Paste the article text instead.")

