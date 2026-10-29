import json
import os
import time
import httpx

def configured():
    return bool(os.getenv("LLM_API_KEY") and os.getenv("LLM_MODEL"))

def error_summary(error):
    if isinstance(error, httpx.TimeoutException):
        return "The API request timed out. Try again."
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return {401:"The API rejected authentication.",403:"The API denied access.",404:"The configured model or endpoint was not found.",429:"The API rate or quota limit was reached."}.get(status, f"The API returned HTTP {status}.")
    if isinstance(error, (ValueError,TypeError,AttributeError)):
        return "The API returned an unexpected response format. Try again."
    return "The API request could not complete. Try again."

def complete_json(system, payload):
    if not configured():
        raise RuntimeError("No LLM configured")
    base = os.getenv("LLM_BASE_URL","https://api.openai.com/v1").rstrip("/")
    # Credentials stay on the server. Never return upstream response bodies/errors.
    payload_body = {
        "model":os.environ["LLM_MODEL"],
        "temperature":0.1,
        "response_format":{"type":"json_object"},
        "messages":[{"role":"system","content":system+" Return one JSON object at the root, with the requested named fields; do not return a bare array."},{"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
    }
    for attempt in range(3):
        response = httpx.post(base + "/chat/completions", headers={"Authorization":"Bearer " + os.environ["LLM_API_KEY"]}, json=payload_body, timeout=60)
        if response.status_code not in (500,502,503,504) or attempt == 2:
            break
        time.sleep(2 ** attempt)
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    return json.loads(raw)

