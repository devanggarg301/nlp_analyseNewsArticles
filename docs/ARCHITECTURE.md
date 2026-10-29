# Architecture and API

## Data flow

Article input → editable preview → validated article → SQLite workspace
→ background comparison → exact-offset passages → classifier / NER / quotation candidates
→ embedding alignment / NLI / topic coverage → persisted analysis → evidence-linked UI.

Question → balanced retrieval per selected article → adjacent context → optional LLM
→ validated citation IDs → answer or extractive/insufficient-evidence fallback.

The React frontend is built by Vite. FastAPI serves the built assets and the API on one
loopback origin. SQLite uses WAL mode. Two background workers run comparisons. Editing
during an active workspace analysis is blocked; editing articles invalidates prior
results and chat. Interrupted jobs are marked failed on server restart.

## Interfaces

Interactive OpenAPI documentation: http://127.0.0.1:8000/docs.

| Method | Endpoint | Behaviour |
|---|---|---|
| GET | /api/health | Server, model and API-configuration status; no secrets |
| GET / POST | /api/workspaces | List or create named event workspace |
| GET | /api/workspaces/{id} | Articles, stored analysis, chat |
| POST | /api/extract | Public URL extraction preview; no save |
| POST | /api/preview-upload | UTF-8 TXT/CSV preview; no save |
| POST | /api/workspaces/{id}/articles | Add validated article |
| PATCH / DELETE | /api/workspaces/{id}/articles/{article} | Edit/remove; clear old results |
| POST | /api/workspaces/{id}/analyze | Relevance advisory or background job |
| GET | /api/jobs/{id} | Progress and terminal state |
| POST | /api/workspaces/{id}/entities/merge | User alias correction; clear old judgements |
| POST | /api/workspaces/{id}/chat | Evidence-linked response for current workspace |
| GET | /api/workspaces/{id}/export?format=markdown or json | Download results |
| GET | /api/evaluation | Recorded metrics and limitations |
| POST | /api/examples/riverton | Separate fictional demo workspace |

Article fields: title, publisher, text, optional URL/date, source_type.
Passage fields: stable ID, article ID, text, original start/end offsets, sentence position.
Every interpretation references passage IDs. New article revisions invalidate analysis;
passage IDs are stable within an unchanged article revision.

## Public URL retrieval

Only HTTP/HTTPS and ports 80/443 are accepted. Credentials in URLs and non-public
resolved addresses are rejected. Each redirect is validated. The socket connects to
the already-validated IP while TLS and Host use the publisher hostname, preventing a
second DNS resolution from changing the connection target. Responses are bounded to
2 MB and HTML/text. No login, browser challenge, or paywall bypass is attempted.

## Persistence and secrets

runtime/prism.sqlite3 stores article text, analyses and chat locally.
runtime/indexes stores FAISS files keyed by content and IDs. Each source retains
publication metadata and exact original text. The application is local and has no
authentication; keep it bound to 127.0.0.1. Do not expose it publicly without adding
authentication, quotas and deployment hardening.

The optional .env is read only by Python. API keys are excluded from Git, exports
and packaging. The provider receives selected article evidence only when an
LLM-backed feature is requested or analysis is run with credentials configured.
Treat uploaded text as untrusted prompt content; prompts direct the model to ignore
embedded instructions. This mitigates, but cannot prove resistance to, prompt injection.

## Failure behaviour

URL failure → pasted-text fallback. Model module failure → warning with other outputs
retained where possible. Embedding-index failure → lexical retrieval. LLM failure →
entity mentions without inferred sentiment, or source-linked excerpts. Invalid
citation IDs are discarded; an uncited generated response is not displayed.

