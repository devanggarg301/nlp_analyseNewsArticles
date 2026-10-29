# Perspective Prism

A local full-stack NLP application for comparing how publishers cover the same event.

**Start here:** run `Start-Prism.ps1`, then open http://127.0.0.1:8000.
The prepared project on this machine includes dependencies, frontend assets, trained
baseline classifiers and downloaded semantic models. The portable ZIP excludes large
neural weights, runtime workspaces and research corpora; see setup below.

The companion **Perspective-Prism-trained-RoBERTa.zip** contains the actual fine-tuned
checkpoint and its runtime selection manifest. Extract it into the same parent folder
as the source ZIP, merging the perspective-prism directory, to restore this experiment
without retraining. The original prepared project already contains these weights.

## What works

- Public live article URLs, editable extraction preview, pasted text, PDF, Word (.docx), TXT and CSV uploads.
- SQLite workspaces, duplicate checks, two-to-five-article comparisons, background jobs.
- Trained BABE sentence classifier with visible scores and baseline comparisons.
- Local spaCy NER, MiniLM sentence embeddings, FAISS retrieval and pretrained NLI.
- Side-by-side readers; exact character-offset evidence; entity alias correction.
- Attributed quotation candidates, corpus topics, coverage and aligned claims.
- Basic API-backed RAG and structured entity portrayal when configured.
- Extractive source-balanced retrieval when an API is absent or unavailable.
- Markdown and JSON exports, with evidence and model provenance.
- Fictional demonstration articles, reproducible scripts and automated tests.

The RoBERTa experiment and its status are recorded in `models/metrics.json`.
The application never displays invented benchmark metrics or filled-in human labels.

## Run on this machine

In PowerShell, from this folder:

~~~powershell
.\Start-Prism.ps1
~~~

This starts a hidden local server and opens the app. Logs are in `runtime/`.
Stop the launcher-owned server with `Stop-Prism.ps1`.

For a visible development server:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
~~~

For frontend hot reload, use a second terminal:

~~~powershell
cd frontend
npm.cmd run dev
~~~

The Vite dev server is http://127.0.0.1:5173 and proxies /api to port 8000.

## Setup on another machine

Requirements: Python 3.11+ (tested here on 3.13), Node.js 20.19+ or 22.12+,
internet for dependencies/model downloads. The bundled source ZIP includes the built
frontend and trained baseline files. A GPU is optional; inference runs on CPU.

~~~powershell
.\Setup-Prism.ps1 -WithModels
.\Start-Prism.ps1
~~~

Minimal mode: omit `-WithModels`. This preserves the trained baseline detector but uses
transparent lexical/name heuristics for unavailable neural components.

On macOS/Linux:

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements-models.txt
cd frontend
npm ci
npm run build
cd ..
.venv/bin/python scripts/prepare_models.py
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
~~~

## Enable generated answers and entity sentiment

Copy `.env.example` to `.env`, enter your provider's API key, model identifier, and
compatible chat-completions base URL. Restart the backend. Keys stay in the backend.
The provider must support JSON-object responses.

Without credentials, every entity judgement remains **unclear** rather than invented.
Ask Prism returns retrieved excerpts and labels them **extractive mode**.
With credentials it generates cited answers and context-sensitive target judgements.
API failures preserve the comparison and fall back to excerpts. No live provider call
can be verified without valid credentials.

## A five-minute demonstration

1. Open **New comparison → Load fictional sample**, or the initial sample button.
2. Click **Compare coverage**. First-run neural loading can take longer.
3. Inspect the three readers and click a highlighted sentence.
4. Open **Claim comparison** and inspect both passages and their attribution.
5. Open **Coverage** to compare topics and quoted voices.
6. Ask **What do the articles report about bus fares?**
7. Click a citation to inspect the exact original evidence.
8. Export Markdown or JSON.
9. Open **Model evaluation** for measured results.

For live news, create a separate event workspace, add two-to-five URLs, inspect the
extracted text, and compare. Do not mix the fictional demo with unrelated real stories.
Some publishers block automated requests or require login; use pasted text instead.
A successfully tested live URL is listed in `docs/VERIFICATION.md`.

CSV columns: `title,publisher,text`; optional `url,published_at`.
Upload one article per PDF or DOCX (up to 10 MB, PDFs up to 30 pages), review extracted text and enter the publisher. Scanned PDFs need OCR first; older .doc files must be saved as .docx. Each article is capped at 3,000 words. TXT and CSV use UTF-8.

## Train and reproduce experiments

~~~powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/train.py
# Full initial experiment including a transformer:
.\.venv\Scripts\python.exe scripts/train.py --transformer --epochs 3
~~~

For today's initial run, the transformer can be trained with `--epochs 1 --max-length 128`.
This is an initial experiment, not evidence of convergence. Default training uses three
epochs and a validation-selected checkpoint. CUDA is used only if installed PyTorch
supports it; the presence of an NVIDIA GPU alone is insufficient.

The script groups all sentences sharing a source article (`news_link`) and unions
near-duplicates before a seeded group split. BABE does not provide reliable event IDs,
so **event-level separation is not guaranteed**. Ambiguous labels and conflicting exact
duplicates are dropped. Split IDs/text, counts, confusion matrices, precision, recall,
and macro-F1 are saved. Selection uses validation macro-F1, not test performance.
Do not repeatedly tune against the reported test scores.

Restart the backend after training so cached models refresh.

## NLP implementation and limitations

- **Biased-language detection:** TF-IDF Logistic Regression / Linear SVM and optional
  RoBERTa fine-tuning. The selected runtime model is in `models/selected.json`.
- **Rationales:** exact-substring lexicon or LLM suggestions. They are not trained token
  predictions or faithful causal explanations of the sentence classifier.
  The interface separates model flags from independent phrase cues. A sentence can
  contain a cue even when its classifier score is below the decision threshold.
- **NER:** local spaCy. Unique surname aliases are merged conservatively; ambiguous
  pronouns remain unresolved. Users can merge groups, clearing old target judgements.
- **Quotation attribution:** deterministic candidates; not a trained attribution model.
- **Portrayal:** structured LLM interpretation over target mentions and cited passages.
  Citations are validated structurally, not proven semantically.
- **Claims:** MiniLM retrieves related sentences. NLI runs in both directions, with
  conservative temporal/participant guards. Similarity is not entailment; flagged
  contradictions require human review.
- **Coverage:** a general topic inventory plus extracted noun phrases and embedding
  matches. It is not exhaustive frame detection or proof of intentional omission.
- **RAG:** sentence-level passages with adjacent context, hybrid semantic/lexical top-four retrieval with citable adjacent passages
  per source, local FAISS and API generation. Overview and affected-place questions use the bounded full article collection instead of a similarity cutoff. It uses the current question and workspace,
  not external browsing or autonomous agents. Citation support needs human evaluation.
- **Scores:** model scores are uncalibrated. Similarity cutoffs are initial defaults
  and have not been tuned against a human-labelled domain-specific validation set.
- **Language/domain:** English only. BABE emphasizes US news; Indian news transfer
  performance has not been measured.

## Tests and human evaluation

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests -q
npm.cmd run build --prefix frontend
~~~

Tests exercise job completion, source offsets, quotation attribution, duplicate/size
limits, input formats, public-URL checks and redirect validation, NLI context guards,
citations, RAG success/failure and insufficient evidence.

`evaluation/` contains 100 entity, 100 claim, and 30 RAG unlabelled slots plus an
annotation protocol. These are **templates**, not completed human evaluation. Two
annotators must label held-out events independently and adjudicate disagreements.
`scripts/evaluate.py` scores populated adjudicated classification results.

## Project layout

- `backend/`: API, persistent storage, safe extraction, model adapters, analysis and RAG.
- `frontend/`: React/TypeScript application and production build.
- `scripts/`: datasets, model setup, training, annotation templates, scoring, packaging.
- `models/`: runtime model manifest, trained files, experimental metrics and local weights.
- `docs/` and `evaluation/`: architecture, report, demo instructions and annotation protocol.

See `docs/PROJECT_REPORT.md` for the academic report and work packages.
Source and dataset attribution are documented in `docs/SOURCES.md`.



## Clone from GitHub
The repository includes trained baseline files and a portable default model manifest.
Large neural weights, downloaded corpora, runtime workspaces and .env are excluded.
Run Setup-Prism.ps1 -WithModels, create your own .env, then Start-Prism.ps1.
The prepared local project can still use its fine-tuned RoBERTa weights. To use that
model on another machine, restore the separate trained checkpoint archive or rerun
training. Never commit API keys.

## Entity portrayal and duplicate names
Entity portrayal describes how a source presents a named person or organisation;
it is not an overall truth or bias rating. Inspect the cited wording for each label.
Use Combine duplicate names only for aliases of the same entity (for example IMD
and India Meteorological Department). This correction persists on reanalysis.
Combining different people, organisations or places would make the comparison misleading.

## Publication dates
Live URLs detect publication dates from metadata, Article JSON-LD and explicitly
labelled publication times. PDF, DOCX, TXT and pasted text can recover dates from
an explicit Published label near the article header. Review the detected date in
the input preview, or use Edit article > Publication date to enter a known date.
Existing saved articles can display a recovered date without rerunning analysis.
Missing dates remain unavailable: update timestamps, file creation times and dates
of events in the article are not substituted for publication dates.

## Numerical claim checks
Claim comparison now has a separate count-comparison pass for reported deaths and
injuries. It supports digits and common number words, exact counts, lower/upper
bounds and approximate counts. The check runs independently of embedding cutoffs
and NLI, with exact source passage links. The dashboard separates language cues
from potential claim conflicts.

For the same event and scope, exactly five deaths versus exactly three deaths at
the same stated time is flagged as a potential contradiction. At least three versus
five is numerically compatible. Different stated times, places or populations are
marked unresolved for review. This is not a universal numeric fact checker: complex
multi-count sentences, implicit scope, time zones, all date formats, negation and
attribution still require human review. Check publication dates and whether a later
report updates an earlier count. Re-run saved comparisons to populate the new checks.
