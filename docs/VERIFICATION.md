# Verification record

## Publication date recovery
- Full regression suite: 102 passing tests. Date tests cover explicit publication
  labels, Article JSON-LD, meta/time tags, invalid calendar dates, DOCX/TXT previews,
  manual date precedence and recovery in saved workspace responses.
- Updated timestamps, event dates and document creation dates are not used as
  publication dates. Missing dates remain unavailable.
- Recovered the saved NDTV article's explicit Published On date as 28 October 2026
  in the running website without clearing its analysis or chat.

## Numerical claim update
- Full regression suite: 76 passing tests. Targeted checks cover the exact five-versus-three
  example, numeric/word forms, compatible lower bounds, different times/locations,
  approximate totals, subgroup counts and negated assertions.
- Numeric candidates bypass sentence similarity filtering and do not depend on NLI.
- The overview now separates language cues from potential claim conflicts.

## Document-count and API-status regression update
- Regression suite expanded to **48 passing tests**; the production build passes.
- Document/docs/file/article/source count questions use workspace metadata rather than
  searching article sentences. Qualified counts remain content questions.
- Saved pre-API notes are explicitly distinguished from current configuration.
- Reanalysis of unchanged evidence preserves the user's chat history.
- Gemini array/object response variations are handled with evidence validation.
- Temporary provider 500/502/503/504 responses have bounded retries; authentication
  errors are not retried. Safe provider-error explanations distinguish service issues
  from missing configuration.
- A live generated answer cited three source passages; a live enrichment retest
  produced 21 entity judgements after the response-shape fix.
- Human evaluation remains pending. The measurements below describe the initial build.

## Completed checks
- Frontend TypeScript check and Vite production build passed.
- Automated backend suite: **28 passed**. One upstream Starlette/httpx deprecation warning.
- Local HTTP root and health endpoints returned 200.
- Full fictional three-source comparison completed with 34 passages,
  24 aligned pairs and 11 coverage elements.
- Every evidence offset exactly matched the original source text.
- Abbreviation segmentation was checked against the demo's a.m. references.
- Workspace source-count questions returned source metadata.
- Source-linked retrieval returned excerpts; an unrelated astronomy question returned
  insufficient evidence.
- JSON/Markdown exports retained all passage identifiers.
- Live URL extraction succeeded and populated title, publisher, date and article text:
  https://www.theguardian.com/world/2025/jan/15/gaza-ceasefire-deal-agreed-between-hamas-and-israel-qatari-pm-says
- No real publisher article text was saved into the fictional sample or source archive.

## Measured classifier results
Dataset: BABE SG2 expert sentence labels.
Split: Source-document and near-duplicate group split; event identifiers unavailable.
Counts: {"train": 2553, "validation": 534, "test": 580}.

| Model | Validation macro-F1 | Test macro-F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.728 | 0.725 |
| TF-IDF + Linear SVM | 0.710 | 0.697 |
| Fine-tuned RoBERTa-base | 0.811 | 0.843 |

Selected by validation macro-F1: **Fine-tuned RoBERTa-base**.
Transformer status: Fine-tuned for 1 epochs; best validation checkpoint selected.

## Limits of verification
The fictional fixture produced zero RoBERTa flags at the default 0.5 threshold and
five sentences with independent lexical cues. The UI distinguishes those cues from
classifier predictions. This illustrates a domain-transfer limitation rather than a
reason to adjust the threshold against a demonstration example.

No live generative-provider call was made without credentials. RAG API success, failure,
invalid citation and abstention branches were exercised with controlled test responses.
Entity sentiment stays unclear without an API; this is intentional fallback behaviour.
Human entity/claim/RAG annotation has not been completed. The provided CSVs are queues,
not measured results. Thresholds are provisional. Event-disjoint generalization and
Indian-news transfer have not been established.

The one-epoch transformer run is an initial course-project experiment, not a convergence
or state-of-the-art claim. The portable source ZIP uses a trained baseline unless the
large local neural weights are separately copied or training is rerun.

## Document and RAG update
- 57 regression tests pass, including PDF/DOCX upload and generic overview questions.
- Live Gemini overview and affected-location questions succeeded against the user's cyclone workspace with passage citations. Location answers identified geographic regions and distinguished forecast risks from reported effects.
- Interface uses a light sage palette with muted teal controls; entity merge guidance explains alias corrections.
