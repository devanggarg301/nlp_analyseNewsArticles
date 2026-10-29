# Perspective Prism — implementation report

## Abstract
Perspective Prism is an English-language news comparison system. It accepts articles
about one event, analyzes evaluative language, aligns related statements, and presents
coverage differences with inspectable evidence. It combines supervised sentence
classification, pretrained entity recognition, semantic embeddings, NLI, and optional
LLM-based target sentiment and retrieval-augmented answers.

## Problem and audience
Readers encounter different wording, quoted voices and topic emphasis across
publishers. A single overall sentiment label cannot explain these differences.
The project supports students and readers investigating how the selected coverage
is expressed. It analyzes language and source claims rather than adjudicating truth.

## Implemented features
1. Live URL extraction, editable preview, pasted text and PDF/DOCX/TXT/CSV imports.
2. Persistent event workspaces with two-to-five sources and duplicate checks.
3. BABE-trained biased-language detection and visible model provenance.
4. Side-by-side sentence readers and exact-offset evidence navigation.
5. Pretrained named entities, simple aliases, manual entity merging.
6. Quote/reported-speech attribution candidates with uncertainty.
7. API-based target portrayal with supporting entity mention passages.
8. Cross-source embedding alignment and bidirectional pretrained NLI.
9. Topic/coverage matrix, headline/opening indicators and quoted-source comparison.
10. Balanced retrieval, optional generated answers and clickable passage citations.
11. Markdown/JSON export and model experiment display.
12. Background progress and recoverable fallback behaviour.

## Methodology
Bias classification compares TF-IDF Logistic Regression, TF-IDF Linear SVM and a
RoBERTa fine-tuning pipeline. Source-document groups and near-duplicate groups are
kept together in train/validation/test partitions. Model selection uses validation
macro-F1. Counts and measured results are in models/metrics.json; these are the
authoritative values, rather than manually typed scores in the application.

spaCy supplies named entities and noun chunks. MiniLM embeddings support semantic
alignment and a FAISS evidence index. A pretrained NLI model examines candidate pairs
in both directions, with conservative context checks. Sentence-level extraction
retains exact source offsets. Entity portrayal uses an API-based model when configured
and remains unclear otherwise. RAG searches each source separately and uses the
retrieved passages as evidence for generated claims.

## Academic contribution
The substantial work is the integration and evaluation of these components, including
a real supervised training experiment, leakage-aware splits, evidence traceability,
speaker attribution handling, target-specific analysis, context-sensitive claim
comparison, balanced retrieval, and reproducible provenance.

This is an engineering course project using established methods. It does not claim a
new state-of-the-art bias-detection algorithm.

## Six-member work packages
| Member | Reviewable ownership |
|---|---|
| 1 | Dataset preparation, split audits, independent annotation coordination |
| 2 | Baselines, RoBERTa experiments, model-selection and error analysis |
| 3 | Entity extraction, aliases, attribution and target portrayal evaluation |
| 4 | Passage alignment, NLI context guards and coverage analysis |
| 5 | RAG retrieval, citation correctness and unanswerable-question testing |
| 6 | Frontend/backend integration, exports, accessibility and system tests |

These are proposed responsibilities, not a record asserting that six people already
performed the work. Record actual contributions in the submission.

## Evaluation
Automated tests cover API integration, evidence offsets, input validation, public-URL
restrictions, quotation handling, NLI guards and RAG fallbacks. Exact results are in
docs/VERIFICATION.md. Classification metrics are produced from held-out source groups.

Human evaluation requires independent labels for 100 entity examples, 100 aligned
claim pairs and 30 RAG questions. The included templates are unfilled. Annotation
agreement, target-sentiment F1, NLI F1, retrieval recall and semantic citation support
must not be reported as measured until annotation is completed.

## Limitations and future work
- BABE source-article grouping does not guarantee event-level disjointness.
- One initial transformer epoch is not a convergence study.
- Entity portrayal and generated answers require a configured available API.
- Quotation attribution and rationale spans are not separately trained models.
- Coverage topics and retrieval cutoffs are provisional and require evaluation.
- NLI cannot independently verify real-world claims.
- Full coreference, multilingual analysis, scanned-PDF OCR and autonomous browsing are excluded.
- Public hosting requires authentication and additional operational controls.

Optional extensions include headline/body framing comparison, additional annotated
frames, multilingual models and carefully evaluated coreference resolution.

