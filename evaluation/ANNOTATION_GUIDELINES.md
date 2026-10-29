# Human evaluation protocol

The CSV queues contain **empty slots**, not completed annotations or measured results.
Use the BASIL official release (https://github.com/launchnlp/BASIL) to choose held-out
same-event article groups. Retain their source event IDs. Do not reuse these groups for
training, prompt tuning, threshold selection, or example demonstrations.

## Sampling
- Entity: 100 target–passage examples across at least 10 events. Include quoted attacks,
  negative event descriptions, pronouns, neutral mentions, and competing targets.
- Claims: 100 cross-source pairs across at least 10 events. Include compatible claims,
  genuine candidate contradictions, unrelated details, different dates and entities.
- RAG: 30 questions across at least 10 events; include 10 unanswerable questions.
  Record exact supporting passage IDs and whether all required evidence was retrieved.

## Independent labels
Two annotators work separately without seeing one another's answers. Hide publisher
names while labelling where feasible. Record disagreements before discussion; a third
reviewer or documented joint adjudication supplies the final label. Retain both originals.

Entity labels: favourable, unfavourable, neutral, mixed, unclear.
Judge sentiment towards the target, not the overall emotional content. Record who
expresses the judgement (narrator, named speaker, unknown).

Claim labels: supporting, potential_contradiction, unresolved.
Read surrounding context, time references, participants and speaker attribution. Entailment
is directional. Missing information is not contradiction.

RAG: supported, partially_supported, unsupported, insufficient_evidence.
Check each substantive answer claim against the actual cited passage. A valid citation ID
does not establish semantic support. Calculate citation support precision manually.
Compute retrieval Recall@k as relevant gold passages retrieved / total gold passages.
For unanswerable questions, check that the answer declines to invent facts.

## Reporting
Report macro-F1 for entity and NLI labels, per-class results, agreement (Cohen's kappa),
retrieval Recall@k, citation support and abstention performance. Publish the denominators.
Never treat an LLM's own judgement as human ground truth. State the tested language,
events, source distribution and model versions.

