# Research and model attribution

- BABE: Spinde et al., Neural Media Bias Detection Using Distant Supervision With
  BABE — Bias Annotations By Experts (2021).
  https://aclanthology.org/2021.findings-emnlp.101/
  Data downloaded from the Media Bias Identification Benchmark repository:
  https://github.com/Media-Bias-Group/MBIB/tree/main/datasets/010_BABE
- BASIL: Fan et al., In Plain Sight: Media Bias Through the Lens of Factual Reporting
  (2019). https://aclanthology.org/D19-1664/
  Official maintained release: https://github.com/launchnlp/BASIL
  Referenced for article-level research and future held-out annotation, not used to
  fabricate completed evaluations.
- Sentence embeddings: sentence-transformers/all-MiniLM-L6-v2
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Natural language inference: cross-encoder/nli-distilroberta-base
  https://huggingface.co/cross-encoder/nli-distilroberta-base
- Named entities: spaCy en_core_web_sm 3.8.0
  https://spacy.io/models/en
- Transformer initialization: FacebookAI/roberta-base
  https://huggingface.co/FacebookAI/roberta-base

Review upstream licences and dataset access/redistribution terms before distributing
research corpora or model weights. The source archive intentionally excludes the
downloaded original corpus and large pretrained neural weights. Fictional demo texts
were authored for this project and do not represent real publishers.

No left/right publisher labels are used as ground truth for individual sentences.
No live article text is bundled into the curated demo.

