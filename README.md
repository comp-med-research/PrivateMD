---
title: PrivateMD
emoji: 🩺
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
python_version: 3.10
app_file: app.py
pinned: false
tags:
  - build-small
  - backyard-ai
  - healthcare
  - fhir
  - local-first
---

# PrivateMD

PrivateMD is an evidence-grounded, multimodal medical copilot for sensitive patient records. It runs on local synthetic FHIR bundles, imaging metadata, and DNA CSV files from the Coherent/Synthea dataset to help clinicians inspect patient journeys and identify personalized treatment review opportunities.

The prototype is intentionally conservative: it does not diagnose, prescribe, or send records to a hosted model. It extracts patient-specific evidence from local files, creates a longitudinal timeline, flags rule-based review opportunities, and answers clinician questions with cited chart evidence.

## Grounding Architecture

PrivateMD uses a local-first RAG pipeline designed for clinical traceability:

- Typed evidence chunks from FHIR resources including conditions, medications, observations, diagnostic reports, encounters, procedures, care plans, and imaging metadata.
- Hybrid retrieval with BM25-style lexical scoring, clinical entity expansion, resource-type priors, and recency weighting.
- Google LangExtract integration for structured, source-grounded clinical extraction when enabled.
- Optional small Gemma generation layer over retrieved evidence, with deterministic synthesis as the default fallback.
- Source citations remain visible in the answer and in the retrieved evidence table.

LangExtract can be enabled with a local Ollama model such as `gemma2:2b`:

```bash
export PRIVATE_MD_USE_LANGEXTRACT=1
export LANGEXTRACT_MODEL_ID=gemma2:2b
export LANGEXTRACT_MODEL_URL=http://localhost:11434
python app.py
```

LangExtract requires Python 3.10 or newer. The Hugging Face Space metadata pins Python 3.10; on older local Python versions, PrivateMD falls back to deterministic entity extraction.

Optional local Gemma answer synthesis can be enabled when `transformers`, `torch`, and local model weights are available:

```bash
export PRIVATE_MD_GENERATOR=gemma
export PRIVATE_MD_GEMMA_MODEL=google/gemma-2-2b-it
python app.py
```

## Hackathon Fit

- Track: Backyard AI
- Interface: Gradio Space
- Model policy: no required hosted model; compatible with under-32B local Gemma models as an optional extension
- Privacy angle: local-first analysis over records already mounted in the Space
- Bonus directions: custom UI, best agent, OpenAI/Codex-attributed development

## Dataset

This app uses the Coherent Synthetic Data Set included under `data/coherent-11-07-2022`, which contains synthetic longitudinal FHIR Bundles, DICOM references, and DNA CSV files. The dataset README lists the Creative Commons Attribution 4.0 International License and MITRE public release details.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Development Attribution

PrivateMD was built in collaboration with OpenAI Codex, including project scaffolding, FHIR parsing logic, Gradio interface development, local verification, and repository setup.

## Submission Checklist

- Add demo video link here.
- Add social post link here.
- Update Space tags/frontmatter for the final prize categories.
- Mention any optional local model added after the deterministic evidence layer.
