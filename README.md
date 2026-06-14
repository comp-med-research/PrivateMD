---
title: PrivateMD
emoji: 🩺
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
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

- Typed evidence chunks from FHIR resources including conditions, medications, observations, diagnostic reports, encounters, procedures, care plans, imaging metadata, DNA summaries, and derived lab trends.
- Query planning with clinical alias expansion and optional Google LangExtract extraction.
- Multi-stage retrieval with fielded BM25, resource-type priors, temporal intent scoring, graph expansion through encounters/facets, and MMR diversification to avoid repeated near-duplicate evidence.
- Optional Gemma 3 4B generation layer over retrieved evidence, with deterministic synthesis as the fallback when no local model server is running.
- Source citations remain visible in the answer and in the retrieved evidence table.

LangExtract and the local generator are designed around Gemma 3 4B. With Ollama:

```bash
ollama pull gemma3:4b
```

Then run PrivateMD with LangExtract and Gemma generation:

```bash
export PRIVATE_MD_USE_LANGEXTRACT=1
export LANGEXTRACT_MODEL_ID=gemma3:4b
export LANGEXTRACT_MODEL_URL=http://localhost:11434
export PRIVATE_MD_GENERATOR=ollama
export PRIVATE_MD_OLLAMA_MODEL=gemma3:4b
python app.py
```

LangExtract requires Python 3.10 or newer. This repo has been tested locally with Python 3.11.

Optional Hugging Face Transformers generation can use the official 4B instruction model when `transformers`, `torch`, and accepted Gemma model access are available:

```bash
export PRIVATE_MD_GENERATOR=hf
export PRIVATE_MD_GEMMA_MODEL=google/gemma-3-4b-it
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
python3.11 -m venv .venv
source .venv/bin/activate
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
