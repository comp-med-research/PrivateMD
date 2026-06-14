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

PrivateMD now uses a single chat flow for grounding:

- A clinician asks one chart question.
- PrivateMD builds a compact source document from the original local patient record: demographics, conditions, medication requests, observations, procedures, imaging, and genomics summaries.
- Google LangExtract extracts question-relevant entities and relationships directly from that source document.
- PrivateMD synthesizes the visible answer from the extracted spans and their source lines; the chat path does not use the RAG pipeline or a separate generation model.
- The page shows the embedded highlighted source visualization first, followed by the answer, the exact source document sent to LangExtract, the extracted entities/relations table, and the source-line evidence table.

The LangExtract source document is designed for clinical traceability:

- Every source line has a bracket citation such as `[1]` and a FHIR/DNA source reference.
- Extracted spans keep `claim_group` relationship attributes.
- The highlighted visualization is embedded directly in the Chat tab.
- The evidence table preserves the original source line, date, resource type, and FHIR source.

LangExtract defaults to the smaller `gemma2:2b` Ollama model because Google's local LangExtract examples use that path and it is fast enough for focused line-by-line extraction.

With Ollama:

```bash
ollama pull gemma2:2b
```

Then run PrivateMD with local LangExtract:

```bash
export LANGEXTRACT_CHAT_MODEL_ID=gemma2:2b
export LANGEXTRACT_MODEL_URL=http://localhost:11434
python app.py
```

The Chat tab returns:

- A grounded answer for clinician review.
- The original chart source document sent to LangExtract.
- Extracted entities and relations grouped by `claim_group`.
- The source-line evidence table used to ground the answer.
- An embedded interactive LangExtract visualization.

LangExtract requires Python 3.10 or newer. This repo has been tested locally with Python 3.11.

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
