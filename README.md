---
title: PrivateMD
emoji: 🩺
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
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

## Hackathon Fit

- Track: Backyard AI
- Interface: Gradio Space
- Model policy: no required hosted model; compatible with under-32B local models as an optional extension
- Privacy angle: local-first analysis over records already mounted in the Space
- Bonus directions: custom UI, best agent, OpenAI/Codex-attributed development

## Dataset

This app uses the Coherent Synthetic Data Set included under `data/coherent-11-07-2022`, which contains synthetic longitudinal FHIR Bundles, DICOM references, and DNA CSV files. The dataset README lists the Creative Commons Attribution 4.0 International License and MITRE public release details.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Submission Checklist

- Add demo video link here.
- Add social post link here.
- Update Space tags/frontmatter for the final prize categories.
- Mention any optional local model added after the deterministic evidence layer.
