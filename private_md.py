from __future__ import annotations

import json
import re
import math
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


DATA_DIR = Path("data/coherent-11-07-2022")
FHIR_DIR = DATA_DIR / "fhir"
DNA_DIR = DATA_DIR / "dna"


@dataclass(frozen=True)
class PatientChoice:
    label: str
    path: str


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    resource_type: str
    date: str
    title: str
    text: str
    source: str
    entities: Tuple[str, ...]
    resource: Dict[str, Any]


RESOURCE_PRIORS = {
    "Condition": 1.35,
    "MedicationRequest": 1.3,
    "Observation": 1.2,
    "DiagnosticReport": 1.15,
    "CarePlan": 1.15,
    "Procedure": 1.05,
    "ImagingStudy": 1.05,
    "Encounter": 0.9,
}

CLINICAL_ALIASES = {
    "a1c": {"a1c", "hba1c", "hemoglobin", "glycemic", "diabetes"},
    "anticoagulation": {"warfarin", "inr", "prothrombin", "anticoagulation", "bleeding"},
    "heart failure": {"heart", "failure", "congestive", "sodium", "volume", "edema"},
    "blood pressure": {"blood", "pressure", "systolic", "diastolic", "hypertension"},
    "obesity": {"bmi", "body", "mass", "weight", "obesity"},
    "genomics": {"gene", "variant", "pathogenic", "genomic", "dna"},
    "imaging": {"image", "imaging", "radiography", "modality", "dicom"},
}


def _entries(bundle: Dict[str, Any], resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
    resources = [entry.get("resource", {}) for entry in bundle.get("entry", [])]
    if resource_type:
        return [resource for resource in resources if resource.get("resourceType") == resource_type]
    return resources


def _coding_text(value: Dict[str, Any]) -> str:
    if not value:
        return ""
    if value.get("text"):
        return value["text"]
    for coding in value.get("coding", []):
        if coding.get("display"):
            return coding["display"]
        if coding.get("code"):
            return coding["code"]
    return ""


def _date(value: Optional[str]) -> str:
    if not value:
        return ""
    return value[:10]


def _parse_date(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return datetime.min


def _quantity(resource: Dict[str, Any]) -> str:
    quantity = resource.get("valueQuantity") or {}
    if not quantity:
        if "valueString" in resource:
            return str(resource["valueString"])
        if "valueCodeableConcept" in resource:
            return _coding_text(resource["valueCodeableConcept"])
        return ""
    value = quantity.get("value")
    unit = quantity.get("unit") or quantity.get("code") or ""
    return f"{value:g} {unit}".strip() if isinstance(value, (int, float)) else f"{value} {unit}".strip()


def _patient_name(patient: Dict[str, Any]) -> str:
    names = patient.get("name", [])
    if not names:
        return patient.get("id", "Unknown patient")
    name = names[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")
    return f"{given} {family}".strip() or patient.get("id", "Unknown patient")


def _age(patient: Dict[str, Any]) -> str:
    birth = patient.get("birthDate")
    if not birth:
        return "Unknown age"
    born = _parse_date(birth)
    if born == datetime.min:
        return "Unknown age"
    today = datetime.utcnow()
    years = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return f"{years} years"


def _source(resource: Dict[str, Any]) -> str:
    resource_type = resource.get("resourceType", "Resource")
    resource_id = resource.get("id", "unknown")
    code = _coding_text(resource.get("code", {}))
    date = (
        resource.get("recordedDate")
        or resource.get("effectiveDateTime")
        or resource.get("authoredOn")
        or resource.get("started")
        or resource.get("performedDateTime")
        or resource.get("period", {}).get("start")
        or resource.get("performedPeriod", {}).get("start")
    )
    label = f"{resource_type}/{resource_id}"
    if code:
        label += f" - {code}"
    if date:
        label += f" ({_date(date)})"
    return label


def load_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def patient_choices(limit: int = 250) -> List[PatientChoice]:
    choices: List[Tuple[bool, PatientChoice]] = []
    for path in sorted(FHIR_DIR.glob("*.json")):
        try:
            bundle = load_bundle(str(path))
            patient = _entries(bundle, "Patient")[0]
            deceased = bool(patient.get("deceasedDateTime") or patient.get("deceasedBoolean"))
            status = "deceased" if deceased else "active"
            label = f"{_patient_name(patient)} | {_age(patient)} | {patient.get('gender', 'unknown')} | {status} | {patient.get('id')}"
        except Exception:
            deceased = True
            label = path.stem
        choices.append((deceased, PatientChoice(label=label, path=str(path))))
    choices = sorted(choices, key=lambda item: (item[0], item[1].label.lower()))
    return [choice for _, choice in choices[:limit]]


def _latest_observations(observations: Iterable[Dict[str, Any]], names: Iterable[str]) -> List[Dict[str, Any]]:
    wanted = [name.lower() for name in names]
    matches = [
        obs
        for obs in observations
        if any(name in _coding_text(obs.get("code", {})).lower() for name in wanted)
    ]
    return sorted(matches, key=lambda obs: _parse_date(obs.get("effectiveDateTime")), reverse=True)


def _dna_summary(patient_id: str) -> Tuple[str, List[str]]:
    files = list(DNA_DIR.glob(f"*{patient_id}_dna.csv"))
    if not files:
        return "No matching DNA file found for this patient.", []
    dna = pd.read_csv(files[0])
    variants = dna[dna["VARIANT"] == True]  # noqa: E712 - pandas wants literal comparison here.
    pathogenic = variants[variants["CLINICAL_SIGNIFICANCE"].str.contains("Pathogenic", na=False)]
    uncertain = variants[variants["CLINICAL_SIGNIFICANCE"].str.contains("Uncertain", na=False)]
    genes = sorted(set(pathogenic["GENE"].dropna().astype(str)))[:8]
    text = (
        f"{len(variants)} variant-positive rows in {files[0].name}; "
        f"{len(pathogenic)} pathogenic/likely pathogenic rows; "
        f"{len(uncertain)} uncertain-significance rows."
    )
    if genes:
        text += f" Genes flagged: {', '.join(genes)}."
    sources = [f"DNA/{files[0].name}"]
    return text, sources


def _timeline(bundle: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for resource in _entries(bundle):
        resource_type = resource.get("resourceType")
        if resource_type not in {"Condition", "MedicationRequest", "Procedure", "Encounter", "Observation", "ImagingStudy", "CarePlan"}:
            continue
        date = (
            resource.get("onsetDateTime")
            or resource.get("recordedDate")
            or resource.get("authoredOn")
            or resource.get("started")
            or resource.get("effectiveDateTime")
            or resource.get("period", {}).get("start")
            or resource.get("performedPeriod", {}).get("start")
        )
        if not date:
            continue
        if resource_type == "MedicationRequest":
            label = _coding_text(resource.get("medicationCodeableConcept", {}))
        elif resource_type == "Encounter":
            label = _coding_text((resource.get("type") or [{}])[0])
        elif resource_type == "CarePlan":
            label = _coding_text((resource.get("category") or [{}])[-1])
        elif resource_type == "ImagingStudy":
            series = resource.get("series") or [{}]
            label = _coding_text(series[0].get("bodySite", {})) or _coding_text(series[0].get("modality", {}))
        else:
            label = _coding_text(resource.get("code", {}))
        rows.append(
            {
                "date": _date(date),
                "type": resource_type,
                "event": label,
                "status": resource.get("status") or resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", ""),
                "source": f"{resource_type}/{resource.get('id', '')}",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("date", ascending=False).head(80)


def _opportunities(bundle: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    conditions = _entries(bundle, "Condition")
    observations = _entries(bundle, "Observation")
    medications = _entries(bundle, "MedicationRequest")
    careplans = _entries(bundle, "CarePlan")
    evidence: List[str] = []
    rows: List[Dict[str, str]] = []

    condition_text = " | ".join(_coding_text(c.get("code", {})).lower() for c in conditions)
    medication_text = " | ".join(_coding_text(m.get("medicationCodeableConcept", {})).lower() for m in medications)

    def add(priority: str, signal: str, action: str, sources: List[Dict[str, Any]]) -> None:
        rows.append(
            {
                "priority": priority,
                "signal": signal,
                "clinical opportunity": action,
                "evidence": "\n".join(_source(source) for source in sources[:4]),
            }
        )
        evidence.extend(_source(source) for source in sources[:4])

    bmi = _latest_observations(observations, ["Body Mass Index"])
    if bmi:
        try:
            value = float((bmi[0].get("valueQuantity") or {}).get("value"))
        except (TypeError, ValueError):
            value = 0
        if value >= 30:
            add(
                "High",
                f"Latest BMI is {_quantity(bmi[0])}.",
                "Review weight-management plan, cardiometabolic screening, and medication contributors.",
                [bmi[0]],
            )

    systolic = _latest_observations(observations, ["Systolic Blood Pressure"])
    diastolic = _latest_observations(observations, ["Diastolic Blood Pressure"])
    if systolic and diastolic:
        s_val = float((systolic[0].get("valueQuantity") or {}).get("value", 0))
        d_val = float((diastolic[0].get("valueQuantity") or {}).get("value", 0))
        if s_val >= 140 or d_val >= 90 or "hypertension" in condition_text:
            add(
                "Medium",
                f"Latest BP is {s_val:g}/{d_val:g} mmHg.",
                "Consider medication adherence, home BP log, renal function, and follow-up interval.",
                [systolic[0], diastolic[0]],
            )

    a1c = _latest_observations(observations, ["Hemoglobin A1c"])
    if "diabetes" in condition_text:
        add(
            "High" if not a1c else "Medium",
            "Diabetes appears in the condition history." + (f" Latest A1c: {_quantity(a1c[0])}." if a1c else " No A1c observation found in this bundle."),
            "Check glycemic trend, kidney protection, statin eligibility, and retinal/foot screening status.",
            ([a1c[0]] if a1c else []) + conditions,
        )

    if "warfarin" in medication_text:
        inr = _latest_observations(observations, ["INR", "Prothrombin"])
        warfarin = [m for m in medications if "warfarin" in _coding_text(m.get("medicationCodeableConcept", {})).lower()]
        add(
            "High",
            "Warfarin exposure is present." + (f" Latest coagulation signal: {_quantity(inr[0])}." if inr else " No INR-like observation found."),
            "Verify anticoagulation indication, INR monitoring cadence, interactions, and bleeding-risk counseling.",
            warfarin + inr,
        )

    if "heart failure" in condition_text or "congestive heart failure" in condition_text:
        heart_careplans = [cp for cp in careplans if "heart failure" in json.dumps(cp).lower()]
        add(
            "High",
            "Heart failure is present in longitudinal history.",
            "Review guideline-directed therapy, volume trend, sodium education, renal function, and recent decompensation.",
            conditions + heart_careplans,
        )

    if not rows:
        rows.append(
            {
                "priority": "Review",
                "signal": "No high-salience rule fired in this small ruleset.",
                "clinical opportunity": "Use the timeline and copilot query to inspect active problems, medications, imaging, and labs.",
                "evidence": "FHIR bundle parsed successfully.",
            }
        )
    return pd.DataFrame(rows), sorted(set(evidence))


def _snapshot(bundle: Dict[str, Any]) -> str:
    patient = _entries(bundle, "Patient")[0]
    conditions = _entries(bundle, "Condition")
    meds = _entries(bundle, "MedicationRequest")
    imaging = _entries(bundle, "ImagingStudy")
    observations = _entries(bundle, "Observation")
    dna_text, _ = _dna_summary(patient["id"])
    active_conditions = [
        _coding_text(c.get("code", {}))
        for c in conditions
        if c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code") == "active"
    ]
    current_meds = [
        _coding_text(m.get("medicationCodeableConcept", {}))
        for m in meds
        if m.get("status") in {"active", "completed", "stopped"}
    ]
    latest_vitals = []
    for name in ["Body Mass Index", "Systolic Blood Pressure", "Diastolic Blood Pressure", "Body Weight"]:
        obs = _latest_observations(observations, [name])
        if obs:
            latest_vitals.append(f"{_coding_text(obs[0].get('code', {}))}: {_quantity(obs[0])} on {_date(obs[0].get('effectiveDateTime'))}")
    return "\n".join(
        [
            f"### {_patient_name(patient)}",
            f"**Demographics:** {_age(patient)}, {patient.get('gender', 'unknown')}, born {patient.get('birthDate', 'unknown')}",
            f"**Record depth:** {len(conditions)} conditions, {len(meds)} medication requests, {len(observations)} observations, {len(imaging)} imaging studies.",
            f"**Active problems:** {', '.join(active_conditions[:8]) or 'None marked active'}",
            f"**Medication history:** {', '.join(current_meds[:8]) or 'No medication requests found'}",
            f"**Latest vitals/labs:** {'; '.join(latest_vitals) or 'No vital signs found'}",
            f"**Genomics:** {dna_text}",
        ]
    )


def _context_lines(bundle: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for resource in _entries(bundle, "Condition"):
        lines.append(f"Condition: {_coding_text(resource.get('code', {}))}; status={resource.get('clinicalStatus', {}).get('coding', [{}])[0].get('code', '')}; source={_source(resource)}")
    for resource in _entries(bundle, "MedicationRequest"):
        lines.append(f"Medication: {_coding_text(resource.get('medicationCodeableConcept', {}))}; status={resource.get('status', '')}; source={_source(resource)}")
    for resource in sorted(_entries(bundle, "Observation"), key=lambda obs: _parse_date(obs.get("effectiveDateTime")), reverse=True)[:80]:
        lines.append(f"Observation: {_coding_text(resource.get('code', {}))}={_quantity(resource)}; source={_source(resource)}")
    for resource in _entries(bundle, "ImagingStudy"):
        series = resource.get("series") or [{}]
        lines.append(f"Imaging: {_coding_text(series[0].get('modality', {}))} {_coding_text(series[0].get('bodySite', {}))}; source={_source(resource)}")
    return lines


def _tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2]


def _clinical_entities(text: str) -> Set[str]:
    tokens = set(_tokens(text))
    entities = set(tokens)
    lower = text.lower()
    for label, aliases in CLINICAL_ALIASES.items():
        if label in lower or tokens.intersection(aliases):
            entities.add(label)
            entities.update(aliases)
    return entities


def _langextract_entities(text: str) -> Set[str]:
    """Use Google LangExtract when configured; otherwise stay fully local and deterministic.

    LangExtract is intentionally optional because cloud extraction requires API keys and
    local extraction requires an Ollama model such as gemma2:2b.
    """
    if os.getenv("PRIVATE_MD_USE_LANGEXTRACT", "").lower() not in {"1", "true", "yes"}:
        return set()
    try:
        lx = import_module("langextract")
    except ImportError:
        return set()

    prompt = (
        "Extract clinical entities from this chart text. Keep extractions verbatim and grounded "
        "to the source. Classes: condition, medication, lab, procedure, imaging, gene, risk_factor."
    )
    try:
        result = lx.extract(
            text_or_documents=text[:4000],
            prompt_description=prompt,
            examples=[],
            model_id=os.getenv("LANGEXTRACT_MODEL_ID", "gemma2:2b"),
            model_url=os.getenv("LANGEXTRACT_MODEL_URL", "http://localhost:11434"),
            extraction_passes=1,
            max_workers=1,
            max_char_buffer=1000,
        )
    except Exception:
        return set()

    extracted: Set[str] = set()
    for extraction in getattr(result, "extractions", []) or []:
        if getattr(extraction, "char_interval", True) is None:
            continue
        value = getattr(extraction, "extraction_text", "")
        if value:
            extracted.update(_tokens(value))
            extracted.add(value.lower())
    return extracted


def _resource_title(resource: Dict[str, Any]) -> str:
    resource_type = resource.get("resourceType", "")
    if resource_type == "MedicationRequest":
        return _coding_text(resource.get("medicationCodeableConcept", {}))
    if resource_type == "Encounter":
        return _coding_text((resource.get("type") or [{}])[0])
    if resource_type == "CarePlan":
        return _coding_text((resource.get("category") or [{}])[-1])
    if resource_type == "ImagingStudy":
        series = resource.get("series") or [{}]
        return " ".join(
            part
            for part in [
                _coding_text(series[0].get("modality", {})),
                _coding_text(series[0].get("bodySite", {})),
            ]
            if part
        )
    if resource_type == "DiagnosticReport":
        return _coding_text(resource.get("code", {}))
    return _coding_text(resource.get("code", {})) or resource_type


def _resource_date(resource: Dict[str, Any]) -> str:
    value = (
        resource.get("recordedDate")
        or resource.get("effectiveDateTime")
        or resource.get("authoredOn")
        or resource.get("started")
        or resource.get("issued")
        or resource.get("onsetDateTime")
        or resource.get("performedDateTime")
        or resource.get("period", {}).get("start")
        or resource.get("performedPeriod", {}).get("start")
    )
    return _date(value)


def _chunk_text(resource: Dict[str, Any]) -> str:
    resource_type = resource.get("resourceType", "")
    title = _resource_title(resource)
    fields = [f"{resource_type}: {title}"]
    date = _resource_date(resource)
    if date:
        fields.append(f"date: {date}")
    if resource_type == "Observation":
        fields.append(f"value: {_quantity(resource)}")
    if resource_type == "MedicationRequest":
        fields.append(f"status: {resource.get('status', '')}")
        requester = resource.get("requester", {}).get("display")
        if requester:
            fields.append(f"requester: {requester}")
    if resource_type == "Condition":
        fields.append(f"clinical status: {resource.get('clinicalStatus', {}).get('coding', [{}])[0].get('code', '')}")
    if resource_type == "CarePlan":
        text = re.sub(r"<[^>]+>", " ", resource.get("text", {}).get("div", ""))
        if text:
            fields.append(f"plan text: {' '.join(text.split())}")
    if resource_type == "DiagnosticReport":
        conclusion = resource.get("conclusion")
        if conclusion:
            fields.append(f"conclusion: {conclusion}")
    if resource_type == "ImagingStudy":
        series = resource.get("series") or [{}]
        fields.append(f"series: {_coding_text(series[0].get('modality', {}))} {_coding_text(series[0].get('bodySite', {}))}")
    return "; ".join(part for part in fields if part)


def _rag_chunks(bundle: Dict[str, Any]) -> List[EvidenceChunk]:
    useful_types = {
        "Condition",
        "MedicationRequest",
        "Observation",
        "DiagnosticReport",
        "Procedure",
        "Encounter",
        "ImagingStudy",
        "CarePlan",
    }
    chunks: List[EvidenceChunk] = []
    for resource in _entries(bundle):
        resource_type = resource.get("resourceType", "")
        if resource_type not in useful_types:
            continue
        title = _resource_title(resource)
        text = _chunk_text(resource)
        entities = _clinical_entities(text)
        chunks.append(
            EvidenceChunk(
                chunk_id=f"{resource_type}/{resource.get('id', len(chunks))}",
                resource_type=resource_type,
                date=_resource_date(resource),
                title=title,
                text=text,
                source=_source(resource),
                entities=tuple(sorted(entities)),
                resource=resource,
            )
        )
    return chunks


def _bm25_score(query_terms: List[str], chunk_terms: List[str], avg_len: float, doc_count: int, document_frequency: Dict[str, int]) -> float:
    if not query_terms or not chunk_terms:
        return 0.0
    k1 = 1.4
    b = 0.72
    length = len(chunk_terms)
    term_counts = {term: chunk_terms.count(term) for term in set(chunk_terms)}
    score = 0.0
    for term in set(query_terms):
        frequency = term_counts.get(term, 0)
        if not frequency:
            continue
        df = document_frequency.get(term, 0)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        score += idf * (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + b * length / max(avg_len, 1)))
    return score


def retrieve_evidence(bundle: Dict[str, Any], question: str, top_k: int = 10) -> List[Tuple[float, EvidenceChunk]]:
    chunks = _rag_chunks(bundle)
    query_entities = _clinical_entities(question) | _langextract_entities(question)
    query_terms = sorted(query_entities | set(_tokens(question)))
    tokenized = {chunk.chunk_id: _tokens(chunk.text) for chunk in chunks}
    doc_count = len(chunks)
    avg_len = sum(len(terms) for terms in tokenized.values()) / max(doc_count, 1)
    document_frequency: Dict[str, int] = {}
    for terms in tokenized.values():
        for term in set(terms):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    scored: List[Tuple[float, EvidenceChunk]] = []
    for chunk in chunks:
        terms = tokenized[chunk.chunk_id]
        lexical = _bm25_score(query_terms, terms, avg_len, doc_count, document_frequency)
        entity_overlap = len(set(chunk.entities).intersection(query_entities))
        resource_prior = RESOURCE_PRIORS.get(chunk.resource_type, 1.0)
        recency = 0.0
        parsed = _parse_date(chunk.date)
        if parsed != datetime.min:
            recency = min(max((parsed.year - 1950) / 90, 0), 0.35)
        score = (lexical + entity_overlap * 1.8 + recency) * resource_prior
        if score > 0:
            scored.append((score, chunk))
    if not scored:
        scored = [(0.1, chunk) for chunk in sorted(chunks, key=lambda item: item.date, reverse=True)[:top_k]]
    return sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]


def _evidence_frame(scored_chunks: List[Tuple[float, EvidenceChunk]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": rank,
                "score": round(score, 3),
                "type": chunk.resource_type,
                "date": chunk.date,
                "evidence": chunk.text,
                "source": chunk.source,
            }
            for rank, (score, chunk) in enumerate(scored_chunks, start=1)
        ]
    )


def _local_gemma_answer(question: str, scored_chunks: List[Tuple[float, EvidenceChunk]]) -> Optional[str]:
    """Optional local generation hook for small Gemma models.

    Set PRIVATE_MD_GENERATOR=gemma and PRIVATE_MD_GEMMA_MODEL to a local HF model id
    only when transformers/torch and model weights are available.
    """
    if os.getenv("PRIVATE_MD_GENERATOR", "").lower() != "gemma":
        return None
    try:
        transformers = import_module("transformers")
    except ImportError:
        return None
    model_id = os.getenv("PRIVATE_MD_GEMMA_MODEL", "google/gemma-2-2b-it")
    context = "\n".join(f"[{i}] {chunk.text}" for i, (_, chunk) in enumerate(scored_chunks[:6], start=1))
    prompt = (
        "You are PrivateMD, a clinician-facing chart review assistant. Use only the cited local evidence. "
        "Do not diagnose or prescribe. Return concise review considerations with citation numbers.\n\n"
        f"Question: {question}\n\nEvidence:\n{context}\n\nAnswer:"
    )
    try:
        pipe = transformers.pipeline("text-generation", model=model_id, device_map="auto")
        output = pipe(prompt, max_new_tokens=220, do_sample=False, return_full_text=False)
    except Exception:
        return None
    if output and isinstance(output, list):
        return output[0].get("generated_text", "").strip() or None
    return None


def answer_question(path: str, question: str) -> Tuple[str, pd.DataFrame]:
    bundle = load_bundle(path)
    if not question.strip():
        question = "What are the most important issues in this chart?"
    scored_chunks = retrieve_evidence(bundle, question)
    generated = _local_gemma_answer(question, scored_chunks)
    evidence_lines = [
        f"- [{rank}] {chunk.text}  \n  Source: `{chunk.source}`"
        for rank, (_, chunk) in enumerate(scored_chunks[:8], start=1)
    ]
    langextract_state = "enabled" if os.getenv("PRIVATE_MD_USE_LANGEXTRACT", "").lower() in {"1", "true", "yes"} else "available but off"
    generator_state = "Gemma local generation" if generated else "deterministic synthesis"
    answer = generated or (
        "The retrieved evidence points to these chart-backed review areas:\n"
        + "\n".join(evidence_lines[:5])
    )
    return (
        "PrivateMD used a local hybrid RAG pipeline over typed FHIR evidence chunks. "
        "This prototype does not diagnose or prescribe; it surfaces record-backed considerations for clinician review.\n\n"
        f"**Question:** {question}\n\n"
        f"**Pipeline:** BM25-style retrieval + clinical entity expansion + resource weighting + recency boost; LangExtract {langextract_state}; {generator_state}.\n\n"
        f"**Grounded answer:**\n{answer}\n\n"
        "**Retrieved evidence:**\n"
        + "\n".join(evidence_lines)
        + "\n\n**Clinician next step:** verify the cited FHIR sources in the patient context before acting.",
        _evidence_frame(scored_chunks),
    )


def analyze_patient(path: str) -> Tuple[str, pd.DataFrame, pd.DataFrame, str]:
    bundle = load_bundle(path)
    snapshot = _snapshot(bundle)
    timeline = _timeline(bundle)
    opportunities, evidence = _opportunities(bundle)
    patient = _entries(bundle, "Patient")[0]
    dna_text, dna_sources = _dna_summary(patient["id"])
    source_markdown = "\n".join(
        ["### Evidence Sources", "- " + "\n- ".join((evidence + dna_sources)[:25] or ["FHIR bundle parsed successfully."]), "", f"### Genomics Detail\n{dna_text}"]
    )
    return snapshot, timeline, opportunities, source_markdown
