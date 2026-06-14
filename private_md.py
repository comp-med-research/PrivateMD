from __future__ import annotations

import json
import re
import math
import os
import urllib.error
import urllib.request
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
    facets: Tuple[str, ...]
    encounter_id: str
    resource: Dict[str, Any]


@dataclass(frozen=True)
class QueryPlan:
    original: str
    intent: str
    terms: Tuple[str, ...]
    entities: Tuple[str, ...]
    must_have: Tuple[str, ...]
    preferred_types: Tuple[str, ...]


RESOURCE_PRIORS = {
    "Condition": 1.35,
    "MedicationRequest": 1.3,
    "Observation": 1.2,
    "DiagnosticReport": 1.15,
    "CarePlan": 1.15,
    "Procedure": 1.05,
    "ImagingStudy": 1.05,
    "Encounter": 0.9,
    "Genomics": 1.2,
    "Trend": 1.25,
}

CLINICAL_ALIASES = {
    "a1c": {"a1c", "hba1c", "hemoglobin", "glycemic", "diabetes", "glucose"},
    "anticoagulation": {"warfarin", "inr", "prothrombin", "anticoagulation", "bleeding", "clot", "stroke"},
    "heart failure": {"chf", "congestive", "volume", "edema", "dyspnea"},
    "blood pressure": {"bp", "systolic", "diastolic", "hypertension"},
    "obesity": {"bmi", "weight", "obesity"},
    "genomics": {"gene", "variant", "pathogenic", "genomic", "dna"},
    "imaging": {"image", "imaging", "radiography", "modality", "dicom"},
    "kidney": {"kidney", "renal", "creatinine", "egfr", "albuminuria", "microalbumin"},
    "lipids": {"cholesterol", "ldl", "hdl", "triglyceride", "statin", "lipid"},
}

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "should", "review",
    "patient", "before", "after", "about", "into", "their", "there", "while", "when",
    "does", "have", "has", "are", "was", "were", "can", "could", "would",
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
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]


def _clinical_entities(text: str) -> Set[str]:
    tokens = set(_tokens(text))
    entities = set(tokens)
    lower = text.lower()
    for label, aliases in CLINICAL_ALIASES.items():
        phrase_hit = label in lower
        alias_hits = tokens.intersection(aliases)
        if phrase_hit or alias_hits:
            entities.add(label)
            entities.update(alias_hits)
    return entities


def _langextract_entities(text: str) -> Tuple[Set[str], str]:
    """Use Google LangExtract for query planning when explicitly enabled."""
    if os.getenv("PRIVATE_MD_USE_LANGEXTRACT", "").lower() not in {"1", "true", "yes"}:
        return set(), "off"
    try:
        lx = import_module("langextract")
        lx_data = import_module("langextract.data")
        lx_prompt_validation = import_module("langextract.prompt_validation")
    except ImportError:
        return set(), "not installed"

    prompt = (
        "Extract concise clinical query concepts. Classes: condition, medication, lab, "
        "procedure, imaging, gene, risk_factor, temporal_intent. Use exact text spans."
    )
    examples = [
        lx_data.ExampleData(
            text="What should I review for diabetes and anticoagulation before changing warfarin?",
            extractions=[
                lx_data.Extraction("condition", "diabetes"),
                lx_data.Extraction("medication", "warfarin"),
                lx_data.Extraction("risk_factor", "anticoagulation"),
            ],
        ),
        lx_data.ExampleData(
            text="Show the latest A1c trend and kidney protection opportunities.",
            extractions=[
                lx_data.Extraction("lab", "A1c"),
                lx_data.Extraction("condition", "kidney"),
                lx_data.Extraction("temporal_intent", "latest"),
            ],
        ),
    ]
    model_id = os.getenv("LANGEXTRACT_MODEL_ID", "gemma3:4b")
    model_url = os.getenv("LANGEXTRACT_MODEL_URL", "http://localhost:11434")
    status = f"enabled via {model_id}"
    if "gemini" not in model_id.lower():
        status += " local"
    resolver_params = {"suppress_parse_errors": True}
    language_model_params = {"format_type": "json"}
    kwargs = dict(
        text_or_documents=text[:2000],
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        model_url=model_url,
        extraction_passes=1,
        max_workers=1,
        max_char_buffer=800,
        temperature=0.0,
        resolver_params=resolver_params,
        language_model_params=language_model_params,
        prompt_validation_level=lx_prompt_validation.PromptValidationLevel.OFF,
        show_progress=False,
    )
    try:
        result = lx.extract(**kwargs)
    except Exception as exc:
        return set(), f"configured but unavailable ({exc.__class__.__name__})"

    extracted: Set[str] = set()
    for extraction in getattr(result, "extractions", []) or []:
        if getattr(extraction, "char_interval", True) is None:
            continue
        value = getattr(extraction, "extraction_text", "")
        if value:
            extracted.update(_tokens(value))
            extracted.add(value.lower())
    return extracted, status


def _query_plan(question: str) -> QueryPlan:
    langextract_terms, _ = _langextract_entities(question)
    entities = _clinical_entities(question) | langextract_terms
    terms = set(_tokens(question)) | entities
    lower = question.lower()
    intent = "focused_review"
    if any(term in lower for term in ["latest", "current", "now", "recent"]):
        intent = "latest_status"
    elif any(term in lower for term in ["trend", "trajectory", "over time", "journey"]):
        intent = "trend"
    elif any(term in lower for term in ["before", "change", "start", "stop", "switch"]):
        intent = "treatment_decision"

    preferred_types: List[str] = []
    if terms.intersection({"warfarin", "inr", "anticoagulation", "medication"}):
        preferred_types.extend(["MedicationRequest", "Observation", "Condition"])
    if terms.intersection({"a1c", "diabetes", "glucose", "kidney", "renal"}):
        preferred_types.extend(["Observation", "Condition", "CarePlan"])
    if terms.intersection({"image", "imaging", "dicom", "radiography"}):
        preferred_types.extend(["ImagingStudy", "DiagnosticReport"])
    if terms.intersection({"gene", "variant", "pathogenic", "dna", "genomic"}):
        preferred_types.append("Genomics")
    if not preferred_types:
        preferred_types = ["Condition", "MedicationRequest", "Observation", "CarePlan"]

    must_have = sorted(label for label, aliases in CLINICAL_ALIASES.items() if terms.intersection(aliases | {label}))
    return QueryPlan(
        original=question,
        intent=intent,
        terms=tuple(sorted(terms)),
        entities=tuple(sorted(entities)),
        must_have=tuple(must_have),
        preferred_types=tuple(dict.fromkeys(preferred_types)),
    )


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


def _encounter_id(resource: Dict[str, Any]) -> str:
    reference = resource.get("encounter", {}).get("reference", "")
    if ":" in reference:
        return reference.rsplit(":", 1)[-1]
    if "/" in reference:
        return reference.rsplit("/", 1)[-1]
    return reference


def _facets(resource: Dict[str, Any], text: str) -> Tuple[str, ...]:
    facets = {resource.get("resourceType", "")}
    lower = text.lower()
    for label, aliases in CLINICAL_ALIASES.items():
        if label in lower or set(_tokens(lower)).intersection(aliases):
            facets.add(label)
    if any(term in lower for term in ["value:", "hemoglobin", "systolic", "diastolic", "cholesterol"]):
        facets.add("measured")
    if any(term in lower for term in ["active", "completed", "stopped"]):
        facets.add("status")
    return tuple(sorted(facet for facet in facets if facet))


def _trend_chunks(bundle: Dict[str, Any]) -> List[EvidenceChunk]:
    observations = _entries(bundle, "Observation")
    groups = {
        "A1c trend": ["hemoglobin a1c", "hba1c"],
        "Blood pressure trend": ["systolic blood pressure", "diastolic blood pressure"],
        "BMI trend": ["body mass index"],
        "Renal function trend": ["creatinine", "glomerular", "egfr"],
    }
    chunks: List[EvidenceChunk] = []
    for title, aliases in groups.items():
        matches = [
            obs
            for obs in observations
            if any(alias in _coding_text(obs.get("code", {})).lower() for alias in aliases)
        ]
        if not matches:
            continue
        ordered = sorted(matches, key=lambda obs: _parse_date(obs.get("effectiveDateTime")), reverse=True)
        facts = [
            f"{_date(obs.get('effectiveDateTime'))}: {_coding_text(obs.get('code', {}))} {_quantity(obs)}"
            for obs in ordered[:8]
        ]
        text = f"Trend: {title}; latest-to-older values: " + " | ".join(facts)
        chunks.append(
            EvidenceChunk(
                chunk_id=f"Trend/{title.replace(' ', '_')}",
                resource_type="Trend",
                date=_resource_date(ordered[0]),
                title=title,
                text=text,
                source=f"Derived trend from {len(matches)} Observation resources",
                entities=tuple(sorted(_clinical_entities(text))),
                facets=_facets({"resourceType": "Trend"}, text),
                encounter_id="",
                resource={"resourceType": "Trend", "id": title, "members": [obs.get("id") for obs in ordered[:8]]},
            )
        )
    return chunks


def _genomics_chunks(bundle: Dict[str, Any]) -> List[EvidenceChunk]:
    patient = _entries(bundle, "Patient")[0]
    files = list(DNA_DIR.glob(f"*{patient['id']}_dna.csv"))
    if not files:
        return []
    dna = pd.read_csv(files[0])
    variants = dna[dna["VARIANT"] == True]  # noqa: E712
    if variants.empty:
        text = f"Genomics: no variant-positive rows found in {files[0].name}"
    else:
        pathogenic = variants[variants["CLINICAL_SIGNIFICANCE"].str.contains("Pathogenic", na=False)]
        genes = sorted(set(pathogenic["GENE"].dropna().astype(str)))[:12]
        top_rows = variants.head(8)
        details = [
            f"{row.GENE} {row.INDEX_PREFIX} {row.CLINICAL_SIGNIFICANCE}"
            for row in top_rows.itertuples()
        ]
        text = (
            f"Genomics: {len(variants)} variant-positive rows; {len(pathogenic)} pathogenic rows; "
            f"flagged genes: {', '.join(genes) or 'none'}; examples: {' | '.join(details)}"
        )
    return [
        EvidenceChunk(
            chunk_id=f"Genomics/{patient['id']}",
            resource_type="Genomics",
            date="",
            title="DNA variant summary",
            text=text,
            source=f"DNA/{files[0].name}",
            entities=tuple(sorted(_clinical_entities(text))),
            facets=("Genomics", "genomics"),
            encounter_id="",
            resource={"resourceType": "Genomics", "id": patient["id"]},
        )
    ]


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
                facets=_facets(resource, text),
                encounter_id=_encounter_id(resource),
                resource=resource,
            )
        )
    return chunks + _trend_chunks(bundle) + _genomics_chunks(bundle)


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


def _fielded_score(plan: QueryPlan, chunk: EvidenceChunk, tokenized: Dict[str, List[str]], avg_len: float, doc_count: int, document_frequency: Dict[str, int]) -> float:
    query_terms = list(plan.terms)
    text_score = _bm25_score(query_terms, tokenized[chunk.chunk_id], avg_len, doc_count, document_frequency)
    title_score = 2.5 * _bm25_score(query_terms, _tokens(chunk.title), 6.0, doc_count, document_frequency)
    entity_overlap = len(set(chunk.entities).intersection(plan.entities))
    facet_overlap = len(set(chunk.facets).intersection(plan.must_have))
    type_boost = 2.0 if chunk.resource_type in plan.preferred_types else 0.0
    resource_prior = RESOURCE_PRIORS.get(chunk.resource_type, 1.0)
    parsed = _parse_date(chunk.date)
    recency = min(max((parsed.year - 1950) / 90, 0), 0.45) if parsed != datetime.min else 0.0
    if plan.intent == "latest_status":
        recency *= 2.0
    trend_boost = 0.0
    if chunk.resource_type == "Trend" and set(chunk.facets).intersection(plan.must_have):
        trend_boost = 14.0
    elif plan.intent == "trend" and chunk.resource_type == "Trend":
        trend_boost = 8.0
    decision_boost = 1.5 if plan.intent == "treatment_decision" and chunk.resource_type in {"MedicationRequest", "Observation", "Condition"} else 0.0
    return (text_score + title_score + entity_overlap * 2.25 + facet_overlap * 1.6 + type_boost + recency + trend_boost + decision_boost) * resource_prior


def _chunk_similarity(left: EvidenceChunk, right: EvidenceChunk) -> float:
    left_terms = set(_tokens(left.text)) | set(left.entities)
    right_terms = set(_tokens(right.text)) | set(right.entities)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms.intersection(right_terms)) / len(left_terms.union(right_terms))


def _cluster_key(chunk: EvidenceChunk) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", chunk.title.lower()).strip()
    if chunk.resource_type == "Observation" and title:
        return f"Observation:{title}"
    return f"{chunk.resource_type}:{title or chunk.chunk_id}"


def _graph_expand(candidates: List[Tuple[float, EvidenceChunk]], all_chunks: List[EvidenceChunk]) -> List[Tuple[float, EvidenceChunk]]:
    selected_ids = {chunk.chunk_id for _, chunk in candidates}
    selected_encounters = {chunk.encounter_id for _, chunk in candidates if chunk.encounter_id}
    selected_facets = set().union(*(set(chunk.facets) for _, chunk in candidates[:8])) if candidates else set()
    expanded = list(candidates)
    for chunk in all_chunks:
        if chunk.chunk_id in selected_ids:
            continue
        encounter_match = chunk.encounter_id and chunk.encounter_id in selected_encounters
        facet_match = bool(set(chunk.facets).intersection(selected_facets))
        if encounter_match or facet_match and chunk.resource_type in {"Condition", "MedicationRequest", "CarePlan", "Trend"}:
            expanded.append((1.25 if encounter_match else 0.85, chunk))
            selected_ids.add(chunk.chunk_id)
    return expanded


def _mmr_select(scored: List[Tuple[float, EvidenceChunk]], top_k: int) -> List[Tuple[float, EvidenceChunk]]:
    if not scored:
        return []
    remaining = sorted(scored, key=lambda item: item[0], reverse=True)
    selected: List[Tuple[float, EvidenceChunk]] = []
    seen_sources: Set[str] = set()
    cluster_counts: Dict[str, int] = {}
    while remaining and len(selected) < top_k:
        best_index = 0
        best_score = -1e9
        for index, (score, chunk) in enumerate(remaining):
            cluster = _cluster_key(chunk)
            if cluster_counts.get(cluster, 0) >= 2:
                continue
            diversity_penalty = max((_chunk_similarity(chunk, chosen) for _, chosen in selected), default=0.0)
            cluster_penalty = 4.0 * cluster_counts.get(cluster, 0)
            source_bonus = 0.35 if chunk.resource_type not in seen_sources else 0.0
            mmr = 0.72 * score - 0.28 * diversity_penalty - cluster_penalty + source_bonus
            if mmr > best_score:
                best_score = mmr
                best_index = index
        if best_score == -1e9:
            break
        selected.append(remaining.pop(best_index))
        cluster_counts[_cluster_key(selected[-1][1])] = cluster_counts.get(_cluster_key(selected[-1][1]), 0) + 1
        seen_sources.add(selected[-1][1].resource_type)
    return selected


def retrieve_evidence(bundle: Dict[str, Any], question: str, top_k: int = 12) -> Tuple[List[Tuple[float, EvidenceChunk]], QueryPlan, str]:
    chunks = _rag_chunks(bundle)
    plan = _query_plan(question)
    _, langextract_state = _langextract_entities(question)
    tokenized = {chunk.chunk_id: _tokens(chunk.text) for chunk in chunks}
    doc_count = len(chunks)
    avg_len = sum(len(terms) for terms in tokenized.values()) / max(doc_count, 1)
    document_frequency: Dict[str, int] = {}
    for terms in tokenized.values():
        for term in set(terms):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    scored: List[Tuple[float, EvidenceChunk]] = []
    for chunk in chunks:
        score = _fielded_score(plan, chunk, tokenized, avg_len, doc_count, document_frequency)
        if score > 0:
            scored.append((score, chunk))
    if not scored:
        scored = [(0.1, chunk) for chunk in sorted(chunks, key=lambda item: item.date, reverse=True)[:top_k]]
    expanded = _graph_expand(sorted(scored, key=lambda item: item[0], reverse=True)[:30], chunks)
    deduped = {chunk.chunk_id: max(score, 0.01) for score, chunk in expanded}
    rescored = [(deduped[chunk.chunk_id], chunk) for chunk in chunks if chunk.chunk_id in deduped]
    return _mmr_select(rescored, top_k), plan, langextract_state


def _evidence_frame(scored_chunks: List[Tuple[float, EvidenceChunk]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": rank,
                "score": round(score, 3),
                "type": chunk.resource_type,
                "date": chunk.date,
                "evidence": chunk.text,
                "facets": ", ".join(chunk.facets),
                "source": chunk.source,
            }
            for rank, (score, chunk) in enumerate(scored_chunks, start=1)
        ]
    )


def _grounding_prompt(question: str, plan: QueryPlan, scored_chunks: List[Tuple[float, EvidenceChunk]]) -> str:
    context = "\n".join(
        f"[{i}] type={chunk.resource_type}; date={chunk.date or 'n/a'}; source={chunk.source}; evidence={chunk.text}"
        for i, (_, chunk) in enumerate(scored_chunks[:10], start=1)
    )
    return (
        "You are PrivateMD, a clinician-facing chart review copilot. Use only the cited local evidence. "
        "Do not diagnose, prescribe, or invent missing facts. If evidence is insufficient, say exactly what is missing. "
        "Write a concise clinical review with sections: Key evidence, Treatment review opportunities, Missing/uncertain data, Next chart checks. "
        "Every factual claim must cite bracket numbers like [1] or [2].\n\n"
        f"Question: {question}\n"
        f"Query intent: {plan.intent}\n"
        f"Clinical concepts: {', '.join(plan.must_have or plan.entities)}\n\n"
        f"Retrieved local evidence:\n{context}\n\n"
        "Answer:"
    )


def _ollama_gemma_answer(prompt: str) -> Optional[str]:
    model = os.getenv("PRIVATE_MD_OLLAMA_MODEL", "gemma3:4b")
    url = os.getenv("PRIVATE_MD_OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 500},
        }
    ).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    return (data.get("response") or "").strip() or None


def _hf_gemma_answer(prompt: str) -> Optional[str]:
    try:
        transformers = import_module("transformers")
    except ImportError:
        return None
    model_id = os.getenv("PRIVATE_MD_GEMMA_MODEL", "google/gemma-3-4b-it")
    try:
        pipe = transformers.pipeline("image-text-to-text", model=model_id, device_map="auto")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        output = pipe(text=messages, max_new_tokens=500, do_sample=False)
    except Exception:
        return None
    if output and isinstance(output, list):
        generated = output[0].get("generated_text", "")
        if isinstance(generated, list) and generated:
            return generated[-1].get("content", "").strip() or None
        return str(generated).strip() or None
    return None


def _local_gemma_answer(question: str, plan: QueryPlan, scored_chunks: List[Tuple[float, EvidenceChunk]]) -> Tuple[Optional[str], str]:
    prompt = _grounding_prompt(question, plan, scored_chunks)
    generator = os.getenv("PRIVATE_MD_GENERATOR", "ollama").lower()
    if generator in {"ollama", "gemma", "gemma-ollama"}:
        answer = _ollama_gemma_answer(prompt)
        if answer:
            return answer, f"Gemma 3 4B via Ollama ({os.getenv('PRIVATE_MD_OLLAMA_MODEL', 'gemma3:4b')})"
    if generator in {"hf", "transformers", "gemma-hf"}:
        answer = _hf_gemma_answer(prompt)
        if answer:
            return answer, f"Gemma 3 4B via Transformers ({os.getenv('PRIVATE_MD_GEMMA_MODEL', 'google/gemma-3-4b-it')})"
    return None, "Gemma 3 4B unavailable; deterministic synthesis"


def _deterministic_answer(scored_chunks: List[Tuple[float, EvidenceChunk]], plan: QueryPlan) -> str:
    by_type: Dict[str, List[EvidenceChunk]] = {}
    for _, chunk in scored_chunks:
        by_type.setdefault(chunk.resource_type, []).append(chunk)
    lines = ["**Key evidence**"]
    for index, (_, chunk) in enumerate(scored_chunks[:6], start=1):
        lines.append(f"- [{index}] {chunk.text}")
    lines.append("\n**Treatment review opportunities**")
    if "anticoagulation" in plan.must_have:
        lines.append("- Verify anticoagulation indication, recent INR/prothrombin evidence, bleeding risks, and medication interactions using the cited medication/lab evidence.")
    if "a1c" in plan.must_have or "kidney" in plan.must_have:
        lines.append("- Review glycemic trajectory, renal-protection signals, and whether diabetes care-plan evidence matches recent labs.")
    if "blood pressure" in plan.must_have:
        lines.append("- Review BP trajectory and follow-up/adherence context rather than relying on a single reading.")
    if len(lines) == 8:
        lines.append("- Use the retrieved condition, medication, lab, and trend evidence to decide which chart gap to inspect next.")
    lines.append("\n**Missing/uncertain data**")
    lines.append("- This local pass cannot confirm symptoms, adherence, contraindications, or clinician intent unless those appear in the cited FHIR bundle.")
    lines.append("\n**Next chart checks**")
    lines.append("- Open the cited FHIR sources and verify dates, status, and units before acting.")
    return "\n".join(lines)


def answer_question(path: str, question: str) -> Tuple[str, pd.DataFrame]:
    bundle = load_bundle(path)
    if not question.strip():
        question = "What are the most important issues in this chart?"
    scored_chunks, plan, langextract_state = retrieve_evidence(bundle, question)
    generated, generator_state = _local_gemma_answer(question, plan, scored_chunks)
    evidence_lines = [
        f"- [{rank}] {chunk.text}  \n  Source: `{chunk.source}`"
        for rank, (_, chunk) in enumerate(scored_chunks[:8], start=1)
    ]
    answer = generated or _deterministic_answer(scored_chunks, plan)
    return (
        "PrivateMD used an advanced local RAG pipeline over typed FHIR, trend, imaging, and genomics evidence chunks. "
        "This prototype does not diagnose or prescribe; it surfaces record-backed considerations for clinician review.\n\n"
        f"**Question:** {question}\n\n"
        f"**Pipeline:** LangExtract query planning `{langextract_state}`; fielded BM25; clinical alias expansion; resource priors; temporal intent scoring; graph expansion; MMR diversification; generation `{generator_state}`.\n\n"
        f"**Query Plan:** intent `{plan.intent}`; concepts `{', '.join(plan.must_have or plan.entities[:10])}`; preferred evidence `{', '.join(plan.preferred_types)}`.\n\n"
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
