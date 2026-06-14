from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATA_DIR = Path("data/coherent-11-07-2022")
FHIR_DIR = DATA_DIR / "fhir"
DNA_DIR = DATA_DIR / "dna"


@dataclass(frozen=True)
class PatientChoice:
    label: str
    path: str


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


def answer_question(path: str, question: str) -> str:
    bundle = load_bundle(path)
    question_terms = [term for term in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(term) > 2]
    lines = _context_lines(bundle)
    scored = []
    for line in lines:
        lower = line.lower()
        score = sum(term in lower for term in question_terms)
        if score:
            scored.append((score, line))
    top = [line for _, line in sorted(scored, reverse=True)[:8]] or lines[:8]
    if not question.strip():
        question = "What are the most important issues in this chart?"
    return (
        "PrivateMD found the most relevant local chart evidence below. "
        "This prototype does not diagnose or prescribe; it surfaces record-backed considerations for clinician review.\n\n"
        f"**Question:** {question}\n\n"
        "**Grounded answer:**\n"
        + "\n".join(f"- {line}" for line in top)
        + "\n\n**Clinician next step:** verify the cited FHIR sources in the patient context before acting."
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
