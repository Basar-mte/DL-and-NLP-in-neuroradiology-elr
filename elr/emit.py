"""Emit an evidence-linked report as an interoperable artefact (Section 6.3).

The manuscript's claim is that a report-generating system is defined as much
by its interfaces as by its model, and that free text returned by an API is
not deployable. This module is the demonstration that the alternative is
constructible with existing standards rather than aspirational: it takes the
same ``Report`` object the metrics operate on and emits

  * a DICOM Comprehensive 3D SR object, findings coded against SNOMED CT,
    measurements retained as numeric content items, and the referenced image
    regions persisted as SCOORD/IMAGE content so each statement carries its
    evidence; and
  * an HL7 FHIR DiagnosticReport bundle whose observations carry the same
    codes, so a finding can trigger a follow-up recommendation or a registry
    entry rather than merely appearing in a document.

Nothing here is novel. That is the point: the substrate the formulation
requires already exists, and its absence from published systems is a choice
rather than a limitation.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import (
    ExplicitVRLittleEndian,
    UID,
    generate_uid,
)

from .model import Finding, Measurement, Report

# Comprehensive 3D SR Storage
COMPREHENSIVE_3D_SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.34"

# Coding scheme designators
SCT = "SCT"      # SNOMED CT
DCM = "DCM"      # DICOM Controlled Terminology
UCUM = "UCUM"    # Unified Code for Units of Measure


def _code(value: str, scheme: str, meaning: str) -> Dataset:
    """A DICOM Code Sequence item."""
    ds = Dataset()
    ds.CodeValue = value
    ds.CodingSchemeDesignator = scheme
    ds.CodeMeaning = meaning
    return ds


def _container(concept: Dataset, relationship: Optional[str] = None) -> Dataset:
    ds = Dataset()
    ds.ValueType = "CONTAINER"
    ds.ConceptNameCodeSequence = DicomSequence([concept])
    ds.ContinuityOfContent = "SEPARATE"
    if relationship:
        ds.RelationshipType = relationship
    ds.ContentSequence = DicomSequence([])
    return ds


def _text(concept: Dataset, value: str, relationship: str = "CONTAINS") -> Dataset:
    ds = Dataset()
    ds.RelationshipType = relationship
    ds.ValueType = "TEXT"
    ds.ConceptNameCodeSequence = DicomSequence([concept])
    ds.TextValue = value
    return ds


def _code_item(concept: Dataset, value: Dataset, relationship: str = "CONTAINS") -> Dataset:
    ds = Dataset()
    ds.RelationshipType = relationship
    ds.ValueType = "CODE"
    ds.ConceptNameCodeSequence = DicomSequence([concept])
    ds.ConceptCodeSequence = DicomSequence([value])
    return ds


def _num(concept: Dataset, m: Measurement, relationship: str = "CONTAINS") -> Dataset:
    ds = Dataset()
    ds.RelationshipType = relationship
    ds.ValueType = "NUM"
    ds.ConceptNameCodeSequence = DicomSequence([concept])
    mv = Dataset()
    mv.NumericValue = str(m.value)
    mv.MeasurementUnitsCodeSequence = DicomSequence(
        [_code(m.ucum or m.unit, UCUM, m.unit)]
    )
    ds.MeasuredValueSequence = DicomSequence([mv])
    return ds


def _scoord(finding: Finding, relationship: str = "CONTAINS") -> Optional[Dataset]:
    """The image region supporting the finding, as a SCOORD content item.

    This is the content item that makes the report auditable. Without it the
    statement is prose; with it, a reader can resolve the assertion back to
    the pixels it was made from.
    """
    if finding.region is None:
        return None
    x0, y0, x1, y1 = finding.region.bbox
    ds = Dataset()
    ds.RelationshipType = relationship
    ds.ValueType = "SCOORD"
    ds.ConceptNameCodeSequence = DicomSequence(
        [_code("111030", DCM, "Image Region")]
    )
    ds.GraphicType = "POLYLINE"
    ds.GraphicData = [x0, y0, x1, y0, x1, y1, x0, y1, x0, y0]

    ref = Dataset()
    ref.RelationshipType = "SELECTED FROM"
    ref.ValueType = "IMAGE"
    ref_img = Dataset()
    ref_img.ReferencedSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.4")  # MR Image Storage
    ref_img.ReferencedSOPInstanceUID = UID(finding.region.series_uid)
    ref_img.ReferencedFrameNumber = str(finding.region.frame)
    ref.ReferencedSOPSequence = DicomSequence([ref_img])
    ds.ContentSequence = DicomSequence([ref])
    return ds


def build_sr(
    report: Report,
    patient_id: str = "ANON",
    patient_name: str = "Anonymous^Patient",
    manufacturer: str = "elr reference implementation",
    verifying_observer: Optional[str] = None,
) -> FileDataset:
    """Build a DICOM Comprehensive 3D SR from an evidence-linked Report.

    ``verifying_observer`` records the radiologist who verified and signed the
    report. The manuscript is explicit that this person remains responsible
    for the content, so the field is part of the artefact rather than an
    optional extra.
    """
    now = datetime.now()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = UID(COMPREHENSIVE_3D_SR_SOP_CLASS)
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.StudyInstanceUID = report.study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "SR"
    ds.SeriesNumber = 9001
    ds.InstanceNumber = 1
    ds.Manufacturer = manufacturer
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime
    ds.SpecificCharacterSet = "ISO_IR 192"

    # An unverified SR is explicitly marked as such. A drafting system's
    # output is not a signed report until a radiologist verifies it.
    if verifying_observer:
        ds.VerificationFlag = "VERIFIED"
        obs = Dataset()
        obs.VerifyingOrganization = "Institution"
        obs.VerificationDateTime = now.strftime("%Y%m%d%H%M%S")
        obs.VerifyingObserverName = verifying_observer
        ds.VerifyingObserverSequence = DicomSequence([obs])
    else:
        ds.VerificationFlag = "UNVERIFIED"

    ds.CompletionFlag = "PARTIAL" if report.abstained else "COMPLETE"

    # Root container: Radiology Report
    ds.ValueType = "CONTAINER"
    ds.ConceptNameCodeSequence = DicomSequence(
        [_code("11540-2", "LN", "Radiology Report")]
    )
    ds.ContinuityOfContent = "SEPARATE"

    content: list[Dataset] = []

    if report.narrative:
        content.append(
            _text(_code("121070", DCM, "Findings"), report.narrative)
        )

    for i, f in enumerate(report.findings, start=1):
        item = _container(_code("125007", DCM, "Measurement Group"), "CONTAINS")
        sub: list[Dataset] = []

        # The coded concept: this is what makes the finding machine-actionable.
        sub.append(
            _code_item(
                _code("121071", DCM, "Finding"),
                _code(f.code, SCT, f.display),
            )
        )
        sub.append(_text(_code("121073", DCM, "Finding Description"), f.text))

        # Presence or explicit absence, retained rather than inferred from
        # prose. SCT 408729009 "Finding context" with the standard context
        # values; a negated finding must survive into the structured report as
        # a coded absence, because "no acute haemorrhage" and silence about
        # haemorrhage are different clinical statements.
        sub.append(
            _code_item(
                _code("408729009", SCT, "Finding context"),
                _code("410516002", SCT, "Known absent") if f.absent
                else _code("410515003", SCT, "Known present"),
            )
        )

        if f.measurement is not None:
            sub.append(
                _num(_code(f.measurement.name, SCT, f.measurement.name), f.measurement)
            )

        region = _scoord(f)
        if region is not None:
            sub.append(region)

        if f.confidence is not None:
            conf = Measurement(name="Confidence", value=f.confidence, unit="1", ucum="1")
            sub.append(_num(_code("113058", DCM, "Confidence"), conf))

        item.ContentSequence = DicomSequence(sub)
        content.append(item)

    ds.ContentSequence = DicomSequence(content)
    return ds


def build_fhir(report: Report, patient_id: str = "ANON") -> dict:
    """Build an HL7 FHIR DiagnosticReport bundle carrying the same findings.

    Each finding becomes an Observation coded with the same SNOMED CT concept
    used in the SR, so the two artefacts describe one report rather than two
    parallel ones. Measurements travel as valueQuantity with a UCUM unit; the
    image region travels in ``derivedFrom`` so the evidence link survives the
    hop into the electronic record.
    """
    entries: list[dict] = []
    obs_refs: list[dict] = []

    for i, f in enumerate(report.findings, start=1):
        oid = f"obs-{i}"
        obs: dict = {
            "resourceType": "Observation",
            "id": oid,
            "status": "preliminary" if not report.abstained else "registered",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "imaging",
                    "display": "Imaging",
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": f.code,
                    "display": f.display,
                }],
                "text": f.text,
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "interpretation": [{
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": "410516002" if f.absent else "410515003",
                    "display": "Known absent" if f.absent else "Known present",
                }]
            }],
        }
        if f.measurement is not None:
            obs["valueQuantity"] = {
                "value": f.measurement.value,
                "unit": f.measurement.unit,
                "system": "http://unitsofmeasure.org",
                "code": f.measurement.ucum or f.measurement.unit,
            }
        if f.region is not None:
            # The evidence link, carried into the record rather than dropped
            # at the boundary.
            obs["derivedFrom"] = [{
                "reference": f"ImagingStudy/{report.study_uid}",
                "display": (
                    f"series {f.region.series_uid} frame {f.region.frame} "
                    f"bbox {tuple(round(v, 1) for v in f.region.bbox)}"
                ),
            }]
        else:
            obs["note"] = [{"text": "No image region referenced: statement is not evidence-linked."}]
        if f.confidence is not None:
            obs.setdefault("extension", []).append({
                "url": "http://example.org/fhir/StructureDefinition/model-confidence",
                "valueDecimal": f.confidence,
            })
        entries.append({"resource": obs, "request": {"method": "POST", "url": "Observation"}})
        obs_refs.append({"reference": f"Observation/{oid}"})

    diagnostic = {
        "resourceType": "DiagnosticReport",
        "id": "dr-1",
        "status": "preliminary",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "RAD",
                "display": "Radiology",
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "11540-2",
                "display": "Radiology Report",
            }]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "result": obs_refs,
        "conclusion": report.narrative,
    }
    entries.insert(0, {"resource": diagnostic,
                       "request": {"method": "POST", "url": "DiagnosticReport"}})

    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def write_sr(report: Report, path: str, **kw) -> str:
    ds = build_sr(report, **kw)
    ds.save_as(path, enforce_file_format=True)
    return path


def write_fhir(report: Report, path: str, **kw) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_fhir(report, **kw), fh, indent=2)
    return path
