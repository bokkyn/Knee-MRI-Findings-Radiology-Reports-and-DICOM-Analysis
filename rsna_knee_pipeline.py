"""Auditable 2.5D training and inference pipeline for RSNA knee MRI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pydicom.tag import Tag
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
PREPROCESS_VERSION = "knee-mri-2.5d-v1"
PARSER_ACCEPTED_STATUSES = {"SILVER"}
PARSER_BLOCKING_STATUSES = {"UNCERTAIN", "CONFLICT"}
QWEN_ACCEPTED_SOURCES = {"LLM_POS", "LLM_NEG"}
IMAGE_METADATA_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
    "SeriesDescription",
    "ProtocolName",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "InstanceNumber",
    "NumberOfFrames",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PhotometricInterpretation",
    "BitsStored",
    "PixelRepresentation",
    "RescaleSlope",
    "RescaleIntercept",
    "PatientID",
]

LOGGER = logging.getLogger("rsna_knee")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _valid_binary(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number not in (0.0, 1.0):
        return None
    return number


def _valid_score(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _unique_index(frame: pd.DataFrame, key: str, source: str) -> dict[str, dict[str, Any]]:
    if key not in frame.columns:
        raise ValueError(f"{source} is missing required key column {key!r}")
    values = frame[key].fillna("").astype(str).str.strip()
    empty_count = int(values.eq("").sum())
    if empty_count:
        raise ValueError(f"{source} contains {empty_count} rows with an empty {key}")
    dupes = values[values.duplicated(keep=False) & values.ne("")]
    if not dupes.empty:
        raise ValueError(
            f"{source} contains {dupes.nunique()} duplicate {key} values; "
            "refusing ambiguous label joins"
        )
    indexed: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        uid = _clean(row.get(key))
        if uid:
            indexed[uid] = row
    return indexed


def merge_labels(
    train_csv: Path,
    parser_csv: Path | None,
    qwen_csv: Path | None,
    parser_confidence_threshold: float = 0.85,
    parser_research_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one auditable long-form row per study and target."""
    if not 0.0 <= parser_confidence_threshold <= 1.0:
        raise ValueError("parser_confidence_threshold must be in [0, 1]")
    gold_frame = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
    missing_targets = [target for target in TARGETS if target not in gold_frame.columns]
    if missing_targets:
        raise ValueError(f"train.csv is missing target columns: {missing_targets}")
    gold = _unique_index(gold_frame, "StudyInstanceUID", str(train_csv))
    parser = (
        _unique_index(pd.read_csv(parser_csv, dtype=str, keep_default_na=False), "StudyInstanceUID", str(parser_csv))
        if parser_csv
        else {}
    )
    research = (
        _unique_index(
            pd.read_csv(parser_research_csv, dtype=str, keep_default_na=False),
            "StudyInstanceUID",
            str(parser_research_csv),
        )
        if parser_research_csv
        else {}
    )
    qwen = (
        _unique_index(pd.read_csv(qwen_csv, dtype=str, keep_default_na=False), "StudyInstanceUID", str(qwen_csv))
        if qwen_csv
        else {}
    )

    records: list[dict[str, Any]] = []
    for uid, gold_row in gold.items():
        parser_row = parser.get(uid, {})
        research_row = research.get(uid, {})
        qwen_row = qwen.get(uid, {})
        for target in TARGETS:
            gold_raw = _clean(gold_row.get(target))
            gold_value = _valid_binary(gold_raw)
            gold_invalid = bool(gold_raw) and gold_value is None
            parser_value = _valid_binary(parser_row.get(target))
            parser_status = _clean(parser_row.get(f"{target}_status")).upper()
            parser_confidence = _valid_score(parser_row.get(f"{target}_confidence"))
            parser_key = "Bakers" if target == "Baker's" else target.replace(" ", "_")
            research_status = _clean(
                research_row.get(f"{parser_key}_NLP_status")
            ).upper()
            research_value = _valid_binary(research_row.get(f"{parser_key}_NLP"))
            qwen_score = _valid_score(qwen_row.get(f"{target}_LLM"))
            qwen_source = _clean(qwen_row.get(f"{target}_SOURCE")).upper()
            source = ""
            status = ""
            value: float | None = None
            confidence: float | None = None
            accepted = False
            reason = ""

            
            if gold_invalid:
                source, status = "GOLD", "INVALID"
                reason = f"invalid non-empty value in source train.csv: {gold_raw!r}"
            elif gold_value is not None:
                source, status, value, confidence, accepted = "GOLD", "GOLD", gold_value, 1.0, True
                if research_status in PARSER_BLOCKING_STATUSES:
                    reason = f"GOLD retained; parser {research_status} output is excluded"
                elif parser_status == "GOLD" and parser_value is not None and parser_value != gold_value:
                    reason = "GOLD retained; generated table disagrees with source train.csv"
                else:
                    reason = "authoritative train.csv label"
            elif research_status in PARSER_BLOCKING_STATUSES:
                source, status = "PARSER", research_status
                reason = f"parser research status {research_status} is explicitly excluded"
            elif parser_status in PARSER_BLOCKING_STATUSES:
                source, status = "PARSER", parser_status
                reason = f"parser status {parser_status} is explicitly excluded"
            elif parser_status in PARSER_ACCEPTED_STATUSES:
                if parser_value is None:
                    source, status, reason = "PARSER", parser_status, "accepted status has invalid or missing value"
                elif parser_confidence is None:
                    source, status, reason = "PARSER", parser_status, "confidence is missing or invalid"
                elif parser_confidence < parser_confidence_threshold:
                    source, status, confidence = "PARSER", parser_status, parser_confidence
                    reason = "confidence is below configured parser threshold"
                elif research and research_status not in {"POSITIVE_HIGH", "NEGATIVE_HIGH"}:
                    source, status = "PARSER", research_status or parser_status
                    reason = "labels_final SILVER status conflicts with parser research status"
                elif research_status in {"POSITIVE_HIGH", "NEGATIVE_HIGH"} and (
                    research_value is None
                    or research_value != parser_value
                    or (research_status == "POSITIVE_HIGH" and research_value != 1.0)
                    or (research_status == "NEGATIVE_HIGH" and research_value != 0.0)
                ):
                    source, status, reason = "PARSER", "CONFLICT", "parser value/status disagree across generated tables"
                else:
                    source, status, value, confidence, accepted = (
                        "SILVER",
                        parser_status,
                        parser_value,
                        parser_confidence,
                        True,
                    )
                    reason = "parser SILVER status and confidence threshold accepted"
            elif parser_status and parser_status not in {
                "UNRESOLVED",
                "SKIPPED_LANGUAGE",
                "UNKNOWN",
                "MISSING",
            }:
                source, status = "PARSER", parser_status
                reason = f"unrecognized parser status {parser_status!r}; masked"
            elif qwen_source in QWEN_ACCEPTED_SOURCES and qwen_score is not None:
                expected_source = "LLM_POS" if qwen_score >= 0.75 else "LLM_NEG" if qwen_score <= 0.25 else ""
                if qwen_source != expected_source:
                    source, status = "QWEN", qwen_source
                    reason = "Qwen source category does not agree with documented score threshold"
                else:
                    source, status, value, confidence, accepted = (
                        "QWEN",
                        qwen_source,
                        qwen_score,
                        None,
                        True,
                    )
                    reason = "continuous Qwen score retained as an opt-in soft target"
            elif qwen_source:
                source, status = "QWEN", qwen_source
                reason = "Qwen score/status is uncertain, invalid, or not explicitly accepted"
            elif parser_status:
                source, status = "PARSER", parser_status
                reason = f"parser status {parser_status!r} is not an accepted label"
            else:
                source, status, reason = "NONE", "MISSING", "no usable source label"

            records.append(
                {
                    "StudyInstanceUID": uid,
                    "target": target,
                    "label_value": value,
                    "label_source": source,
                    "label_status": status or "MISSING",
                    "parser_source_status": research_status or parser_status,
                    "confidence": confidence,
                    "is_gold": source == "GOLD" and accepted,
                    "is_accepted": accepted,
                    "is_masked": not accepted,
                    "reason": reason,
                }
            )

    merged = pd.DataFrame.from_records(records)
    stats: list[dict[str, Any]] = []
    for target in TARGETS:
        rows = merged[merged["target"] == target]
        gold_rows = rows[rows["label_source"] == "GOLD"]
        silver_rows = rows[rows["label_source"] == "SILVER"]
        qwen_rows = rows[rows["label_source"] == "QWEN"]
        stats.append(
            {
                "target": target,
                "gold_positive": int((gold_rows["label_value"] == 1).sum()),
                "gold_negative": int((gold_rows["label_value"] == 0).sum()),
                "accepted_silver_positive": int((silver_rows["label_value"] == 1).sum()),
                "accepted_silver_negative": int((silver_rows["label_value"] == 0).sum()),
                "accepted_qwen_soft": int(qwen_rows["is_accepted"].sum()),
                "excluded_uncertain": int(
                    (
                        (rows["parser_source_status"] == "UNCERTAIN")
                        | (rows["label_status"] == "UNCERTAIN")
                    ).sum()
                    + (rows["label_status"] == "LLM_UNCERTAIN").sum()
                ),
                "excluded_conflict": int(
                    (
                        (rows["parser_source_status"] == "CONFLICT")
                        | (rows["label_status"] == "CONFLICT")
                    ).sum()
                ),
                "missing_or_invalid": int(
                    (
                        rows["label_status"].isin(
                            {
                                "MISSING",
                                "INVALID",
                                "UNRESOLVED",
                                "SKIPPED_LANGUAGE",
                                "UNKNOWN",
                            }
                        )
                        | rows["reason"].str.contains(
                            "invalid|missing",
                            case=False,
                            regex=True,
                        )
                    ).sum()
                ),
                "other_masked": int(
                    (
                        rows["is_masked"]
                        & (
                            ~rows["label_status"].isin(
                                {
                                    "UNCERTAIN",
                                    "LLM_UNCERTAIN",
                                    "CONFLICT",
                                    "MISSING",
                                    "INVALID",
                                    "UNRESOLVED",
                                    "SKIPPED_LANGUAGE",
                                    "UNKNOWN",
                                }
                            )
                        )
                    ).sum()
                ),
            }
        )
    return merged, pd.DataFrame(stats)


def save_label_artifacts(
    merged: pd.DataFrame, stats: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "labels_merged.csv", index=False)
    stats.to_csv(output_dir / "label_statistics.csv", index=False)


def _sequence(value: Any, expected: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if len(values) != expected or not all(math.isfinite(v) for v in values):
        return None
    return values


@dataclass(frozen=True)
class DicomInstance:
    path: str
    study_uid: str
    series_uid: str
    sop_uid: str
    modality: str
    series_description: str
    protocol_name: str
    manufacturer: str
    manufacturer_model: str
    magnetic_field_strength: str
    instance_number: int | None
    number_of_frames: int
    pixel_data_present: bool
    rows: int | None
    columns: int | None
    pixel_spacing: tuple[float, ...] | None
    slice_thickness: float | None
    spacing_between_slices: float | None
    image_position: tuple[float, ...] | None
    image_orientation: tuple[float, ...] | None
    photometric_interpretation: str
    bits_stored: int | None
    pixel_representation: int | None
    rescale_slope: float
    rescale_intercept: float
    transfer_syntax_uid: str
    patient_group: str


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_text(value: Any) -> str:
    text = _clean(value)
    return text.replace("\n", " ").replace("\r", " ")


def read_dicom_metadata(path: Path) -> DicomInstance:
    ds = pydicom.dcmread(
        str(path),
        defer_size=1024,
        force=False,
        specific_tags=[*IMAGE_METADATA_TAGS, "PixelData"],
    )
    transfer = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "")
    patient = _safe_text(getattr(ds, "PatientID", ""))
    return DicomInstance(
        path=str(path.resolve()),
        study_uid=_safe_text(getattr(ds, "StudyInstanceUID", "")),
        series_uid=_safe_text(getattr(ds, "SeriesInstanceUID", "")),
        sop_uid=_safe_text(getattr(ds, "SOPInstanceUID", "")),
        modality=_safe_text(getattr(ds, "Modality", "")),
        series_description=_safe_text(getattr(ds, "SeriesDescription", "")),
        protocol_name=_safe_text(getattr(ds, "ProtocolName", "")),
        manufacturer=_safe_text(getattr(ds, "Manufacturer", "")),
        manufacturer_model=_safe_text(getattr(ds, "ManufacturerModelName", "")),
        magnetic_field_strength=_safe_text(getattr(ds, "MagneticFieldStrength", "")),
        instance_number=_optional_int(getattr(ds, "InstanceNumber", None)),
        number_of_frames=max(1, _optional_int(getattr(ds, "NumberOfFrames", 1)) or 1),
        pixel_data_present=Tag(0x7FE00010) in ds,
        rows=_optional_int(getattr(ds, "Rows", None)),
        columns=_optional_int(getattr(ds, "Columns", None)),
        pixel_spacing=_sequence(getattr(ds, "PixelSpacing", None), 2),
        slice_thickness=_optional_float(getattr(ds, "SliceThickness", None)),
        spacing_between_slices=_optional_float(getattr(ds, "SpacingBetweenSlices", None)),
        image_position=_sequence(getattr(ds, "ImagePositionPatient", None), 3),
        image_orientation=_sequence(getattr(ds, "ImageOrientationPatient", None), 6),
        photometric_interpretation=_safe_text(
            getattr(ds, "PhotometricInterpretation", "")
        ),
        bits_stored=_optional_int(getattr(ds, "BitsStored", None)),
        pixel_representation=_optional_int(getattr(ds, "PixelRepresentation", None)),
        rescale_slope=_optional_float(getattr(ds, "RescaleSlope", 1.0), 1.0) or 1.0,
        rescale_intercept=_optional_float(getattr(ds, "RescaleIntercept", 0.0), 0.0) or 0.0,
        transfer_syntax_uid=_safe_text(transfer),
        patient_group=patient,
    )


def scan_dicom_files(
    roots: Sequence[Path], max_files: int | None = None
) -> tuple[list[DicomInstance], list[dict[str, str]]]:
    records: list[DicomInstance] = []
    errors: list[dict[str, str]] = []
    seen: set[Path] = set()
    index = 0
    for root in roots:
        if not root.exists():
            LOGGER.warning("DICOM root does not exist: %s", root)
            continue
        if root.is_file():
            paths: Iterable[Path] = [root]
        else:
            def walk_files() -> Iterable[Path]:
                for current, dirs, files in os.walk(root):
                    dirs.sort()
                    for filename in sorted(files):
                        yield Path(current) / filename

            paths = walk_files()
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if max_files is not None and index >= max_files:
                return records, errors
            index += 1
            try:
                record = read_dicom_metadata(path)
                if not record.study_uid or not record.series_uid:
                    errors.append(
                        {"path": str(path), "error": "missing StudyInstanceUID or SeriesInstanceUID"}
                    )
                    continue
                records.append(record)
            except Exception as exc:
                errors.append(
                    {"path": str(path), "error": f"{type(exc).__name__}: {str(exc)[:240]}"}
                )
            if index % 10000 == 0:
                LOGGER.info("Read DICOM metadata for %d files", index)
    return records, errors


def _frame_position(record: DicomInstance, frame_index: int) -> float | None:
    if not record.image_position or not record.image_orientation:
        return None
    orientation = record.image_orientation
    row = np.asarray(orientation[:3], dtype=np.float64)
    column = np.asarray(orientation[3:], dtype=np.float64)
    normal = np.cross(row, column)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm < 1e-8:
        return None
    position = np.asarray(record.image_position, dtype=np.float64)
    return float(np.dot(position, normal / norm)) + frame_index * 1e-5


def sort_series_instances(
    instances: Sequence[DicomInstance],
) -> tuple[list[tuple[DicomInstance, int]], str]:
    """Sort files/frames geometrically, with an explicit documented fallback."""
    frames = [
        (instance, frame)
        for instance in instances
        for frame in range(instance.number_of_frames)
    ]
    geometric = [
        _frame_position(instance, frame) for instance, frame in frames
    ]
    if frames and all(position is not None for position in geometric):
        return (
            [
                item
                for _, item in sorted(
                    zip(geometric, frames), key=lambda pair: float(pair[0])
                )
            ],
            "ImagePositionPatient projection on slice normal",
        )
    return (
        sorted(
            frames,
            key=lambda item: (
                item[0].instance_number is None,
                item[0].instance_number if item[0].instance_number is not None else 0,
                item[1],
                item[0].path,
            ),
        ),
        "InstanceNumber, frame index, then path (geometry unavailable)",
    )


def _as_csv_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return json.dumps(value)
    return value


def write_dicom_audit(
    instances: Sequence[DicomInstance],
    errors: Sequence[dict[str, str]],
    output_dir: Path,
    label_uids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    sop_counts = Counter(item.sop_uid for item in instances if item.sop_uid)
    for record in instances:
        item = {
            "path": record.path,
            "StudyInstanceUID": record.study_uid,
            "SeriesInstanceUID": record.series_uid,
            "SOPInstanceUID": record.sop_uid,
            "Modality": record.modality,
            "SeriesDescription": record.series_description,
            "ProtocolName": record.protocol_name,
            "Manufacturer": record.manufacturer,
            "ManufacturerModelName": record.manufacturer_model,
            "MagneticFieldStrength": record.magnetic_field_strength,
            "InstanceNumber": record.instance_number,
            "NumberOfFrames": record.number_of_frames,
            "PixelDataPresent": record.pixel_data_present,
            "Rows": record.rows,
            "Columns": record.columns,
            "PixelSpacing": _as_csv_value(record.pixel_spacing),
            "SliceThickness": record.slice_thickness,
            "SpacingBetweenSlices": record.spacing_between_slices,
            "ImagePositionPatient": _as_csv_value(record.image_position),
            "ImageOrientationPatient": _as_csv_value(record.image_orientation),
            "PhotometricInterpretation": record.photometric_interpretation,
            "BitsStored": record.bits_stored,
            "PixelRepresentation": record.pixel_representation,
            "RescaleSlope": record.rescale_slope,
            "RescaleIntercept": record.rescale_intercept,
            "TransferSyntaxUID": record.transfer_syntax_uid,
        }
        item["duplicate_sop_instance_uid"] = bool(
            record.sop_uid and sop_counts[record.sop_uid] > 1
        )
        item["label_match"] = (
            record.study_uid in label_uids if label_uids is not None else None
        )
        audit_rows.append(item)
    audit = pd.DataFrame(audit_rows)

    series_rows: list[dict[str, Any]] = []
    study_rows: list[dict[str, Any]] = []
    by_series: dict[tuple[str, str], list[DicomInstance]] = defaultdict(list)
    by_study: dict[str, list[DicomInstance]] = defaultdict(list)
    for record in instances:
        by_series[(record.study_uid, record.series_uid)].append(record)
        by_study[record.study_uid].append(record)
    for (study_uid, series_uid), rows in sorted(by_series.items()):
        ordered, ordering = sort_series_instances(rows)
        series_rows.append(
            {
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "file_count": len(rows),
                "frame_count": len(ordered),
                "distinct_sop_count": len({x.sop_uid for x in rows if x.sop_uid}),
                "modalities": "|".join(sorted({x.modality for x in rows if x.modality})),
                "series_descriptions": "|".join(
                    sorted({x.series_description for x in rows if x.series_description})
                ),
                "manufacturers": "|".join(sorted({x.manufacturer for x in rows if x.manufacturer})),
                "dimensions": "|".join(
                    sorted({f"{x.rows}x{x.columns}" for x in rows if x.rows and x.columns})
                ),
                "has_pixel_dimensions": any(x.rows and x.columns for x in rows),
                "has_pixel_data": any(x.pixel_data_present for x in rows),
                "has_complete_geometry": all(
                    x.image_position is not None and x.image_orientation is not None for x in rows
                ),
                "ordering_method": ordering,
                "label_match": study_uid in label_uids if label_uids is not None else None,
            }
        )
    for study_uid, rows in sorted(by_study.items()):
        study_rows.append(
            {
                "StudyInstanceUID": study_uid,
                "file_count": len(rows),
                "series_count": len({x.series_uid for x in rows}),
                "frame_count": sum(x.number_of_frames for x in rows),
                "files_without_pixel_data": sum(
                    not x.pixel_data_present for x in rows
                ),
                "modalities": "|".join(sorted({x.modality for x in rows if x.modality})),
                "label_match": study_uid in label_uids if label_uids is not None else None,
            }
        )
    series = pd.DataFrame(series_rows)
    studies = pd.DataFrame(study_rows)
    pd.DataFrame(errors, columns=["path", "error"]).to_csv(
        output_dir / "dicom_errors.csv", index=False
    )
    audit.to_csv(output_dir / "dicom_audit.csv", index=False)
    series.to_csv(output_dir / "series_summary.csv", index=False)
    studies.to_csv(output_dir / "study_summary.csv", index=False)
    return audit, series, studies


def _parse_bool(value: Any) -> bool | None:
    text = _clean(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def select_series(
    instances: Sequence[DicomInstance], series_metadata: pd.DataFrame | None
) -> tuple[dict[str, list[tuple[DicomInstance, int]]], pd.DataFrame]:
    metadata: dict[str, dict[str, Any]] = {}
    if series_metadata is not None and not series_metadata.empty:
        required = {"StudyInstanceUID", "SeriesInstanceUID"}
        if required.issubset(series_metadata.columns):
            for row in series_metadata.to_dict(orient="records"):
                series_uid = _clean(row.get("SeriesInstanceUID"))
                if series_uid:
                    metadata[series_uid] = row
    grouped: dict[tuple[str, str], list[DicomInstance]] = defaultdict(list)
    for item in instances:
        grouped[(item.study_uid, item.series_uid)].append(item)

    candidates: dict[str, list[tuple[int, str, list[tuple[DicomInstance, int]], dict[str, Any]]]] = defaultdict(list)
    for (study_uid, series_uid), rows in grouped.items():
        ordered, ordering = sort_series_instances(rows)
        meta = metadata.get(series_uid, {})
        fluid = _parse_bool(meta.get("Fluid_Sensitive"))
        fat_suppression = _parse_bool(meta.get("Fat_Suppression"))
        plane = _clean(meta.get("Anatomical_Plane")).title()
        rank = (
            (40 if fluid else 0)
            + (30 if fat_suppression else 0)
            + {"Sagittal": 20, "Coronal": 12, "Axial": 5}.get(plane, 0)
            + (5 if any(r.modality.upper() == "MR" for r in rows) else 0)
            + min(len(ordered), 100) / 100.0
        )
        candidates[study_uid].append(
            (
                int(rank * 100),
                series_uid,
                ordered,
                {
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": series_uid,
                    "file_count": len(rows),
                    "frame_count": len(ordered),
                    "Fluid_Sensitive": fluid,
                    "Fat_Suppression": fat_suppression,
                    "Anatomical_Plane": plane,
                    "selection_score": rank,
                    "ordering_method": ordering,
                },
            )
        )

    selected: dict[str, list[tuple[DicomInstance, int]]] = {}
    selection_rows: list[dict[str, Any]] = []
    for study_uid, options in sorted(candidates.items()):
        options.sort(key=lambda option: (-option[0], option[1]))
        for index, (_, series_uid, ordered, details) in enumerate(options):
            is_selected = index == 0
            reason = (
                "highest available fluid-sensitive/fat-suppressed/anatomical-plane score"
                if is_selected
                else "lower-ranked candidate; initial model uses one series per study"
            )
            selection_rows.append(
                {**details, "selected": is_selected, "reason": reason}
            )
            if is_selected:
                selected[study_uid] = ordered
    if series_metadata is not None and not series_metadata.empty:
        present_series = {series_uid for _, series_uid in grouped}
        for row in series_metadata.to_dict(orient="records"):
            series_uid = _clean(row.get("SeriesInstanceUID"))
            if not series_uid or series_uid in present_series:
                continue
            selection_rows.append(
                {
                    "StudyInstanceUID": _clean(row.get("StudyInstanceUID")),
                    "SeriesInstanceUID": series_uid,
                    "file_count": 0,
                    "frame_count": 0,
                    "Fluid_Sensitive": _parse_bool(row.get("Fluid_Sensitive")),
                    "Fat_Suppression": _parse_bool(row.get("Fat_Suppression")),
                    "Anatomical_Plane": _clean(row.get("Anatomical_Plane")).title(),
                    "selection_score": 0.0,
                    "ordering_method": "",
                    "selected": False,
                    "reason": "series metadata has no matching DICOM files",
                }
            )
    return selected, pd.DataFrame(selection_rows)


def save_dicom_preview(
    selected_series: dict[str, list[tuple[DicomInstance, int]]],
    output_dir: Path,
    max_studies: int = 3,
) -> tuple[Path | None, list[dict[str, str]]]:
    """Save a small grayscale montage of representative adjacent slices for visual QA."""
    study_ids = sorted(selected_series)[:max_studies]
    if not study_ids:
        return None, []
    blank = np.zeros(len(TARGETS), dtype=np.float32)
    labels = {
        uid: (blank.copy(), blank.copy(), blank.copy(), blank.copy()) for uid in study_ids
    }
    config = PipelineConfig(
        image_size=192,
        slices_per_study=3,
        batch_size=1,
        num_workers=0,
        num_epochs=1,
        pretrained=False,
        imagenet_normalization=False,
    )
    dataset = StudyDataset(
        study_ids, selected_series, labels, config, training=False
    )
    previews: list[tuple[int, list[np.ndarray]]] = []
    errors: list[dict[str, str]] = []
    for index in range(len(dataset)):
        try:
            item = dataset[index]
            center_stack = item["images"][item["images"].shape[0] // 2]
            channels = [
                np.clip(channel.numpy() * 255.0, 0, 255).astype(np.uint8)
                for channel in center_stack
            ]
            previews.append((index, channels))
        except Exception as exc:
            errors.append(
                {
                    "path": selected_series[study_ids[index]][0][0].path,
                    "error": f"preview decode failed: {type(exc).__name__}: {str(exc)[:200]}",
                }
            )
            LOGGER.warning("Could not create DICOM preview for study index %d: %s", index, exc)
        errors.extend(dataset.pixel_errors)
        dataset.pixel_errors.clear()
    if not previews:
        return None, errors
    cell = 192
    margin = 28
    canvas = Image.new(
        "RGB",
        (cell * 3, (cell + margin) * len(previews)),
        color=(24, 24, 24),
    )
    draw = ImageDraw.Draw(canvas)
    for row, (study_number, channels) in enumerate(previews):
        y = row * (cell + margin)
        draw.text((4, y + 3), f"Study sample {study_number + 1}: adjacent 2.5D slices", fill="white")
        for column, channel in enumerate(channels):
            gray = Image.fromarray(channel)
            canvas.paste(gray.convert("RGB"), (column * cell, y + margin))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dicom_preview.png"
    canvas.save(path)
    return path, errors


def _study_label_matrices(
    merged: pd.DataFrame,
    weights: dict[str, float],
    use_silver: bool,
    use_qwen: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    target_index = {target: i for i, target in enumerate(TARGETS)}
    for uid, rows in merged.groupby("StudyInstanceUID", sort=False):
        targets = np.zeros(len(TARGETS), dtype=np.float32)
        mask = np.zeros(len(TARGETS), dtype=np.float32)
        source_weights = np.zeros(len(TARGETS), dtype=np.float32)
        gold_mask = np.zeros(len(TARGETS), dtype=np.float32)
        for row in rows.to_dict(orient="records"):
            index = target_index[row["target"]]
            source = row["label_source"]
            value = _valid_score(row["label_value"])
            if value is None:
                continue
            enabled = (
                source == "GOLD"
                or (source == "SILVER" and use_silver)
                or (source == "QWEN" and use_qwen)
            )
            if enabled:
                targets[index] = value
                mask[index] = 1.0
                source_weights[index] = float(weights[source])
            if source == "GOLD":
                gold_mask[index] = 1.0
        result[uid] = (targets, mask, source_weights, gold_mask)
    return result


def grouped_multilabel_split(
    study_ids: Sequence[str],
    gold_labels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    patient_groups: dict[str, str] | None = None,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str], str]:
    """Greedy multilabel split at patient level when PatientID is available."""
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5")
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for uid in study_ids:
        patient = (patient_groups or {}).get(uid, "")
        groups[patient if patient else uid].append(uid)
    group_ids = list(groups)
    rng.shuffle(group_ids)
    classes = len(TARGETS) * 2
    vectors: dict[str, np.ndarray] = {}
    for group_id, members in groups.items():
        vector = np.zeros(classes, dtype=np.float64)
        for uid in members:
            if uid not in gold_labels:
                continue
            target_values, _, _, gold_mask = gold_labels[uid]
            for target_index in range(len(TARGETS)):
                if gold_mask[target_index]:
                    slot = target_index * 2 + int(target_values[target_index] > 0.5)
                    vector[slot] += 1.0
        vectors[group_id] = vector
    total = sum(vectors.values(), np.zeros(classes, dtype=np.float64))
    desired = total * validation_fraction
    validation_count = max(1, round(len(study_ids) * validation_fraction))
    chosen: list[str] = []
    selected_vector = np.zeros(classes, dtype=np.float64)
    selected_studies = 0
    remaining = set(group_ids)
    while remaining and selected_studies < validation_count:
        deficit = np.maximum(desired - selected_vector, 0.0)
        rare = np.where(deficit > 0, deficit / np.maximum(total, 1.0), 0.0)
        candidates = list(remaining)
        rng.shuffle(candidates)
        best_group = candidates[0]
        best_score = -1.0
        for group_id in candidates:
            vector = vectors[group_id]
            size = len(groups[group_id])
            coverage = float(np.minimum(vector, deficit).dot(1.0 / np.maximum(total, 1.0)))
            size_penalty = abs((selected_studies + size) - validation_count) / max(validation_count, 1)
            score = coverage / max(size, 1) - 0.05 * size_penalty
            if score > best_score:
                best_score, best_group = score, group_id
        chosen.append(best_group)
        selected_vector += vectors[best_group]
        selected_studies += len(groups[best_group])
        remaining.remove(best_group)
    val_ids = sorted(uid for group_id in chosen for uid in groups[group_id])
    val_set = set(val_ids)
    train_ids = sorted(uid for uid in study_ids if uid not in val_set)
    split_digest = hashlib.sha256(
        ("\n".join(val_ids) + f"\nseed={seed}\nfraction={validation_fraction}").encode()
    ).hexdigest()[:16]
    return train_ids, val_ids, split_digest


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != mask.shape or logits.shape != weights.shape:
        raise ValueError("logits, targets, mask, and weights must have identical shapes")
    if logits.ndim != 2 or logits.shape[1] != len(TARGETS):
        raise ValueError(f"expected tensors with shape [batch, {len(TARGETS)}]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values")
    if not torch.isfinite(mask).all() or ((mask < 0) | (mask > 1)).any():
        raise ValueError("mask must contain finite values in [0, 1]")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    active = mask > 0
    if active.any():
        active_targets = targets[active]
        if not torch.isfinite(active_targets).all() or ((active_targets < 0) | (active_targets > 1)).any():
            raise ValueError("unmasked targets must be finite values in [0, 1]")
    safe_targets = torch.where(active, targets, torch.zeros_like(targets))
    per_target = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    weighted_mask = mask * weights
    denominator = weighted_mask.sum()
    return (per_target * weighted_mask).sum() / denominator.clamp_min(1.0)


@dataclass
class PipelineConfig:
    image_size: int = 320
    slices_per_study: int = 5
    batch_size: int = 4
    num_workers: int = 2
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    validation_fraction: float = 0.2
    seed: int = 42
    parser_confidence_threshold: float = 0.85
    gold_weight: float = 1.0
    silver_weight: float = 0.25
    qwen_weight: float = 0.1
    use_silver: bool = False
    use_qwen_soft_labels: bool = False
    gold_only: bool = True
    smoke_test: bool = False
    max_studies: int | None = None
    max_batches_per_epoch: int | None = None
    pretrained: bool = True
    imagenet_normalization: bool = True
    cache_images: bool = False
    cache_max_gb: float = 5.0

    def __post_init__(self) -> None:
        if self.gold_only and (self.use_silver or self.use_qwen_soft_labels):
            raise ValueError("gold_only cannot be combined with SILVER or Qwen labels")
        if self.image_size < 64 or self.slices_per_study < 1:
            raise ValueError("image_size must be >=64 and slices_per_study >=1")
        if self.batch_size < 1 or self.num_epochs < 1 or self.num_workers < 0:
            raise ValueError("batch_size and num_epochs must be positive; num_workers cannot be negative")
        if min(self.gold_weight, self.silver_weight, self.qwen_weight) < 0:
            raise ValueError("source weights must be non-negative")
        if self.cache_max_gb <= 0:
            raise ValueError("cache_max_gb must be positive")

    @property
    def source_weights(self) -> dict[str, float]:
        return {
            "GOLD": self.gold_weight,
            "SILVER": self.silver_weight,
            "QWEN": self.qwen_weight,
        }


class StudyDataset(Dataset):
    def __init__(
        self,
        study_ids: Sequence[str],
        selected_series: dict[str, list[tuple[DicomInstance, int]]],
        labels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        config: PipelineConfig,
        cache_dir: Path | None = None,
        training: bool = False,
        gold_validation_only: bool = False,
    ) -> None:
        self.study_ids = [
            uid for uid in study_ids if uid in selected_series and uid in labels
        ]
        self.selected_series = selected_series
        self.labels = labels
        self.config = config
        self.cache_dir = cache_dir if config.cache_images else None
        self.training = training
        self.gold_validation_only = gold_validation_only
        self.pixel_errors: list[dict[str, str]] = []
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.study_ids)

    def _cache_key(
        self, uid: str, frames: list[tuple[DicomInstance, int]], centers: list[int]
    ) -> str:
        fingerprint = {
            "version": PREPROCESS_VERSION,
            "size": self.config.image_size,
            "normalization": [1.0, 99.0, self.config.imagenet_normalization],
            "study": uid,
            "series": frames[0][0].series_uid,
            "centers": centers,
            "sources": [
                (frames[i][0].path, Path(frames[i][0].path).stat().st_size, Path(frames[i][0].path).stat().st_mtime_ns)
                for center in centers
                for i in range(max(0, center - 1), min(len(frames), center + 2))
            ],
        }
        return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()

    def _read_frame(self, item: tuple[DicomInstance, int]) -> np.ndarray:
        record, frame_index = item
        ds = pydicom.dcmread(record.path, force=False)
        pixels = np.asarray(ds.pixel_array)
        if pixels.ndim == 2:
            frame = pixels
        elif pixels.ndim >= 3:
            if frame_index >= pixels.shape[0]:
                raise ValueError(f"frame index {frame_index} out of range in {record.path}")
            frame = pixels[frame_index]
            if frame.ndim != 2:
                raise ValueError(f"unsupported non-2D frame shape {pixels.shape} in {record.path}")
        else:
            raise ValueError(f"unsupported pixel array shape {pixels.shape} in {record.path}")
        frame = np.asarray(frame, dtype=np.float32)
        frame = frame * record.rescale_slope + record.rescale_intercept
        frame[~np.isfinite(frame)] = 0.0
        return frame

    def _resize_letterbox(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape
        scale = min(self.config.image_size / max(height, 1), self.config.image_size / max(width, 1))
        out_h = max(1, min(self.config.image_size, round(height * scale)))
        out_w = max(1, min(self.config.image_size, round(width * scale)))
        pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
        pil = pil.resize((out_w, out_h), Image.Resampling.BILINEAR)
        resized = np.asarray(pil, dtype=np.float32) / 255.0
        canvas = np.zeros((self.config.image_size, self.config.image_size), dtype=np.float32)
        y0 = (self.config.image_size - out_h) // 2
        x0 = (self.config.image_size - out_w) // 2
        canvas[y0 : y0 + out_h, x0 : x0 + out_w] = resized
        return canvas

    def _load_stacks(self, uid: str) -> np.ndarray:
        frames = self.selected_series[uid]
        if not frames:
            raise ValueError(f"selected series for study has no frames: {uid}")
        size = len(frames)
        count = max(1, self.config.slices_per_study)
        if size == 1:
            centers = [size // 2]
        else:
            centers = sorted(set(np.linspace(0, size - 1, num=count).round().astype(int).tolist()))
            while len(centers) < count:
                centers.append(centers[-1])
        raw_by_index: dict[int, np.ndarray] = {}
        needed = sorted(
            {
                min(max(center + offset, 0), size - 1)
                for center in centers
                for offset in (-1, 0, 1)
            }
        )
        cache_path: Path | None = None
        if self.cache_dir:
            key = self._cache_key(uid, frames, centers)
            cache_path = self.cache_dir / f"{key}.npz"
            if cache_path.exists():
                try:
                    with np.load(cache_path, allow_pickle=False) as cached:
                        if (
                            str(cached["version"].item()) == PREPROCESS_VERSION
                            and int(cached["image_size"].item()) == self.config.image_size
                        ):
                            return cached["images"].astype(np.float32)
                except (OSError, ValueError, KeyError):
                    pass
                cache_path.unlink(missing_ok=True)
        for index in needed:
            try:
                raw_by_index[index] = self._read_frame(frames[index])
            except Exception as exc:
                record, frame_number = frames[index]
                error = {
                    "path": record.path,
                    "error": (
                        f"pixel decode failed for frame {frame_number}: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    ),
                }
                self.pixel_errors.append(error)
                LOGGER.warning("%s (%s)", error["error"], record.path)
        if not raw_by_index:
            raise RuntimeError(f"all selected frames failed pixel decoding for study {uid}")
        for index in needed:
            if index not in raw_by_index:
                nearest = min(raw_by_index, key=lambda valid: abs(valid - index))
                raw_by_index[index] = raw_by_index[nearest]
        inversion = any(
            frames[index][0].photometric_interpretation.upper() == "MONOCHROME1"
            for index in needed
        )
        if inversion:
            maximum = max(float(np.max(image)) for image in raw_by_index.values())
            raw_by_index = {index: maximum - image for index, image in raw_by_index.items()}
        sample = np.concatenate([raw_by_index[index].reshape(-1) for index in needed])
        if sample.size == 0 or not np.isfinite(sample).any():
            raise ValueError(f"no finite pixels in selected series for study {uid}")
        finite = sample[np.isfinite(sample)]
        low, high = np.percentile(finite, [1.0, 99.0]).astype(float)
        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            low = float(np.min(finite))
            high = float(np.max(finite))
        if high <= low:
            high = low + 1.0
        stacks = []
        for center in centers:
            channels = []
            for offset in (-1, 0, 1):
                index = min(max(center + offset, 0), size - 1)
                image = np.clip((raw_by_index[index] - low) / (high - low), 0.0, 1.0)
                channels.append(self._resize_letterbox(image))
            stacks.append(np.stack(channels, axis=0))
        images = np.stack(stacks, axis=0).astype(np.float32)
        if self.config.imagenet_normalization:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
            images = (images - mean) / std
        if cache_path:
            np.savez_compressed(
                cache_path,
                images=images,
                version=np.asarray(PREPROCESS_VERSION),
                image_size=np.asarray(self.config.image_size),
            )
            self._prune_cache()
        return images

    def _prune_cache(self) -> None:
        if not self.cache_dir:
            return
        limit = int(self.config.cache_max_gb * 1024**3)
        files = sorted(
            (path for path in self.cache_dir.glob("*.npz") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
        )
        total = sum(path.stat().st_size for path in files)
        while total > limit and files:
            victim = files.pop(0)
            size = victim.stat().st_size
            victim.unlink()
            total -= size

    def __getitem__(self, index: int) -> dict[str, Any]:
        uid = self.study_ids[index]
        images = torch.from_numpy(self._load_stacks(uid))
        if self.training:
            angle = random.uniform(-5.0, 5.0)
            translate = [random.uniform(-0.025, 0.025), random.uniform(-0.025, 0.025)]
            theta = torch.tensor(
                [
                    [
                        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), translate[0]],
                        [math.sin(math.radians(angle)), math.cos(math.radians(angle)), translate[1]],
                    ]
                ],
                dtype=images.dtype,
            ).expand(images.shape[0], -1, -1).contiguous()
            grid = F.affine_grid(theta, images.shape, align_corners=False)
            images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            contrast = random.uniform(0.95, 1.05)
            brightness = random.uniform(-0.03, 0.03)
            images = images * contrast + brightness
        targets, mask, weights, gold_mask = self.labels[uid]
        if self.gold_validation_only:
            mask = gold_mask
            weights = gold_mask * self.config.gold_weight
        return {
            "study_uid": uid,
            "images": images,
            "targets": torch.tensor(targets, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "weights": torch.tensor(weights, dtype=torch.float32),
            "gold_mask": torch.tensor(gold_mask, dtype=torch.float32),
        }


class ResNet34StudyModel(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = resnet34(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(TARGETS))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("model expects [batch, slices, 3, height, width]")
        batch, slices, channels, height, width = images.shape
        logits = self.backbone(images.reshape(batch * slices, channels, height, width))
        return logits.reshape(batch, slices, len(TARGETS)).mean(dim=1)


def _collate_studies(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "study_uid": [item["study_uid"] for item in items],
        "images": torch.stack([item["images"] for item in items]),
        "targets": torch.stack([item["targets"] for item in items]),
        "mask": torch.stack([item["mask"] for item in items]),
        "weights": torch.stack([item["weights"] for item in items]),
        "gold_mask": torch.stack([item["gold_mask"] for item in items]),
    }


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    negative = labels == 0
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    uids: list[str] = []
    total_loss = 0.0
    loss_batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            weights = batch["weights"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                logits = model(images)
                loss = masked_bce_loss(logits, targets, mask, weights)
            if mask.sum().item() > 0:
                total_loss += float(loss.item())
                loss_batches += 1
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(targets.cpu().numpy())
            all_masks.append(batch["gold_mask"].numpy())
            uids.extend(batch["study_uid"])
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, len(TARGETS)))
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0, len(TARGETS)))
    masks = np.concatenate(all_masks, axis=0) if all_masks else np.empty((0, len(TARGETS)))
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    per_target = []
    aucs = []
    for target_index, target in enumerate(TARGETS):
        selected = masks[:, target_index] > 0
        y_true = labels[selected, target_index]
        y_score = probabilities[selected, target_index]
        value = _auc(y_true, y_score) if selected.any() else float("nan")
        aucs.append(value)
        per_target.append(
            {
                "target": target,
                "auc": value,
                "gold_positive": int((y_true == 1).sum()),
                "gold_negative": int((y_true == 0).sum()),
                "gold_count": int(selected.sum()),
            }
        )
    valid_auc = [value for value in aucs if math.isfinite(value)]
    macro_auc = float(np.mean(valid_auc)) if valid_auc else float("nan")
    results = {
        "macro_auc": macro_auc,
        "valid_targets": len(valid_auc),
        "gold_loss": total_loss / loss_batches if loss_batches else float("nan"),
        "n_studies": len(uids),
    }
    frame = pd.DataFrame(per_target)
    frame["macro_auc"] = macro_auc
    return results, frame, total_loss / loss_batches if loss_batches else float("nan")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _system_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        report.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_gb": round(total_bytes / 1024**3, 2),
                "gpu_free_memory_gb": round(free_bytes / 1024**3, 2),
            }
        )
    else:
        report.update({"gpu_name": None, "gpu_total_memory_gb": None, "gpu_free_memory_gb": None})
    LOGGER.info("Runtime: %s", json.dumps(report))
    return report


def _load_series_metadata(paths: Sequence[Path]) -> pd.DataFrame | None:
    frames = []
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            if {"StudyInstanceUID", "SeriesInstanceUID"}.issubset(frame.columns):
                frames.append(frame)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["StudyInstanceUID", "SeriesInstanceUID"], keep="last"
    )
    return combined


def _candidate_file(root: Path, names: Sequence[str]) -> Path | None:
    wanted = {name.lower() for name in names}
    candidates: list[Path] = []
    for current, dirs, files in os.walk(root):
        dirs.sort()
        for filename in files:
            if filename.lower() in wanted:
                candidates.append(Path(current) / filename)
        dirs[:] = [
            name
            for name in dirs
            if name.lower() not in {"train_images", "test_images", "train", "test"}
        ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (len(path.parts), str(path).lower()))
    return candidates[0]


def discover_inputs(input_root: Path) -> dict[str, Any]:
    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    train = _candidate_file(input_root, ["train.csv"])
    if train is None:
        raise FileNotFoundError(f"could not find train.csv under {input_root}")
    parser = _candidate_file(input_root, ["labels_final.csv"])
    qwen = _candidate_file(input_root, ["labels_llm_test.csv", "labels_llm_output.csv"])
    parser_research = _candidate_file(input_root, ["labels_research.csv"])
    train_series = _candidate_file(input_root, ["train_series.csv"])
    test_series = _candidate_file(input_root, ["test_series.csv"])
    sample = _candidate_file(input_root, ["sample_submission.csv"])
    image_roots: list[Path] = []
    for current, dirs, _ in os.walk(input_root):
        matched = [
            Path(current) / name
            for name in dirs
            if name.lower() in {"train_images", "test_images", "train", "test"}
        ]
        if matched:
            image_roots.extend(matched)
            dirs[:] = [
                name
                for name in dirs
                if name.lower()
                not in {"train_images", "test_images", "train", "test"}
            ]
    if not image_roots:
        preferred = [
            path
            for path in input_root.iterdir()
            if path.is_dir()
            and (
                path.name.lower() in {"data", "train", "test", "train_images", "test_images"}
                or "image" in path.name.lower()
            )
        ]
        if preferred:
            image_roots = preferred
        elif any(path.is_file() and path.suffix.lower() == ".dcm" for path in input_root.iterdir()):
            image_roots = [input_root]
        else:
            for path in input_root.iterdir():
                if path.is_dir():
                    try:
                        if next(path.rglob("*.dcm"), None) is not None:
                            image_roots.append(path)
                    except OSError:
                        continue
    if not image_roots:
        image_roots = [input_root]
    train_roots = [
        path for path in image_roots if "test" not in path.name.lower()
    ] or image_roots
    test_roots = [path for path in image_roots if "test" in path.name.lower()]
    return {
        "train_csv": train,
        "parser_csv": parser,
        "qwen_csv": qwen,
        "parser_research_csv": parser_research,
        "train_series_csv": train_series,
        "test_series_csv": test_series,
        "sample_submission": sample,
        "train_image_roots": train_roots,
        "test_image_roots": test_roots,
        "all_image_roots": sorted(set(image_roots)),
    }


def _patient_groups(instances: Sequence[DicomInstance]) -> dict[str, str]:
    result: dict[str, str] = {}
    conflicted: set[str] = set()
    for item in instances:
        if item.patient_group and item.study_uid not in conflicted:
            previous = result.get(item.study_uid)
            if previous and previous != item.patient_group:
                LOGGER.warning("Multiple PatientID values in a study; grouping that study by StudyInstanceUID")
                result.pop(item.study_uid, None)
                conflicted.add(item.study_uid)
            elif item.study_uid not in result:
                result[item.study_uid] = item.patient_group
    return result


def _save_selection(selection: pd.DataFrame, output_dir: Path) -> None:
    selection.to_csv(output_dir / "series_selection.csv", index=False)


def _make_loaders(
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    selected_series: dict[str, list[tuple[DicomInstance, int]]],
    labels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    config: PipelineConfig,
    output_dir: Path,
) -> tuple[DataLoader, DataLoader, StudyDataset, StudyDataset]:
    cache_dir = output_dir / "image_cache" if config.cache_images else None
    train_dataset = StudyDataset(
        train_ids, selected_series, labels, config, cache_dir, training=True
    )
    val_dataset = StudyDataset(
        val_ids,
        selected_series,
        labels,
        config,
        cache_dir,
        training=False,
        gold_validation_only=True,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"no usable studies after image/label matching (train={len(train_dataset)}, val={len(val_dataset)})"
        )
    workers = max(0, config.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=_collate_studies,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=_collate_studies,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def train_model(
    config: PipelineConfig,
    inputs: dict[str, Any],
    output_dir: Path,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _system_report()
    _seed_everything(config.seed)
    merged, label_stats = merge_labels(
        inputs["train_csv"],
        inputs["parser_csv"],
        inputs["qwen_csv"],
        config.parser_confidence_threshold,
        inputs["parser_research_csv"],
    )
    save_label_artifacts(merged, label_stats, output_dir)
    label_fingerprint = hashlib.sha256(
        merged.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    LOGGER.info("Indexing DICOM headers; pixel data are not loaded during this step")
    instances, errors = scan_dicom_files(inputs["all_image_roots"])
    if not instances:
        raise RuntimeError("no readable DICOM instances found; verify the Kaggle input selection")
    dicom_audit, _, studies = write_dicom_audit(
        instances, errors, output_dir, set(merged["StudyInstanceUID"].unique())
    )
    LOGGER.info(
        "DICOM metadata: %d files, %d studies, %d errors",
        len(instances),
        len(studies),
        len(errors),
    )
    metadata_paths = [inputs["train_series_csv"], inputs["test_series_csv"]]
    series_metadata = _load_series_metadata([p for p in metadata_paths if p is not None])
    selected_series, selection = select_series(instances, series_metadata)
    _save_selection(selection, output_dir)
    if config.smoke_test:
        preview_path, preview_errors = save_dicom_preview(selected_series, output_dir)
        errors.extend(preview_errors)
        if preview_path:
            LOGGER.info("Saved representative DICOM preview to %s", preview_path)
        if preview_errors:
            pd.DataFrame(errors, columns=["path", "error"]).drop_duplicates().to_csv(
                output_dir / "dicom_errors.csv", index=False
            )
    _project_audit(
        inputs,
        merged,
        label_stats,
        instances,
        errors,
        output_dir / "project_audit.md",
    )

    labels = _study_label_matrices(
        merged,
        config.source_weights,
        use_silver=config.use_silver and not config.gold_only,
        use_qwen=config.use_qwen_soft_labels and not config.gold_only,
    )
    usable_studies = [
        uid
        for uid, (_, mask, _, _) in labels.items()
        if mask.sum() > 0 and uid in selected_series
    ]
    if config.max_studies:
        usable_studies = sorted(usable_studies)[: config.max_studies]
    gold_studies = [
        uid
        for uid, (_, _, _, gold_mask) in labels.items()
        if gold_mask.sum() > 0 and uid in selected_series
    ]
    if config.max_studies:
        gold_studies = [uid for uid in gold_studies if uid in set(usable_studies)]
    if len(gold_studies) < 4:
        raise RuntimeError(
            f"only {len(gold_studies)} GOLD-labeled studies have selected DICOMs; "
            "at least 4 are needed for a GOLD-only validation split"
        )
    patients = _patient_groups(instances)
    if patients:
        LOGGER.info("Patient-level grouping available for %d studies", len(patients))
    else:
        LOGGER.warning("No PatientID values found; split is grouped by StudyInstanceUID only")
    train_ids, val_ids, split_id = grouped_multilabel_split(
        gold_studies,
        labels,
        patients,
        config.validation_fraction,
        config.seed,
    )
    val_patient_groups = {
        patients[uid] for uid in val_ids if uid in patients
    }
    val_study_set = set(val_ids)
    train_ids = sorted(
        uid
        for uid in usable_studies
        if uid not in val_study_set
        and (uid not in patients or patients[uid] not in val_patient_groups)
    )
    val_ids = sorted(val_ids)
    
    if set(train_ids) & set(val_ids):
        raise AssertionError("train/validation study leakage detected")
    if patients:
        train_patients = {patients[uid] for uid in train_ids if uid in patients}
        val_patients = {patients[uid] for uid in val_ids if uid in patients}
        if train_patients & val_patients:
            raise AssertionError("train/validation patient leakage detected")
    pd.DataFrame(
        [
            *({"StudyInstanceUID": uid, "split": "train"} for uid in train_ids),
            *({"StudyInstanceUID": uid, "split": "validation"} for uid in val_ids),
        ]
    ).to_csv(output_dir / "train_validation_split.csv", index=False)

    model_config = {
        **asdict(config),
        "targets": TARGETS,
        "preprocess_version": PREPROCESS_VERSION,
        "label_version": "labels_merged-v2",
        "label_fingerprint": label_fingerprint,
        "split_id": split_id,
        "train_studies": len(train_ids),
        "validation_studies": len(val_ids),
        "validation_metric": "macro mean of available per-target GOLD ROC-AUCs",
        "validation_labels": "GOLD only",
        "series_count": len(selected_series),
        "dicom_read_errors": len(errors),
    }
    _write_json(output_dir / "model_config.json", model_config)
    LOGGER.info(
        "Split %s: train=%d validation=%d; all validation metrics use GOLD only",
        split_id,
        len(train_ids),
        len(val_ids),
    )
    train_loader, val_loader, train_dataset, val_dataset = _make_loaders(
        train_ids, val_ids, selected_series, labels, config, output_dir
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = ResNet34StudyModel(pretrained=config.pretrained).to(device)
    except Exception as exc:
        raise RuntimeError(
            "could not initialize ResNet34 weights. For offline smoke testing set pretrained=False; "
            "for the baseline provide cached/downloadable torchvision ResNet34 weights."
        ) from exc
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_metric = -float("inf")
    best_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if checkpoint.get("targets") != TARGETS:
            raise ValueError("checkpoint targets do not match this pipeline")
        if checkpoint.get("preprocess_version") != PREPROCESS_VERSION:
            raise ValueError("checkpoint preprocessing version does not match this pipeline")
        if checkpoint.get("split_id") != split_id:
            raise ValueError("checkpoint split ID does not match the current train/validation split")
        if checkpoint.get("label_fingerprint") != label_fingerprint:
            raise ValueError("checkpoint labels do not match the current merged label data")
        old_config = checkpoint.get("config", {})
        for key in ("image_size", "slices_per_study", "imagenet_normalization"):
            if old_config.get(key) != getattr(config, key):
                raise ValueError(f"checkpoint {key} does not match the current configuration")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_selection_metric", -float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
        LOGGER.info("Resumed checkpoint at epoch %d", start_epoch)

    log_rows: list[dict[str, Any]] = []
    validation_rows: list[pd.DataFrame] = []
    best_path = output_dir / "best_model.pth"
    stale_epochs = 0
    start_time = time.perf_counter()
    epochs = min(config.num_epochs, 2) if config.smoke_test else config.num_epochs
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        seen_batches = 0
        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            weights = batch["weights"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = masked_bce_loss(logits, targets, mask, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            seen_batches += 1
            if config.max_batches_per_epoch and seen_batches >= config.max_batches_per_epoch:
                break
        if not losses:
            raise RuntimeError("training produced no batches")
        val_summary, val_targets, val_loss = evaluate(
            model, val_loader, device, mixed_precision=device.type == "cuda"
        )
        validation_rows.append(
            val_targets.assign(epoch=epoch + 1, split_id=split_id)
        )
        metric = val_summary["macro_auc"]
        selection_metric = metric if math.isfinite(metric) else -val_loss
        scheduler.step(metric if math.isfinite(metric) else -val_loss)
        elapsed = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "training_batches": seen_batches,
            "gold_validation_loss": val_loss,
            "macro_gold_auc": metric,
            "valid_auc_targets": val_summary["valid_targets"],
            "validation_studies": val_summary["n_studies"],
            "epoch_seconds": elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "split_id": split_id,
            "selection_metric": "macro_auc" if math.isfinite(metric) else "negative_gold_loss_fallback",
        }
        log_rows.append(record)
        LOGGER.info(
            "Epoch %d/%d train_loss=%.5f val_gold_auc=%s valid_targets=%d time=%.1fs",
            epoch + 1,
            epochs,
            record["train_loss"],
            f"{metric:.4f}" if math.isfinite(metric) else "NaN",
            val_summary["valid_targets"],
            elapsed,
        )
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                    ),
                    "epoch": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_val_auc": metric,
                    "best_selection_metric": best_metric,
                    "config": model_config,
                    "seed": config.seed,
                    "preprocess_version": PREPROCESS_VERSION,
                    "targets": TARGETS,
                    "label_version": "labels_merged-v2",
                    "label_fingerprint": label_fingerprint,
                    "split_id": split_id,
                },
                best_path,
            )
        else:
            stale_epochs += 1
        pd.DataFrame(log_rows).to_csv(output_dir / "training_log.csv", index=False)
        if validation_rows:
            pd.concat(validation_rows, ignore_index=True).to_csv(
                output_dir / "validation_results.csv", index=False
            )
        if stale_epochs >= config.early_stopping_patience and not config.smoke_test:
            LOGGER.info("Early stopping after %d stale epochs", stale_epochs)
            break

    if not best_path.exists():
        raise RuntimeError("training completed without producing best_model.pth")
    elapsed_total = time.perf_counter() - start_time
    best_record = next(
        (row for row in log_rows if row["epoch"] == best_epoch), None
    )
    best_auc = best_record["macro_gold_auc"] if best_record else None
    if best_auc is not None and not math.isfinite(best_auc):
        best_auc = None
    report = {
        "status": "PASS",
        "smoke_test": config.smoke_test,
        "elapsed_seconds": round(elapsed_total, 2),
        "train_studies": len(train_dataset),
        "validation_studies": len(val_dataset),
        "best_epoch": best_epoch,
        "best_val_auc": float(best_auc) if best_auc is not None else None,
        "training_batches": sum(int(row["training_batches"]) for row in log_rows),
        "epochs_completed": len(log_rows),
        "checks": {
            "dicom_metadata_indexed": len(instances) > 0,
            "study_level_split_disjoint": not bool(set(train_ids) & set(val_ids)),
            "model_checkpoint_saved": best_path.exists(),
            "validation_gold_only": True,
        },
        "limitations": [
            "A smoke-test subset is an execution check, not a performance estimate.",
            "Validation macro-AUC averages only targets with both GOLD classes present.",
            "A validation target without both classes has NaN AUC and is excluded from the macro mean.",
        ],
    }
    if config.smoke_test:
        reloaded = ResNet34StudyModel(pretrained=False).to(device)
        saved_checkpoint = torch.load(
            best_path, map_location=device, weights_only=False
        )
        reloaded.load_state_dict(saved_checkpoint["model_state_dict"])
        report["checks"]["checkpoint_reloaded"] = True
        validation_preview = predict_studies(
            reloaded,
            val_dataset.study_ids[:1],
            selected_series,
            config,
            device,
            output_dir,
        )
        validation_scores = validation_preview[TARGETS].to_numpy(dtype=np.float64)
        report["checks"]["finite_probability_inference"] = bool(
            np.isfinite(validation_scores).all()
            and (validation_scores >= 0).all()
            and (validation_scores <= 1).all()
        )
        sample_path = inputs["sample_submission"]
        if sample_path is None:
            report["checks"]["official_submission_preview"] = None
        else:
            template = pd.read_csv(sample_path)
            if template.empty:
                raise ValueError("sample_submission.csv contains no rows")
            identifier = template.columns[0]
            test_uid = str(template.iloc[0][identifier])
            if test_uid not in selected_series:
                raise ValueError(
                    "first official sample ID has no available test DICOM series; "
                    "cannot validate submission inference"
                )
            test_prediction = predict_studies(
                reloaded,
                [test_uid],
                selected_series,
                config,
                device,
                output_dir,
            ).rename(columns={"StudyInstanceUID": identifier})
            preview = validate_submission(
                template.iloc[:1].copy(), test_prediction, identifier
            )
            preview.to_csv(output_dir / "smoke_submission_preview.csv", index=False)
            report["checks"]["official_submission_preview"] = len(preview) == 1
            report["submission_preview_rows"] = len(preview)
    pixel_errors = train_dataset.pixel_errors + val_dataset.pixel_errors
    if pixel_errors:
        existing_errors = pd.read_csv(output_dir / "dicom_errors.csv")
        combined_errors = pd.concat(
            [existing_errors, pd.DataFrame(pixel_errors)], ignore_index=True
        )
        combined_errors.drop_duplicates(subset=["path", "error"], inplace=True)
        combined_errors.to_csv(output_dir / "dicom_errors.csv", index=False)
        report["pixel_decode_errors"] = len(pixel_errors)
        report["warnings"] = [
            "Some individual frames failed decoding and were replaced by the nearest valid slice."
        ]
    failed_checks = [
        name for name, result in report["checks"].items() if result is False
    ]
    if failed_checks:
        report["status"] = "FAIL"
        report["failure_reason"] = "failed checks: " + ", ".join(failed_checks)
    if config.smoke_test:
        _write_smoke_report(output_dir / "smoke_test_report.txt", report)
    _write_json(output_dir / "training_summary.json", report)
    if failed_checks:
        raise RuntimeError(report["failure_reason"])
    LOGGER.info("Finished; best checkpoint: %s", best_path)
    return report


def _write_smoke_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "RSNA Knee pipeline smoke-test report",
        f"Status: {report['status']}",
        f"Elapsed seconds: {report['elapsed_seconds']}",
        f"Train studies: {report['train_studies']}",
        f"Validation studies: {report['validation_studies']}",
        f"Epochs completed: {report.get('epochs_completed', 0)}",
        f"Training batches: {report.get('training_batches', 0)}",
        f"Best epoch: {report['best_epoch']}",
        f"Best GOLD validation AUC: {report['best_val_auc']}",
        f"Pixel decode errors: {report.get('pixel_decode_errors', 0)}",
        "Checks:",
    ]
    lines.extend(
        f"  {name}: {'N/A' if ok is None else 'PASS' if ok else 'FAIL'}"
        for name, ok in report["checks"].items()
    )
    if report.get("errors"):
        lines.append("Errors:")
        lines.extend(f"  - {error}" for error in report["errors"])
    if report.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in report["warnings"])
    lines.extend(["Limitations:", *[f"  - {item}" for item in report["limitations"]]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke_test(
    config: PipelineConfig,
    inputs: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if not config.smoke_test:
        raise ValueError("run_smoke_test requires config.smoke_test=True")
    started = time.perf_counter()
    try:
        return train_model(config, inputs, output_dir)
    except Exception as exc:
        report = {
            "status": "FAIL",
            "smoke_test": True,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "train_studies": 0,
            "validation_studies": 0,
            "best_epoch": None,
            "best_val_auc": None,
            "checks": {"end_to_end_smoke_test": False},
            "errors": [f"{type(exc).__name__}: {str(exc)}"],
            "limitations": [
                "This report records a failed execution check, not model performance."
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_smoke_report(output_dir / "smoke_test_report.txt", report)
        _write_json(output_dir / "training_summary.json", report)
        raise RuntimeError(
            f"Smoke test failed; see {output_dir / 'smoke_test_report.txt'}"
        ) from exc


def validate_submission(
    template: pd.DataFrame,
    predictions: pd.DataFrame,
    id_column: str | None = None,
) -> pd.DataFrame:
    """Return a validated wide-format submission in template row/column order."""
    if template.empty or predictions.empty:
        raise ValueError("sample submission and prediction table must both be non-empty")
    identifier = id_column or next(
        (name for name in ("StudyInstanceUID", "study_id", "ID", "id") if name in template.columns),
        template.columns[0],
    )
    if identifier not in predictions.columns:
        raise ValueError(f"prediction table is missing identifier column {identifier!r}")
    target_columns = [column for column in template.columns if column != identifier]
    if set(target_columns) != set(TARGETS) or len(target_columns) != len(TARGETS):
        raise ValueError(
            "sample_submission must contain the identifier plus exactly the 12 documented target columns; "
            f"found {template.columns.tolist()}"
        )
    if template[identifier].isna().any() or template[identifier].duplicated().any():
        raise ValueError("sample_submission identifiers are missing or duplicated")
    if predictions[identifier].isna().any() or predictions[identifier].duplicated().any():
        raise ValueError("prediction identifiers are missing or duplicated")
    expected = set(template[identifier].astype(str))
    actual = set(predictions[identifier].astype(str))
    if expected != actual:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise ValueError(f"prediction ID mismatch: {missing} missing, {extra} unexpected")
    indexed = predictions.set_index(identifier)
    output = template[[identifier]].copy()
    for target in target_columns:
        values = pd.to_numeric(indexed.loc[template[identifier], target], errors="coerce").to_numpy()
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"submission scores for {target!r} must be finite values in [0, 1]")
        output[target] = values
    if output.columns.tolist() != template.columns.tolist() or len(output) != len(template):
        raise AssertionError("submission does not preserve the official template shape")
    return output


def predict_studies(
    model: nn.Module,
    study_ids: Sequence[str],
    selected_series: dict[str, list[tuple[DicomInstance, int]]],
    config: PipelineConfig,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    empty_labels = {
        uid: (
            np.zeros(len(TARGETS), dtype=np.float32),
            np.zeros(len(TARGETS), dtype=np.float32),
            np.zeros(len(TARGETS), dtype=np.float32),
            np.zeros(len(TARGETS), dtype=np.float32),
        )
        for uid in study_ids
    }
    dataset = StudyDataset(
        study_ids,
        selected_series,
        empty_labels,
        config,
        output_dir / "image_cache" if config.cache_images else None,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(0, config.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_studies,
    )
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(batch["images"].to(device))
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
            for uid, scores in zip(batch["study_uid"], probabilities):
                records.append({"StudyInstanceUID": uid, **dict(zip(TARGETS, scores.tolist()))})
    return pd.DataFrame(records)


def generate_submission(
    checkpoint_path: Path,
    inputs: dict[str, Any],
    output_dir: Path,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    template_path = inputs["sample_submission"]
    if template_path is None:
        raise FileNotFoundError(
            "sample_submission.csv is not present. Submission generation is intentionally blocked "
            "until the official template is available."
        )
    template = pd.read_csv(template_path)
    test_roots = inputs["test_image_roots"]
    if not test_roots:
        raise FileNotFoundError("no test image directory was found under the input root")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_config = checkpoint.get("config", {})
    config = PipelineConfig(
        image_size=int(checkpoint_config.get("image_size", 320)),
        slices_per_study=int(checkpoint_config.get("slices_per_study", 5)),
        batch_size=int(checkpoint_config.get("batch_size", 4)),
        num_workers=int(checkpoint_config.get("num_workers", 2)),
        pretrained=False,
        imagenet_normalization=bool(checkpoint_config.get("imagenet_normalization", True)),
        cache_images=bool(checkpoint_config.get("cache_images", False)),
        **(config_overrides or {}),
    )
    test_studies = set(template.iloc[:, 0].dropna().astype(str))
    instances, errors = scan_dicom_files(test_roots)
    found_ids = {instance.study_uid for instance in instances}
    if test_studies - found_ids:
        raise ValueError(
            f"test images missing for {len(test_studies - found_ids)} official sample IDs"
        )
    metadata = _load_series_metadata(
        [inputs["test_series_csv"]] if inputs["test_series_csv"] else []
    )
    selected, selection = select_series(instances, metadata)
    selection.to_csv(output_dir / "series_selection_test.csv", index=False)
    model = ResNet34StudyModel(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions = predict_studies(
        model, sorted(test_studies), selected, config, device, output_dir
    )
    identifier = template.columns[0]
    predictions = predictions.rename(columns={"StudyInstanceUID": identifier})
    submission = validate_submission(template, predictions, identifier)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    pd.DataFrame(errors, columns=["path", "error"]).to_csv(
        output_dir / "dicom_errors_test.csv", index=False
    )
    LOGGER.info(
        "Saved validated submission with %d rows to %s", len(submission), output_path
    )
    return output_path


def _project_audit(
    inputs: dict[str, Any],
    merged: pd.DataFrame,
    stats: pd.DataFrame,
    instances: Sequence[DicomInstance],
    errors: Sequence[dict[str, str]],
    output_path: Path,
) -> None:
    train = pd.read_csv(inputs["train_csv"], dtype=str, keep_default_na=False)
    series = pd.read_csv(inputs["train_series_csv"], dtype=str, keep_default_na=False) if inputs["train_series_csv"] else pd.DataFrame()
    parser = pd.read_csv(inputs["parser_csv"], dtype=str, keep_default_na=False) if inputs["parser_csv"] else pd.DataFrame()
    llm = pd.read_csv(inputs["qwen_csv"], dtype=str, keep_default_na=False) if inputs["qwen_csv"] else pd.DataFrame()
    project_root = inputs["train_csv"].parent
    source_paths = {
        "train.csv": inputs["train_csv"],
        "train_series.csv": inputs["train_series_csv"],
        "labels_final.csv": inputs["parser_csv"],
        "labels_research.csv": inputs["parser_research_csv"],
        "labels_llm_test.csv": inputs["qwen_csv"],
        "sample_submission.csv": inputs["sample_submission"],
    }
    prior_outputs: list[str] = []
    current_keys = {
        "dicom_sve_slike.csv": (
            "SOPInstanceUID",
            {item.sop_uid for item in instances if item.sop_uid},
        ),
        "dicom_serije.csv": (
            "SeriesInstanceUID",
            {item.series_uid for item in instances if item.series_uid},
        ),
        "dicom_studiji.csv": (
            "StudyInstanceUID",
            {item.study_uid for item in instances if item.study_uid},
        ),
        "language_analysis.csv": (
            "StudyInstanceUID",
            set(train["StudyInstanceUID"]),
        ),
    }
    previous_dir = project_root / "dicom_analiza"
    for filename, (key, current) in current_keys.items():
        previous_path = (
            previous_dir / filename
            if filename.startswith("dicom_")
            else project_root / filename
        )
        if not previous_path.exists():
            continue
        previous_frame = pd.read_csv(previous_path, low_memory=False)
        if key not in previous_frame.columns:
            prior_outputs.append(
                f"- `{previous_path}`: {len(previous_frame):,} rows; key `{key}` is absent, so no key comparison was possible."
            )
            continue
        previous_keys = set(previous_frame[key].dropna().astype(str))
        prior_outputs.append(
            f"- `{previous_path}`: {len(previous_frame):,} rows; "
            f"{len(previous_keys & current):,}/{len(previous_keys):,} unique `{key}` values match the current source; "
            f"{len(previous_keys - current):,} prior-only and {len(current - previous_keys):,} current-only."
        )
    schema_lines: list[str] = []
    jsonl_lines: list[str] = []
    for label, path in source_paths.items():
        if path is None:
            schema_lines.append(f"- `{label}`: not found.")
            continue
        frame = pd.read_csv(path, low_memory=False)
        key = "SeriesInstanceUID" if label == "train_series.csv" else "StudyInstanceUID"
        duplicate_count = (
            int(frame[key].duplicated(keep=False).sum())
            if key in frame.columns
            else 0
        )
        schema_lines.extend(
            [
                f"- `{path}`: {len(frame):,} rows, {len(frame.columns)} columns, "
                f"{path.stat().st_size:,} bytes, {duplicate_count:,} rows with duplicate `{key}` values.",
                "  - Columns:",
            ]
        )
        for column in frame.columns:
            empty_count = int(
                frame[column].isna().sum()
                + frame[column].astype("string").str.strip().eq("").fillna(False).sum()
            )
            schema_lines.append(
                f"    - `{column}`: dtype={frame[column].dtype}, empty={empty_count:,}, "
                f"unique={frame[column].nunique(dropna=False):,}."
            )
            if column.endswith("_status") or column.endswith("_SOURCE"):
                counts = (
                    frame[column]
                    .fillna("<empty>")
                    .astype(str)
                    .replace("", "<empty>")
                    .value_counts()
                    .to_dict()
                )
                schema_lines.append(f"      - Values: {counts}.")
    for path in sorted(project_root.glob("*.jsonl")):
        rows = 0
        malformed = 0
        keys: set[str] = set()
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    rows += 1
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            keys.update(str(key) for key in record)
                    except json.JSONDecodeError:
                        malformed += 1
            jsonl_lines.append(
                f"- `{path}`: {rows:,} JSONL records, {path.stat().st_size:,} bytes, "
                f"{malformed:,} malformed lines; top-level keys: {sorted(keys)}."
            )
        except OSError as exc:
            jsonl_lines.append(f"- `{path}`: could not read ({type(exc).__name__}: {exc}).")
    workbook_lines: list[str] = []
    workbook_paths = sorted(project_root.glob("*.xlsx")) + sorted(
        (project_root / "dicom_analiza").glob("*.xlsx")
    )
    if workbook_paths and importlib.util.find_spec("openpyxl") is None:
        workbook_lines.append(
            "- XLSX workbooks were found, but `openpyxl` is unavailable; their sheets were not inspected."
        )
    else:
        for workbook_path in workbook_paths:
            try:
                workbook = pd.ExcelFile(workbook_path)
                sheets = []
                for sheet_name in workbook.sheet_names:
                    sheet = pd.read_excel(workbook, sheet_name=sheet_name)
                    sheets.append(
                        f"{sheet_name}: {len(sheet):,} rows x {len(sheet.columns)} columns"
                    )
                workbook_lines.append(
                    f"- `{workbook_path}` ({workbook_path.stat().st_size:,} bytes): "
                    + "; ".join(sheets)
                    + "."
                )
            except (OSError, ValueError, ImportError) as exc:
                workbook_lines.append(
                    f"- `{workbook_path}`: workbook inspection failed "
                    f"({type(exc).__name__}: {str(exc)[:180]})."
                )

    gold_studies = set(merged.loc[merged["is_gold"], "StudyInstanceUID"])
    dicom_studies = {item.study_uid for item in instances}
    gold_dicom_overlap = len(gold_studies & dicom_studies)
    series_metadata = _load_series_metadata(
        [path for path in (inputs["train_series_csv"], inputs["test_series_csv"]) if path]
    )
    dicom_series = {item.series_uid for item in instances}
    missing_series = (
        len(set(series_metadata["SeriesInstanceUID"]) - dicom_series)
        if series_metadata is not None
        else None
    )
    project_files = sorted(
        path.name
        for path in project_root.iterdir()
        if path.is_file() and path.suffix.lower() != ".dcm"
    )
    lines = [
        "# RSNA Knee project audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Project inputs",
        "",
        f"- Training labels: `{inputs['train_csv']}` ({len(train):,} rows; {len(train.columns)} columns).",
        f"- Training series: `{inputs['train_series_csv']}` ({len(series):,} rows; {len(series.columns)} columns).",
        f"- Parser result: `{inputs['parser_csv']}` ({len(parser):,} rows; {len(parser.columns)} columns).",
        f"- Qwen result: `{inputs['qwen_csv']}` ({len(llm):,} rows; {len(llm.columns)} columns).",
        f"- Parser research table: `{inputs['parser_research_csv']}`.",
        f"- Sample submission: `{inputs['sample_submission']}`.",
        f"- DICOM roots: {', '.join(str(path) for path in inputs['all_image_roots'])}.",
        f"- Other top-level project files: {', '.join(project_files)}.",
        "",
        "## CSV/JSON schema inventory",
        "",
        *schema_lines,
        "",
        "### Raw JSON/JSONL artifacts",
        "",
        *(jsonl_lines or ["- No JSONL files found at the project root."]),
        "",
        "### Excel workbook inventory",
        "",
        *(workbook_lines or ["- No XLSX workbooks found."]),
        "",
        "### Comparison with prior CSV outputs",
        "",
        *(prior_outputs or ["- No matching prior CSV outputs were found."]),
        "",
        "## Source and label findings",
        "",
        f"- `train.csv` has {train['StudyInstanceUID'].nunique():,} unique study IDs. It contains the 12 target columns and only non-empty values from the structured GOLD set.",
        f"- `labels_final.csv` has {parser['StudyInstanceUID'].nunique() if not parser.empty else 0:,} unique studies. Its accepted parser labels are rows with status `SILVER`, valid binary value, and confidence at/above 0.85; `UNCERTAIN`, `CONFLICT`, unresolved, unknown, and skipped-language states are masked.",
        f"- `labels_llm_test.csv` has {llm['StudyInstanceUID'].nunique() if not llm.empty else 0:,} unique studies. Qwen values are continuous scores; the result file stores them in `*_LLM` and the source categories in `*_SOURCE`.",
        "- The prior Qwen prompt produces contextual scores for findings that are not mentioned. Such scores are not verified negatives. Qwen is therefore opt-in, lower-weight, preserves score values, and requires the stored `LLM_POS`/`LLM_NEG` source category to agree with the score threshold; it is disabled in the default GOLD-only run.",
        "- For each study/target, the merge applies source precedence GOLD > accepted SILVER > accepted Qwen. An explicit parser `UNCERTAIN` or `CONFLICT` state is masked, not filled by another weak source. Missing values are never treated as negatives.",
        "- Parser `UNCERTAIN`/`CONFLICT` counts in the statistics describe parser outputs and can overlap a GOLD-selected study/target because GOLD remains authoritative.",
        "",
        "### Per-target current counts",
        "",
        "| Target | GOLD + | GOLD - | SILVER + | SILVER - | Qwen soft | UNCERTAIN excluded | CONFLICT excluded | Missing/invalid | Other masked |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            "| "
            + " | ".join(
                str(row[column])
                for column in [
                    "target",
                    "gold_positive",
                    "gold_negative",
                    "accepted_silver_positive",
                    "accepted_silver_negative",
                    "accepted_qwen_soft",
                    "excluded_uncertain",
                    "excluded_conflict",
                    "missing_or_invalid",
                    "other_masked",
                ]
            )
            + " |"
            for row in stats.to_dict(orient="records")
        ],
        "",
        "## Series metadata",
        "",
        f"- `train_series.csv`: {len(series):,} rows; {series['StudyInstanceUID'].nunique() if not series.empty else 0:,} studies; `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` are present: {all(c in series.columns for c in ['Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane'])}.",
        f"- Series metadata values: {series[['Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']].nunique(dropna=False).to_dict() if not series.empty and all(c in series.columns for c in ['Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']) else 'not available'}.",
        "",
        "## DICOM findings",
        "",
        f"- Metadata scan read {len(instances):,} DICOM files across {len({x.study_uid for x in instances}):,} studies and {len({(x.study_uid, x.series_uid) for x in instances}):,} series.",
        f"- Per-file metadata errors: {len(errors):,}. Pixel bytes are deferred; `PixelDataPresent` checks the element tag without decoding it.",
        f"- Files without a PixelData element: {sum(not x.pixel_data_present for x in instances):,}. Series-metadata records with no matching local DICOM series: {missing_series if missing_series is not None else 'not available'}.",
        f"- Representative DICOM preview: `{output_path.parent / 'dicom_preview.png'}`."
        if instances and (output_path.parent / "dicom_preview.png").exists()
        else "- Representative DICOM preview: not generated.",
        f"- Studies with PatientID metadata: {len({x.study_uid for x in instances if x.patient_group}):,}. Patient IDs are not emitted in audit CSVs.",
        f"- Studies with image metadata and matching label IDs: {len({x.study_uid for x in instances} & set(train['StudyInstanceUID'])):,}.",
        f"- GOLD-labeled studies with local DICOM: {gold_dicom_overlap:,}.",
        f"- Training label studies without local DICOM: {len(set(train['StudyInstanceUID']) - dicom_studies):,}.",
        "- Full Kaggle DICOM decode and official test submission were not verified unless those inputs are present under the discovered roots.",
        "",
        "## Existing project scripts and artifacts",
        "",
        "- `weak.py` is the EN/ES rule-based parser; its source defines a SILVER confidence threshold of 0.85. It records `POSITIVE_HIGH`, `NEGATIVE_HIGH`, `UNCERTAIN`, `CONFLICT`, and related states in the research table, then materializes `SILVER` in `labels_final.csv`.",
        "- `kaggle_kod.py` and `llm_test_v6.py` are report-to-Qwen inference scripts; they do not provide MRI model training. The new pipeline keeps their outputs read-only.",
        "- `dicom_analiza` contains prior DICOM audit CSV/XLSX/TXT outputs; the new audit outputs are generated separately.",
        "",
        "## Limitations and assumptions",
        "",
        "- The local directory may contain only a partial DICOM sample; counts here describe only discovered local files.",
        "- This is a competition pipeline and is not a clinically validated diagnostic system.",
        "- The validation split is study-level and uses PatientID grouping only when a consistent DICOM PatientID is available. If no PatientID exists, it falls back to StudyInstanceUID groups.",
        "- Per-series intensity scaling uses the selected representative slices for a bounded streaming baseline, not all pixels in a series.",
        "- The official sample submission was not found locally, so exact test output is validated only when Kaggle's `sample_submission.csv` is available.",
        "- `/kaggle/working/` is session-scoped. Save notebook outputs or publish them as a Kaggle Dataset to retain checkpoints and reports.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(inputs: dict[str, Any], output_dir: Path, audit_dicom: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged, stats = merge_labels(
        inputs["train_csv"],
        inputs["parser_csv"],
        inputs["qwen_csv"],
        parser_research_csv=inputs["parser_research_csv"],
    )
    save_label_artifacts(merged, stats, output_dir)
    instances: list[DicomInstance] = []
    errors: list[dict[str, str]] = []
    if audit_dicom:
        instances, errors = scan_dicom_files(inputs["all_image_roots"])
        write_dicom_audit(
            instances, errors, output_dir, set(merged["StudyInstanceUID"].unique())
        )
        metadata_paths = [inputs["train_series_csv"], inputs["test_series_csv"]]
        metadata = _load_series_metadata([p for p in metadata_paths if p])
        selected, selection = select_series(instances, metadata)
        _save_selection(selection, output_dir)
        preview_path, preview_errors = save_dicom_preview(selected, output_dir)
        errors.extend(preview_errors)
        if preview_errors:
            pd.DataFrame(errors, columns=["path", "error"]).drop_duplicates().to_csv(
                output_dir / "dicom_errors.csv", index=False
            )
        if preview_path:
            LOGGER.info("Saved representative DICOM preview to %s", preview_path)
    _project_audit(
        inputs,
        merged,
        stats,
        instances,
        errors,
        output_dir / "project_audit.md",
    )
    LOGGER.info("Audit artifacts saved to %s", output_dir)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-dicom", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--use-silver", action="store_true")
    parser.add_argument("--use-qwen-soft-labels", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-studies", type=int)
    parser.add_argument("--max-batches-per-epoch", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--parser-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--gold-weight", type=float, default=1.0)
    parser.add_argument("--silver-weight", type=float, default=0.25)
    parser.add_argument("--qwen-weight", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _args()
    inputs = discover_inputs(args.input_root)
    LOGGER.info(
        "Discovered train=%s parser=%s qwen=%s sample_submission=%s roots=%s",
        inputs["train_csv"],
        inputs["parser_csv"],
        inputs["qwen_csv"],
        inputs["sample_submission"],
        inputs["all_image_roots"],
    )
    if args.audit_only:
        run_audit(inputs, args.output_dir, args.audit_dicom)
        return
    smoke = args.smoke_test
    use_silver = args.use_silver
    use_qwen = args.use_qwen_soft_labels
    gold_only = args.gold_only or not (use_silver or use_qwen)
    if smoke:
        use_silver = use_qwen = False
        gold_only = True
    config = PipelineConfig(
        image_size=args.image_size,
        batch_size=args.batch_size if not smoke else min(args.batch_size, 2),
        num_workers=0 if smoke else args.num_workers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        parser_confidence_threshold=args.parser_confidence_threshold,
        gold_weight=args.gold_weight,
        silver_weight=args.silver_weight,
        qwen_weight=args.qwen_weight,
        use_silver=use_silver,
        use_qwen_soft_labels=use_qwen,
        gold_only=gold_only,
        smoke_test=smoke,
        max_studies=args.max_studies or (64 if smoke else None),
        max_batches_per_epoch=args.max_batches_per_epoch or (3 if smoke else None),
        pretrained=not args.no_pretrained,
        cache_images=args.cache_images,
    )
    if smoke:
        run_smoke_test(config, inputs, args.output_dir)
    else:
        train_model(config, inputs, args.output_dir, args.resume_from_checkpoint)
    if args.make_submission:
        generate_submission(
            args.output_dir / "best_model.pth", inputs, args.output_dir
        )


if __name__ == "__main__":
    main()
