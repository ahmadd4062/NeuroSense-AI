from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from django.db import transaction

from speech_app.models import (
    RecordingPrediction,
    ScreeningPrediction,
)


class PredictionPersistenceError(Exception):
    """Raised when prediction results cannot be saved."""


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    if hasattr(value, "to_dict"):
        result = value.to_dict()

        if isinstance(result, Mapping):
            return dict(result)

    if is_dataclass(value):
        return asdict(value)

    raise PredictionPersistenceError(
        f"Unable to serialize object of type "
        f"{type(value).__name__}."
    )


def _get_value(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)

    return getattr(value, name, default)


def _normalize_model_prediction(value: Any) -> str:
    if value is None:
        return ""

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    return str(value)


@transaction.atomic
def save_screening_prediction(
    *,
    screening_id: str,
    prediction_collection: Any,
    fusion_result: Any,
    healthy_threshold: float = 0.23,
    parkinson_threshold: float = 0.27,
    metadata: Mapping[str, Any] | None = None,
) -> ScreeningPrediction:
    """
    Save or replace one completed screening prediction.

    Re-running the same screening_id updates the parent record and
    replaces its individual recording predictions.
    """
    normalized_screening_id = str(screening_id).strip()

    if not normalized_screening_id:
        raise PredictionPersistenceError(
            "screening_id is required."
        )

    final_score = _get_value(
        fusion_result,
        "final_score",
        None,
    )

    if final_score is None:
        final_score = _get_value(
            fusion_result,
            "fused_score",
            None,
        )

    if final_score is None:
        final_score = _get_value(
            fusion_result,
            "score",
            None,
        )

    final_label = _get_value(
        fusion_result,
        "final_label",
        None,
    )

    if final_label is None:
        final_label = _get_value(
            fusion_result,
            "predicted_label",
            None,
        )

    if final_label is None:
        final_label = _get_value(
            fusion_result,
            "classification",
            None,
        )

    if final_score is None:
        raise PredictionPersistenceError(
            "Fusion result does not contain a final score."
        )

    pd_risk_score = _get_value(
        fusion_result,
        "pd_risk_score",
        None,
    )

    if pd_risk_score is None:
        raise PredictionPersistenceError(
            "Fusion result does not contain a PD risk score."
        )

    if not final_label:
        raise PredictionPersistenceError(
            "Fusion result does not contain a final label."
        )

    successful_recordings = int(
        _get_value(
            prediction_collection,
            "successful_predictions",
            0,
        )
    )

    failed_recordings = int(
        _get_value(
            prediction_collection,
            "failed_predictions",
            0,
        )
    )

    screening, _created = (
        ScreeningPrediction.objects.update_or_create(
            screening_id=normalized_screening_id,
            defaults={
                # Preserve the original fused Linear-SVM decision score.
                "final_score": float(final_score),

                # Store the separate threshold-aligned PD risk value used
                # by the recommendation/final multimodal fusion system.
                "pd_risk_score": float(pd_risk_score),

                "final_label": str(final_label),
                "healthy_threshold": float(
                    healthy_threshold
                ),
                "parkinson_threshold": float(
                    parkinson_threshold
                ),
                "successful_recordings": (
                    successful_recordings
                ),
                "failed_recordings": failed_recordings,
                "fusion_data": _to_mapping(fusion_result),
                "metadata": dict(metadata or {}),
            },
        )
    )

    # A completed rerun should replace stale slot results.
    screening.recording_predictions.all().delete()

    recording_results = _get_value(
        prediction_collection,
        "recording_results",
        [],
    )

    rows: list[RecordingPrediction] = []

    for result in recording_results:
        rows.append(
            RecordingPrediction(
                screening=screening,
                slot_number=int(
                    _get_value(result, "slot_number", 0)
                ),
                filename=str(
                    _get_value(result, "filename", "") or ""
                ),
                audio_path=str(
                    _get_value(result, "audio_path", "") or ""
                ),
                decision_score=_get_value(
                    result,
                    "decision_score",
                    None,
                ),
                predicted_label=str(
                    _get_value(
                        result,
                        "predicted_label",
                        "",
                    )
                    or ""
                ),
                model_prediction=(
                    _normalize_model_prediction(
                        _get_value(
                            result,
                            "model_prediction",
                            None,
                        )
                    )
                ),
                probability=_get_value(
                    result,
                    "probability",
                    None,
                ),
                extraction_successful=bool(
                    _get_value(
                        result,
                        "extraction_successful",
                        False,
                    )
                ),
                prediction_successful=bool(
                    _get_value(
                        result,
                        "prediction_successful",
                        False,
                    )
                ),
                feature_count=int(
                    _get_value(
                        result,
                        "feature_count",
                        0,
                    )
                ),
                error=str(
                    _get_value(result, "error", "") or ""
                ),
            )
        )

    RecordingPrediction.objects.bulk_create(rows)

    return screening