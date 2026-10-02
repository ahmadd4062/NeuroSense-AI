from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .model_service import (
    DEFAULT_HEALTHY_THRESHOLD,
    DEFAULT_PARKINSON_THRESHOLD,
    PredictionCollectionResult,
    PredictionThresholds,
    RecordingPredictionResult,
    decision_score_to_pd_risk,
    get_prediction_thresholds,
)


MINIMUM_VALID_RECORDINGS = 2
MAXIMUM_RECORDINGS = 3

DEFAULT_LOW_VARIABILITY = 0.02
DEFAULT_MODERATE_VARIABILITY = 0.05


class FusionServiceError(Exception):
    pass


class InsufficientPredictionsError(FusionServiceError):
    pass


class InvalidFusionInputError(FusionServiceError):
    pass


@dataclass(frozen=True)
class FusionConfig:
    minimum_valid_recordings: int = MINIMUM_VALID_RECORDINGS
    maximum_recordings: int = MAXIMUM_RECORDINGS

    healthy_maximum: float = DEFAULT_HEALTHY_THRESHOLD
    parkinson_minimum: float = DEFAULT_PARKINSON_THRESHOLD

    low_variability_threshold: float = DEFAULT_LOW_VARIABILITY
    moderate_variability_threshold: float = DEFAULT_MODERATE_VARIABILITY

    def __post_init__(self) -> None:
        if self.minimum_valid_recordings < 2:
            raise InvalidFusionInputError(
                "At least two valid recordings must be required."
            )

        if self.maximum_recordings < self.minimum_valid_recordings:
            raise InvalidFusionInputError(
                "Maximum recordings cannot be lower than the minimum."
            )

        if self.healthy_maximum >= self.parkinson_minimum:
            raise InvalidFusionInputError(
                "Healthy threshold must be lower than Parkinson threshold."
            )

        if self.low_variability_threshold < 0:
            raise InvalidFusionInputError(
                "Low variability threshold cannot be negative."
            )

        if (
            self.moderate_variability_threshold
            < self.low_variability_threshold
        ):
            raise InvalidFusionInputError(
                "Moderate variability threshold cannot be lower "
                "than the low variability threshold."
            )

    @property
    def thresholds(self) -> PredictionThresholds:
        return PredictionThresholds(
            healthy_maximum=self.healthy_maximum,
            parkinson_minimum=self.parkinson_minimum,
        )


def _default_fusion_config() -> FusionConfig:
    """Build fusion thresholds from the deployed model artifact.

    This keeps category interpretation in sync with the exact model
    configuration instead of duplicating threshold values in views.
    """
    thresholds = get_prediction_thresholds()

    return FusionConfig(
        healthy_maximum=thresholds.healthy_maximum,
        parkinson_minimum=thresholds.parkinson_minimum,
    )


@dataclass
class FusionRecording:
    slot_number: int
    decision_score: float
    predicted_label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FusionResult:
    final_label: str | None
    fused_score: float | None
    pd_risk_score: float | None

    score_mean: float | None
    score_std: float | None
    score_min: float | None
    score_max: float | None
    score_range: float | None
    score_median: float | None

    confidence_score: float
    confidence_level: str
    consistency_level: str

    category_agreement: float
    unanimous_agreement: bool

    valid_recording_count: int
    required_recording_count: int

    valid_slot_numbers: list[int] = field(default_factory=list)
    individual_scores: list[float] = field(default_factory=list)
    individual_labels: list[str] = field(default_factory=list)
    recordings: list[FusionRecording] = field(default_factory=list)

    fusion_successful: bool = True
    requires_more_recordings: bool = False
    warning: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_label": self.final_label,
            "fused_score": self.fused_score,
            "pd_risk_score": self.pd_risk_score,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_range": self.score_range,
            "score_median": self.score_median,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
            "consistency_level": self.consistency_level,
            "category_agreement": self.category_agreement,
            "unanimous_agreement": self.unanimous_agreement,
            "valid_recording_count": self.valid_recording_count,
            "required_recording_count": self.required_recording_count,
            "valid_slot_numbers": list(self.valid_slot_numbers),
            "individual_scores": list(self.individual_scores),
            "individual_labels": list(self.individual_labels),
            "recordings": [
                recording.to_dict()
                for recording in self.recordings
            ],
            "fusion_successful": self.fusion_successful,
            "requires_more_recordings": self.requires_more_recordings,
            "warning": self.warning,
            "error": self.error,
        }


def _validate_score(score: Any) -> float:
    try:
        normalized_score = float(score)
    except (TypeError, ValueError) as error:
        raise InvalidFusionInputError(
            "Decision score must be numeric."
        ) from error

    if not math.isfinite(normalized_score):
        raise InvalidFusionInputError(
            "Decision score must be finite."
        )

    return normalized_score


def _classify_score(
    score: float,
    config: FusionConfig,
) -> str:
    return config.thresholds.classify(score)


def _calculate_category_agreement(
    labels: Sequence[str],
) -> float:
    if not labels:
        return 0.0

    counts: dict[str, int] = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    highest_count = max(counts.values())

    return float(highest_count / len(labels))


def _calculate_consistency_level(
    score_std: float,
    config: FusionConfig,
) -> str:
    if score_std <= config.low_variability_threshold:
        return "High"

    if score_std <= config.moderate_variability_threshold:
        return "Moderate"

    return "Low"


def _distance_from_nearest_boundary(
    score: float,
    config: FusionConfig,
) -> float:
    if score <= config.healthy_maximum:
        return config.healthy_maximum - score

    if score >= config.parkinson_minimum:
        return score - config.parkinson_minimum

    lower_distance = score - config.healthy_maximum
    upper_distance = config.parkinson_minimum - score

    return min(lower_distance, upper_distance)


def _calculate_confidence_score(
    fused_score: float,
    score_std: float,
    category_agreement: float,
    recording_count: int,
    config: FusionConfig,
) -> float:
    agreement_component = category_agreement * 45.0

    if score_std <= config.low_variability_threshold:
        consistency_component = 30.0
    elif score_std <= config.moderate_variability_threshold:
        variability_span = (
            config.moderate_variability_threshold
            - config.low_variability_threshold
        )

        if variability_span <= 0:
            consistency_component = 20.0
        else:
            ratio = (
                score_std - config.low_variability_threshold
            ) / variability_span

            consistency_component = 30.0 - (ratio * 15.0)
    else:
        consistency_component = max(
            0.0,
            15.0 - (
                (
                    score_std
                    - config.moderate_variability_threshold
                )
                * 100.0
            ),
        )

    boundary_distance = _distance_from_nearest_boundary(
        fused_score,
        config,
    )

    threshold_width = (
        config.parkinson_minimum
        - config.healthy_maximum
    )

    if threshold_width <= 0:
        boundary_component = 0.0
    else:
        normalized_distance = min(
            boundary_distance / threshold_width,
            1.0,
        )

        boundary_component = normalized_distance * 15.0

    recording_component = (
        min(
            recording_count,
            config.maximum_recordings,
        )
        / config.maximum_recordings
    ) * 10.0

    confidence = (
        agreement_component
        + consistency_component
        + boundary_component
        + recording_component
    )

    return round(
        float(np.clip(confidence, 0.0, 100.0)),
        2,
    )


def _confidence_level(
    confidence_score: float,
) -> str:
    if confidence_score >= 80.0:
        return "High"

    if confidence_score >= 60.0:
        return "Moderate"

    return "Low"


def _build_failed_result(
    valid_recording_count: int,
    required_recording_count: int,
    error: str,
    recordings: Sequence[FusionRecording] | None = None,
) -> FusionResult:
    recording_items = list(recordings or [])

    return FusionResult(
        final_label=None,
        fused_score=None,
        pd_risk_score=None,
        score_mean=None,
        score_std=None,
        score_min=None,
        score_max=None,
        score_range=None,
        score_median=None,
        confidence_score=0.0,
        confidence_level="Unavailable",
        consistency_level="Unavailable",
        category_agreement=0.0,
        unanimous_agreement=False,
        valid_recording_count=valid_recording_count,
        required_recording_count=required_recording_count,
        valid_slot_numbers=[
            item.slot_number
            for item in recording_items
        ],
        individual_scores=[
            item.decision_score
            for item in recording_items
        ],
        individual_labels=[
            item.predicted_label
            for item in recording_items
        ],
        recordings=recording_items,
        fusion_successful=False,
        requires_more_recordings=True,
        warning=None,
        error=error,
    )


def fuse_recordings(
    recordings: Sequence[FusionRecording],
    *,
    config: FusionConfig | None = None,
) -> FusionResult:
    active_config = config or _default_fusion_config()

    normalized_recordings: list[FusionRecording] = []
    seen_slots: set[int] = set()

    for recording in recordings:
        slot_number = int(recording.slot_number)

        if slot_number < 1:
            raise InvalidFusionInputError(
                "Slot number must be at least 1."
            )

        if slot_number in seen_slots:
            raise InvalidFusionInputError(
                f"Duplicate slot number: {slot_number}"
            )

        seen_slots.add(slot_number)

        decision_score = _validate_score(
            recording.decision_score
        )

        predicted_label = _classify_score(
            decision_score,
            active_config,
        )

        normalized_recordings.append(
            FusionRecording(
                slot_number=slot_number,
                decision_score=decision_score,
                predicted_label=predicted_label,
            )
        )

    normalized_recordings.sort(
        key=lambda item: item.slot_number
    )

    if (
        len(normalized_recordings)
        > active_config.maximum_recordings
    ):
        normalized_recordings = normalized_recordings[
            :active_config.maximum_recordings
        ]

    recording_count = len(normalized_recordings)

    if recording_count < active_config.minimum_valid_recordings:
        return _build_failed_result(
            valid_recording_count=recording_count,
            required_recording_count=(
                active_config.minimum_valid_recordings
            ),
            error=(
                "At least "
                f"{active_config.minimum_valid_recordings} valid "
                "recordings are required for fusion."
            ),
            recordings=normalized_recordings,
        )

    scores = np.asarray(
        [
            recording.decision_score
            for recording in normalized_recordings
        ],
        dtype=np.float64,
    )

    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_range = float(score_max - score_min)
    score_median = float(np.median(scores))

    final_label = _classify_score(
        score_mean,
        active_config,
    )

    # Keep the original fused decision score unchanged.
    # The postprocessor converts that score into the threshold-aligned
    # PD risk value used by the recommendation/final fusion system.
    pd_risk_score = decision_score_to_pd_risk(
        score_mean
    )

    individual_labels = [
        recording.predicted_label
        for recording in normalized_recordings
    ]

    category_agreement = _calculate_category_agreement(
        individual_labels
    )

    unanimous_agreement = (
        len(set(individual_labels)) == 1
    )

    consistency_level = _calculate_consistency_level(
        score_std,
        active_config,
    )

    confidence_score = _calculate_confidence_score(
        fused_score=score_mean,
        score_std=score_std,
        category_agreement=category_agreement,
        recording_count=recording_count,
        config=active_config,
    )

    confidence_level = _confidence_level(
        confidence_score
    )

    warning: str | None = None

    if not unanimous_agreement:
        warning = (
            "The recordings produced different prediction "
            "categories."
        )

    if consistency_level == "Low":
        warning = (
            "The recording scores have high variability. "
            "The result should be interpreted cautiously."
        )

    return FusionResult(
        final_label=final_label,
        fused_score=score_mean,
        pd_risk_score=pd_risk_score,
        score_mean=score_mean,
        score_std=score_std,
        score_min=score_min,
        score_max=score_max,
        score_range=score_range,
        score_median=score_median,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        consistency_level=consistency_level,
        category_agreement=round(
            category_agreement,
            4,
        ),
        unanimous_agreement=unanimous_agreement,
        valid_recording_count=recording_count,
        required_recording_count=(
            active_config.minimum_valid_recordings
        ),
        valid_slot_numbers=[
            recording.slot_number
            for recording in normalized_recordings
        ],
        individual_scores=[
            recording.decision_score
            for recording in normalized_recordings
        ],
        individual_labels=individual_labels,
        recordings=normalized_recordings,
        fusion_successful=True,
        requires_more_recordings=False,
        warning=warning,
        error=None,
    )


def fuse_prediction_results(
    prediction_results: Iterable[
        RecordingPredictionResult
    ],
    *,
    config: FusionConfig | None = None,
) -> FusionResult:
    recordings: list[FusionRecording] = []

    for result in prediction_results:
        if not result.prediction_successful:
            continue

        if result.decision_score is None:
            continue

        score = _validate_score(
            result.decision_score
        )

        recordings.append(
            FusionRecording(
                slot_number=int(result.slot_number),
                decision_score=score,
                predicted_label=(
                    result.predicted_label
                    or ""
                ),
            )
        )

    return fuse_recordings(
        recordings,
        config=config,
    )


def fuse_prediction_collection(
    prediction_collection: PredictionCollectionResult,
    *,
    config: FusionConfig | None = None,
) -> FusionResult:
    return fuse_prediction_results(
        prediction_collection.recording_results,
        config=config,
    )


def fuse_scores(
    decision_scores: Sequence[float],
    *,
    slot_numbers: Sequence[int] | None = None,
    config: FusionConfig | None = None,
) -> FusionResult:
    active_config = config or _default_fusion_config()

    scores = list(decision_scores)

    if slot_numbers is None:
        slots = list(
            range(1, len(scores) + 1)
        )
    else:
        slots = [
            int(slot)
            for slot in slot_numbers
        ]

    if len(scores) != len(slots):
        raise InvalidFusionInputError(
            "Decision scores and slot numbers must have "
            "the same length."
        )

    recordings = [
        FusionRecording(
            slot_number=slot,
            decision_score=_validate_score(score),
            predicted_label=_classify_score(
                _validate_score(score),
                active_config,
            ),
        )
        for slot, score in zip(
            slots,
            scores,
            strict=True,
        )
    ]

    return fuse_recordings(
        recordings,
        config=active_config,
    )


def build_fusion_summary(
    result: FusionResult,
) -> dict[str, Any]:
    if not result.fusion_successful:
        message = (
            result.error
            or "Fusion could not be completed."
        )
    elif result.final_label == "Healthy":
        message = (
            "The combined voice screening score is within "
            "the healthy range."
        )
    elif result.final_label == "Parkinson":
        message = (
            "The combined voice screening score is within "
            "the Parkinson range."
        )
    else:
        message = (
            "The combined voice screening score is within "
            "the borderline range."
        )

    return {
        "fusion_successful": result.fusion_successful,
        "final_label": result.final_label,
        "fused_score": result.fused_score,
        "pd_risk_score": result.pd_risk_score,
        "confidence_score": result.confidence_score,
        "confidence_level": result.confidence_level,
        "consistency_level": result.consistency_level,
        "category_agreement": result.category_agreement,
        "unanimous_agreement": result.unanimous_agreement,
        "valid_recording_count": (
            result.valid_recording_count
        ),
        "valid_slot_numbers": list(
            result.valid_slot_numbers
        ),
        "individual_scores": list(
            result.individual_scores
        ),
        "requires_more_recordings": (
            result.requires_more_recordings
        ),
        "warning": result.warning,
        "error": result.error,
        "message": message,
    }