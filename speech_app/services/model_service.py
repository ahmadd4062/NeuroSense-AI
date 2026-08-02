from __future__ import annotations

import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

from .audio_module.feature_extraction import (
    FeatureCollectionResult,
    FeatureExtractionResult,
    FeatureExtractionConfig,
    extract_features,
    extract_features_safely,
)


logger = logging.getLogger(__name__)


DEFAULT_MODEL_FILENAME = "final_linear_svm_matched.joblib"
DEFAULT_HEALTHY_THRESHOLD = 0.23
DEFAULT_PARKINSON_THRESHOLD = 0.27

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: "LoadedModelArtifact | None" = None


class ModelServiceError(Exception):
    pass


class ModelArtifactNotFoundError(ModelServiceError):
    pass


class ModelArtifactLoadError(ModelServiceError):
    pass


class ModelConfigurationError(ModelServiceError):
    pass


class ModelFeatureSchemaError(ModelServiceError):
    pass


class ModelPredictionError(ModelServiceError):
    pass


@dataclass(frozen=True)
class PredictionThresholds:
    healthy_maximum: float = DEFAULT_HEALTHY_THRESHOLD
    parkinson_minimum: float = DEFAULT_PARKINSON_THRESHOLD

    def __post_init__(self) -> None:
        if not math.isfinite(self.healthy_maximum):
            raise ModelConfigurationError(
                "Healthy threshold must be finite."
            )

        if not math.isfinite(self.parkinson_minimum):
            raise ModelConfigurationError(
                "Parkinson threshold must be finite."
            )

        if self.healthy_maximum >= self.parkinson_minimum:
            raise ModelConfigurationError(
                "Healthy threshold must be lower than Parkinson threshold."
            )

    def classify(self, score: float) -> str:
        normalized_score = float(score)

        if not math.isfinite(normalized_score):
            raise ModelPredictionError(
                "Prediction score is not finite."
            )

        if normalized_score <= self.healthy_maximum:
            return "Healthy"

        if normalized_score >= self.parkinson_minimum:
            return "Parkinson"

        return "Borderline"

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class LoadedModelArtifact:
    artifact_path: str
    estimator: Any
    feature_names: list[str]
    thresholds: PredictionThresholds
    positive_class: Any = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_extraction_config: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "feature_names": list(self.feature_names),
            "feature_count": self.feature_count,
            "thresholds": self.thresholds.to_dict(),
            "positive_class": self.positive_class,
            "metadata": dict(self.metadata),
            "feature_extraction_config": dict(
                self.feature_extraction_config
            ),
            "estimator_type": type(self.estimator).__name__,
        }


@dataclass
class RecordingPredictionResult:
    slot_number: int
    audio_path: str
    filename: str
    decision_score: float | None
    predicted_label: str | None
    model_prediction: Any = None
    probability: float | None = None
    extraction_successful: bool = True
    prediction_successful: bool = True
    feature_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionCollectionResult:
    total_recordings: int
    successful_predictions: int
    failed_predictions: int
    recording_results: list[RecordingPredictionResult] = field(
        default_factory=list
    )
    processing_successful: bool = True
    error: str | None = None

    @property
    def valid_scores(self) -> list[float]:
        return [
            float(result.decision_score)
            for result in self.recording_results
            if result.prediction_successful
            and result.decision_score is not None
        ]

    @property
    def valid_slot_numbers(self) -> list[int]:
        return [
            result.slot_number
            for result in self.recording_results
            if result.prediction_successful
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_recordings": self.total_recordings,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "recording_results": [
                result.to_dict()
                for result in self.recording_results
            ],
            "valid_scores": self.valid_scores,
            "valid_slot_numbers": self.valid_slot_numbers,
            "processing_successful": self.processing_successful,
            "error": self.error,
        }


def get_model_path() -> Path:
    configured_path = getattr(
        settings,
        "PARKINSON_MODEL_PATH",
        None,
    )

    if configured_path:
        model_path = Path(configured_path)
    else:
        model_path = (
            Path(settings.BASE_DIR)
            / "speech_app"
            / "models"
            / DEFAULT_MODEL_FILENAME
        )

    return model_path.expanduser().resolve()


def _first_available(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]

    return None


def _normalize_feature_names(
    feature_names: Iterable[Any],
) -> list[str]:
    normalized = [
        str(name).strip()
        for name in feature_names
    ]

    if not normalized:
        raise ModelFeatureSchemaError(
            "The model artifact contains no feature names."
        )

    if any(not name for name in normalized):
        raise ModelFeatureSchemaError(
            "The model feature schema contains an empty name."
        )

    if len(normalized) != len(set(normalized)):
        raise ModelFeatureSchemaError(
            "The model feature schema contains duplicate names."
        )

    return normalized


def _extract_estimator(artifact: Any) -> Any:
    if isinstance(artifact, Mapping):
        estimator = _first_available(
            artifact,
            (
                "model",
                "estimator",
                "pipeline",
                "classifier",
                "svm_model",
                "trained_model",
            ),
        )

        if estimator is not None:
            return estimator

    if hasattr(artifact, "predict"):
        return artifact

    raise ModelArtifactLoadError(
        "No prediction estimator was found in the model artifact."
    )


def _extract_feature_names(
    artifact: Any,
    estimator: Any,
) -> list[str]:
    if isinstance(artifact, Mapping):
        raw_feature_names = _first_available(
            artifact,
            (
                "feature_names",
                "expected_feature_names",
                "selected_features",
                "model_features",
                "feature_columns",
                "columns",
            ),
        )

        if raw_feature_names is not None:
            return _normalize_feature_names(
                list(raw_feature_names)
            )

    estimator_feature_names = getattr(
        estimator,
        "feature_names_in_",
        None,
    )

    if estimator_feature_names is not None:
        return _normalize_feature_names(
            list(estimator_feature_names)
        )

    if hasattr(estimator, "named_steps"):
        for step in reversed(
            list(estimator.named_steps.values())
        ):
            step_feature_names = getattr(
                step,
                "feature_names_in_",
                None,
            )

            if step_feature_names is not None:
                return _normalize_feature_names(
                    list(step_feature_names)
                )

    raise ModelFeatureSchemaError(
        "Feature names were not found in the model artifact. "
        "The exact training feature order is required."
    )


def _extract_thresholds(
    artifact: Any,
) -> PredictionThresholds:
    healthy_threshold = DEFAULT_HEALTHY_THRESHOLD
    parkinson_threshold = DEFAULT_PARKINSON_THRESHOLD

    if isinstance(artifact, Mapping):
        threshold_mapping = _first_available(
            artifact,
            (
                "thresholds",
                "prediction_thresholds",
                "deployment_thresholds",
                "classification_thresholds",
            ),
        )

        if isinstance(threshold_mapping, Mapping):
            healthy_value = _first_available(
                threshold_mapping,
                (
                    "healthy_maximum",
                    "healthy_threshold",
                    "healthy_max",
                    "lower_threshold",
                    "low",
                ),
            )

            parkinson_value = _first_available(
                threshold_mapping,
                (
                    "parkinson_minimum",
                    "parkinson_threshold",
                    "parkinson_min",
                    "upper_threshold",
                    "high",
                ),
            )

            if healthy_value is not None:
                healthy_threshold = float(healthy_value)

            if parkinson_value is not None:
                parkinson_threshold = float(parkinson_value)

        direct_healthy = _first_available(
            artifact,
            (
                "healthy_threshold",
                "healthy_maximum",
                "lower_threshold",
            ),
        )

        direct_parkinson = _first_available(
            artifact,
            (
                "parkinson_threshold",
                "parkinson_minimum",
                "upper_threshold",
            ),
        )

        if direct_healthy is not None:
            healthy_threshold = float(direct_healthy)

        if direct_parkinson is not None:
            parkinson_threshold = float(direct_parkinson)

    django_config = getattr(
        settings,
        "PARKINSON_PREDICTION_THRESHOLDS",
        None,
    )

    if isinstance(django_config, Mapping):
        healthy_threshold = float(
            django_config.get(
                "healthy_maximum",
                healthy_threshold,
            )
        )

        parkinson_threshold = float(
            django_config.get(
                "parkinson_minimum",
                parkinson_threshold,
            )
        )

    return PredictionThresholds(
        healthy_maximum=healthy_threshold,
        parkinson_minimum=parkinson_threshold,
    )


def _extract_positive_class(
    artifact: Any,
    estimator: Any,
) -> Any:
    if isinstance(artifact, Mapping):
        configured_class = _first_available(
            artifact,
            (
                "positive_class",
                "positive_label",
                "parkinson_class",
                "target_positive_class",
            ),
        )

        if configured_class is not None:
            return configured_class

    classes = getattr(estimator, "classes_", None)

    if classes is not None and len(classes) == 2:
        if 1 in classes:
            return 1

        for class_value in classes:
            normalized = str(class_value).strip().lower()

            if normalized in {
                "parkinson",
                "parkinsons",
                "pd",
                "positive",
                "disease",
            }:
                return class_value

        return classes[-1]

    return 1


def _extract_metadata(
    artifact: Any,
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        return {}

    raw_metadata = _first_available(
        artifact,
        (
            "metadata",
            "model_metadata",
            "training_metadata",
            "deployment_metadata",
        ),
    )

    if isinstance(raw_metadata, Mapping):
        return dict(raw_metadata)

    return {}


def _extract_feature_config(
    artifact: Any,
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        return {}

    raw_config = _first_available(
        artifact,
        (
            "feature_extraction_config",
            "extraction_config",
            "audio_feature_config",
            "feature_config",
        ),
    )

    if isinstance(raw_config, Mapping):
        return dict(raw_config)

    return {}


def load_model_artifact(
    *,
    force_reload: bool = False,
) -> LoadedModelArtifact:
    global _MODEL_CACHE

    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE

    with _MODEL_LOCK:
        if _MODEL_CACHE is not None and not force_reload:
            return _MODEL_CACHE

        model_path = get_model_path()

        if not model_path.exists():
            raise ModelArtifactNotFoundError(
                f"Model artifact was not found: {model_path}"
            )

        if not model_path.is_file():
            raise ModelArtifactNotFoundError(
                f"Model artifact path is not a file: {model_path}"
            )

        try:
            artifact = joblib.load(model_path)
        except Exception as error:
            raise ModelArtifactLoadError(
                f"Unable to load model artifact: {error}"
            ) from error

        estimator = _extract_estimator(artifact)
        feature_names = _extract_feature_names(
            artifact,
            estimator,
        )

        expected_feature_count = getattr(
            estimator,
            "n_features_in_",
            None,
        )

        if (
            expected_feature_count is not None
            and int(expected_feature_count) != len(feature_names)
        ):
            raise ModelFeatureSchemaError(
                "The number of stored feature names does not match "
                f"the estimator. Names: {len(feature_names)}, "
                f"estimator: {expected_feature_count}."
            )

        loaded_artifact = LoadedModelArtifact(
            artifact_path=str(model_path),
            estimator=estimator,
            feature_names=feature_names,
            thresholds=_extract_thresholds(artifact),
            positive_class=_extract_positive_class(
                artifact,
                estimator,
            ),
            metadata=_extract_metadata(artifact),
            feature_extraction_config=_extract_feature_config(
                artifact
            ),
        )

        _MODEL_CACHE = loaded_artifact

        logger.info(
            "Loaded model artifact with %s features from %s.",
            loaded_artifact.feature_count,
            model_path,
        )

        return loaded_artifact


def clear_model_cache() -> None:
    global _MODEL_CACHE

    with _MODEL_LOCK:
        _MODEL_CACHE = None


def get_expected_feature_names() -> list[str]:
    return list(
        load_model_artifact().feature_names
    )


def get_prediction_thresholds() -> PredictionThresholds:
    return load_model_artifact().thresholds


def _build_feature_config(
    loaded_artifact: LoadedModelArtifact,
) -> FeatureExtractionConfig:
    config_values: dict[str, Any] = {}

    django_config = getattr(
        settings,
        "AUDIO_FEATURE_EXTRACTION",
        None,
    )

    if isinstance(django_config, Mapping):
        config_values.update(django_config)

    config_values.update(
        loaded_artifact.feature_extraction_config
    )

    valid_fields = set(
        FeatureExtractionConfig.__dataclass_fields__
    )

    filtered_values = {
        key: value
        for key, value in config_values.items()
        if key in valid_fields
    }

    return FeatureExtractionConfig(
        **filtered_values
    )


def _validate_feature_dataframe(
    dataframe: pd.DataFrame,
    loaded_artifact: LoadedModelArtifact,
) -> pd.DataFrame:
    if dataframe.empty:
        raise ModelFeatureSchemaError(
            "Model input contains no rows."
        )

    expected_names = loaded_artifact.feature_names
    received_names = list(dataframe.columns)

    if received_names != expected_names:
        missing = [
            name
            for name in expected_names
            if name not in received_names
        ]

        unexpected = [
            name
            for name in received_names
            if name not in expected_names
        ]

        raise ModelFeatureSchemaError(
            "Model input feature schema mismatch. "
            f"Missing: {missing[:10]}. "
            f"Unexpected: {unexpected[:10]}."
        )

    try:
        numeric_dataframe = dataframe.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ModelFeatureSchemaError(
            "Model input contains non-numeric feature values."
        ) from error

    feature_array = numeric_dataframe.to_numpy()

    if not np.all(np.isfinite(feature_array)):
        raise ModelFeatureSchemaError(
            "Model input contains NaN or infinite values."
        )

    return numeric_dataframe


def _positive_class_index(
    estimator: Any,
    positive_class: Any,
    column_count: int,
) -> int:
    classes = getattr(estimator, "classes_", None)

    if classes is not None:
        class_values = list(classes)

        if positive_class in class_values:
            return class_values.index(positive_class)

        normalized_positive = str(
            positive_class
        ).strip().lower()

        for index, class_value in enumerate(class_values):
            if (
                str(class_value).strip().lower()
                == normalized_positive
            ):
                return index

    if column_count == 2:
        return 1

    raise ModelPredictionError(
        "Unable to identify the Parkinson positive-class column."
    )


def _extract_decision_score(
    estimator: Any,
    model_input: pd.DataFrame,
    positive_class: Any,
) -> float:
    if not hasattr(estimator, "decision_function"):
        raise ModelPredictionError(
            "The estimator does not expose decision_function()."
        )

    try:
        raw_scores = estimator.decision_function(
            model_input
        )
    except Exception as error:
        raise ModelPredictionError(
            f"decision_function failed: {error}"
        ) from error

    score_array = np.asarray(
        raw_scores,
        dtype=np.float64,
    )

    if score_array.ndim == 0:
        score = float(score_array)

    elif score_array.ndim == 1:
        if score_array.size != 1:
            raise ModelPredictionError(
                "Unexpected decision score shape."
            )

        score = float(score_array[0])

    elif score_array.ndim == 2:
        if score_array.shape[0] != 1:
            raise ModelPredictionError(
                "Expected one prediction row."
            )

        if score_array.shape[1] == 1:
            score = float(score_array[0, 0])
        else:
            index = _positive_class_index(
                estimator,
                positive_class,
                score_array.shape[1],
            )
            score = float(score_array[0, index])

    else:
        raise ModelPredictionError(
            "Unsupported decision score shape."
        )

    if not math.isfinite(score):
        raise ModelPredictionError(
            "The model returned a non-finite decision score."
        )

    return score


def _extract_probability(
    estimator: Any,
    model_input: pd.DataFrame,
    positive_class: Any,
) -> float | None:
    if not hasattr(estimator, "predict_proba"):
        return None

    try:
        raw_probabilities = estimator.predict_proba(
            model_input
        )
    except Exception:
        logger.exception(
            "Model probability extraction failed."
        )
        return None

    probability_array = np.asarray(
        raw_probabilities,
        dtype=np.float64,
    )

    if (
        probability_array.ndim != 2
        or probability_array.shape[0] != 1
    ):
        return None

    try:
        positive_index = _positive_class_index(
            estimator,
            positive_class,
            probability_array.shape[1],
        )
    except ModelPredictionError:
        return None

    probability = float(
        probability_array[0, positive_index]
    )

    if not math.isfinite(probability):
        return None

    return probability


def predict_feature_result(
    feature_result: FeatureExtractionResult,
    *,
    loaded_artifact: LoadedModelArtifact | None = None,
) -> RecordingPredictionResult:
    artifact = loaded_artifact or load_model_artifact()

    if not feature_result.extraction_successful:
        return RecordingPredictionResult(
            slot_number=feature_result.slot_number,
            audio_path=feature_result.audio_path,
            filename=feature_result.filename,
            decision_score=None,
            predicted_label=None,
            extraction_successful=False,
            prediction_successful=False,
            feature_count=0,
            error=(
                feature_result.error
                or "Feature extraction was unsuccessful."
            ),
        )

    try:
        dataframe = feature_result.to_dataframe()
        model_input = _validate_feature_dataframe(
            dataframe,
            artifact,
        )

        decision_score = _extract_decision_score(
            artifact.estimator,
            model_input,
            artifact.positive_class,
        )

        model_prediction = artifact.estimator.predict(
            model_input
        )

        prediction_array = np.asarray(
            model_prediction
        ).reshape(-1)

        raw_prediction = (
            prediction_array[0].item()
            if hasattr(prediction_array[0], "item")
            else prediction_array[0]
        )

        probability = _extract_probability(
            artifact.estimator,
            model_input,
            artifact.positive_class,
        )

        predicted_label = artifact.thresholds.classify(
            decision_score
        )

        return RecordingPredictionResult(
            slot_number=feature_result.slot_number,
            audio_path=feature_result.audio_path,
            filename=feature_result.filename,
            decision_score=decision_score,
            predicted_label=predicted_label,
            model_prediction=raw_prediction,
            probability=probability,
            extraction_successful=True,
            prediction_successful=True,
            feature_count=len(feature_result.feature_names),
            error=None,
        )

    except Exception as error:
        logger.exception(
            "Prediction failed for slot %s.",
            feature_result.slot_number,
        )

        return RecordingPredictionResult(
            slot_number=feature_result.slot_number,
            audio_path=feature_result.audio_path,
            filename=feature_result.filename,
            decision_score=None,
            predicted_label=None,
            model_prediction=None,
            probability=None,
            extraction_successful=True,
            prediction_successful=False,
            feature_count=len(feature_result.feature_names),
            error=str(error),
        )


def predict_audio_recording(
    audio_path: str | Path,
    slot_number: int,
    *,
    loaded_artifact: LoadedModelArtifact | None = None,
    safe: bool = True,
) -> RecordingPredictionResult:
    artifact = loaded_artifact or load_model_artifact()
    extraction_config = _build_feature_config(artifact)

    if safe:
        feature_result = extract_features_safely(
            audio_path=audio_path,
            slot_number=slot_number,
            expected_feature_names=artifact.feature_names,
            config=extraction_config,
        )
    else:
        feature_result = extract_features(
            audio_path=audio_path,
            slot_number=slot_number,
            expected_feature_names=artifact.feature_names,
            config=extraction_config,
        )

    return predict_feature_result(
        feature_result,
        loaded_artifact=artifact,
    )


def predict_feature_collection(
    collection: FeatureCollectionResult,
    *,
    loaded_artifact: LoadedModelArtifact | None = None,
) -> PredictionCollectionResult:
    artifact = loaded_artifact or load_model_artifact()

    results = [
        predict_feature_result(
            feature_result,
            loaded_artifact=artifact,
        )
        for feature_result in collection.recording_results
    ]

    successful_count = sum(
        result.prediction_successful
        for result in results
    )

    failed_count = len(results) - successful_count

    return PredictionCollectionResult(
        total_recordings=len(results),
        successful_predictions=successful_count,
        failed_predictions=failed_count,
        recording_results=results,
        processing_successful=successful_count > 0,
        error=(
            None
            if failed_count == 0
            else f"{failed_count} recording prediction(s) failed."
        ),
    )


def predict_audio_collection(
    recording_paths: Sequence[str | Path],
    *,
    loaded_artifact: LoadedModelArtifact | None = None,
    continue_on_error: bool = True,
) -> PredictionCollectionResult:
    artifact = loaded_artifact or load_model_artifact()

    if not recording_paths:
        raise ModelPredictionError(
            "No recording paths were supplied."
        )

    results: list[RecordingPredictionResult] = []

    for slot_number, audio_path in enumerate(
        recording_paths,
        start=1,
    ):
        result = predict_audio_recording(
            audio_path=audio_path,
            slot_number=slot_number,
            loaded_artifact=artifact,
            safe=continue_on_error,
        )

        results.append(result)

        if (
            not continue_on_error
            and not result.prediction_successful
        ):
            raise ModelPredictionError(
                result.error or "Prediction failed."
            )

    successful_count = sum(
        result.prediction_successful
        for result in results
    )

    failed_count = len(results) - successful_count

    return PredictionCollectionResult(
        total_recordings=len(results),
        successful_predictions=successful_count,
        failed_predictions=failed_count,
        recording_results=results,
        processing_successful=successful_count > 0,
        error=(
            None
            if failed_count == 0
            else f"{failed_count} recording prediction(s) failed."
        ),
    )


def predict_valid_recordings(
    quality_results: Iterable[Any],
    *,
    duplicate_slot_numbers: Iterable[int] | None = None,
    loaded_artifact: LoadedModelArtifact | None = None,
    continue_on_error: bool = True,
) -> PredictionCollectionResult:
    artifact = loaded_artifact or load_model_artifact()

    excluded_slots = {
        int(slot)
        for slot in (duplicate_slot_numbers or [])
    }

    results: list[RecordingPredictionResult] = []

    for quality_result in quality_results:
        if not bool(
            getattr(quality_result, "is_valid", False)
        ):
            continue

        slot_number = getattr(
            quality_result,
            "slot_number",
            None,
        )

        audio_path = getattr(
            quality_result,
            "audio_path",
            None,
        )

        if slot_number is None or not audio_path:
            raise ModelPredictionError(
                "A valid quality result is missing "
                "'slot_number' or 'audio_path'."
            )

        normalized_slot = int(slot_number)

        if normalized_slot in excluded_slots:
            continue

        prediction_result = predict_audio_recording(
            audio_path=audio_path,
            slot_number=normalized_slot,
            loaded_artifact=artifact,
            safe=continue_on_error,
        )

        results.append(prediction_result)

    successful_count = sum(
        result.prediction_successful
        for result in results
    )

    failed_count = len(results) - successful_count

    return PredictionCollectionResult(
        total_recordings=len(results),
        successful_predictions=successful_count,
        failed_predictions=failed_count,
        recording_results=results,
        processing_successful=successful_count > 0,
        error=(
            "No valid non-duplicate recordings were available."
            if not results
            else (
                None
                if failed_count == 0
                else f"{failed_count} recording prediction(s) failed."
            )
        ),
    )


def inspect_model_artifact() -> dict[str, Any]:
    artifact = load_model_artifact()

    estimator = artifact.estimator
    classes = getattr(estimator, "classes_", None)
    feature_count = getattr(
        estimator,
        "n_features_in_",
        None,
    )

    return {
        **artifact.to_dict(),
        "estimator_classes": (
            [
                value.item()
                if hasattr(value, "item")
                else value
                for value in classes
            ]
            if classes is not None
            else None
        ),
        "estimator_feature_count": (
            int(feature_count)
            if feature_count is not None
            else None
        ),
        "supports_decision_function": hasattr(
            estimator,
            "decision_function",
        ),
        "supports_predict_proba": hasattr(
            estimator,
            "predict_proba",
        ),
    }


def build_prediction_summary(
    result: PredictionCollectionResult,
) -> dict[str, Any]:
    return {
        "total_recordings": result.total_recordings,
        "successful_predictions": (
            result.successful_predictions
        ),
        "failed_predictions": (
            result.failed_predictions
        ),
        "valid_slot_numbers": result.valid_slot_numbers,
        "decision_scores": result.valid_scores,
        "processing_successful": (
            result.processing_successful
        ),
        "message": (
            "All recordings were predicted successfully."
            if result.processing_successful
            and result.failed_predictions == 0
            else (
                "Prediction completed, but some recordings failed."
                if result.processing_successful
                else "No recording could be predicted."
            )
        ),
    }