"""
Production feature extraction for the NeuroSense Linear SVM model.

This module reproduces the sustained-vowel feature engineering used by the
training notebook and preserves the public API expected by model_service.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from scipy.signal import welch


logger = logging.getLogger(__name__)

EXPECTED_SAMPLE_RATE = 8000
EXPECTED_CHANNELS = 1
PITCH_FLOOR = 60.0
PITCH_CEILING = 400.0
PITCH_TIME_STEP = 0.01

TRAINING_FEATURE_NAMES = [
    "pitch_mean",
    "pitch_median",
    "pitch_std",
    "pitch_min",
    "pitch_max",
    "jitter_local",
    "jitter_absolute",
    "jitter_rap",
    "jitter_ppq5",
    "shimmer_local",
    "shimmer_db",
    "shimmer_apq3",
    "shimmer_apq5",
    "shimmer_apq11",
    "hnr_mean",
    "energy_rms_mean",
    "energy_rms_std",
    "pitch_coefficient_variation",
    "tremor_power_ratio",
    *[
        feature_name
        for coefficient in range(1, 14)
        for feature_name in (
            f"mfcc_{coefficient}_mean",
            f"mfcc_{coefficient}_std",
        )
    ],
]


class FeatureExtractionError(Exception):
    """Base exception for feature-extraction failures."""


class InvalidFeatureInputError(FeatureExtractionError):
    """Raised when an input recording is missing or invalid."""


class AudioSchemaError(FeatureExtractionError):
    """Raised when the standardized audio format is incorrect."""


class OpenSmileExtractionError(FeatureExtractionError):
    """Retained for backward compatibility."""


class ParselmouthExtractionError(FeatureExtractionError):
    """Raised when Praat/Parselmouth extraction fails."""


class LibrosaExtractionError(FeatureExtractionError):
    """Raised when librosa extraction fails."""


class FeatureAlignmentError(FeatureExtractionError):
    """Raised when extracted features cannot match the model schema."""


@dataclass(frozen=True)
class FeatureExtractionConfig:
    expected_sample_rate: int = EXPECTED_SAMPLE_RATE
    expected_channels: int = EXPECTED_CHANNELS

    f0_min_hz: float = PITCH_FLOOR
    f0_max_hz: float = PITCH_CEILING

    frame_length: int = 256
    hop_length: int = 80
    n_fft: int = 256
    n_mfcc: int = 13

    # Kept for compatibility with existing settings/model_service.
    include_opensmile: bool = False
    include_parselmouth: bool = True
    include_librosa: bool = True

    replace_non_finite: bool = True
    non_finite_replacement: float = 0.0

    strict_feature_alignment: bool = True
    missing_feature_fill_value: float = 0.0
    reject_unexpected_features: bool = False


@dataclass
class FeatureExtractionResult:
    slot_number: int
    audio_path: str
    filename: str

    feature_values: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    opensmile_feature_count: int = 0
    parselmouth_feature_count: int = 0
    librosa_feature_count: int = 0
    total_feature_count: int = 0

    missing_expected_features: list[str] = field(default_factory=list)
    unexpected_features: list[str] = field(default_factory=list)
    replaced_non_finite_features: list[str] = field(default_factory=list)

    extraction_successful: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            name: self.feature_values[name]
            for name in self.feature_names
        }])

    def to_numpy(
        self,
        dtype: np.dtype[Any] | type = np.float64,
    ) -> np.ndarray:
        return np.asarray([[
            self.feature_values[name]
            for name in self.feature_names
        ]], dtype=dtype)


@dataclass
class FeatureCollectionResult:
    total_recordings: int
    successful_recordings: int
    failed_recordings: int

    feature_names: list[str] = field(default_factory=list)
    recording_results: list[FeatureExtractionResult] = field(default_factory=list)

    processing_successful: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_recordings": self.total_recordings,
            "successful_recordings": self.successful_recordings,
            "failed_recordings": self.failed_recordings,
            "feature_names": list(self.feature_names),
            "recording_results": [
                result.to_dict() for result in self.recording_results
            ],
            "processing_successful": self.processing_successful,
            "error": self.error,
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        slots: list[int] = []

        for result in self.recording_results:
            if not result.extraction_successful:
                continue
            rows.append({
                name: result.feature_values[name]
                for name in self.feature_names
            })
            slots.append(result.slot_number)

        dataframe = pd.DataFrame(rows, columns=self.feature_names)

        if slots:
            dataframe.index = slots
            dataframe.index.name = "slot_number"

        return dataframe


def validate_slot_number(slot_number: int) -> int:
    try:
        normalized = int(slot_number)
    except (TypeError, ValueError) as error:
        raise InvalidFeatureInputError(
            "Slot number must be an integer."
        ) from error

    if normalized < 1:
        raise InvalidFeatureInputError(
            "Slot number must be at least 1."
        )

    return normalized


def validate_audio_path(audio_path: str | Path) -> Path:
    path = Path(audio_path).expanduser().resolve()

    if not path.exists() or not path.is_file():
        raise InvalidFeatureInputError(
            f"Audio file does not exist: {path}"
        )

    if path.suffix.lower() != ".wav":
        raise InvalidFeatureInputError(
            "Feature extraction requires a standardized WAV file. "
            f"Received: {path.name}"
        )

    if path.stat().st_size <= 0:
        raise InvalidFeatureInputError(
            f"Audio file is empty: {path.name}"
        )

    return path


def validate_config(
    config: FeatureExtractionConfig,
) -> FeatureExtractionConfig:
    if config.expected_sample_rate <= 0:
        raise InvalidFeatureInputError(
            "Expected sample rate must be greater than zero."
        )

    if config.expected_channels != 1:
        raise InvalidFeatureInputError(
            "This model requires mono audio."
        )

    if config.f0_min_hz <= 0:
        raise InvalidFeatureInputError(
            "Minimum pitch must be greater than zero."
        )

    if config.f0_max_hz <= config.f0_min_hz:
        raise InvalidFeatureInputError(
            "Maximum pitch must exceed minimum pitch."
        )

    if config.frame_length <= 0 or config.hop_length <= 0:
        raise InvalidFeatureInputError(
            "Frame and hop lengths must be positive."
        )

    if config.n_fft <= 0 or config.n_mfcc <= 0:
        raise InvalidFeatureInputError(
            "FFT and MFCC counts must be positive."
        )

    return config


def validate_standardized_audio(
    audio_path: str | Path,
    config: FeatureExtractionConfig,
) -> sf.SoundFile:
    path = validate_audio_path(audio_path)

    try:
        with sf.SoundFile(str(path), mode="r") as audio_file:
            sample_rate = int(audio_file.samplerate)
            channels = int(audio_file.channels)
            frames = int(audio_file.frames)
            subtype = str(audio_file.subtype or "").upper()
            file_format = str(audio_file.format or "").upper()
    except (RuntimeError, OSError) as error:
        raise AudioSchemaError(
            f"Unable to read standardized WAV: {path.name}"
        ) from error

    if sample_rate != config.expected_sample_rate:
        raise AudioSchemaError(
            f"Expected {config.expected_sample_rate} Hz but received "
            f"{sample_rate} Hz for {path.name}."
        )

    if channels != config.expected_channels:
        raise AudioSchemaError(
            f"Expected mono audio but received {channels} channels "
            f"for {path.name}."
        )

    if frames <= 0:
        raise AudioSchemaError(
            f"Audio file contains no frames: {path.name}"
        )

    if file_format != "WAV":
        raise AudioSchemaError(
            f"Expected WAV format for {path.name}."
        )

    if subtype != "PCM_16":
        raise AudioSchemaError(
            f"Expected PCM 16-bit WAV but received {subtype} "
            f"for {path.name}."
        )

    return sf.SoundFile(str(path), mode="r")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)

    return converted if math.isfinite(converted) else float(default)


def sanitize_feature_mapping(
    features: Mapping[str, Any],
    replacement_value: float = 0.0,
) -> tuple[dict[str, float], list[str]]:
    sanitized: dict[str, float] = {}
    replaced: list[str] = []

    for raw_name, raw_value in features.items():
        name = str(raw_name)

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = float(replacement_value)
            replaced.append(name)
        else:
            if not math.isfinite(value):
                value = float(replacement_value)
                replaced.append(name)

        sanitized[name] = value

    return sanitized, replaced


def load_standardized_audio(
    audio_path: str | Path,
    config: FeatureExtractionConfig,
) -> tuple[np.ndarray, int]:
    """
    Reproduce the training notebook's post-load preprocessing:
    silence trim, DC-offset removal, and peak normalization.
    """
    path = validate_audio_path(audio_path)

    try:
        signal, original_sample_rate = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )
    except (RuntimeError, OSError) as error:
        raise InvalidFeatureInputError(
            f"Unable to load audio file: {path.name}"
        ) from error

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    if signal.size == 0:
        raise InvalidFeatureInputError(
            f"Audio file contains no samples: {path.name}"
        )

    signal = np.nan_to_num(signal)

    if int(original_sample_rate) != config.expected_sample_rate:
        signal = librosa.resample(
            signal,
            orig_sr=int(original_sample_rate),
            target_sr=config.expected_sample_rate,
        )

    signal, _ = librosa.effects.trim(
        signal,
        top_db=30,
    )

    if signal.size == 0:
        raise InvalidFeatureInputError(
            "The recording is empty after silence trimming."
        )

    signal = signal - np.mean(signal)

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = 0.99 * signal / peak

    return signal.astype(np.float32), config.expected_sample_rate


def safe_praat_call(*arguments: Any) -> float:
    try:
        value = call(*arguments)
        return float(value) if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def calculate_tremor_features(
    pitch_values: np.ndarray,
    time_step: float = PITCH_TIME_STEP,
) -> tuple[float, float]:
    pitch_values = np.asarray(pitch_values, dtype=float)

    voiced_mask = (
        np.isfinite(pitch_values)
        & (pitch_values > 0)
    )

    if int(voiced_mask.sum()) < 10:
        return float("nan"), float("nan")

    frame_indices = np.arange(len(pitch_values))

    interpolated_pitch = np.interp(
        frame_indices,
        frame_indices[voiced_mask],
        pitch_values[voiced_mask],
    )

    normalized_pitch = (
        interpolated_pitch
        - np.mean(interpolated_pitch)
    )

    pitch_mean = float(np.mean(interpolated_pitch))

    pitch_cv = (
        float(np.std(interpolated_pitch) / pitch_mean)
        if pitch_mean != 0
        else float("nan")
    )

    frequencies, power = welch(
        normalized_pitch,
        fs=1.0 / time_step,
        nperseg=min(256, len(normalized_pitch)),
    )

    tremor_band = (
        (frequencies >= 3)
        & (frequencies <= 12)
    )

    total_band = (
        (frequencies >= 0.5)
        & (frequencies <= 20)
    )

    total_power = float(np.sum(power[total_band]))
    tremor_power = float(np.sum(power[tremor_band]))

    tremor_ratio = (
        tremor_power / total_power
        if total_power > 0
        else float("nan")
    )

    return float(pitch_cv), float(tremor_ratio)


def extract_sustained_vowel_features(
    audio_path: str | Path,
    config: FeatureExtractionConfig,
) -> dict[str, float]:
    """
    Reproduce the training notebook's exact 45-feature calculation.
    The deployment model currently selects 37 of these features.
    """
    signal, sample_rate = load_standardized_audio(
        audio_path,
        config,
    )

    try:
        sound = parselmouth.Sound(
            signal,
            sampling_frequency=sample_rate,
        )

        pitch = sound.to_pitch(
            time_step=PITCH_TIME_STEP,
            pitch_floor=config.f0_min_hz,
            pitch_ceiling=config.f0_max_hz,
        )

        pitch_values = np.asarray(
            pitch.selected_array["frequency"],
            dtype=np.float64,
        )

        voiced_pitch = pitch_values[
            np.isfinite(pitch_values)
            & (pitch_values > 0)
        ]

        if voiced_pitch.size == 0:
            pitch_mean = pitch_median = pitch_std = float("nan")
            pitch_min = pitch_max = float("nan")
        else:
            pitch_mean = float(np.mean(voiced_pitch))
            pitch_median = float(np.median(voiced_pitch))
            pitch_std = float(np.std(voiced_pitch))
            pitch_min = float(np.min(voiced_pitch))
            pitch_max = float(np.max(voiced_pitch))

        point_process = call(
            sound,
            "To PointProcess (periodic, cc)",
            config.f0_min_hz,
            config.f0_max_hz,
        )

        jitter_local = safe_praat_call(
            point_process,
            "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3,
        )

        jitter_absolute = safe_praat_call(
            point_process,
            "Get jitter (local, absolute)",
            0, 0, 0.0001, 0.02, 1.3,
        )

        jitter_rap = safe_praat_call(
            point_process,
            "Get jitter (rap)",
            0, 0, 0.0001, 0.02, 1.3,
        )

        jitter_ppq5 = safe_praat_call(
            point_process,
            "Get jitter (ppq5)",
            0, 0, 0.0001, 0.02, 1.3,
        )

        shimmer_local = safe_praat_call(
            [sound, point_process],
            "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )

        shimmer_db = safe_praat_call(
            [sound, point_process],
            "Get shimmer (local_dB)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )

        shimmer_apq3 = safe_praat_call(
            [sound, point_process],
            "Get shimmer (apq3)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )

        shimmer_apq5 = safe_praat_call(
            [sound, point_process],
            "Get shimmer (apq5)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )

        shimmer_apq11 = safe_praat_call(
            [sound, point_process],
            "Get shimmer (apq11)",
            0, 0, 0.0001, 0.02, 1.3, 1.6,
        )

        harmonicity = call(
            sound,
            "To Harmonicity (cc)",
            PITCH_TIME_STEP,
            config.f0_min_hz,
            0.1,
            1.0,
        )

        hnr_values = np.asarray(
            harmonicity.values,
            dtype=np.float64,
        ).flatten()

        valid_hnr = hnr_values[
            np.isfinite(hnr_values)
            & (hnr_values > -100)
        ]

        hnr_mean = (
            float(np.mean(valid_hnr))
            if valid_hnr.size > 0
            else float("nan")
        )

    except Exception as error:
        raise ParselmouthExtractionError(
            f"Parselmouth extraction failed for "
            f"{Path(audio_path).name}: {error}"
        ) from error

    try:
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sample_rate,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=200,
        )

        rms = librosa.feature.rms(
            y=signal,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
        )[0]

    except Exception as error:
        raise LibrosaExtractionError(
            f"librosa extraction failed for "
            f"{Path(audio_path).name}: {error}"
        ) from error

    pitch_cv, tremor_ratio = calculate_tremor_features(
        pitch_values,
        time_step=PITCH_TIME_STEP,
    )

    features: dict[str, float] = {
        "pitch_mean": pitch_mean,
        "pitch_median": pitch_median,
        "pitch_std": pitch_std,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,

        "jitter_local": jitter_local * 100,
        "jitter_absolute": jitter_absolute,
        "jitter_rap": jitter_rap * 100,
        "jitter_ppq5": jitter_ppq5 * 100,

        "shimmer_local": shimmer_local * 100,
        "shimmer_db": shimmer_db,
        "shimmer_apq3": shimmer_apq3 * 100,
        "shimmer_apq5": shimmer_apq5 * 100,
        "shimmer_apq11": shimmer_apq11 * 100,

        "hnr_mean": hnr_mean,

        "energy_rms_mean": float(np.mean(rms)),
        "energy_rms_std": float(np.std(rms)),

        "pitch_coefficient_variation": pitch_cv,
        "tremor_power_ratio": tremor_ratio,
    }

    for coefficient in range(config.n_mfcc):
        features[f"mfcc_{coefficient + 1}_mean"] = float(
            np.mean(mfcc[coefficient])
        )
        features[f"mfcc_{coefficient + 1}_std"] = float(
            np.std(mfcc[coefficient])
        )

    return features


def normalize_expected_feature_names(
    expected_feature_names: Sequence[str] | None,
) -> list[str] | None:
    if expected_feature_names is None:
        return None

    normalized = [
        str(name).strip()
        for name in expected_feature_names
    ]

    if not normalized:
        raise FeatureAlignmentError(
            "Expected feature-name sequence cannot be empty."
        )

    if any(not name for name in normalized):
        raise FeatureAlignmentError(
            "Expected feature names cannot contain empty values."
        )

    if len(normalized) != len(set(normalized)):
        raise FeatureAlignmentError(
            "Expected feature schema contains duplicate names."
        )

    return normalized


def align_features_to_schema(
    extracted_features: Mapping[str, float],
    expected_feature_names: Sequence[str],
    *,
    strict: bool = True,
    missing_fill_value: float = 0.0,
    reject_unexpected: bool = False,
) -> tuple[dict[str, float], list[str], list[str]]:
    expected_names = normalize_expected_feature_names(
        expected_feature_names
    )

    if expected_names is None:
        raise FeatureAlignmentError(
            "Expected feature names are required for alignment."
        )

    expected_set = set(expected_names)
    extracted_set = set(extracted_features)

    missing = [
        name for name in expected_names
        if name not in extracted_set
    ]
    unexpected = sorted(extracted_set - expected_set)

    if strict and missing:
        preview = ", ".join(missing[:10])
        if len(missing) > 10:
            preview += f", ... ({len(missing)} total)"
        raise FeatureAlignmentError(
            "Extracted features do not match the trained model. "
            f"Missing expected features: {preview}"
        )

    if reject_unexpected and unexpected:
        preview = ", ".join(unexpected[:10])
        if len(unexpected) > 10:
            preview += f", ... ({len(unexpected)} total)"
        raise FeatureAlignmentError(
            f"Unexpected features were extracted: {preview}"
        )

    aligned = {
        name: safe_float(
            extracted_features.get(name, missing_fill_value),
            default=missing_fill_value,
        )
        for name in expected_names
    }

    return aligned, missing, unexpected


def extract_features(
    audio_path: str | Path,
    slot_number: int,
    *,
    expected_feature_names: Sequence[str] | None = None,
    config: FeatureExtractionConfig | None = None,
) -> FeatureExtractionResult:
    active_config = validate_config(
        config or FeatureExtractionConfig()
    )

    normalized_slot = validate_slot_number(slot_number)
    path = validate_audio_path(audio_path)

    metadata_handle = validate_standardized_audio(
        path,
        active_config,
    )
    metadata_handle.close()

    combined_features = extract_sustained_vowel_features(
        path,
        active_config,
    )

    replaced_non_finite: list[str] = []

    if active_config.replace_non_finite:
        combined_features, replaced_non_finite = (
            sanitize_feature_mapping(
                combined_features,
                active_config.non_finite_replacement,
            )
        )

    expected_names = normalize_expected_feature_names(
        expected_feature_names
    )

    missing: list[str] = []
    unexpected: list[str] = []

    if expected_names is not None:
        final_features, missing, unexpected = (
            align_features_to_schema(
                combined_features,
                expected_names,
                strict=active_config.strict_feature_alignment,
                missing_fill_value=(
                    active_config.missing_feature_fill_value
                ),
                reject_unexpected=(
                    active_config.reject_unexpected_features
                ),
            )
        )
        final_names = list(expected_names)
    else:
        final_names = list(TRAINING_FEATURE_NAMES)
        final_features = {
            name: combined_features[name]
            for name in final_names
        }

    parselmouth_names = {
        "pitch_mean",
        "pitch_median",
        "pitch_std",
        "pitch_min",
        "pitch_max",
        "jitter_local",
        "jitter_absolute",
        "jitter_rap",
        "jitter_ppq5",
        "shimmer_local",
        "shimmer_db",
        "shimmer_apq3",
        "shimmer_apq5",
        "shimmer_apq11",
        "hnr_mean",
        "pitch_coefficient_variation",
        "tremor_power_ratio",
    }

    parselmouth_count = sum(
        name in parselmouth_names
        for name in final_names
    )
    librosa_count = len(final_names) - parselmouth_count

    logger.info(
        "Extracted %s model-compatible features from slot %s (%s).",
        len(final_names),
        normalized_slot,
        path.name,
    )

    return FeatureExtractionResult(
        slot_number=normalized_slot,
        audio_path=str(path),
        filename=path.name,
        feature_values=final_features,
        feature_names=final_names,
        opensmile_feature_count=0,
        parselmouth_feature_count=parselmouth_count,
        librosa_feature_count=librosa_count,
        total_feature_count=len(final_names),
        missing_expected_features=missing,
        unexpected_features=unexpected,
        replaced_non_finite_features=replaced_non_finite,
        extraction_successful=True,
        error=None,
    )


def extract_features_safely(
    audio_path: str | Path,
    slot_number: int,
    *,
    expected_feature_names: Sequence[str] | None = None,
    config: FeatureExtractionConfig | None = None,
) -> FeatureExtractionResult:
    try:
        return extract_features(
            audio_path=audio_path,
            slot_number=slot_number,
            expected_feature_names=expected_feature_names,
            config=config,
        )
    except Exception as error:
        logger.exception(
            "Feature extraction failed for slot %s.",
            slot_number,
        )

        path = Path(audio_path)

        return FeatureExtractionResult(
            slot_number=int(slot_number),
            audio_path=str(path),
            filename=path.name,
            extraction_successful=False,
            error=str(error),
        )


def extract_recording_collection(
    recording_paths: Sequence[str | Path],
    *,
    expected_feature_names: Sequence[str] | None = None,
    config: FeatureExtractionConfig | None = None,
    continue_on_error: bool = True,
) -> FeatureCollectionResult:
    paths = list(recording_paths)

    if not paths:
        raise InvalidFeatureInputError(
            "No recording paths were supplied."
        )

    results: list[FeatureExtractionResult] = []

    for slot_number, audio_path in enumerate(paths, start=1):
        extractor = (
            extract_features_safely
            if continue_on_error
            else extract_features
        )
        results.append(
            extractor(
                audio_path=audio_path,
                slot_number=slot_number,
                expected_feature_names=expected_feature_names,
                config=config,
            )
        )

    return _build_collection_result(
        results,
        expected_feature_names,
    )


def extract_features_from_valid_quality_results(
    quality_results: Iterable[Any],
    *,
    expected_feature_names: Sequence[str] | None = None,
    duplicate_slot_numbers: Iterable[int] | None = None,
    config: FeatureExtractionConfig | None = None,
    continue_on_error: bool = True,
) -> FeatureCollectionResult:
    excluded_slots = {
        int(slot)
        for slot in (duplicate_slot_numbers or [])
    }

    results: list[FeatureExtractionResult] = []

    for quality_result in quality_results:
        if not bool(getattr(quality_result, "is_valid", False)):
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
            raise InvalidFeatureInputError(
                "A valid quality result is missing "
                "'slot_number' or 'audio_path'."
            )

        normalized_slot = validate_slot_number(slot_number)

        if normalized_slot in excluded_slots:
            continue

        extractor = (
            extract_features_safely
            if continue_on_error
            else extract_features
        )

        results.append(
            extractor(
                audio_path=audio_path,
                slot_number=normalized_slot,
                expected_feature_names=expected_feature_names,
                config=config,
            )
        )

    if not results:
        return FeatureCollectionResult(
            total_recordings=0,
            successful_recordings=0,
            failed_recordings=0,
            feature_names=(
                list(expected_feature_names)
                if expected_feature_names is not None
                else []
            ),
            recording_results=[],
            processing_successful=False,
            error=(
                "No valid non-duplicate recordings were available "
                "for feature extraction."
            ),
        )

    return _build_collection_result(
        results,
        expected_feature_names,
    )


def _build_collection_result(
    results: Sequence[FeatureExtractionResult],
    expected_feature_names: Sequence[str] | None,
) -> FeatureCollectionResult:
    successful = [
        result for result in results
        if result.extraction_successful
    ]
    failed = [
        result for result in results
        if not result.extraction_successful
    ]

    expected_names = normalize_expected_feature_names(
        expected_feature_names
    )

    if expected_names is not None:
        feature_names = expected_names
    elif successful:
        feature_names = list(successful[0].feature_names)
    else:
        feature_names = []

    return FeatureCollectionResult(
        total_recordings=len(results),
        successful_recordings=len(successful),
        failed_recordings=len(failed),
        feature_names=feature_names,
        recording_results=list(results),
        processing_successful=bool(successful),
        error=(
            None
            if not failed
            else f"{len(failed)} recording(s) failed feature extraction."
        ),
    )
