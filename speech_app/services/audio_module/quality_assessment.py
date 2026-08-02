"""
Audio quality assessment service for NeuroSense.

This module evaluates standardized WAV recordings before feature
extraction and prediction.

Blocking conditions
-------------------
- Duration below 3 seconds
- RMS below the configured minimum
- Estimated SNR below 8 dB
- Silence above 45%
- Excessive clipping
- Unexpected sample rate or channel count
- Invalid, unreadable, or empty audio

Warning condition
-----------------
- Duration above 5 seconds

Recordings longer than 5 seconds are not trimmed. They remain valid
when all blocking checks pass.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf
from django.conf import settings


logger = logging.getLogger(__name__)


# ============================================================
# DEFAULT QUALITY CONFIGURATION
# ============================================================

DEFAULT_MINIMUM_DURATION_SECONDS = 3.0
DEFAULT_RECOMMENDED_MAXIMUM_DURATION_SECONDS = 5.0

DEFAULT_MINIMUM_SNR_DB = 8.0
DEFAULT_MAXIMUM_SILENCE_PERCENTAGE = 45.0

DEFAULT_MINIMUM_RMS_DBFS = -40.0

DEFAULT_CLIPPING_AMPLITUDE_THRESHOLD = 0.99
DEFAULT_MAXIMUM_CLIPPING_PERCENTAGE = 1.0

DEFAULT_SILENCE_THRESHOLD_DBFS = -40.0

DEFAULT_FRAME_LENGTH_MS = 25.0
DEFAULT_HOP_LENGTH_MS = 10.0

DEFAULT_EXPECTED_SAMPLE_RATE = 8_000
DEFAULT_EXPECTED_CHANNELS = 1


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class AudioQualityError(Exception):
    """Base exception for audio-quality failures."""


class InvalidQualityConfigurationError(AudioQualityError):
    """Raised when quality thresholds are invalid."""


class AudioQualityProcessingError(AudioQualityError):
    """Raised when an audio recording cannot be assessed."""


# ============================================================
# CONFIGURATION OBJECT
# ============================================================

@dataclass(frozen=True)
class AudioQualityConfig:
    """Thresholds used during audio-quality assessment."""

    minimum_duration_seconds: float = (
        DEFAULT_MINIMUM_DURATION_SECONDS
    )

    recommended_maximum_duration_seconds: float = (
        DEFAULT_RECOMMENDED_MAXIMUM_DURATION_SECONDS
    )

    minimum_snr_db: float = DEFAULT_MINIMUM_SNR_DB

    maximum_silence_percentage: float = (
        DEFAULT_MAXIMUM_SILENCE_PERCENTAGE
    )

    minimum_rms_dbfs: float = DEFAULT_MINIMUM_RMS_DBFS

    clipping_amplitude_threshold: float = (
        DEFAULT_CLIPPING_AMPLITUDE_THRESHOLD
    )

    maximum_clipping_percentage: float = (
        DEFAULT_MAXIMUM_CLIPPING_PERCENTAGE
    )

    silence_threshold_dbfs: float = (
        DEFAULT_SILENCE_THRESHOLD_DBFS
    )

    frame_length_ms: float = DEFAULT_FRAME_LENGTH_MS
    hop_length_ms: float = DEFAULT_HOP_LENGTH_MS

    expected_sample_rate: int = (
        DEFAULT_EXPECTED_SAMPLE_RATE
    )

    expected_channels: int = DEFAULT_EXPECTED_CHANNELS

    def validate(self) -> None:
        """Validate the complete quality configuration."""

        if self.minimum_duration_seconds <= 0:
            raise InvalidQualityConfigurationError(
                "Minimum duration must be greater than zero."
            )

        if (
            self.recommended_maximum_duration_seconds
            < self.minimum_duration_seconds
        ):
            raise InvalidQualityConfigurationError(
                "Recommended maximum duration cannot be lower "
                "than the minimum duration."
            )

        if not 0 <= self.maximum_silence_percentage <= 100:
            raise InvalidQualityConfigurationError(
                "Maximum silence percentage must be between "
                "0 and 100."
            )

        if not 0 < self.clipping_amplitude_threshold <= 1:
            raise InvalidQualityConfigurationError(
                "Clipping amplitude threshold must be within "
                "(0, 1]."
            )

        if not 0 <= self.maximum_clipping_percentage <= 100:
            raise InvalidQualityConfigurationError(
                "Maximum clipping percentage must be between "
                "0 and 100."
            )

        if self.frame_length_ms <= 0:
            raise InvalidQualityConfigurationError(
                "Frame length must be greater than zero."
            )

        if self.hop_length_ms <= 0:
            raise InvalidQualityConfigurationError(
                "Hop length must be greater than zero."
            )

        if self.expected_sample_rate <= 0:
            raise InvalidQualityConfigurationError(
                "Expected sample rate must be greater than zero."
            )

        if self.expected_channels <= 0:
            raise InvalidQualityConfigurationError(
                "Expected channel count must be greater than zero."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return configuration as a dictionary."""
        return asdict(self)


def get_audio_quality_config() -> AudioQualityConfig:
    """
    Build quality configuration from Django settings.

    Optional settings.py configuration:

        AUDIO_QUALITY = {
            "minimum_duration_seconds": 3.0,
            "recommended_maximum_duration_seconds": 5.0,
            "minimum_snr_db": 8.0,
            "maximum_silence_percentage": 45.0,
            "minimum_rms_dbfs": -40.0,
            "clipping_amplitude_threshold": 0.99,
            "maximum_clipping_percentage": 1.0,
            "silence_threshold_dbfs": -40.0,
            "frame_length_ms": 25.0,
            "hop_length_ms": 10.0,
            "expected_sample_rate": 8000,
            "expected_channels": 1,
        }
    """

    configured_values = getattr(
        settings,
        "AUDIO_QUALITY",
        {},
    )

    if configured_values is None:
        configured_values = {}

    if not isinstance(configured_values, dict):
        raise InvalidQualityConfigurationError(
            "AUDIO_QUALITY must be a dictionary."
        )

    config = AudioQualityConfig(
        minimum_duration_seconds=float(
            configured_values.get(
                "minimum_duration_seconds",
                DEFAULT_MINIMUM_DURATION_SECONDS,
            )
        ),
        recommended_maximum_duration_seconds=float(
            configured_values.get(
                "recommended_maximum_duration_seconds",
                DEFAULT_RECOMMENDED_MAXIMUM_DURATION_SECONDS,
            )
        ),
        minimum_snr_db=float(
            configured_values.get(
                "minimum_snr_db",
                DEFAULT_MINIMUM_SNR_DB,
            )
        ),
        maximum_silence_percentage=float(
            configured_values.get(
                "maximum_silence_percentage",
                DEFAULT_MAXIMUM_SILENCE_PERCENTAGE,
            )
        ),
        minimum_rms_dbfs=float(
            configured_values.get(
                "minimum_rms_dbfs",
                DEFAULT_MINIMUM_RMS_DBFS,
            )
        ),
        clipping_amplitude_threshold=float(
            configured_values.get(
                "clipping_amplitude_threshold",
                DEFAULT_CLIPPING_AMPLITUDE_THRESHOLD,
            )
        ),
        maximum_clipping_percentage=float(
            configured_values.get(
                "maximum_clipping_percentage",
                DEFAULT_MAXIMUM_CLIPPING_PERCENTAGE,
            )
        ),
        silence_threshold_dbfs=float(
            configured_values.get(
                "silence_threshold_dbfs",
                DEFAULT_SILENCE_THRESHOLD_DBFS,
            )
        ),
        frame_length_ms=float(
            configured_values.get(
                "frame_length_ms",
                DEFAULT_FRAME_LENGTH_MS,
            )
        ),
        hop_length_ms=float(
            configured_values.get(
                "hop_length_ms",
                DEFAULT_HOP_LENGTH_MS,
            )
        ),
        expected_sample_rate=int(
            configured_values.get(
                "expected_sample_rate",
                DEFAULT_EXPECTED_SAMPLE_RATE,
            )
        ),
        expected_channels=int(
            configured_values.get(
                "expected_channels",
                DEFAULT_EXPECTED_CHANNELS,
            )
        ),
    )

    config.validate()

    return config


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class AudioQualityResult:
    """Quality-assessment result for one recording."""

    slot_number: int
    audio_path: str
    filename: str

    processing_successful: bool = False
    is_valid: bool = False
    quality_status: str = "processing_failure"

    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    sample_count: int | None = None

    rms: float | None = None
    rms_dbfs: float | None = None
    peak_amplitude: float | None = None

    clipped_sample_count: int | None = None
    clipping_percentage: float | None = None

    estimated_snr_db: float | None = None
    silence_percentage: float | None = None

    blocking_reasons: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a serializable dictionary."""
        return asdict(self)


# ============================================================
# SIGNAL HELPERS
# ============================================================

def amplitude_to_dbfs(
    amplitude: float | np.ndarray,
    epsilon: float = 1e-12,
) -> float | np.ndarray:
    """
    Convert full-scale linear amplitude to dBFS.
    """

    safe_amplitude = np.maximum(
        np.asarray(amplitude, dtype=np.float64),
        epsilon,
    )

    result = 20.0 * np.log10(safe_amplitude)

    if np.ndim(result) == 0:
        return float(result)

    return result


def create_audio_frames(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    """
    Divide one-dimensional audio into overlapping frames.

    The final frame is padded with zeros when needed.
    """

    signal = np.asarray(
        audio,
        dtype=np.float64,
    ).reshape(-1)

    if signal.size == 0:
        raise AudioQualityProcessingError(
            "Cannot frame an empty audio signal."
        )

    if frame_length <= 0 or hop_length <= 0:
        raise AudioQualityProcessingError(
            "Frame and hop lengths must be positive."
        )

    if signal.size <= frame_length:
        padded_signal = np.pad(
            signal,
            (0, frame_length - signal.size),
            mode="constant",
        )

        return padded_signal.reshape(1, -1)

    frame_count = int(
        np.ceil(
            (signal.size - frame_length)
            / hop_length
        )
    ) + 1

    required_length = (
        (frame_count - 1) * hop_length
        + frame_length
    )

    if required_length > signal.size:
        signal = np.pad(
            signal,
            (0, required_length - signal.size),
            mode="constant",
        )

    indices = (
        np.arange(frame_length)[None, :]
        + hop_length
        * np.arange(frame_count)[:, None]
    )

    return signal[indices]


def calculate_frame_rms(
    frames: np.ndarray,
) -> np.ndarray:
    """Calculate RMS amplitude for every frame."""

    frame_array = np.asarray(
        frames,
        dtype=np.float64,
    )

    if frame_array.ndim != 2:
        raise AudioQualityProcessingError(
            "Audio frames must be two-dimensional."
        )

    return np.sqrt(
        np.mean(
            np.square(frame_array),
            axis=1,
        )
    )


def estimate_snr_db(
    frame_rms: np.ndarray,
    silence_threshold_dbfs: float,
    epsilon: float = 1e-12,
) -> float:
    """
    Estimate signal-to-noise ratio using frame-level RMS.

    The quietest 20% of frames estimate background noise.
    Frames above the silence threshold estimate the signal.
    """

    rms_values = np.asarray(
        frame_rms,
        dtype=np.float64,
    )

    rms_values = rms_values[
        np.isfinite(rms_values)
    ]

    if rms_values.size == 0:
        return float("-inf")

    frame_dbfs = amplitude_to_dbfs(
        rms_values,
        epsilon=epsilon,
    )

    active_frames = rms_values[
        frame_dbfs > silence_threshold_dbfs
    ]

    if active_frames.size == 0:
        return float("-inf")

    sorted_rms = np.sort(rms_values)

    noise_frame_count = max(
        1,
        int(np.ceil(sorted_rms.size * 0.20)),
    )

    noise_frames = sorted_rms[
        :noise_frame_count
    ]

    signal_level = float(
        np.sqrt(
            np.mean(
                np.square(active_frames)
            )
        )
    )

    noise_level = float(
        np.sqrt(
            np.mean(
                np.square(noise_frames)
            )
        )
    )

    signal_level = max(
        signal_level,
        epsilon,
    )

    noise_level = max(
        noise_level,
        epsilon,
    )

    return float(
        20.0
        * np.log10(
            signal_level / noise_level
        )
    )


# ============================================================
# SINGLE-RECORDING ASSESSMENT
# ============================================================

def assess_audio_quality(
    audio_path: str | Path,
    slot_number: int,
    config: AudioQualityConfig | None = None,
) -> AudioQualityResult:
    """
    Assess one standardized WAV recording.

    Parameters
    ----------
    audio_path:
        Standardized WAV path.

    slot_number:
        Recording position.

    config:
        Optional quality configuration.

    Returns
    -------
    AudioQualityResult
        Metrics, warnings, blocking reasons, and validity.
    """

    active_config = (
        config
        if config is not None
        else get_audio_quality_config()
    )

    active_config.validate()

    path = Path(audio_path).resolve()

    result = AudioQualityResult(
        slot_number=int(slot_number),
        audio_path=str(path),
        filename=path.name,
    )

    try:
        if not path.exists():
            raise FileNotFoundError(
                f"Audio file does not exist: {path}"
            )

        if not path.is_file():
            raise AudioQualityProcessingError(
                f"Audio path is not a file: {path}"
            )

        if path.stat().st_size <= 0:
            raise AudioQualityProcessingError(
                f"Audio file is empty: {path.name}"
            )

        audio, sample_rate = sf.read(
            file=str(path),
            dtype="float32",
            always_2d=True,
        )

        if audio.size == 0:
            raise AudioQualityProcessingError(
                "Decoded audio contains no samples."
            )

        if not np.all(np.isfinite(audio)):
            raise AudioQualityProcessingError(
                "Audio contains NaN or infinite values."
            )

        channel_count = int(audio.shape[1])

        if channel_count == 1:
            mono_audio = audio[:, 0]
        else:
            mono_audio = np.mean(
                audio,
                axis=1,
            )

        mono_audio = np.asarray(
            mono_audio,
            dtype=np.float64,
        ).reshape(-1)

        sample_count = int(
            mono_audio.size
        )

        if sample_count <= 0:
            raise AudioQualityProcessingError(
                "Audio contains zero usable samples."
            )

        duration_seconds = (
            sample_count / float(sample_rate)
        )

        rms = float(
            np.sqrt(
                np.mean(
                    np.square(mono_audio)
                )
            )
        )

        rms_dbfs = float(
            amplitude_to_dbfs(rms)
        )

        peak_amplitude = float(
            np.max(
                np.abs(mono_audio)
            )
        )

        clipping_mask = (
            np.abs(mono_audio)
            >= active_config.clipping_amplitude_threshold
        )

        clipped_sample_count = int(
            np.count_nonzero(clipping_mask)
        )

        clipping_percentage = float(
            100.0
            * clipped_sample_count
            / sample_count
        )

        frame_length = max(
            1,
            int(
                round(
                    sample_rate
                    * active_config.frame_length_ms
                    / 1000.0
                )
            ),
        )

        hop_length = max(
            1,
            int(
                round(
                    sample_rate
                    * active_config.hop_length_ms
                    / 1000.0
                )
            ),
        )

        frames = create_audio_frames(
            audio=mono_audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        frame_rms = calculate_frame_rms(
            frames
        )

        frame_dbfs = amplitude_to_dbfs(
            frame_rms
        )

        silence_percentage = float(
            100.0
            * np.mean(
                frame_dbfs
                <= active_config.silence_threshold_dbfs
            )
        )

        estimated_snr = estimate_snr_db(
            frame_rms=frame_rms,
            silence_threshold_dbfs=(
                active_config.silence_threshold_dbfs
            ),
        )

        result.duration_seconds = round(
            duration_seconds,
            4,
        )

        result.sample_rate = int(
            sample_rate
        )

        result.channels = channel_count
        result.sample_count = sample_count

        result.rms = round(
            rms,
            8,
        )

        result.rms_dbfs = round(
            rms_dbfs,
            3,
        )

        result.peak_amplitude = round(
            peak_amplitude,
            8,
        )

        result.clipped_sample_count = (
            clipped_sample_count
        )

        result.clipping_percentage = round(
            clipping_percentage,
            4,
        )

        result.estimated_snr_db = (
            round(estimated_snr, 3)
            if np.isfinite(estimated_snr)
            else float("-inf")
        )

        result.silence_percentage = round(
            silence_percentage,
            3,
        )

        if (
            sample_rate
            != active_config.expected_sample_rate
        ):
            result.blocking_reasons.append(
                "unexpected_sample_rate"
            )

        if (
            channel_count
            != active_config.expected_channels
        ):
            result.blocking_reasons.append(
                "unexpected_channel_count"
            )

        if (
            duration_seconds
            < active_config.minimum_duration_seconds
        ):
            result.blocking_reasons.append(
                "duration_below_minimum"
            )

        if (
            duration_seconds
            > active_config.recommended_maximum_duration_seconds
        ):
            result.warnings.append(
                "duration_above_recommended_maximum"
            )

        if (
            rms_dbfs
            < active_config.minimum_rms_dbfs
        ):
            result.blocking_reasons.append(
                "rms_below_minimum"
            )

        if (
            clipping_percentage
            > active_config.maximum_clipping_percentage
        ):
            result.blocking_reasons.append(
                "excessive_clipping"
            )

        if (
            not np.isfinite(estimated_snr)
            or estimated_snr
            < active_config.minimum_snr_db
        ):
            result.blocking_reasons.append(
                "snr_below_minimum"
            )

        if (
            silence_percentage
            > active_config.maximum_silence_percentage
        ):
            result.blocking_reasons.append(
                "silence_above_maximum"
            )

        result.processing_successful = True
        result.is_valid = (
            len(result.blocking_reasons) == 0
        )

        if result.is_valid:
            result.quality_status = (
                "valid_with_warning"
                if result.warnings
                else "valid"
            )
        else:
            result.quality_status = "rejected"

    except Exception as error:
        logger.exception(
            "Quality assessment failed for slot %s: %s",
            slot_number,
            path,
        )

        result.processing_successful = False
        result.is_valid = False
        result.quality_status = "processing_failure"
        result.blocking_reasons = [
            "processing_failure"
        ]
        result.error = (
            f"{type(error).__name__}: {error}"
        )

    return result


# ============================================================
# COLLECTION ASSESSMENT
# ============================================================

def assess_recording_collection(
    recording_paths: Sequence[str | Path],
    config: AudioQualityConfig | None = None,
) -> list[AudioQualityResult]:
    """
    Assess every recording in a collection.

    Returns
    -------
    list[AudioQualityResult]
        One result per supplied recording.
    """

    paths = list(recording_paths)

    if not paths:
        raise AudioQualityProcessingError(
            "No recordings were supplied for quality assessment."
        )

    active_config = (
        config
        if config is not None
        else get_audio_quality_config()
    )

    results: list[AudioQualityResult] = []

    for slot_number, recording_path in enumerate(
        paths,
        start=1,
    ):
        result = assess_audio_quality(
            audio_path=recording_path,
            slot_number=slot_number,
            config=active_config,
        )

        results.append(result)

    return results


def assess_processed_recordings(
    processed_recordings: Iterable[Any],
    config: AudioQualityConfig | None = None,
) -> list[AudioQualityResult]:
    """
    Assess ProcessedAudio objects returned by audio_processing.py.
    """

    recording_paths: list[str] = []

    for recording in processed_recordings:
        standardized_path = getattr(
            recording,
            "standardized_path",
            None,
        )

        if not standardized_path:
            raise AudioQualityProcessingError(
                "A processed recording does not contain "
                "'standardized_path'."
            )

        recording_paths.append(
            str(standardized_path)
        )

    return assess_recording_collection(
        recording_paths=recording_paths,
        config=config,
    )


# ============================================================
# RESULT HELPERS
# ============================================================

def get_valid_quality_results(
    quality_results: Iterable[AudioQualityResult],
) -> list[AudioQualityResult]:
    """Return only valid recording results."""

    return [
        result
        for result in quality_results
        if result.is_valid
    ]


def get_rejected_quality_results(
    quality_results: Iterable[AudioQualityResult],
) -> list[AudioQualityResult]:
    """Return only rejected recording results."""

    return [
        result
        for result in quality_results
        if not result.is_valid
    ]


def get_rejected_slot_numbers(
    quality_results: Iterable[AudioQualityResult],
) -> list[int]:
    """Return slot numbers that require replacement."""

    return [
        result.slot_number
        for result in quality_results
        if not result.is_valid
    ]


def build_quality_summary(
    quality_results: Iterable[AudioQualityResult],
) -> dict[str, Any]:
    """
    Build a serializable collection-level quality summary.
    """

    results = list(quality_results)

    valid_results = get_valid_quality_results(
        results
    )

    rejected_results = get_rejected_quality_results(
        results
    )

    return {
        "total_recordings": len(results),
        "valid_recordings": len(valid_results),
        "rejected_recordings": len(rejected_results),
        "valid_slot_numbers": [
            result.slot_number
            for result in valid_results
        ],
        "rejected_slot_numbers": [
            result.slot_number
            for result in rejected_results
        ],
        "results": [
            result.to_dict()
            for result in results
        ],
    }