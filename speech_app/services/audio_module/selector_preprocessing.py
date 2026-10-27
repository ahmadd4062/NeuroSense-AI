"""NeuroSense candidate preprocessing v1: shared by training and inference.

Energy-based boundary selection is not a vowel detector or disease classifier.
Call preprocess_audio() on RAW input once. Keep this module with any retrained model.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import io
import math
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PIPELINE_VERSION = "neurosense_active_span_v1"

class InvalidAudioError(ValueError):
    pass

@dataclass(frozen=True)
class PreprocessConfig:
    target_sr: int = 8000
    min_duration_s: float = 1.5
    frame_s: float = 0.025
    reference_percentile: float = 95.0
    active_level_ratio: float = 0.30
    target_peak: float = 0.99
    silence_epsilon: float = 1e-7
    clipping_level: float = 0.999
    clipping_warning_fraction: float = 0.01

    def __post_init__(self):
        if not isinstance(self.target_sr, int) or isinstance(self.target_sr, bool) or self.target_sr <= 0:
            raise ValueError("target_sr must be a positive integer")
        for name in ("min_duration_s", "frame_s", "silence_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if round(self.frame_s * self.target_sr) < 1:
            raise ValueError("frame_s produces an empty frame")
        if not 0 < self.reference_percentile <= 100:
            raise ValueError("reference_percentile must be in (0, 100]")
        if not 0 < self.active_level_ratio <= 1:
            raise ValueError("active_level_ratio must be in (0, 1]")
        if not 0 < self.target_peak < 1:
            raise ValueError("target_peak must be in (0, 1)")
        if not 0 < self.clipping_level <= 1 or not 0 <= self.clipping_warning_fraction <= 1:
            raise ValueError("Invalid clipping diagnostic settings")

DEFAULT_CONFIG = PreprocessConfig()

def decode_audio(path, *, ffmpeg_executable="ffmpeg", timeout_seconds=120):
    """Read WAV/FLAC/OGG etc.; FFmpeg fallback for formats libsndfile cannot read.

    FFmpeg emits FLOAT WAV at native rate/channel count. Do not quantize to PCM16
    before measuring clipping/quietness. Install ffmpeg for compressed inputs.
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidAudioError(f"Audio file not found: {path}")
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError):
        try:
            run = subprocess.run(
                [str(ffmpeg_executable), "-v", "error", "-i", str(path), "-map", "0:a:0",
                 "-vn", "-f", "wav", "-acodec", "pcm_f32le", "pipe:1"],
                check=True, capture_output=True, timeout=timeout_seconds,
            )
            y, sr = sf.read(io.BytesIO(run.stdout), dtype="float32", always_2d=True)
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
            raise InvalidAudioError(f"Cannot decode {path.name}") from error
    return y, int(sr)

def standardize_array(samples, sample_rate, config=DEFAULT_CONFIG):
    """Validate, measure source quality, average channels and resample once."""
    cfg = config
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or min(a.shape, default=0) == 0:
        raise InvalidAudioError("Empty audio or unsupported shape; use samples x channels")
    if not np.isfinite(a).all():
        raise InvalidAudioError("Audio contains NaN or infinity; it was not silently repaired")
    if not isinstance(sample_rate, (int, np.integer)) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise InvalidAudioError("Sample rate must be a positive integer")
    if len(a) < math.ceil(cfg.min_duration_s * sample_rate):
        raise InvalidAudioError(f"Raw audio is shorter than {cfg.min_duration_s:g} seconds")
    warnings = []
    clipped = float(np.mean(np.abs(a) >= cfg.clipping_level))
    if clipped > cfg.clipping_warning_fraction:
        warnings.append("possible_source_clipping")
    mono = a.mean(axis=1)
    channel_rms = float(np.sqrt(np.mean(a * a)))
    mono_rms = float(np.sqrt(np.mean(mono * mono)))
    if a.shape[1] > 1 and channel_rms > cfg.silence_epsilon and mono_rms < 0.1 * channel_rms:
        warnings.append("possible_channel_cancellation")
    if sample_rate != cfg.target_sr:
        common = math.gcd(int(sample_rate), cfg.target_sr)
        mono = resample_poly(mono, cfg.target_sr // common, int(sample_rate) // common)
    if not np.isfinite(mono).all():
        raise InvalidAudioError("Non-finite samples after resampling")
    decoded_hash = hashlib.sha256(np.asarray(mono, dtype="<f4").tobytes()).hexdigest()
    audit = {
        "pipeline_version": PIPELINE_VERSION,
        "source_sr": int(sample_rate), "source_channels": int(a.shape[1]),
        "raw_duration_s": len(a) / sample_rate,
        "source_peak": float(np.max(np.abs(a))),
        "source_rms": channel_rms,
        "source_clip_fraction": clipped,
        "decoded_mono_8k_sha256": decoded_hash,
        "warnings": warnings,
    }
    return mono, audit

def select_active_span(mono, sample_rate, config=DEFAULT_CONFIG):
    """Keep first-to-last energetic frame, including every sample in between.

    No lowest-CV window, pitch cutoff, internal pause deletion, padding or denoising.
    The last partial frame is included. This uses energy, not speech recognition.
    """
    cfg = config
    y = np.asarray(mono, dtype=np.float64)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise InvalidAudioError("Selector requires a finite mono array")
    if sample_rate != cfg.target_sr:
        raise InvalidAudioError("Standardize the sample rate before selecting")
    minimum = math.ceil(cfg.min_duration_s * sample_rate)
    if len(y) < minimum:
        raise InvalidAudioError(f"Audio is shorter than {cfg.min_duration_s:g} seconds")
    # Removing DC here prevents a constant offset from looking like voice activity.
    centered = y - np.mean(y)
    frame = round(cfg.frame_s * sample_rate)
    starts = np.arange(0, len(centered), frame)
    sizes = np.minimum(frame, len(centered) - starts)
    rms = np.sqrt(np.add.reduceat(centered * centered, starts) / sizes)
    reference = float(np.percentile(rms, cfg.reference_percentile))
    if reference <= cfg.silence_epsilon:
        raise InvalidAudioError("No measurable energy above the numerical silence floor")
    active = np.flatnonzero(rms >= cfg.active_level_ratio * reference)
    first = int(starts[active[0]])
    last = int(min(len(centered), starts[active[-1]] + frame))
    if last - first < minimum:
        raise InvalidAudioError(f"Selected span is shorter than {cfg.min_duration_s:g} seconds")
    span = centered[first:last].copy()
    info = {
        "start_sample_8k": first, "end_sample_8k_exclusive": last,
        "start_s": first / sample_rate, "end_s": last / sample_rate,
        "kept_duration_s": len(span) / sample_rate,
        "kept_fraction": len(span) / len(y),
        "reference_rms": reference,
        "active_threshold_rms": cfg.active_level_ratio * reference,
        "active_frame_fraction": float(len(active) / len(rms)),
    }
    return span, info

def normalize_span(span, config=DEFAULT_CONFIG):
    y = np.asarray(span, dtype=np.float64)
    y = y - np.mean(y)
    peak = float(np.max(np.abs(y)))
    if not math.isfinite(peak) or peak <= config.silence_epsilon:
        raise InvalidAudioError("Selected audio is numerically silent")
    return np.asarray(config.target_peak * y / peak, dtype=np.float32)

def preprocess_array(samples, sample_rate, config=DEFAULT_CONFIG):
    mono, audit = standardize_array(samples, sample_rate, config)
    span, selection = select_active_span(mono, config.target_sr, config)
    audit.update(selection)
    audit["pre_normalization_span_rms"] = float(np.sqrt(np.mean(span * span)))
    audit["pre_normalization_span_peak"] = float(np.max(np.abs(span)))
    y = normalize_span(span, config)
    audit["output_sr"] = config.target_sr
    audit["output_peak"] = float(np.max(np.abs(y)))
    audit["status"] = "processed"  # Not a claim of verified vowel/clinical quality.
    return y, config.target_sr, audit

def preprocess_audio(path, config=DEFAULT_CONFIG):
    samples, sr = decode_audio(path)
    return preprocess_array(samples, sr, config)

def preprocess_reference_audio(path, config=DEFAULT_CONFIG):
    """Reference arm: common decoding/resampling, original librosa trim top_db=30.

    This is a reference preprocessing arm, not an exact rerun of every old app gate.
    """
    import librosa
    samples, sr = decode_audio(path)
    mono, audit = standardize_array(samples, sr, config)
    span, bounds = librosa.effects.trim(np.asarray(mono, dtype=np.float32), top_db=30)
    if len(span) < math.ceil(config.min_duration_s * config.target_sr):
        raise InvalidAudioError("Reference span is shorter than the shared duration minimum")
    y = normalize_span(span, config)
    audit.update({"pipeline_version": "neurosense_trim_reference_v1",
                  "start_s": int(bounds[0]) / config.target_sr,
                  "end_s": int(bounds[1]) / config.target_sr,
                  "kept_duration_s": len(y) / config.target_sr,
                  "status": "processed"})
    return y, config.target_sr, audit

def config_dict(config=DEFAULT_CONFIG):
    return asdict(config)
