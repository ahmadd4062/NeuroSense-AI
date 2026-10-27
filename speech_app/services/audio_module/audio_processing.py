"""
Audio input and standardization service for NeuroSense.

Responsibilities
----------------
- Receive Django uploaded files.
- Receive browser microphone audio bytes.
- Validate filename, extension and file size.
- Save raw audio safely.
- Convert WAV, MP4, OGG and WebM audio using FFmpeg.
- Standardize every recording to:
    - WAV
    - Mono
    - 8000 Hz
    - lossless floating-point WAV for feature extraction
- Validate the standardized output.
Process uploaded recordings for the audio screening pipeline.
This module performs only audio processing and standardization.
Workflow decisions such as recording replacement are handled by views.py.

This module does not perform:
- Audio quality assessment
- Duplicate detection
- Feature extraction
- Model prediction
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import soundfile as sf
import numpy as np
from scipy.io import wavfile
from .selector_preprocessing import decode_audio, standardize_array, InvalidAudioError
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile


logger = logging.getLogger(__name__)


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

TARGET_SAMPLE_RATE = 8_000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_f64le"  # Internal precision; no amplitude normalization here.
TARGET_EXTENSION = ".wav"

REQUIRED_RECORDINGS = 3

MAX_AUDIO_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

SUPPORTED_UPLOAD_EXTENSIONS = {
    ".wav",
    ".mp4",
    ".ogg",
    ".webm",
}

SUPPORTED_MICROPHONE_EXTENSIONS = {
    ".webm",
    ".wav",
    ".ogg",
    ".mp4",
}

VALID_AUDIO_SOURCES = {
    "web_upload",
    "web_microphone",
}


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class AudioProcessingError(Exception):
    """Base exception for audio-processing failures."""


class UnsupportedAudioFormatError(AudioProcessingError):
    """Raised when the supplied audio format is unsupported."""


class AudioFileTooLargeError(AudioProcessingError):
    """Raised when an audio file exceeds the configured size limit."""


class AudioConversionError(AudioProcessingError):
    """Raised when FFmpeg cannot convert an audio file."""


class InvalidStandardizedAudioError(AudioProcessingError):
    """Raised when converted WAV output does not match requirements."""


class InvalidAudioInputError(AudioProcessingError):
    """Raised when an audio input is missing, empty or malformed."""


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass(frozen=True)
class ProcessedAudio:
    """
    Metadata for one successfully standardized recording.
    """

    slot_number: int
    source: str
    original_filename: str
    raw_path: str
    standardized_path: str
    standardized_filename: str
    sample_rate: int
    channels: int
    sample_count: int
    duration_seconds: float
    subtype: str
    file_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a serializable dictionary."""
        return asdict(self)


# ============================================================
# DIRECTORY HELPERS
# ============================================================

def get_audio_work_directory() -> Path:
    """
    Return the directory used for temporary audio processing.

    The directory can be configured in settings.py:

        AUDIO_WORK_DIR = BASE_DIR / "media" / "audio_work"

    When AUDIO_WORK_DIR is not configured, the module uses:

        MEDIA_ROOT / "audio_work"
    """

    configured_directory = getattr(
        settings,
        "AUDIO_WORK_DIR",
        None,
    )

    if configured_directory:
        work_directory = Path(configured_directory)
    else:
        media_root = getattr(
            settings,
            "MEDIA_ROOT",
            None,
        )

        if media_root:
            work_directory = Path(media_root) / "audio_work"
        else:
            work_directory = (
                Path(settings.BASE_DIR)
                / "media"
                / "audio_work"
            )

    work_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return work_directory.resolve()


def create_request_audio_directory(
    request_id: str | None = None,
) -> Path:
    """
    Create an isolated directory for one screening request.

    Parameters
    ----------
    request_id:
        Optional request identifier. A UUID is generated when omitted.

    Returns
    -------
    pathlib.Path
        Directory containing raw and standardized subdirectories.
    """

    safe_request_id = (
        sanitize_identifier(request_id)
        if request_id
        else uuid.uuid4().hex
    )

    request_directory = (
        get_audio_work_directory()
        / safe_request_id
    )

    (request_directory / "raw").mkdir(
        parents=True,
        exist_ok=True,
    )

    (request_directory / "standardized").mkdir(
        parents=True,
        exist_ok=True,
    )

    return request_directory.resolve()


# ============================================================
# VALIDATION HELPERS
# ============================================================

def sanitize_identifier(value: str) -> str:
    """
    Produce a safe identifier containing letters, numbers,
    underscores and hyphens.
    """

    cleaned = "".join(
        character
        for character in str(value)
        if character.isalnum()
        or character in {"_", "-"}
    )

    if not cleaned:
        raise InvalidAudioInputError(
            "The supplied identifier is invalid."
        )

    return cleaned[:100]


def sanitize_filename(filename: str) -> str:
    """
    Remove directory components from a supplied filename.
    """

    safe_filename = Path(str(filename)).name.strip()

    if not safe_filename:
        raise InvalidAudioInputError(
            "The audio filename is missing."
        )

    return safe_filename


def normalize_source(source: str) -> str:
    """
    Validate and normalize an audio source identifier.
    """

    normalized_source = str(source).strip().lower()

    if normalized_source not in VALID_AUDIO_SOURCES:
        raise InvalidAudioInputError(
            f"Unsupported audio source: {source}. "
            f"Expected one of: {sorted(VALID_AUDIO_SOURCES)}"
        )

    return normalized_source


def get_allowed_extensions(
    source: str,
) -> set[str]:
    """
    Return extensions allowed for the selected input source.
    """

    normalized_source = normalize_source(source)

    if normalized_source == "web_microphone":
        return SUPPORTED_MICROPHONE_EXTENSIONS

    return SUPPORTED_UPLOAD_EXTENSIONS


def validate_audio_extension(
    filename: str,
    source: str,
) -> str:
    """
    Validate an audio filename extension.

    Returns
    -------
    str
        Lowercase validated extension.
    """

    safe_filename = sanitize_filename(filename)
    extension = Path(safe_filename).suffix.lower()

    if not extension:
        raise UnsupportedAudioFormatError(
            f"The file '{safe_filename}' has no extension."
        )

    allowed_extensions = get_allowed_extensions(source)

    if extension not in allowed_extensions:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{extension}' for "
            f"'{safe_filename}'. Allowed formats: "
            f"{sorted(allowed_extensions)}"
        )

    return extension


def validate_file_size(
    file_size_bytes: int,
) -> None:
    """
    Validate that an audio file is non-empty and within the limit.
    """

    try:
        file_size_bytes = int(file_size_bytes)
    except (TypeError, ValueError) as error:
        raise InvalidAudioInputError(
            "Audio file size must be an integer."
        ) from error

    if file_size_bytes <= 0:
        raise InvalidAudioInputError(
            "The supplied audio file is empty."
        )

    if file_size_bytes > MAX_AUDIO_FILE_SIZE_BYTES:
        maximum_mb = (
            MAX_AUDIO_FILE_SIZE_BYTES
            / (1024 * 1024)
        )

        raise AudioFileTooLargeError(
            f"Audio file exceeds the maximum size of "
            f"{maximum_mb:.0f} MB."
        )


def validate_slot_number(
    slot_number: int,
) -> int:
    """
    Validate a recording slot number.
    """

    try:
        normalized_slot = int(slot_number)
    except (TypeError, ValueError) as error:
        raise InvalidAudioInputError(
            "Recording slot number must be an integer."
        ) from error

    if normalized_slot < 1:
        raise InvalidAudioInputError(
            "Recording slot number must be at least 1."
        )

    return normalized_slot


# ============================================================
# FFMPEG HELPERS
# ============================================================

def get_ffmpeg_executable() -> str:
    """
    Return the configured FFmpeg executable.

    Optionally configure an explicit path in settings.py:

        FFMPEG_BINARY = r"C:\\ffmpeg\\bin\\ffmpeg.exe"

    Otherwise, FFmpeg is located through the system PATH.
    """

    configured_binary = getattr(
        settings,
        "FFMPEG_BINARY",
        None,
    )

    if configured_binary:
        binary_path = Path(configured_binary)

        if not binary_path.exists():
            raise AudioProcessingError(
                f"Configured FFmpeg binary was not found: "
                f"{binary_path}"
            )

        return str(binary_path)

    detected_binary = shutil.which("ffmpeg")

    if not detected_binary:
        raise AudioProcessingError(
            "FFmpeg was not found in the system PATH."
        )

    return detected_binary


def verify_ffmpeg_available() -> str:
    """
    Verify that FFmpeg can be executed.

    Returns
    -------
    str
        First line of the FFmpeg version output.
    """

    ffmpeg_binary = get_ffmpeg_executable()

    try:
        completed_process = subprocess.run(
            [
                ffmpeg_binary,
                "-version",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        raise AudioProcessingError(
            "FFmpeg could not be executed."
        ) from error

    if completed_process.returncode != 0:
        error_message = (
            completed_process.stderr.strip()
            or "Unknown FFmpeg error."
        )

        raise AudioProcessingError(
            f"FFmpeg availability check failed: "
            f"{error_message}"
        )

    version_line = (
        completed_process.stdout.splitlines()[0]
        if completed_process.stdout
        else "FFmpeg available"
    )

    return version_line


# ============================================================
# RAW AUDIO STORAGE
# ============================================================

def save_django_uploaded_file(
    uploaded_file: UploadedFile,
    destination_directory: Path,
    slot_number: int,
    source: str = "web_upload",
) -> Path:
    """
    Save one Django UploadedFile safely.

    Parameters
    ----------
    uploaded_file:
        Django UploadedFile received through request.FILES.

    destination_directory:
        Directory in which the raw file is stored.

    slot_number:
        Recording position.

    source:
        Input source. Normally ``web_upload``.

    Returns
    -------
    pathlib.Path
        Saved raw-audio path.
    """

    if not isinstance(uploaded_file, UploadedFile):
        raise InvalidAudioInputError(
            "Expected a Django UploadedFile instance."
        )

    normalized_source = normalize_source(source)
    slot_number = validate_slot_number(slot_number)

    original_filename = sanitize_filename(
        uploaded_file.name
    )

    extension = validate_audio_extension(
        filename=original_filename,
        source=normalized_source,
    )

    validate_file_size(
        uploaded_file.size
    )

    destination_directory = Path(
        destination_directory
    )
    
    for existing_file in destination_directory.glob(
        f"slot_{slot_number}_*"
    ):
        existing_file.unlink(
            missing_ok=True
        )
    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_path = (
        destination_directory
        / (
            f"slot_{slot_number}_"
            f"{uuid.uuid4().hex[:12]}"
            f"{extension}"
        )
    )

    bytes_written = 0

    try:
        with destination_path.open("wb") as destination_file:
            for chunk in uploaded_file.chunks():
                if not chunk:
                    continue

                bytes_written += len(chunk)

                if bytes_written > MAX_AUDIO_FILE_SIZE_BYTES:
                    raise AudioFileTooLargeError(
                        "Audio file exceeded the maximum "
                        "allowed size while being saved."
                    )

                destination_file.write(chunk)

    except Exception:
        destination_path.unlink(
            missing_ok=True
        )
        raise

    validate_file_size(bytes_written)

    return destination_path.resolve()


def save_audio_bytes(
    audio_bytes: bytes | bytearray,
    filename: str,
    destination_directory: Path,
    slot_number: int,
    source: str = "web_microphone",
) -> Path:
    """
    Save raw audio bytes received from a browser microphone.

    Parameters
    ----------
    audio_bytes:
        Encoded WebM, WAV, OGG or MP4 audio bytes.

    filename:
        Browser-provided filename, such as ``recording_1.webm``.

    destination_directory:
        Directory in which the raw audio is saved.

    slot_number:
        Recording position.

    source:
        Input source. Normally ``web_microphone``.

    Returns
    -------
    pathlib.Path
        Saved raw-audio path.
    """

    if not isinstance(
        audio_bytes,
        (bytes, bytearray),
    ):
        raise InvalidAudioInputError(
            "Microphone audio content must be bytes."
        )

    normalized_source = normalize_source(source)
    slot_number = validate_slot_number(slot_number)

    extension = validate_audio_extension(
        filename=filename,
        source=normalized_source,
    )

    validate_file_size(
        len(audio_bytes)
    )

    destination_directory = Path(
        destination_directory
    )

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_path = (
        destination_directory
        / (
            f"slot_{slot_number}_"
            f"{uuid.uuid4().hex[:12]}"
            f"{extension}"
        )
    )

    destination_path.write_bytes(
        bytes(audio_bytes)
    )

    return destination_path.resolve()


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_to_standard_wav(input_path: str | Path, output_path: str | Path,
                            timeout_seconds: int = 60) -> Path:
    """Decode raw input, average channels, and resample exactly as in training.

    Lossless float64 WAV is an internal interchange format. It avoids PCM16
    rounding before pitch/jitter/MFCC extraction. Selection/normalization happens
    later, once, in load_standardized_audio(). Scipy's WAV writer is deterministic
    and adds no time-varying PEAK metadata, preserving duplicate-file detection.
    """
    source_path = Path(input_path).resolve()
    destination_path = Path(output_path).resolve()
    if not source_path.is_file():
        raise InvalidAudioInputError(f"Input audio does not exist: {source_path}")
    if source_path == destination_path:
        raise InvalidAudioInputError("Raw input and standardized output must be different paths.")
    validate_file_size(source_path.stat().st_size)
    if destination_path.suffix.lower() != '.wav':
        raise InvalidAudioInputError("Standardized output must use the .wav extension.")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.unlink(missing_ok=True)
    try:
        samples, source_sr = decode_audio(source_path,
            ffmpeg_executable=get_ffmpeg_executable(), timeout_seconds=timeout_seconds)
        mono, _ = standardize_array(samples, source_sr)
        wavfile.write(str(destination_path), TARGET_SAMPLE_RATE, np.asarray(mono, dtype=np.float64))
        validate_standardized_wav(destination_path)
    except (InvalidAudioError, OSError, ValueError) as error:
        destination_path.unlink(missing_ok=True)
        raise AudioConversionError(f"Unable to standardize {source_path.name}: {error}") from error
    return destination_path


# ============================================================
# STANDARDIZED WAV VALIDATION
# ============================================================

def inspect_standardized_wav(
    wav_path: str | Path,
) -> dict[str, Any]:
    """
    Read metadata from a standardized WAV file.
    """

    path = Path(wav_path).resolve()

    if not path.exists():
        raise InvalidStandardizedAudioError(
            f"Standardized audio does not exist: {path}"
        )

    if not path.is_file():
        raise InvalidStandardizedAudioError(
            f"Standardized audio path is not a file: {path}"
        )

    try:
        audio_info = sf.info(str(path))
    except Exception as error:
        raise InvalidStandardizedAudioError(
            f"Unable to read standardized WAV: {path.name}"
        ) from error

    sample_count = int(audio_info.frames)
    sample_rate = int(audio_info.samplerate)
    channels = int(audio_info.channels)
    duration_seconds = float(audio_info.duration)
    subtype = str(audio_info.subtype or "")
    file_format = str(audio_info.format or "")

    return {
        "path": str(path),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_count": sample_count,
        "duration_seconds": duration_seconds,
        "subtype": subtype,
        "format": file_format,
        "file_size_bytes": path.stat().st_size,
    }


def validate_standardized_wav(
    wav_path: str | Path,
) -> dict[str, Any]:
    """
    Confirm that a WAV file matches deployment requirements.

    Returns
    -------
    dict
        Validated audio metadata.
    """

    metadata = inspect_standardized_wav(
        wav_path
    )

    validation_errors: list[str] = []

    if metadata["format"].upper() != "WAV":
        validation_errors.append(
            f"format is {metadata['format']}, expected WAV"
        )

    if metadata["sample_rate"] != TARGET_SAMPLE_RATE:
        validation_errors.append(
            f"sample rate is {metadata['sample_rate']} Hz, "
            f"expected {TARGET_SAMPLE_RATE} Hz"
        )

    if metadata["channels"] != TARGET_CHANNELS:
        validation_errors.append(
            f"channel count is {metadata['channels']}, "
            f"expected {TARGET_CHANNELS}"
        )

    if metadata["subtype"].upper() not in {"PCM_16", "FLOAT", "DOUBLE"}:
        validation_errors.append(
            f"subtype is {metadata['subtype']}, "
            "expected PCM_16, FLOAT or DOUBLE"
        )

    if metadata["sample_count"] <= 0:
        validation_errors.append(
            "audio contains no samples"
        )

    if metadata["duration_seconds"] <= 0:
        validation_errors.append(
            "audio duration is zero"
        )

    if validation_errors:
        raise InvalidStandardizedAudioError(
            "Standardized WAV validation failed: "
            + "; ".join(validation_errors)
        )

    return metadata


# ============================================================
# SINGLE-RECORDING PROCESSING
# ============================================================

def process_uploaded_recording(
    uploaded_file: UploadedFile,
    slot_number: int,
    request_directory: str | Path,
) -> ProcessedAudio:
    """
    Save and standardize one Django file upload.
    """

    request_directory = Path(
        request_directory
    ).resolve()

    original_filename = sanitize_filename(
        uploaded_file.name
    )

    raw_path = save_django_uploaded_file(
        uploaded_file=uploaded_file,
        destination_directory=(
            request_directory / "raw"
        ),
        slot_number=slot_number,
        source="web_upload",
    )

    standardized_path = (
        request_directory
        / "standardized"
        / f"slot_{slot_number}_standardized.wav"
    )

    try:
        standardized_path = convert_to_standard_wav(
            input_path=raw_path,
            output_path=standardized_path,
        )

        metadata = validate_standardized_wav(
            standardized_path
        )

    except Exception:
        standardized_path.unlink(
            missing_ok=True
        )
        raise

    return ProcessedAudio(
        slot_number=validate_slot_number(slot_number),
        source="web_upload",
        original_filename=original_filename,
        raw_path=str(raw_path),
        standardized_path=str(standardized_path),
        standardized_filename=standardized_path.name,
        sample_rate=metadata["sample_rate"],
        channels=metadata["channels"],
        sample_count=metadata["sample_count"],
        duration_seconds=round(
            metadata["duration_seconds"],
            6,
        ),
        subtype=metadata["subtype"],
        file_size_bytes=metadata["file_size_bytes"],
    )


def process_microphone_recording(
    audio_bytes: bytes | bytearray,
    filename: str,
    slot_number: int,
    request_directory: str | Path,
) -> ProcessedAudio:
    """
    Save and standardize one browser microphone recording.

    The frontend should normally send a WebM Blob.
    """

    request_directory = Path(
        request_directory
    ).resolve()

    original_filename = sanitize_filename(
        filename
    )

    raw_path = save_audio_bytes(
        audio_bytes=audio_bytes,
        filename=original_filename,
        destination_directory=(
            request_directory / "raw"
        ),
        slot_number=slot_number,
        source="web_microphone",
    )

    standardized_path = (
        request_directory
        / "standardized"
        / f"slot_{slot_number}_standardized.wav"
    )

    try:
        standardized_path = convert_to_standard_wav(
            input_path=raw_path,
            output_path=standardized_path,
        )

        metadata = validate_standardized_wav(
            standardized_path
        )

    except Exception:
        standardized_path.unlink(
            missing_ok=True
        )
        raise

    return ProcessedAudio(
        slot_number=validate_slot_number(slot_number),
        source="web_microphone",
        original_filename=original_filename,
        raw_path=str(raw_path),
        standardized_path=str(standardized_path),
        standardized_filename=standardized_path.name,
        sample_rate=metadata["sample_rate"],
        channels=metadata["channels"],
        sample_count=metadata["sample_count"],
        duration_seconds=round(
            metadata["duration_seconds"],
            6,
        ),
        subtype=metadata["subtype"],
        file_size_bytes=metadata["file_size_bytes"],
    )


# ============================================================
# THREE-RECORDING COLLECTION
# ============================================================

def process_three_uploaded_recordings(
    uploaded_files: Sequence[tuple[int, UploadedFile]],
    request_id: str | None = None,
) -> tuple[Path, list[ProcessedAudio]]:
    """
    Process uploaded recordings.

    Initial request:
        Processes slots 1,2,3.

    Replacement request:
        Processes only uploaded slots, then reloads the remaining
        processed recordings from disk.
    """

    uploaded_items = list(uploaded_files)

    if not uploaded_items:
        raise InvalidAudioInputError(
            "No uploaded recordings were supplied."
        )

    request_directory = create_request_audio_directory(
        request_id=request_id
    )

    processed_by_slot: dict[int, ProcessedAudio] = {}

    #
    # Process uploaded recordings.
    #
    for slot_number, uploaded_file in uploaded_items:

        processed = process_uploaded_recording(
            uploaded_file=uploaded_file,
            slot_number=slot_number,
            request_directory=request_directory,
        )

        processed_by_slot[slot_number] = processed

    #
    # Reload existing processed recordings.
    #
    for slot_number in range(1, 4):

        if slot_number in processed_by_slot:
            continue

        standardized_path = (
            request_directory
            / "standardized"
            / f"slot_{slot_number}_standardized.wav"
        )

        raw_directory = (
            request_directory
            / "raw"
        )

        raw_candidates = sorted(
            raw_directory.glob(
                f"slot_{slot_number}_*"
            )
        )

        if (
            not standardized_path.exists()
            or not raw_candidates
        ):
            raise AudioProcessingError(
                f"Missing processed recording for slot {slot_number}."
            )

        metadata = validate_standardized_wav(
            standardized_path
        )

        processed_by_slot[slot_number] = ProcessedAudio(
            slot_number=slot_number,
            source="web_upload",
            original_filename=raw_candidates[-1].name,
            raw_path=str(raw_candidates[-1]),
            standardized_path=str(standardized_path),
            standardized_filename=standardized_path.name,
            sample_rate=metadata["sample_rate"],
            channels=metadata["channels"],
            sample_count=metadata["sample_count"],
            duration_seconds=round(
                metadata["duration_seconds"],
                6,
            ),
            subtype=metadata["subtype"],
            file_size_bytes=metadata["file_size_bytes"],
        )

    processed_recordings = [
        processed_by_slot[slot]
        for slot in sorted(processed_by_slot)
    ]

    return request_directory, processed_recordings

def process_three_microphone_recordings(
    microphone_items: Sequence[Mapping[str, Any]],
    request_id: str | None = None,
) -> tuple[Path, list[ProcessedAudio]]:
    """
    Process exactly three browser microphone recordings.

    Each item must have this structure:

        {
            "filename": "recording_1.webm",
            "content": b"..."
        }

    Returns
    -------
    tuple
        ``(request_directory, processed_recordings)``
    """

    items = list(microphone_items)

    if len(items) != REQUIRED_RECORDINGS:
        raise InvalidAudioInputError(
            f"Exactly {REQUIRED_RECORDINGS} microphone "
            f"recordings are required. Received: {len(items)}"
        )

    request_directory = create_request_audio_directory(
        request_id=request_id
    )

    processed_recordings: list[ProcessedAudio] = []

    try:
        for slot_number, item in enumerate(
            items,
            start=1,
        ):
            if not isinstance(item, Mapping):
                raise InvalidAudioInputError(
                    f"Microphone item {slot_number} must "
                    "be a mapping."
                )

            filename = item.get(
                "filename",
                f"recording_{slot_number}.webm",
            )

            content = item.get("content")

            if content is None:
                raise InvalidAudioInputError(
                    f"Microphone recording {slot_number} "
                    "does not contain audio content."
                )

            processed_recording = (
                process_microphone_recording(
                    audio_bytes=content,
                    filename=str(filename),
                    slot_number=slot_number,
                    request_directory=request_directory,
                )
            )

            processed_recordings.append(
                processed_recording
            )

    except Exception:
        cleanup_audio_request_directory(
            request_directory
        )
        raise

    return request_directory, processed_recordings


# ============================================================
# CLEANUP
# ============================================================

def cleanup_audio_request_directory(
    request_directory: str | Path,
) -> None:
    """
    Delete one request's raw and standardized audio files.

    Call this only after the complete screening pipeline has
    finished and the files are no longer required.
    """

    path = Path(
        request_directory
    ).resolve()

    work_directory = get_audio_work_directory()

    try:
        path.relative_to(work_directory)
    except ValueError as error:
        raise AudioProcessingError(
            "Refusing to delete a directory outside "
            "the configured audio work directory."
        ) from error

    if path == work_directory:
        raise AudioProcessingError(
            "Refusing to delete the root audio work directory."
        )

    if path.exists():
        shutil.rmtree(path)

        logger.info(
            "Deleted audio request directory: %s",
            path,
        )


def remove_raw_audio_files(
    processed_recordings: Iterable[ProcessedAudio],
) -> None:
    """
    Delete raw inputs while preserving standardized WAV files.
    """

    for recording in processed_recordings:
        raw_path = Path(recording.raw_path)

        try:
            raw_path.unlink(
                missing_ok=True
            )
        except OSError:
            logger.exception(
                "Could not delete raw audio file: %s",
                raw_path,
            )