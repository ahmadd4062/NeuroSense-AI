"""
Duplicate recording detection service for NeuroSense.

This module detects exact duplicate recordings by calculating a
SHA-256 hash from each standardized WAV file.

Important
---------
Duplicate detection is performed only after audio standardization.

This means the comparison is based on the final deployment format:

- WAV
- Mono
- 8000 Hz
- PCM 16-bit

The module detects byte-identical standardized recordings. It does
not perform acoustic similarity detection, speaker verification,
feature similarity, or machine-learning comparison.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_HASH_ALGORITHM = "sha256"
DEFAULT_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MB


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class DuplicateDetectionError(Exception):
    """Base exception for duplicate-detection failures."""


class InvalidDuplicateInputError(DuplicateDetectionError):
    """Raised when duplicate-detection input is invalid."""


class AudioHashingError(DuplicateDetectionError):
    """Raised when an audio file cannot be hashed."""


# ============================================================
# RESULT OBJECTS
# ============================================================

@dataclass(frozen=True)
class AudioHashResult:
    """
    Hash metadata for one standardized recording.
    """

    slot_number: int
    audio_path: str
    filename: str
    hash_algorithm: str
    audio_hash: str
    file_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary."""
        return asdict(self)


@dataclass
class DuplicateGroup:
    """
    Group of recordings sharing the same hash.
    """

    audio_hash: str
    slot_numbers: list[int] = field(
        default_factory=list
    )
    audio_paths: list[str] = field(
        default_factory=list
    )
    filenames: list[str] = field(
        default_factory=list
    )

    @property
    def is_duplicate_group(self) -> bool:
        """
        Return True when at least two recordings share the hash.
        """
        return len(self.slot_numbers) > 1

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary."""
        return {
            "audio_hash": self.audio_hash,
            "slot_numbers": list(self.slot_numbers),
            "audio_paths": list(self.audio_paths),
            "filenames": list(self.filenames),
            "is_duplicate_group": self.is_duplicate_group,
        }


@dataclass
class DuplicateDetectionResult:
    """
    Collection-level duplicate-detection result.
    """

    total_recordings: int
    unique_recordings: int
    duplicate_recordings: int

    has_duplicates: bool

    hash_results: list[AudioHashResult] = field(
        default_factory=list
    )

    duplicate_groups: list[DuplicateGroup] = field(
        default_factory=list
    )

    duplicate_slot_numbers: list[int] = field(
        default_factory=list
    )

    original_slot_numbers: list[int] = field(
        default_factory=list
    )

    processing_successful: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary."""
        return {
            "total_recordings": self.total_recordings,
            "unique_recordings": self.unique_recordings,
            "duplicate_recordings": self.duplicate_recordings,
            "has_duplicates": self.has_duplicates,
            "hash_results": [
                result.to_dict()
                for result in self.hash_results
            ],
            "duplicate_groups": [
                group.to_dict()
                for group in self.duplicate_groups
            ],
            "duplicate_slot_numbers": list(
                self.duplicate_slot_numbers
            ),
            "original_slot_numbers": list(
                self.original_slot_numbers
            ),
            "processing_successful": (
                self.processing_successful
            ),
            "error": self.error,
        }


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_hash_algorithm(
    hash_algorithm: str,
) -> str:
    """
    Validate that the requested hashing algorithm is available.

    Parameters
    ----------
    hash_algorithm:
        Hash algorithm name, such as ``sha256``.

    Returns
    -------
    str
        Normalized algorithm name.
    """

    normalized_algorithm = str(
        hash_algorithm
    ).strip().lower()

    if not normalized_algorithm:
        raise InvalidDuplicateInputError(
            "Hash algorithm cannot be empty."
        )

    if normalized_algorithm not in hashlib.algorithms_available:
        raise InvalidDuplicateInputError(
            f"Unsupported hash algorithm: "
            f"{normalized_algorithm}"
        )

    return normalized_algorithm


def validate_chunk_size(
    chunk_size_bytes: int,
) -> int:
    """
    Validate the file-reading chunk size.
    """

    try:
        normalized_chunk_size = int(
            chunk_size_bytes
        )
    except (TypeError, ValueError) as error:
        raise InvalidDuplicateInputError(
            "Chunk size must be an integer."
        ) from error

    if normalized_chunk_size <= 0:
        raise InvalidDuplicateInputError(
            "Chunk size must be greater than zero."
        )

    return normalized_chunk_size


def validate_audio_path(
    audio_path: str | Path,
) -> Path:
    """
    Validate one standardized audio path.

    Returns
    -------
    pathlib.Path
        Resolved file path.
    """

    path = Path(audio_path).resolve()

    if not path.exists():
        raise InvalidDuplicateInputError(
            f"Audio file does not exist: {path}"
        )

    if not path.is_file():
        raise InvalidDuplicateInputError(
            f"Audio path is not a file: {path}"
        )

    if path.stat().st_size <= 0:
        raise InvalidDuplicateInputError(
            f"Audio file is empty: {path.name}"
        )

    if path.suffix.lower() != ".wav":
        raise InvalidDuplicateInputError(
            f"Duplicate detection expects standardized WAV "
            f"files. Received: {path.name}"
        )

    return path


def validate_slot_number(
    slot_number: int,
) -> int:
    """
    Validate a recording slot number.
    """

    try:
        normalized_slot = int(slot_number)
    except (TypeError, ValueError) as error:
        raise InvalidDuplicateInputError(
            "Slot number must be an integer."
        ) from error

    if normalized_slot < 1:
        raise InvalidDuplicateInputError(
            "Slot number must be at least 1."
        )

    return normalized_slot


# ============================================================
# HASHING
# ============================================================

def calculate_file_hash(
    audio_path: str | Path,
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> str:
    """
    Calculate a cryptographic hash for one audio file.

    Parameters
    ----------
    audio_path:
        Standardized WAV path.

    hash_algorithm:
        Hash algorithm. SHA-256 is used by default.

    chunk_size_bytes:
        Number of bytes read per file chunk.

    Returns
    -------
    str
        Lowercase hexadecimal hash.
    """

    path = validate_audio_path(audio_path)

    normalized_algorithm = validate_hash_algorithm(
        hash_algorithm
    )

    normalized_chunk_size = validate_chunk_size(
        chunk_size_bytes
    )

    try:
        hash_object = hashlib.new(
            normalized_algorithm
        )

        with path.open("rb") as audio_file:
            while True:
                chunk = audio_file.read(
                    normalized_chunk_size
                )

                if not chunk:
                    break

                hash_object.update(chunk)

    except OSError as error:
        raise AudioHashingError(
            f"Unable to read audio file for hashing: "
            f"{path.name}"
        ) from error

    except ValueError as error:
        raise AudioHashingError(
            f"Unable to initialize hash algorithm: "
            f"{normalized_algorithm}"
        ) from error

    audio_hash = hash_object.hexdigest()

    if not audio_hash:
        raise AudioHashingError(
            f"Hash calculation returned an empty value "
            f"for {path.name}."
        )

    return audio_hash


def build_audio_hash_result(
    audio_path: str | Path,
    slot_number: int,
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> AudioHashResult:
    """
    Calculate hash metadata for one standardized recording.
    """

    path = validate_audio_path(audio_path)

    normalized_slot = validate_slot_number(
        slot_number
    )

    normalized_algorithm = validate_hash_algorithm(
        hash_algorithm
    )

    audio_hash = calculate_file_hash(
        audio_path=path,
        hash_algorithm=normalized_algorithm,
        chunk_size_bytes=chunk_size_bytes,
    )

    return AudioHashResult(
        slot_number=normalized_slot,
        audio_path=str(path),
        filename=path.name,
        hash_algorithm=normalized_algorithm,
        audio_hash=audio_hash,
        file_size_bytes=path.stat().st_size,
    )


# ============================================================
# COLLECTION DUPLICATE DETECTION
# ============================================================

def detect_duplicate_recordings(
    recording_paths: Sequence[str | Path],
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> DuplicateDetectionResult:
    """
    Detect exact duplicates in a recording collection.

    Duplicate handling rule
    -----------------------
    The first recording in each duplicate group is considered
    the original. All later recordings in that group are marked
    as duplicates and should be replaced.

    Example
    -------
    If slots 1 and 3 have the same hash:

        original_slot_numbers = [1]
        duplicate_slot_numbers = [3]

    Parameters
    ----------
    recording_paths:
        Ordered collection of standardized WAV paths.

    hash_algorithm:
        Hash algorithm used for comparison.

    chunk_size_bytes:
        File-reading chunk size.

    Returns
    -------
    DuplicateDetectionResult
        Duplicate groups and slot-level replacement information.
    """

    paths = list(recording_paths)

    if not paths:
        raise InvalidDuplicateInputError(
            "No recording paths were supplied."
        )

    normalized_algorithm = validate_hash_algorithm(
        hash_algorithm
    )

    normalized_chunk_size = validate_chunk_size(
        chunk_size_bytes
    )

    hash_results: list[AudioHashResult] = []

    for slot_number, audio_path in enumerate(
        paths,
        start=1,
    ):
        hash_result = build_audio_hash_result(
            audio_path=audio_path,
            slot_number=slot_number,
            hash_algorithm=normalized_algorithm,
            chunk_size_bytes=normalized_chunk_size,
        )

        hash_results.append(hash_result)

    groups_by_hash: dict[str, DuplicateGroup] = {}

    for hash_result in hash_results:
        group = groups_by_hash.get(
            hash_result.audio_hash
        )

        if group is None:
            group = DuplicateGroup(
                audio_hash=hash_result.audio_hash
            )

            groups_by_hash[
                hash_result.audio_hash
            ] = group

        group.slot_numbers.append(
            hash_result.slot_number
        )

        group.audio_paths.append(
            hash_result.audio_path
        )

        group.filenames.append(
            hash_result.filename
        )

    duplicate_groups = [
        group
        for group in groups_by_hash.values()
        if group.is_duplicate_group
    ]

    duplicate_slot_numbers: list[int] = []
    original_slot_numbers: list[int] = []

    for duplicate_group in duplicate_groups:
        sorted_slots = sorted(
            duplicate_group.slot_numbers
        )

        original_slot_numbers.append(
            sorted_slots[0]
        )

        duplicate_slot_numbers.extend(
            sorted_slots[1:]
        )

    duplicate_slot_numbers = sorted(
        set(duplicate_slot_numbers)
    )

    original_slot_numbers = sorted(
        set(original_slot_numbers)
    )

    unique_recordings = len(
        groups_by_hash
    )

    total_recordings = len(
        hash_results
    )

    duplicate_recordings = len(
        duplicate_slot_numbers
    )

    return DuplicateDetectionResult(
        total_recordings=total_recordings,
        unique_recordings=unique_recordings,
        duplicate_recordings=duplicate_recordings,
        has_duplicates=bool(
            duplicate_slot_numbers
        ),
        hash_results=hash_results,
        duplicate_groups=duplicate_groups,
        duplicate_slot_numbers=(
            duplicate_slot_numbers
        ),
        original_slot_numbers=(
            original_slot_numbers
        ),
        processing_successful=True,
        error=None,
    )


# ============================================================
# PROCESSED-AUDIO SUPPORT
# ============================================================

def detect_duplicates_in_processed_recordings(
    processed_recordings: Iterable[Any],
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> DuplicateDetectionResult:
    """
    Detect duplicates in ProcessedAudio objects.

    Each object must expose:

        standardized_path
    """

    standardized_paths: list[str] = []

    for recording in processed_recordings:
        standardized_path = getattr(
            recording,
            "standardized_path",
            None,
        )

        if not standardized_path:
            raise InvalidDuplicateInputError(
                "A processed recording does not contain "
                "'standardized_path'."
            )

        standardized_paths.append(
            str(standardized_path)
        )

    return detect_duplicate_recordings(
        recording_paths=standardized_paths,
        hash_algorithm=hash_algorithm,
        chunk_size_bytes=chunk_size_bytes,
    )


# ============================================================
# QUALITY-VALID RECORDING SUPPORT
# ============================================================

def detect_duplicates_in_valid_recordings(
    quality_results: Iterable[Any],
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> DuplicateDetectionResult:
    """
    Detect duplicates only among recordings that passed quality.

    Each quality result must expose:

        is_valid
        audio_path
        slot_number

    Important
    ---------
    Slot numbers are preserved from the original collection.
    """

    valid_items: list[tuple[int, str]] = []

    for result in quality_results:
        is_valid = bool(
            getattr(result, "is_valid", False)
        )

        if not is_valid:
            continue

        slot_number = getattr(
            result,
            "slot_number",
            None,
        )

        audio_path = getattr(
            result,
            "audio_path",
            None,
        )

        if slot_number is None or not audio_path:
            raise InvalidDuplicateInputError(
                "A valid quality result is missing "
                "'slot_number' or 'audio_path'."
            )

        valid_items.append(
            (
                validate_slot_number(slot_number),
                str(audio_path),
            )
        )

    if not valid_items:
        return DuplicateDetectionResult(
            total_recordings=0,
            unique_recordings=0,
            duplicate_recordings=0,
            has_duplicates=False,
            hash_results=[],
            duplicate_groups=[],
            duplicate_slot_numbers=[],
            original_slot_numbers=[],
            processing_successful=True,
            error=None,
        )

    hash_results: list[AudioHashResult] = []

    normalized_algorithm = validate_hash_algorithm(
        hash_algorithm
    )

    normalized_chunk_size = validate_chunk_size(
        chunk_size_bytes
    )

    for slot_number, audio_path in valid_items:
        hash_result = build_audio_hash_result(
            audio_path=audio_path,
            slot_number=slot_number,
            hash_algorithm=normalized_algorithm,
            chunk_size_bytes=normalized_chunk_size,
        )

        hash_results.append(hash_result)

    return build_duplicate_result_from_hash_results(
        hash_results
    )


def build_duplicate_result_from_hash_results(
    hash_results: Sequence[AudioHashResult],
) -> DuplicateDetectionResult:
    """
    Build a duplicate result from existing hash results.

    This helper preserves non-sequential original slot numbers.
    """

    results = list(hash_results)

    if not results:
        return DuplicateDetectionResult(
            total_recordings=0,
            unique_recordings=0,
            duplicate_recordings=0,
            has_duplicates=False,
            hash_results=[],
            duplicate_groups=[],
            duplicate_slot_numbers=[],
            original_slot_numbers=[],
            processing_successful=True,
            error=None,
        )

    seen_slots: set[int] = set()

    groups_by_hash: dict[str, DuplicateGroup] = {}

    for hash_result in results:
        if hash_result.slot_number in seen_slots:
            raise InvalidDuplicateInputError(
                f"Duplicate slot number detected: "
                f"{hash_result.slot_number}"
            )

        seen_slots.add(
            hash_result.slot_number
        )

        group = groups_by_hash.get(
            hash_result.audio_hash
        )

        if group is None:
            group = DuplicateGroup(
                audio_hash=hash_result.audio_hash
            )

            groups_by_hash[
                hash_result.audio_hash
            ] = group

        group.slot_numbers.append(
            hash_result.slot_number
        )

        group.audio_paths.append(
            hash_result.audio_path
        )

        group.filenames.append(
            hash_result.filename
        )

    duplicate_groups = [
        group
        for group in groups_by_hash.values()
        if group.is_duplicate_group
    ]

    duplicate_slot_numbers: list[int] = []
    original_slot_numbers: list[int] = []

    for group in duplicate_groups:
        sorted_slots = sorted(
            group.slot_numbers
        )

        original_slot_numbers.append(
            sorted_slots[0]
        )

        duplicate_slot_numbers.extend(
            sorted_slots[1:]
        )

    duplicate_slot_numbers = sorted(
        set(duplicate_slot_numbers)
    )

    original_slot_numbers = sorted(
        set(original_slot_numbers)
    )

    return DuplicateDetectionResult(
        total_recordings=len(results),
        unique_recordings=len(
            groups_by_hash
        ),
        duplicate_recordings=len(
            duplicate_slot_numbers
        ),
        has_duplicates=bool(
            duplicate_slot_numbers
        ),
        hash_results=results,
        duplicate_groups=duplicate_groups,
        duplicate_slot_numbers=(
            duplicate_slot_numbers
        ),
        original_slot_numbers=(
            original_slot_numbers
        ),
        processing_successful=True,
        error=None,
    )


# ============================================================
# RETRY / REPLACEMENT SUPPORT
# ============================================================

def compare_replacement_recording(
    replacement_audio_path: str | Path,
    replacement_slot_number: int,
    accepted_hash_results: Sequence[AudioHashResult],
    hash_algorithm: str = DEFAULT_HASH_ALGORITHM,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> dict[str, Any]:
    """
    Check whether a replacement recording duplicates any already
    accepted recording.

    Parameters
    ----------
    replacement_audio_path:
        Standardized WAV path for the replacement recording.

    replacement_slot_number:
        Slot being replaced.

    accepted_hash_results:
        Hashes of recordings already accepted in other slots.

    Returns
    -------
    dict
        Duplicate status and matching slot information.
    """

    normalized_slot = validate_slot_number(
        replacement_slot_number
    )

    replacement_hash_result = build_audio_hash_result(
        audio_path=replacement_audio_path,
        slot_number=normalized_slot,
        hash_algorithm=hash_algorithm,
        chunk_size_bytes=chunk_size_bytes,
    )

    matching_slots: list[int] = []

    for accepted_result in accepted_hash_results:
        if (
            accepted_result.slot_number
            == normalized_slot
        ):
            continue

        if (
            accepted_result.audio_hash
            == replacement_hash_result.audio_hash
        ):
            matching_slots.append(
                accepted_result.slot_number
            )

    matching_slots = sorted(
        set(matching_slots)
    )

    return {
        "slot_number": normalized_slot,
        "audio_path": (
            replacement_hash_result.audio_path
        ),
        "audio_hash": (
            replacement_hash_result.audio_hash
        ),
        "is_duplicate": bool(
            matching_slots
        ),
        "matching_slot_numbers": matching_slots,
        "hash_result": (
            replacement_hash_result.to_dict()
        ),
    }


def update_hash_results_for_replacement(
    existing_hash_results: Sequence[AudioHashResult],
    replacement_hash_result: AudioHashResult,
) -> list[AudioHashResult]:
    """
    Replace or append one slot's hash result.

    This is useful after a rejected slot is re-recorded.
    """

    replacement_slot = validate_slot_number(
        replacement_hash_result.slot_number
    )

    updated_results: list[AudioHashResult] = []

    replacement_inserted = False

    for existing_result in existing_hash_results:
        if (
            existing_result.slot_number
            == replacement_slot
        ):
            updated_results.append(
                replacement_hash_result
            )

            replacement_inserted = True
        else:
            updated_results.append(
                existing_result
            )

    if not replacement_inserted:
        updated_results.append(
            replacement_hash_result
        )

    updated_results.sort(
        key=lambda result: result.slot_number
    )

    return updated_results


# ============================================================
# SERIALIZATION HELPERS
# ============================================================

def hash_results_from_mappings(
    items: Iterable[Mapping[str, Any]],
) -> list[AudioHashResult]:
    """
    Rebuild AudioHashResult objects from dictionaries.

    Useful when retry state is stored in the Django session.
    """

    results: list[AudioHashResult] = []

    for item in items:
        try:
            result = AudioHashResult(
                slot_number=int(
                    item["slot_number"]
                ),
                audio_path=str(
                    item["audio_path"]
                ),
                filename=str(
                    item["filename"]
                ),
                hash_algorithm=str(
                    item["hash_algorithm"]
                ),
                audio_hash=str(
                    item["audio_hash"]
                ),
                file_size_bytes=int(
                    item["file_size_bytes"]
                ),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise InvalidDuplicateInputError(
                "Invalid serialized audio hash result."
            ) from error

        results.append(result)

    return results


# ============================================================
# USER-FACING SUMMARY
# ============================================================

def build_duplicate_summary(
    duplicate_result: DuplicateDetectionResult,
) -> dict[str, Any]:
    """
    Build a concise serializable duplicate summary.
    """

    return {
        "has_duplicates": (
            duplicate_result.has_duplicates
        ),
        "total_recordings": (
            duplicate_result.total_recordings
        ),
        "unique_recordings": (
            duplicate_result.unique_recordings
        ),
        "duplicate_recordings": (
            duplicate_result.duplicate_recordings
        ),
        "duplicate_slot_numbers": list(
            duplicate_result.duplicate_slot_numbers
        ),
        "original_slot_numbers": list(
            duplicate_result.original_slot_numbers
        ),
        "message": (
            "Duplicate recordings were detected. "
            "Only the duplicate slots must be replaced."
            if duplicate_result.has_duplicates
            else "No duplicate recordings were detected."
        ),
    }