from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import Any
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST
from speech_app.services.persistence_service import (
    PredictionPersistenceError,
    save_screening_prediction,
)
from .services.audio_module.audio_processing import (
    AudioProcessingError,
    cleanup_audio_request_directory,
    process_three_uploaded_recordings,
)
from .services.audio_module.duplicate_detection import (
    DuplicateDetectionError,
    build_duplicate_summary,
    detect_duplicates_in_valid_recordings,
)
from .services.audio_module.quality_assessment import (
    AudioQualityError,
    assess_processed_recordings,
    build_quality_summary,
)
from .services.fusion_service import (
    FusionServiceError,
    build_fusion_summary,
    fuse_prediction_collection,
)
from .services.model_service import (
    ModelServiceError,
    build_prediction_summary,
    inspect_model_artifact,
    predict_valid_recordings,
)

logger = logging.getLogger(__name__)

REQUIRED_RECORDINGS = 3
MAX_RECORDING_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_RECORDING_EXTENSIONS = {".wav", ".mp4", ".ogg", ".webm"}


class ScreeningRequestError(ValueError):
    """Raised when a screening request is invalid."""


def home(request):
    return render(request, "home.html")


@require_GET
def screening_page(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "speech_app/screening.html",
        {
            "required_recordings": REQUIRED_RECORDINGS,
            "maximum_recording_seconds": 6,
        },
    )


def _json_error(
    message: str,
    *,
    status: int = 400,
    code: str = "screening_error",
    details: dict[str, Any] | None = None,
) -> JsonResponse:
    payload: dict[str, Any] = {
        "success": False,
        "code": code,
        "message": message,
    }

    if details is not None:
        payload["details"] = details

    return JsonResponse(payload, status=status)


def _get_uploaded_recordings(
    request: HttpRequest,
) -> list[tuple[int, Any]]:
    """
    Return uploaded recordings together with their slot numbers.

    Initial screening:
        - Exactly 3 recordings are required.

    Replacement screening:
        - One or more recordings may be uploaded.
        - Only the uploaded slots are returned.
    """

    is_replacement_request = bool(
        request.POST.get("screening_id")
        or request.POST.get("screeningId")
    )

    recordings: list[tuple[int, Any]] = []

    for slot_number in range(1, REQUIRED_RECORDINGS + 1):

        recording = request.FILES.get(
            f"recording_{slot_number}"
        )

        if recording is not None:
            recordings.append(
                (
                    slot_number,
                    recording,
                )
            )

    if not recordings:
        uploaded = request.FILES.getlist("recordings")

        recordings = [
            (
                index,
                recording,
            )
            for index, recording in enumerate(
                uploaded,
                start=1,
            )
        ]

    if is_replacement_request:

        if not recordings:
            raise ScreeningRequestError(
                "At least one replacement recording is required."
            )

    else:

        if len(recordings) != REQUIRED_RECORDINGS:
            raise ScreeningRequestError(
                f"Exactly {REQUIRED_RECORDINGS} recordings are required. "
                f"Received {len(recordings)}."
            )

    for slot_number, recording in recordings:

        filename = str(
            getattr(recording, "name", "") or ""
        )

        extension = Path(filename).suffix.lower()

        file_size = getattr(
            recording,
            "size",
            None,
        )

        if file_size == 0:
            raise ScreeningRequestError(
                f"Recording {slot_number} is empty."
            )

        if (
            file_size is not None
            and file_size > MAX_RECORDING_SIZE_BYTES
        ):
            raise ScreeningRequestError(
                f"Recording {slot_number} exceeds the maximum "
                f"allowed size of "
                f"{MAX_RECORDING_SIZE_BYTES // (1024 * 1024)} MB."
            )

        if extension not in ALLOWED_RECORDING_EXTENSIONS:

            allowed = ", ".join(
                sorted(ALLOWED_RECORDING_EXTENSIONS)
            )

            raise ScreeningRequestError(
                f"Recording {slot_number} has an unsupported format. "
                f"Allowed extensions: {allowed}."
            )

    return recordings
def _serialize_processed_recordings(
    processed_recordings: list[Any],
) -> list[dict[str, Any]]:
    return [
        recording.to_dict()
        for recording in processed_recordings
    ]


def _serialize_quality_results(
    quality_results: list[Any],
) -> list[dict[str, Any]]:
    return [
        result.to_dict()
        for result in quality_results
    ]


def _valid_non_duplicate_slots(
    quality_results: list[Any],
    duplicate_slot_numbers: list[int],
) -> list[int]:
    excluded_slots = {
        int(slot_number)
        for slot_number in duplicate_slot_numbers
    }

    return [
        int(result.slot_number)
        for result in quality_results
        if result.is_valid
        and int(result.slot_number) not in excluded_slots
    ]


def _replacement_slots(
    quality_results: list[Any],
    duplicate_slot_numbers: list[int],
) -> list[int]:
    invalid_slots = [
        int(result.slot_number)
        for result in quality_results
        if not result.is_valid
    ]

    return sorted(
        set(invalid_slots)
        | {
            int(slot_number)
            for slot_number in duplicate_slot_numbers
        }
    )


def _get_quality_message(reason: str) -> str:
    messages = {
        "snr_extremely_low": (
            "Too much background noise was detected. "
            "Please move to a quieter place and record again."
        ),
        "invalid_snr": (
            "We couldn't assess the recording quality. "
            "Please record again."
        ),
        "duration_below_minimum": (
            "The recording is too short. "
            "Please sustain 'Aaaa' for at least 3 seconds."
        ),
        "rms_below_minimum": (
            "Your voice is too quiet. "
            "Please speak a little louder or move closer to the microphone."
        ),
        "excessive_clipping": (
            "Your voice is too loud or too close to the microphone. "
            "Please move slightly away and record again."
        ),
        "silence_above_maximum": (
            "Too much silence was detected. "
            "Please sustain 'Aaaa' continuously while recording."
        ),
        "unexpected_sample_rate": (
            "This audio format could not be processed correctly. "
            "Please record again."
        ),
        "unexpected_channel_count": (
            "This audio format is not supported correctly. "
            "Please record again."
        ),
        "processing_failure": (
            "We couldn't process this recording. "
            "Please record or upload it again."
        ),
    }

    return messages.get(
        reason,
        "The recording quality was not sufficient. Please record again."
    )

def _build_quality_feedback(
    quality_results,
    duplicate_slot_numbers=None,
):
    feedback = []
    duplicate_slots = set(
        int(slot)
        for slot in (duplicate_slot_numbers or [])
    )

    for result in quality_results:
        messages = []

        if not result.is_valid:
            messages.extend(
                _get_quality_message(reason)
                for reason in result.blocking_reasons
            )

        if int(result.slot_number) in duplicate_slots:
            messages.append(
                "This recording appears to be the same as another recording. "
                "Please provide a new and different voice sample."
            )

        if messages:
            feedback.append({
                "slot": result.slot_number,
                "messages": messages,
            })

    return feedback
@require_POST
def analyze_screening(request: HttpRequest) -> JsonResponse:
    request_directory: Path | None = None
    keep_request_audio = False
    existing_screening_id = (
    request.POST.get("screening_id")
    or request.POST.get("screeningId")
    )

    if existing_screening_id:
        screening_id = existing_screening_id
    else:
        screening_id = uuid.uuid4().hex

    try:
        uploaded_recordings = _get_uploaded_recordings(request)

        request_directory, processed_recordings = (
            process_three_uploaded_recordings(
                uploaded_files=uploaded_recordings,
                request_id=screening_id,
            )
        )
        
        quality_results = assess_processed_recordings(
            processed_recordings
        )
        quality_summary = build_quality_summary(
            quality_results
        )

        duplicate_result = detect_duplicates_in_valid_recordings(
            quality_results
        )
        duplicate_summary = build_duplicate_summary(
            duplicate_result
        )

        duplicate_slot_numbers = list(
            duplicate_result.duplicate_slot_numbers
        )
        usable_slot_numbers = _valid_non_duplicate_slots(
            quality_results,
            duplicate_slot_numbers,
        )

        if len(usable_slot_numbers) != REQUIRED_RECORDINGS:
            replacement_slots = _replacement_slots(
                quality_results,
                duplicate_slot_numbers,
            )

            keep_request_audio = True

            return JsonResponse(
                {
                    "success": False,
                    "code": "recordings_need_replacement",
                    "message": (
                        "Some recordings need to be recorded again."
                    ),
                    "screening_id": screening_id,
                    "replacement_required": True,
                    "replacement_slots": replacement_slots,
                    "quality_feedback": _build_quality_feedback(
                        quality_results,
                        duplicate_slot_numbers,
                    ),
                },
                status=422,
            )

        prediction_collection = predict_valid_recordings(
            quality_results=quality_results,
            duplicate_slot_numbers=duplicate_slot_numbers,
            continue_on_error=True,
        )
        prediction_summary = build_prediction_summary(
            prediction_collection
        )

        if (
            prediction_collection.successful_predictions
            != REQUIRED_RECORDINGS
            ):
            failed_prediction_slots = [
                int(result.slot_number)
                for result
                in prediction_collection.recording_results
                if not result.prediction_successful
            ]

            replacement_slots = sorted(
                set(
                    _replacement_slots(
                        quality_results,
                        duplicate_slot_numbers,
                    )
                    + failed_prediction_slots
                )
            )

            keep_request_audio = True

            return JsonResponse(
                {
                    "success": False,
                    "code": "recordings_need_replacement",
                    "message": (
                        "One or more recordings need to be "
                        "recorded again."
                    ),
                    "screening_id": screening_id,
                    "replacement_required": True,
                    "replacement_slots": replacement_slots,
                },
                status=422,
            )

        fusion_result = fuse_prediction_collection(
            prediction_collection
        )
        fusion_summary = build_fusion_summary(
            fusion_result
        )

        try:
            saved_prediction = save_screening_prediction(
                screening_id=screening_id,
                prediction_collection=prediction_collection,
                fusion_result=fusion_result,
                healthy_threshold=0.23,
                parkinson_threshold=0.27,
                metadata={
                    "request_directory": str(request_directory) if request_directory else "",
                },
            )

            logger.info(
                "Prediction saved to database with ID %s",
                saved_prediction.id,
            )

        except PredictionPersistenceError as error:
            logger.exception(
                "Failed to save prediction for screening %s.",
                screening_id,
            )

            return _json_error(
                str(error),
                status=500,
                code="prediction_save_failed",
            )

        if not fusion_result.fusion_successful:
            logger.warning(
                "Fusion failed for screening %s: %s",
                screening_id,
                fusion_result.error,
            )
            return JsonResponse(
                {
                    "success": False,
                    "code": "screening_not_completed",
                    "message": (
                        "Audio screening could not be completed. "
                        "Please try again."
                    ),
                    "screening_id": screening_id,
                },
                status=422,
            )

        # Keep model and fusion outputs on the server. They should be
        # persisted against screening_id when the shared screening
        # database/session layer is connected. Do not expose them here.
        logger.info(
            "Audio screening %s completed successfully.",
            screening_id,
        )

        request.session["audio_module_result"] = {
            'prediction': fusion_result.final_label,  
            'pd_probability': fusion_result.fused_score,
            'control_probability': 1 - fusion_result.fused_score,
            'screening_id': screening_id,
            'model_used': 'Linear SVM'
            }
        request.session.modified = True
        request.session["screening_id"] = screening_id
        request.session.modified = True

        return JsonResponse(
            {
                "success": True,
                "screening_id": screening_id,
                "message": (
                    "Audio screening completed successfully."
                ),
                "next_url": reverse(
                    "speech_app:module_three"
                ),
            },
            status=200,
        )

    except ScreeningRequestError as error:
        return _json_error(
            str(error),
            status=400,
            code="invalid_request",
        )

    except AudioProcessingError as error:
        logger.warning(
            "Audio processing failed for screening %s: %s",
            screening_id,
            error,
        )
        return _json_error(
            str(error),
            status=422,
            code="audio_processing_failed",
        )

    except AudioQualityError as error:
        logger.warning(
            "Quality assessment failed for screening %s: %s",
            screening_id,
            error,
        )
        return _json_error(
            str(error),
            status=422,
            code="quality_assessment_failed",
        )

    except DuplicateDetectionError as error:
        logger.warning(
            "Duplicate detection failed for screening %s: %s",
            screening_id,
            error,
        )
        return _json_error(
            str(error),
            status=422,
            code="duplicate_detection_failed",
        )

    except ModelServiceError as error:
        logger.exception(
            "Model service failed for screening %s.",
            screening_id,
        )
        return _json_error(
            str(error),
            status=500,
            code="model_service_failed",
        )

    except FusionServiceError as error:
        logger.exception(
            "Fusion service failed for screening %s.",
            screening_id,
        )
        return _json_error(
            str(error),
            status=422,
            code="fusion_service_failed",
        )

    except Exception:
        logger.exception(
            "Unexpected screening failure for %s.",
            screening_id,
        )
        return _json_error(
            "An unexpected error occurred during screening.",
            status=500,
            code="internal_server_error",
        )

    finally:
        preserve_audio = bool(
            getattr(
                settings,
                "PRESERVE_SCREENING_AUDIO",
                False,
            )
        )

        if (
            request_directory is not None
            and not preserve_audio
            and not keep_request_audio
        ):
            try:
                cleanup_audio_request_directory(
                    request_directory
                )
            except Exception:
                logger.exception(
                    "Failed to clean screening directory: %s",
                    request_directory,
                )


@require_GET
def screening_health(request: HttpRequest) -> JsonResponse:
    try:
        model_information = inspect_model_artifact()

        return JsonResponse(
            {
                "success": True,
                "status": "ready",
                "model": model_information,
            },
            status=200,
        )

    except ModelServiceError as error:
        logger.exception(
            "Screening health check failed."
        )
        return JsonResponse(
            {
                "success": False,
                "status": "unavailable",
                "message": str(error),
            },
            status=503,
        )

    except Exception:
        logger.exception(
            "Unexpected screening health-check failure."
        )
        return JsonResponse(
            {
                "success": False,
                "status": "unavailable",
                "message": (
                    "The screening service health check failed."
                ),
            },
            status=503,
        )


def module_three_placeholder(request):
    return render(request, "gait_app/gait_screening.html")