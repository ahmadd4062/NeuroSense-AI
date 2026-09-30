from django.shortcuts import render, redirect
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
import os
import tempfile
import logging
import uuid
import numpy as np
from django.contrib.auth.decorators import login_required

from .gait_model import predictor
from .feature_extractor import extract_pose_sequence_from_video, validate_video_quality
from .models import GaitResult
from django.views.decorators.csrf import csrf_exempt

# Set up logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reason code -> user-facing message (EN / UR)
# Kept here so both the "no pose" early return and the quality-check failure
# return path use the same source of truth.
# ---------------------------------------------------------------------------
REASON_MESSAGES = {
    'no_pose': {
        'en': "We couldn't detect a person in your video. Please make sure "
              "you are fully visible and well-lit.",
        'ur': "ہم آپ کی ویڈیو میں کوئی شخص نہیں ڈھونڈ سکے۔ براہ کرم یقینی بنائیں "
              "کہ آپ پوری طرح نظر آ رہے ہیں اور روشنی مناسب ہے۔",
    },
    'wrong_angle': {
        'en': "Please record from the SIDE view. Your video was recorded from "
              "the front or back, which cannot be analyzed. Stand sideways "
              "with your profile facing the camera.",
        'ur': "براہ کرم سائیڈ ویو سے ویڈیو ریکارڈ کریں۔ آپ کی ویڈیو سامنے یا پیچھے "
              "سے بنی تھی، جس کا تجزیہ نہیں ہو سکتا۔ سائیڈ میں کھڑے ہوں اور "
              "کیمرے کی طرف اپنا پروفائل رکھیں۔",
    },
    'not_side_view': {
        'en': "Side view not clearly detected. Turn 90° so the camera sees "
              "your profile (one shoulder facing the camera).",
        'ur': "سائیڈ ویو واضح طور پر نظر نہیں آئی۔ 90 ڈگری مڑیں تاکہ کیمرہ آپ کا "
              "پروفائل دیکھ سکے (ایک کندھا کیمرے کی طرف)۔",
    },
    'no_person': {
        'en': "You were not visible in most of the video. Please stay fully "
              "in frame for the entire recording.",
        'ur': "ویڈیو کے زیادہ تر حصے میں آپ نظر نہیں آ رہے تھے۔ براہ کرم پوری "
              "ریکارڈنگ کے دوران فریم میں رہیں۔",
    },
    'no_full_body': {
        'en': "Full body not visible. Please step back so your head and feet "
              "are both in the frame.",
        'ur': "پورا جسم نظر نہیں آ رہا۔ براہ کرم پیچھے ہٹیں تاکہ آپ کا سر اور "
              "پاؤں دونوں فریم میں آ جائیں۔",
    },
    'no_movement': {
        'en': "We didn't detect clear walking movement. Please walk in a "
              "straight line during the recording.",
        'ur': "ہمیں واضح چلنے کی حرکت نظر نہیں آئی۔ براہ کرم ریکارڈنگ کے دوران "
              "سیدھی لائن میں چلیں۔",
    },
    'low_quality': {
        'en': "Video quality was not sufficient. Please re-record following "
              "the on-screen guidelines.",
        'ur': "ویڈیو کا معیار کافی نہیں تھا۔ براہ کرم اسکرین پر دی گئی ہدایات "
              "کے مطابق دوبارہ ریکارڈ کریں۔",
    },
}


def _get_reason_message(reason_code, request):
    """Return a localized user-facing message for a reason code."""
    current_lang = request.session.get('language', 'en')
    if current_lang not in ('en', 'ur'):
        current_lang = 'en'

    message_map = REASON_MESSAGES.get(reason_code, REASON_MESSAGES['low_quality'])
    return message_map[current_lang]


@api_view(['POST'])
@parser_classes([MultiPartParser])
@csrf_exempt
def gait_predict(request):
    """
    API endpoint for gait analysis.

    Expects:
        - video file in request.FILES
        - screening_id (optional) in request.data

    Returns:
        - prediction: PD or Control
        - pd_probability: float (0-1)
        - control_probability: float (0-1)
        - model_used: string (LSTM)
        - screening_id: string
        - quality: dict (quality_score, feedback, is_valid, reason)
    """
    try:
        # Check if video file is provided
        if 'video' not in request.FILES:
            return Response(
                {
                    'error': 'No video file provided',
                    'reason': 'low_quality',
                    'error_code': 'no_video',
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        video_file = request.FILES['video']

        # Get screening_id from request or generate new
        screening_id = (
            request.data.get('screening_id')
            or request.POST.get('screening_id')
            or request.session.get('screening_id')
        )
        if not screening_id:
            screening_id = uuid.uuid4().hex

        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in video_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name

        logger.info(f"Processing video: {video_file.name}, size: {video_file.size} bytes")

        # Extract pose sequence for LSTM
        pose_sequence = extract_pose_sequence_from_video(tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        if pose_sequence is None:
            user_message = _get_reason_message('no_pose', request)
            return Response(
                {
                    'error': user_message,
                    'reason': 'no_pose',
                    'error_code': 'no_pose',
                    'quality': {
                        'is_valid': False,
                        'score': 0.0,
                        'reason': 'no_pose',
                        'feedback': ['❌ No pose data could be extracted from this video.']
                    }
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate video quality (now returns 4-tuple with reason code)
        is_valid, quality_score, feedback, reason_code = validate_video_quality(pose_sequence)

        # If quality is poor, return feedback without prediction
        if not is_valid:
            user_message = _get_reason_message(reason_code, request)

            logger.info(
                f"Video rejected for screening {screening_id}: "
                f"reason={reason_code}, score={quality_score}"
            )

            return Response(
                {
                    'error': user_message,
                    'reason': reason_code,          # structured code for JS
                    'error_code': 'poor_quality',   # top-level category
                    'quality': {
                        'is_valid': False,
                        'score': quality_score,
                        'feedback': feedback,
                        'reason': reason_code,
                    }
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        # Get prediction from LSTM model
        result = predictor.predict(pose_sequence)

        # Calculate confidence (max probability)
        confidence = max(result['pd_probability'], result['control_probability'])

        # Convert pose sequence to list for JSON storage
        pose_sequence_list = pose_sequence[0].tolist()  # (50, 34)

        # Get user from session or request (if authenticated)
        user = request.user if request.user.is_authenticated else None

        # Save to database (pose sequence stored)
        try:
            GaitResult.objects.create(
                user=user,  # Link to user (matches Fatima and Bushra's database)
                screening_id=screening_id,
                prediction=result['prediction'],
                pd_probability=float(result['pd_probability']),
                control_probability=float(result['control_probability']),
                model_used='LSTM',
                pose_sequence=pose_sequence_list,
                video_fps=30,  # Default, can be extracted from video
                pose_quality_score=quality_score
            )
            logger.info(f"Saved gait result for screening: {screening_id}, User: {user}")
        except Exception as e:
            logger.warning(f"Failed to save gait result: {str(e)}")

        # Store in session for fusion module
        request.session['gait_result'] = {
            'screening_id': screening_id,
            'prediction': result['prediction'],
            'pd_probability': float(result['pd_probability']),
            'control_probability': float(result['control_probability']),
            'model_used': 'LSTM'
        }
        request.session.modified = True

        # Prepare response with quality info
        response_data = {
            'prediction': result['prediction'],
            'pd_probability': result['pd_probability'],
            'control_probability': result['control_probability'],
            'model_used': 'LSTM',
            'screening_id': screening_id,
            'quality': {
                'is_valid': True,
                'score': quality_score,
                'feedback': feedback,
                'reason': reason_code,   # will be "ok" on success
            }
        }

        logger.info(
            f"Prediction: {result['prediction']}, "
            f"Confidence: {confidence:.3f}, Quality: {quality_score:.2f}"
        )

        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Gait prediction error: {str(e)}")
        return Response(
            {
                'error': f'Processing error: {str(e)}',
                'reason': 'low_quality',
                'error_code': 'processing_error',
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@login_required
def gait_screening_page(request):
    """Render the gait screening page with language support"""
    return render(request, 'gait_app/gait_screening.html', {
        'lang': request.LANG  # Pass language to template
    })


@login_required
def gait_results_page(request):
    """Render the gait results page with language support"""
    return render(request, 'gait_app/gait_results.html', {
        'lang': request.LANG  # Pass language to template
    })