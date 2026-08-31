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

from .gait_model import predictor
from .feature_extractor import extract_pose_sequence_from_video, validate_video_quality
from .models import GaitResult
from django.views.decorators.csrf import csrf_exempt

# Set up logging
logger = logging.getLogger(__name__)


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
        - quality: dict (quality_score, feedback, is_valid)
    """
    try:
        # Check if video file is provided
        if 'video' not in request.FILES:
            return Response(
                {'error': 'No video file provided'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        video_file = request.FILES['video']
        
        # Get screening_id from request or generate new
        screening_id = request.data.get('screening_id') or request.POST.get('screening_id')
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
            return Response(
                {
                    'error': 'Could not extract pose from video. '
                             'Please ensure side view, good lighting, '
                             'and the person is clearly visible.',
                    'quality': {
                        'is_valid': False,
                        'score': 0.0,
                        'feedback': ['❌ No pose data could be extracted from this video.']
                    }
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate video quality
        is_valid, quality_score, feedback = validate_video_quality(pose_sequence)
        
        # If quality is poor, return feedback without prediction
        if not is_valid:
            return Response(
                {
                    'error': 'Video quality is poor. Please re-record following the guidelines.',
                    'quality': {
                        'is_valid': False,
                        'score': quality_score,
                        'feedback': feedback
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
                'feedback': feedback
            }
        }
        
        logger.info(f"Prediction: {result['prediction']}, Confidence: {confidence:.3f}, Quality: {quality_score:.2f}")
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Gait prediction error: {str(e)}")
        return Response(
            {'error': f'Processing error: {str(e)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def gait_screening_page(request):
    """Render the gait screening page with language support"""
    return render(request, 'gait_app/gait_screening.html', {
        'lang': request.LANG  # Pass language to template
    })


def gait_results_page(request):
    """Render the gait results page with language support"""
    return render(request, 'gait_app/gait_results.html', {
        'lang': request.LANG  # Pass language to template
    })