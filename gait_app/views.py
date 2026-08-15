from django.shortcuts import render
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

# Set up logging
logger = logging.getLogger(__name__)


def validate_video_quality(pose_sequence, fps=30):
    """
    Validate the quality of extracted pose sequence.
    Returns: (is_valid, quality_score, feedback_list)
    """
    feedback = []
    quality_score = 0.0
    
    if pose_sequence is None:
        return False, 0.0, ["❌ No pose data extracted"]
    
    # Reshape to (frames, 17, 2)
    try:
        poses = pose_sequence[0].reshape(50, 17, 2)
    except:
        return False, 0.0, ["❌ Invalid pose data format"]
    
    # 1. Check if any poses were detected (not all zeros)
    total_zeros = np.sum(np.all(poses == 0, axis=2))
    zero_frames = np.sum(np.all(poses.reshape(50, 34) == 0, axis=1))
    
    if zero_frames > 20:  # More than 40% frames are empty
        feedback.append("❌ No person detected in most frames. Please ensure full body is visible.")
        quality_score += 0.1
    elif zero_frames > 10:
        feedback.append("⚠️ Person not visible in some frames. Try to stay in frame.")
        quality_score += 0.3
    else:
        feedback.append("✅ Person detected throughout the video.")
        quality_score += 0.4
    
    # 2. Check body height (full body visibility)
    shoulder_hip_dist = np.mean(np.abs(poses[:, 5, 1] - poses[:, 11, 1]))  # Shoulder - Hip
    hip_ankle_dist = np.mean(np.abs(poses[:, 11, 1] - poses[:, 15, 1]))    # Hip - Ankle
    
    if shoulder_hip_dist < 0.05 or hip_ankle_dist < 0.05:
        feedback.append("⚠️ Full body not visible. Please step back for side view.")
        quality_score += 0.1
    else:
        feedback.append("✅ Full body visible.")
        quality_score += 0.3
    
    # 3. Check movement (walking)
    hip_center = np.mean(poses[:, 11:13, 0], axis=1)
    movement_range = np.max(hip_center) - np.min(hip_center)
    
    if movement_range < 0.05:
        feedback.append("⚠️ Limited movement detected. Please walk in a straight line.")
        quality_score += 0.1
    else:
        feedback.append("✅ Walking movement detected.")
        quality_score += 0.3
    
    # 4. Check posture (side view)
    shoulder_diff = np.mean(np.abs(poses[:, 5, 0] - poses[:, 6, 0]))
    hip_diff = np.mean(np.abs(poses[:, 11, 0] - poses[:, 12, 0]))
    
    if shoulder_diff < 0.05 or hip_diff < 0.05:
        feedback.append("⚠️ Side view not detected. Please turn sideways.")
        quality_score += 0.1
    else:
        feedback.append("✅ Side view detected.")
        quality_score += 0.3
    
    # 5. Check arm swing
    arm_swing = np.mean(np.abs(poses[:, 9, 0] - poses[:, 5, 0]))  # Wrist - Shoulder
    if arm_swing < 0.02:
        feedback.append("⚠️ Limited arm swing detected. Walk naturally with arms swinging.")
        quality_score += 0.1
    else:
        feedback.append("✅ Arm swing detected.")
        quality_score += 0.3
    
    # Calculate final quality score (0-1)
    quality_score = min(max(quality_score / 1.7, 0), 1)  # Normalize to 0-1
    
    # Determine if video is usable
    is_valid = quality_score >= 0.4
    
    # Add overall status
    if is_valid:
        feedback.insert(0, "✅ Video quality is acceptable for analysis.")
    else:
        feedback.insert(0, "❌ Video quality is poor. Please re-record following the guidelines.")
    
    return is_valid, round(quality_score, 2), feedback


@api_view(['POST'])
@parser_classes([MultiPartParser])
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
        
        # Save to database (pose sequence stored)
        try:
            # Convert pose sequence to list for JSON storage
            pose_sequence_list = pose_sequence[0].tolist()  # (50, 34)
            
            GaitResult.objects.create(
                screening_id=screening_id,
                prediction=result['prediction'],
                pd_probability=float(result['pd_probability']),
                control_probability=float(result['control_probability']),
                model_used='LSTM',
                pose_sequence=pose_sequence_list,
                video_fps=30,  # Default, can be extracted from video
                pose_quality_score=quality_score
            )
            logger.info(f"Saved gait result for screening: {screening_id}")
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
    """Render the gait screening page"""
    return render(request, 'gait_app/gait_screening.html')


def gait_results_page(request):
    """Render the gait results page"""
    return render(request, 'gait_app/gait_results.html')