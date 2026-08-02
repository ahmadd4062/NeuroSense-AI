from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FileUploadParser, MultiPartParser
from rest_framework.response import Response
from rest_framework import status
import os
import tempfile


from .gait_model import predictor
from .feature_extractor import extract_features_from_video

@api_view(['POST'])
@parser_classes([MultiPartParser])
def gait_predict(request):
    """
    Predict Parkinson's risk from walking video
    Expected: video file in request.FILES with key 'video'
    Returns: prediction result with probabilities and features
    """
    try:
        # Check if video file is in request
        if 'video' not in request.FILES:
            return Response(
                {'error': 'No video file provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        video_file = request.FILES['video']
        
        # Get filename from the uploaded file
        filename = video_file.name
        
        # Save video temporarily with original filename
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in video_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Extract features from video
        feature_array, feature_dict = extract_features_from_video(tmp_path)
        
        # Make prediction
        result = predictor.predict(feature_array)
        
        # Add feature importance
        result['features'] = feature_dict
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return Response(result, status=status.HTTP_200_OK)
        
    except ValueError as e:
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return Response(
            {'error': f'Processing error: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


def gait_screening_page(request):
    """Gait video upload page"""
    return render(request, 'gait_app/gait_screening.html')

def gait_results_page(request):
    """Gait results page"""
    return render(request, 'gait_app/gait_results.html')