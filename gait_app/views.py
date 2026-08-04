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

def get_abnormal_features(features):
    """
    Identify which gait features are outside normal range
    Returns: List of abnormal features with their values and descriptions
    """
    
    # Normal ranges for all 15 features
    normal_ranges = {
        'symmetry_index': {
            'min': -20, 'max': 20,
            'description': 'Gait symmetry should be close to 0%'
        },
        'step_time_L_std': {
            'min': 0, 'max': 0.8,
            'description': 'Left step timing should be consistent'
        },
        'step_time_R_std': {
            'min': 0, 'max': 0.8,
            'description': 'Right step timing should be consistent'
        },
        'step_time_L_CV': {
            'min': 0, 'max': 0.5,
            'description': 'Left step variability should be low'
        },
        'step_time_R_CV': {
            'min': 0, 'max': 0.5,
            'description': 'Right step variability should be low'
        },
        'step_asymmetry': {
            'min': 0, 'max': 0.2,
            'description': 'Left and right steps should be similar'
        },
        'cadence': {
            'min': 60, 'max': 130,
            'description': 'Normal walking speed is 60-130 steps/min'
        },
        'step_regularity': {
            'min': 0.5, 'max': 1.0,
            'description': 'Step pattern should be regular'
        },
        'step_ratio': {
            'min': 0.8, 'max': 1.2,
            'description': 'Left/right step time ratio should be balanced'
        },
        'step_diff': {
            'min': -0.2, 'max': 0.2,
            'description': 'Left-right step time difference should be small'
        },
        'total_std': {
            'min': 0, 'max': 1.0,
            'description': 'Combined step variability should be low'
        },
        'std_ratio': {
            'min': 0.5, 'max': 2.0,
            'description': 'Left/right variability ratio should be balanced'
        },
        'cadence_reg': {
            'min': 30, 'max': 100,
            'description': 'Cadence-regularity product should be normal'
        }
    }

    # Friendly names for display
    friendly_names = {
        'symmetry_index': 'Gait Symmetry',
        'step_time_L_std': 'Left Step Variability',
        'step_time_R_std': 'Right Step Variability',
        'step_time_L_CV': 'Left Step Consistency',
        'step_time_R_CV': 'Right Step Consistency',
        'step_asymmetry': 'Step Asymmetry',
        'cadence': 'Walking Speed',
        'step_regularity': 'Step Regularity',
        'step_ratio': 'Step Time Ratio',
        'step_diff': 'Step Time Difference',
        'total_std': 'Total Variability',
        'std_ratio': 'Variability Ratio',
        'cadence_reg': 'Cadence Regularity'
    }

    abnormal = []

    for feature, value in features.items():
        if feature in normal_ranges and feature in friendly_names:
            rule = normal_ranges[feature]
            is_abnormal = False
            severity = 'Normal'

            if 'min' in rule and 'max' in rule:
                if value < rule['min'] or value > rule['max']:
                    is_abnormal = True
                    if value < rule['min'] * 0.5 or value > rule['max'] * 1.5:
                        severity = 'High'
                    else:
                        severity = 'Medium'
            elif 'max' in rule and value > rule['max']:
                is_abnormal = True
                if value > rule['max'] * 1.5:
                    severity = 'High'
                else:
                    severity = 'Medium'
            elif 'min' in rule and value < rule['min']:
                is_abnormal = True
                if value < rule['min'] * 0.5:
                    severity = 'High'
                else:
                    severity = 'Medium'

            if is_abnormal:
                abnormal.append({
                    'feature': feature,
                    'name': friendly_names.get(feature, feature),
                    'value': round(value, 4),
                    'normal_range': f"{rule.get('min', 0)} - {rule.get('max', '∞')}",
                    'severity': severity,
                    'description': rule.get('description', 'Out of normal range')
                })

    return abnormal

@api_view(['POST'])
@parser_classes([MultiPartParser])
def gait_predict(request):
    try:
        if 'video' not in request.FILES:
            return Response({'error': 'No video file provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        video_file = request.FILES['video']
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in video_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Extract features (15 features with auto-scaling)
        feature_array, feature_dict = extract_features_from_video(tmp_path)
        
        # Predict
        result = predictor.predict(feature_array)
        
        # Get abnormal features (using the 15 features)
        abnormal_features = get_abnormal_features(feature_dict)
        
        os.unlink(tmp_path)
        
        response_data = {
            'prediction': result['prediction'],
            'pd_probability': result['pd_probability'],
            'control_probability': result['control_probability'],
            'features': feature_dict,
            'abnormal_features': abnormal_features,
            'scaling_applied': feature_dict.get('scaling_applied', False)
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except ValueError as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': f'Processing error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def gait_screening_page(request):
    return render(request, 'gait_app/gait_screening.html')

def gait_results_page(request):
    return render(request, 'gait_app/gait_results.html')