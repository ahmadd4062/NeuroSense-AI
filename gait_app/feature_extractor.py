import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_features_from_video(video_path):
    """
    Extract 15 gait features from a walking video using MediaPipe Pose.
    Includes auto-scaling for high variability videos.
    Returns: numpy array of 15 features (in correct order for the model)
    """
    
    # Download the pose landmarker model if not present
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print("Downloading pose landmarker model...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        urllib.request.urlretrieve(url, model_path)
        print("Download complete!")
    
    # Create Pose Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO
    )
    
    detector = vision.PoseLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    
    left_ankle_y = []
    right_ankle_y = []
    frame_times = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect pose
        detection_result = detector.detect_for_video(mp_image, int(frame_count * 1000 / fps))
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            h, w, _ = frame.shape
            
            # Left ankle (index 27), Right ankle (index 28)
            l_ankle = landmarks[27]
            r_ankle = landmarks[28]
            
            left_ankle_y.append(l_ankle.y * h)
            right_ankle_y.append(r_ankle.y * h)
            frame_times.append(frame_count / fps)
        
        frame_count += 1
    
    cap.release()
    detector.close()
    
    if len(left_ankle_y) < 20:
        raise ValueError("Not enough frames with pose detected. Please check video quality.")
    
    # ============================================================
    # IMPROVED HEEL STRIKE DETECTION (Local Minima)
    # ============================================================
    def find_heel_strikes(ankle_y, threshold=5):
        """
        Find heel strikes by detecting local MINIMA in ankle height.
        A heel strike occurs when the foot touches the ground.
        """
        if len(ankle_y) < 20:
            return []
        
        strikes = []
        for i in range(3, len(ankle_y) - 3):
            is_min = True
            for j in range(1, 4):
                if ankle_y[i] > ankle_y[i - j] or ankle_y[i] > ankle_y[i + j]:
                    is_min = False
                    break
            if is_min and ankle_y[i] < np.mean(ankle_y) - 5:
                strikes.append(i)
        
        filtered = []
        for s in strikes:
            if not filtered or s - filtered[-1] > 10:
                filtered.append(s)
        
        return filtered
    
    left_strikes = find_heel_strikes(left_ankle_y)
    right_strikes = find_heel_strikes(right_ankle_y)
    
    # Mirror foot if one is not detected
    if len(right_strikes) < 3 and len(left_strikes) >= 3:
        print("⚠️ Right foot not detected. Using left foot data for both sides.")
        right_strikes = left_strikes.copy()
    elif len(left_strikes) < 3 and len(right_strikes) >= 3:
        print("⚠️ Left foot not detected. Using right foot data for both sides.")
        left_strikes = right_strikes.copy()
    
    left_step_times = []
    right_step_times = []
    
    for i in range(1, len(left_strikes)):
        left_step_times.append(frame_times[left_strikes[i]] - frame_times[left_strikes[i-1]])
    for i in range(1, len(right_strikes)):
        right_step_times.append(frame_times[right_strikes[i]] - frame_times[right_strikes[i-1]])
    
    if len(left_step_times) == 0:
        left_step_times = [0.5]
    if len(right_step_times) == 0:
        right_step_times = [0.5]
    
    # ============================================================
    # STEP REGULARITY
    # ============================================================
    all_step_times = left_step_times + right_step_times
    filtered_step_times = [t for t in all_step_times if 0.2 < t < 3.0]
    
    if len(filtered_step_times) > 1:
        step_regularity = 1 - (np.std(filtered_step_times) / (np.mean(filtered_step_times) + 0.001))
        step_regularity = max(0, min(1, step_regularity))
    else:
        step_regularity = 0.7
    
    # ============================================================
    # CALCULATE ALL FEATURES
    # ============================================================
    features = {
        'step_time_L_mean': np.mean(left_step_times),
        'step_time_L_std': np.std(left_step_times) if len(left_step_times) > 1 else 0,
        'step_time_R_mean': np.mean(right_step_times),
        'step_time_R_std': np.std(right_step_times) if len(right_step_times) > 1 else 0,
        'step_time_L_CV': np.std(left_step_times) / (np.mean(left_step_times) + 0.001) if len(left_step_times) > 1 else 0,
        'step_time_R_CV': np.std(right_step_times) / (np.mean(right_step_times) + 0.001) if len(right_step_times) > 1 else 0,
        'step_asymmetry': abs(np.mean(left_step_times) - np.mean(right_step_times)) / (max(np.mean(left_step_times), np.mean(right_step_times)) + 0.001),
        'symmetry_index': (np.mean(right_step_times) - np.mean(left_step_times)) / (np.mean(left_step_times) + np.mean(right_step_times) + 0.001) * 200,
        'cadence': 60 / np.mean(left_step_times) if np.mean(left_step_times) > 0 else 0,
        'step_regularity': step_regularity
    }
    
    # ============================================================
    # AUTO-SCALING LOGIC (for high variability videos)
    # ============================================================
    std = features['step_time_L_std']
    regularity = features['step_regularity']
    
    if std > 0.4 and regularity < 0.5:
        print(f"  🔵 Auto-scaling applied (std={std:.3f}, reg={regularity:.3f})")
        factor = 0.5
        features['step_time_L_std'] = features['step_time_L_std'] * factor
        features['step_time_R_std'] = features['step_time_R_std'] * factor
        features['step_time_L_CV'] = features['step_time_L_CV'] * factor
        features['step_time_R_CV'] = features['step_time_R_CV'] * factor
        features['scaling_applied'] = True
    else:
        features['scaling_applied'] = False
    
    # ============================================================
    # ENGINEERED FEATURES
    # ============================================================
    features['step_ratio'] = features['step_time_L_mean'] / (features['step_time_R_mean'] + 0.001)
    features['step_diff'] = features['step_time_L_mean'] - features['step_time_R_mean']
    features['total_std'] = features['step_time_L_std'] + features['step_time_R_std']
    features['std_ratio'] = features['step_time_L_std'] / (features['step_time_R_std'] + 0.001)
    features['cadence_reg'] = features['cadence'] * features['step_regularity']
    
    # ============================================================
    # FINAL FEATURE ARRAY (15 features in correct order)
    # ============================================================
    feature_names = ['step_time_L_mean', 'step_time_L_std', 'step_time_R_mean', 'step_time_R_std',
                     'step_time_L_CV', 'step_time_R_CV', 'step_asymmetry', 'symmetry_index',
                     'cadence', 'step_regularity', 'step_ratio', 'step_diff', 
                     'total_std', 'std_ratio', 'cadence_reg']
    
    feature_array = np.array([features[name] for name in feature_names])
    
    return feature_array, features