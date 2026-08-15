import cv2
import mediapipe as mp
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from django.conf import settings


def extract_pose_sequence_from_video(video_path, sequence_length=50):
    """
    Extract pose sequence (50 frames × 17 joints × 2 coordinates) from video
    """
    # Model path in gait_app/models/
    model_dir = os.path.join(settings.BASE_DIR, 'gait_app', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'pose_landmarker_lite.task')
    
    if not os.path.exists(model_path):
        print("Downloading pose landmarker model...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        urllib.request.urlretrieve(url, model_path)
        print("Download complete! ✅")
    
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
    
    pose_sequence = []
    frame_count = 0
    body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect_for_video(mp_image, int(frame_count * 1000 / fps))
        
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            h, w, _ = frame.shape
            frame_pose = []
            
            for idx in body_indices:
                if idx < len(landmarks):
                    frame_pose.append([landmarks[idx].x * w, landmarks[idx].y * h])
                else:
                    frame_pose.append([0.0, 0.0])
            
            pose_sequence.append(np.array(frame_pose))
        
        frame_count += 1
        if frame_count > 500:
            break
    
    cap.release()
    detector.close()
    
    if len(pose_sequence) < 30:
        return None
    
    pose_sequence = np.array(pose_sequence)
    
    if len(pose_sequence) > sequence_length:
        indices = np.linspace(0, len(pose_sequence)-1, sequence_length, dtype=int)
        pose_sequence = pose_sequence[indices]
    else:
        pad_length = sequence_length - len(pose_sequence)
        pose_sequence = np.vstack([pose_sequence, np.zeros((pad_length, 17, 2))])
    
    return pose_sequence.reshape(1, sequence_length, -1)


def validate_video_quality(pose_sequence):
    """
    Validate the quality of extracted pose sequence.
    Returns: (is_valid, quality_score, feedback_list)
    
    This is called from views.py before prediction.
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
    zero_frames = np.sum(np.all(poses.reshape(50, 34) == 0, axis=1))
    
    if zero_frames > 20:  # More than 40% frames are empty
        feedback.append("❌ No person detected in most frames. Please ensure full body is visible.")
        quality_score += 0.05
    elif zero_frames > 10:
        feedback.append("⚠️ Person not visible in some frames. Try to stay in frame.")
        quality_score += 0.15
    else:
        feedback.append("✅ Person detected throughout the video.")
        quality_score += 0.30
    
    # 2. Check body height (full body visibility)
    shoulder_hip_dist = np.mean(np.abs(poses[:, 5, 1] - poses[:, 11, 1]))  # Shoulder - Hip
    hip_ankle_dist = np.mean(np.abs(poses[:, 11, 1] - poses[:, 15, 1]))    # Hip - Ankle
    
    if shoulder_hip_dist < 0.05 or hip_ankle_dist < 0.05:
        feedback.append("⚠️ Full body not visible. Please step back for side view.")
        quality_score += 0.05
    else:
        feedback.append("✅ Full body visible.")
        quality_score += 0.25
    
    # 3. Check movement (walking)
    hip_center = np.mean(poses[:, 11:13, 0], axis=1)
    movement_range = np.max(hip_center) - np.min(hip_center)
    
    if movement_range < 0.05:
        feedback.append("⚠️ Limited movement detected. Please walk in a straight line.")
        quality_score += 0.05
    else:
        feedback.append("✅ Walking movement detected.")
        quality_score += 0.25
    
    # 4. Check posture (side view)
    shoulder_diff = np.mean(np.abs(poses[:, 5, 0] - poses[:, 6, 0]))
    hip_diff = np.mean(np.abs(poses[:, 11, 0] - poses[:, 12, 0]))
    
    if shoulder_diff < 0.05 or hip_diff < 0.05:
        feedback.append("⚠️ Side view not detected. Please turn sideways.")
        quality_score += 0.05
    else:
        feedback.append("✅ Side view detected.")
        quality_score += 0.25
    
    # 5. Check arm swing
    arm_swing = np.mean(np.abs(poses[:, 9, 0] - poses[:, 5, 0]))  # Wrist - Shoulder
    if arm_swing < 0.02:
        feedback.append("⚠️ Limited arm swing detected. Walk naturally with arms swinging.")
        quality_score += 0.05
    else:
        feedback.append("✅ Arm swing detected.")
        quality_score += 0.20
    
    # Calculate final quality score (0-1)
    quality_score = min(max(quality_score, 0), 1)
    
    # Determine if video is usable
    is_valid = quality_score >= 0.40
    
    # Add overall status
    if is_valid:
        feedback.insert(0, "✅ Video quality is acceptable for analysis.")
    else:
        feedback.insert(0, "❌ Video quality is poor. Please re-record following the guidelines.")
    
    return is_valid, round(quality_score, 2), feedback