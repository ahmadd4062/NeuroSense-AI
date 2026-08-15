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