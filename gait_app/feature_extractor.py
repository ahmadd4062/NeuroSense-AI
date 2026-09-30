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

    Returns:
        (is_valid, quality_score, feedback_list, reason_code)

    reason_code is one of:
        "ok", "no_pose", "wrong_angle", "not_side_view",
        "no_person", "no_full_body", "no_movement", "low_quality"
    """
    feedback = []
    quality_score = 0.0

    if pose_sequence is None:
        return False, 0.0, ["❌ No pose data extracted"], "no_pose"

    try:
        poses = pose_sequence[0].reshape(50, 17, 2).astype(np.float32)
    except Exception:
        return False, 0.0, ["❌ Invalid pose data format"], "no_pose"

    # ------------------------------------------------------------------
    # NORMALIZE: pose_sequence is in PIXEL coordinates (x = 0..width,
    # y = 0..height). Convert to 0..1 ranges so all thresholds below
    # are resolution-independent.
    # ------------------------------------------------------------------
    # Ignore zero-padding frames when computing reference box
    nonzero_mask = ~np.all(poses.reshape(50, 34) == 0, axis=1)
    if nonzero_mask.sum() == 0:
        return False, 0.0, ["❌ No pose data extracted"], "no_pose"

    valid_poses = poses[nonzero_mask]
    x_min = valid_poses[:, :, 0].min()
    x_max = valid_poses[:, :, 0].max()
    y_min = valid_poses[:, :, 1].min()
    y_max = valid_poses[:, :, 1].max()

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    normalized = poses.copy()
    normalized[:, :, 0] = (poses[:, :, 0] - x_min) / x_range
    normalized[:, :, 1] = (poses[:, :, 1] - y_min) / y_range

    # Zero-padding frames should stay zero after normalization
    normalized[~nonzero_mask] = 0.0

    poses = normalized

    # ---- 1. Person presence ---------------------------------------------
    zero_frames = np.sum(~nonzero_mask)

    if zero_frames > 20:
        feedback.append("❌ No person detected in most frames. Please ensure full body is visible.")
        quality_score += 0.05
        if zero_frames > 35:
            return False, 0.05, feedback, "no_person"
    elif zero_frames > 10:
        feedback.append("⚠️ Person not visible in some frames. Try to stay in frame.")
        quality_score += 0.15
    else:
        feedback.append("✅ Person detected throughout the video.")
        quality_score += 0.25

    # ---- 2. Full body visibility ----------------------------------------
    shoulder_hip_dist = np.mean(np.abs(poses[:, 5, 1] - poses[:, 11, 1]))
    hip_ankle_dist    = np.mean(np.abs(poses[:, 11, 1] - poses[:, 15, 1]))

    if shoulder_hip_dist < 0.05 or hip_ankle_dist < 0.05:
        feedback.append("⚠️ Full body not visible. Please step back for side view.")
        quality_score += 0.05
    else:
        feedback.append("✅ Full body visible.")
        quality_score += 0.20

    # ---- 3. Walking movement --------------------------------------------
    hip_center = np.mean(poses[:, 11:13, 0], axis=1)
    movement_range = np.max(hip_center) - np.min(hip_center)

    if movement_range < 0.05:
        feedback.append("⚠️ Limited movement detected. Please walk in a straight line.")
        quality_score += 0.05
    else:
        feedback.append("✅ Walking movement detected.")
        quality_score += 0.20

    # ---- 4. SIDE-VIEW CHECK ---------------------------------------------
    # In a true side profile, left & right shoulders/hips overlap in X
    # (small horizontal spread). Front/back views spread them apart.
    # Uses MEDIAN to ignore outlier frames (e.g. one arm swinging wide).
    shoulder_x_spread = np.median(np.abs(poses[:, 5, 0] - poses[:, 6, 0]))
    hip_x_spread      = np.median(np.abs(poses[:, 11, 0] - poses[:, 12, 0]))

    # Debug logging (optional — remove after tuning)
    print(f"[GAIT-DEBUG] shoulder_x_spread={shoulder_x_spread:.3f}, "
          f"hip_x_spread={hip_x_spread:.3f}, "
          f"movement_range={movement_range:.3f}")

    # Thresholds tuned from real test data:
    #   side view  → shoulder_x_spread ~ 0.003–0.05, hip_x_spread ~ 0.02–0.05
    #   front/back → shoulder_x_spread > 0.25, hip_x_spread > 0.25
    is_side_view = (
        shoulder_x_spread < 0.15
        and hip_x_spread < 0.18
    )

    is_front_or_back_view = (
        shoulder_x_spread > 0.25
        or hip_x_spread > 0.25
    )

    if is_front_or_back_view:
        feedback.append(
            "❌ Front or back angle detected. Please record from a SIDE view "
            "(camera at waist height, pointed at your profile)."
        )
        return False, round(min(quality_score, 0.39), 2), feedback, "wrong_angle"

    if not is_side_view:
        feedback.append(
            "⚠️ Side view not clearly detected. Turn 90° so the camera sees "
            "your profile (one shoulder facing the camera)."
        )
    else:
        feedback.append("✅ Side view detected.")
        quality_score += 0.30

    # ---- 5. Arm swing ----------------------------------------------------
    arm_swing = np.median(np.abs(poses[:, 9, 0] - poses[:, 5, 0]))
    if arm_swing < 0.02:
        feedback.append("⚠️ Limited arm swing detected. Walk naturally with arms swinging.")
        quality_score += 0.05
    else:
        feedback.append("✅ Arm swing detected.")
        quality_score += 0.05

    # ---- Final score & validity ------------------------------------------
    quality_score = min(max(quality_score, 0), 1)
    is_valid = quality_score >= 0.40 and is_side_view

    if is_valid:
        feedback.insert(0, "✅ Video quality is acceptable for analysis.")
        reason_code = "ok"
    else:
        feedback.insert(0, "❌ Video quality is poor. Please re-record following the guidelines.")

        if not is_side_view and not is_front_or_back_view:
            reason_code = "not_side_view"
        elif zero_frames > 10:
            reason_code = "no_person"
        elif shoulder_hip_dist < 0.05 or hip_ankle_dist < 0.05:
            reason_code = "no_full_body"
        elif movement_range < 0.05:
            reason_code = "no_movement"
        else:
            reason_code = "low_quality"

    return is_valid, round(quality_score, 2), feedback, reason_code