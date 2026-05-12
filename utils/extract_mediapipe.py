import cv2
import mediapipe as mp
import numpy as np

def extract_mediapipe(video_path, normalize_wrist=True):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(image)
        hands_res = hands.process(image)
        kp = []
        
        if pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks.landmark:
                kp.append([lm.x, lm.y, lm.z])
        else:
            kp.extend([[0, 0, 0]] * 33)
            
        if hands_res.multi_hand_landmarks:
            for hand_lm in hands_res.multi_hand_landmarks:
                if normalize_wrist:
                    wx, wy, wz = hand_lm.landmark[0].x, hand_lm.landmark[0].y, hand_lm.landmark[0].z
                    for lm in hand_lm.landmark:
                        kp.append([lm.x - wx, lm.y - wy, lm.z - wz])
                else:
                    for lm in hand_lm.landmark:
                        kp.append([lm.x, lm.y, lm.z])
        else:
            kp.extend([[0, 0, 0]] * 21 * 2)
            
        frames_data.append(np.array(kp).flatten())
        
    cap.release()
    return np.array(frames_data)
