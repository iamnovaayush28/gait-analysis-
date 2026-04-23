import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---- Utility Functions ----
def calculate_angle(a, b, c):
    """Calculates the joint angle between three points (a-b-c)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def count_steps(y_vals, distance=10, prominence=10):
    peaks, _ = find_peaks(-np.array(y_vals), distance=distance, prominence=prominence)
    return len(peaks), peaks

def calculate_speed(peaks, fps=30):
    """Estimates walking/running speed using step frequency."""
    if len(peaks) < 2: return 0
    intervals = np.diff(peaks) / fps
    step_freq = 1 / np.mean(intervals)
    speed = step_freq * 1.2  # pseudo stride length factor
    return round(speed, 2)

# ---- Video Input ----
video_path = r"gait analysis.py\Arka.mp4"  # Change path
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

left_ankle_y, right_ankle_y, shoulder_y = [], [], []
total_steps, last_count = 0, 0
pushup_count, pushup_dir = 0, 0  # 0 = down, 1 = up
activity, posture = "Unknown", "Good"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        # Key joints
        l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
        r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Y positions
        la_y, ra_y = int(l_ankle.y * h), int(r_ankle.y * h)
        sh_y = int((l_shoulder.y + r_shoulder.y) / 2 * h)
        left_ankle_y.append(la_y)
        right_ankle_y.append(ra_y)
        shoulder_y.append(sh_y)

        if len(left_ankle_y) > 100:
            left_ankle_y.pop(0)
            right_ankle_y.pop(0)
            shoulder_y.pop(0)

        # ---- Step Count (Walking / Running) ----
        steps, peaks = count_steps(left_ankle_y)
        if steps > last_count:
            total_steps += (steps - last_count)
        last_count = steps

        # ---- Speed Estimation ----
        speed = calculate_speed(peaks, fps=fps)

        # ---- Posture Evaluation (Both Legs) ----
        left_back_angle = calculate_angle(
            [l_shoulder.x * w, l_shoulder.y * h],
            [l_hip.x * w, l_hip.y * h],
            [l_knee.x * w, l_knee.y * h]
        )
        right_back_angle = calculate_angle(
            [r_shoulder.x * w, r_shoulder.y * h],
            [r_hip.x * w, r_hip.y * h],
            [r_knee.x * w, r_knee.y * h]
        )
        posture_diff = abs(left_back_angle - right_back_angle)

        if posture_diff > 25:
            posture = "Poor (Asymmetrical)"
        elif posture_diff > 15:
            posture = "Average"
        else:
            posture = "Good"

        # ---- Pushup Detection & Counting ----
        sh_range = max(shoulder_y) - min(shoulder_y)
        # Track nose or shoulder Y-movement for pushup
        nose = lm[mp_pose.PoseLandmark.NOSE]
        nose_y = nose.y * h

        if len(shoulder_y) > 10:
            avg_shoulder = np.mean(shoulder_y[-10:])
            if nose_y > avg_shoulder + 25:  # Down phase
                pushup_dir = 0
            elif nose_y < avg_shoulder - 25 and pushup_dir == 0:  # Up phase
                pushup_count += 1
                pushup_dir = 1

        # ---- Activity Detection ----
        if pushup_count > 0 or sh_range > 60 and speed < 0.5:
            activity = "Pushups"
        elif speed >= 3:
            activity = "Running"
        elif 0.5 < speed < 3:
            activity = "Walking"
        elif speed <= 0.3:
            activity = "Standing"
        else:
            activity = "Unknown"

    # ---- Display ----
    cv2.putText(image, f"Activity: {activity}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(image, f"Speed: {speed:.2f}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
    cv2.putText(image, f"Steps: {total_steps}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(image, f"Posture: {posture}", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(image, f"Pushups: {pushup_count}", (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Activity + Speed + Posture + Pushups", image)
    if cv2.waitKey(10) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()