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
    if angle > 180.0:
        angle = 360 - angle
    return angle

def moving_average(x, w=5):
    """Simple edge-handled moving average."""
    if len(x) < 3:
        return np.array(x)
    w = max(1, int(w))
    return np.convolve(x, np.ones(w)/w, mode='same')

def merge_peaks(peaks_l, peaks_r, min_separation=5):
    """
    Combine peaks from left & right ankles and avoid double counting.
    peaks_l/peaks_r are arrays of indices (relative to buffer).
    min_separation = minimum frames between two distinct steps (merge if closer).
    Returns sorted unique merged peak indices.
    """
    if len(peaks_l) == 0 and len(peaks_r) == 0:
        return np.array([], dtype=int)
    combined = np.concatenate((peaks_l, peaks_r)).astype(int)
    combined = np.unique(np.sort(combined))
    merged = []
    last = -999
    for p in combined:
        if p - last <= min_separation:
            # treat as same step (skip)
            continue
        merged.append(p)
        last = p
    return np.array(merged, dtype=int)

def count_steps(left_y, right_y, fps, smooth_w=5, distance_sec=0.35, prominence=10):
    """
    Count steps using both ankles.
    left_y/right_y: lists (buffers) of ankle Y pixel positions (length = buffer length)
    fps: video FPS (for converting seconds to frames)
    Returns: merged_count (int), merged_peaks (np.array of indices relative to buffer)
    """
    if len(left_y) < 3 and len(right_y) < 3:
        return 0, np.array([], dtype=int)

    # smoothing
    left_s = moving_average(left_y, w=smooth_w) if len(left_y) >= 1 else np.array([])
    right_s = moving_average(right_y, w=smooth_w) if len(right_y) >= 1 else np.array([])

    # invert Y so foot contact (if resulting in troughs) becomes peaks — this is robust across setups
    # if your camera produces peaks instead of troughs, remove the negative sign.
    left_sig = -np.array(left_s)
    right_sig = -np.array(right_s)

    distance = max(1, int(distance_sec * fps))
    # find peaks
    try:
        peaks_l, _ = find_peaks(left_sig, distance=distance, prominence=prominence)
    except Exception:
        peaks_l = np.array([], dtype=int)
    try:
        peaks_r, _ = find_peaks(right_sig, distance=distance, prominence=prominence)
    except Exception:
        peaks_r = np.array([], dtype=int)

    # merge left & right peaks
    min_sep_frames = max(1, int(0.12 * fps))  # default ~0.12s between two different foot strikes
    merged = merge_peaks(peaks_l, peaks_r, min_separation=min_sep_frames)

    return len(merged), merged

def calculate_speed_from_peaks(merged_peaks, fps, stride_factor=1.2):
    """
    Estimate speed from merged peak indices (relative to buffer).
    stride_factor: meters per step (coarse). Tweak to match subject height/stride.
    Returns float speed in m/s.
    """
    if len(merged_peaks) < 2:
        return 0.0
    intervals_frames = np.diff(merged_peaks)
    intervals_sec = intervals_frames / fps
    # ignore zero intervals
    intervals_sec = intervals_sec[intervals_sec > 1e-6]
    if len(intervals_sec) == 0:
        return 0.0
    step_freq = 1.0 / np.mean(intervals_sec)  # steps per second
    speed = step_freq * stride_factor
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

# tuning parameters (easy to change)
SMOOTH_W = 5
PEAK_DISTANCE_SEC = 0.35   # minimal time between steps (seconds)
PEAK_PROMINENCE = 10
STRIDE_FACTOR = 1.2        # meters per step (coarse); change to better match subject's stride

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    speed = 0.0  # default if no detection

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

        # limit buffers (keep same as you had)
        if len(left_ankle_y) > 100:
            left_ankle_y.pop(0)
            right_ankle_y.pop(0)
            shoulder_y.pop(0)

        # ---- Improved Step Count (uses both ankles) ----
        steps, peaks = count_steps(left_ankle_y, right_ankle_y, fps,
                                   smooth_w=SMOOTH_W,
                                   distance_sec=PEAK_DISTANCE_SEC,
                                   prominence=PEAK_PROMINENCE)
        # peaks are indices relative to the buffer (0..len-1)
        if steps > last_count:
            total_steps += (steps - last_count)
        last_count = steps

        # ---- Improved Speed Estimation ----
        speed = calculate_speed_from_peaks(peaks, fps=fps, stride_factor=STRIDE_FACTOR)

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
        sh_range = max(shoulder_y) - min(shoulder_y) if shoulder_y else 0
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
        if pushup_count > 0 or (sh_range > 60 and speed < 0.5):
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
    
    
    cv2.putText(image, f"Steps: {total_steps}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.putText(image, f"Posture: {posture}", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(image, f"Pushups: {pushup_count}", (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Activity + Speed + Posture + Pushups", image)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()