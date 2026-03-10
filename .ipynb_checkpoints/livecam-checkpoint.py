import argparse
import collections
import os
import pickle
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import numpy as np

# get arguments for model and camera
parser = argparse.ArgumentParser()
parser.add_argument("--model",   default="xgb_model.pkl")
parser.add_argument("--le",      default="xgb_le.pkl")
parser.add_argument("--scaler",  default="xgb_scaler.pkl")
parser.add_argument("--camera",  type=int, default=0)
args = parser.parse_args()

# Download MediaPipe pose model if needed for keypoints
MODEL_PATH = "pose_landmarker_full.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading pose model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        MODEL_PATH,
    )
    print("Downloaded!")

# declare features -- match training features
FEATURE_COLS = [
    "left_knee_angle",      "right_knee_angle",
    "left_elbow_angle",     "right_elbow_angle",
    "left_hip_angle",       "right_hip_angle",
    "left_body_line",       "right_body_line",
    "left_shoulder_angle",  "right_shoulder_angle",
    "torso_angle",
    "left_knee_bend",       "right_knee_bend",
    "left_shoulder_shrug",  "right_shoulder_shrug",
    "head_drop",
]

# compute angles for the features (match training)
def compute_angle(a, b, c):
    # angle at point b formed by a-b-c (degrees)
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_a   = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def compute_torso_angle(kp):
    sx = (kp["left_shoulder_x"]  + kp["right_shoulder_x"]) / 2
    sy = (kp["left_shoulder_y"]  + kp["right_shoulder_y"]) / 2
    hx = (kp["left_hip_x"]       + kp["right_hip_x"])      / 2
    hy = (kp["left_hip_y"]       + kp["right_hip_y"])      / 2
    dx, dy = sx - hx, sy - hy
    return np.degrees(np.arctan2(abs(dx), abs(dy)))


def kp_xy(kp, name):
    return (kp[f"{name}_x"], kp[f"{name}_y"])


def angle3(kp, a, b, c):
    return compute_angle(kp_xy(kp, a), kp_xy(kp, b), kp_xy(kp, c))


def extract_features(kp: dict) -> np.ndarray:
    """
    Given a flat dict of {landmark_x, landmark_y, ...},
    return a 1-D numpy array matching FEATURE_COLS order.
    """
    feats = {}
    for side in ["left", "right"]:
        feats[f"{side}_knee_angle"]     = angle3(kp, f"{side}_hip",      f"{side}_knee",     f"{side}_ankle")
        feats[f"{side}_elbow_angle"]    = angle3(kp, f"{side}_shoulder",  f"{side}_elbow",    f"{side}_wrist")
        feats[f"{side}_hip_angle"]      = angle3(kp, f"{side}_shoulder",  f"{side}_hip",      f"{side}_knee")
        feats[f"{side}_body_line"]      = angle3(kp, f"{side}_shoulder",  f"{side}_hip",      f"{side}_ankle")
        feats[f"{side}_shoulder_angle"] = angle3(kp, f"{side}_elbow",     f"{side}_shoulder", f"{side}_hip")
        feats[f"{side}_knee_bend"]      = kp[f"{side}_knee_x"] - kp[f"{side}_ankle_x"]
        feats[f"{side}_shoulder_shrug"] = kp[f"{side}_ear_y"]  - kp[f"{side}_shoulder_y"]

    feats["torso_angle"] = compute_torso_angle(kp)
    feats["head_drop"]   = kp["nose_y"] - kp["left_shoulder_y"]

    return np.array([feats[col] for col in FEATURE_COLS], dtype=np.float32)


# MediaPipe landmark index → name mapping
KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Connections for skeleton drawing
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (27, 29), (27, 31),              # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),              # right leg
]

FACE_INDICES = set(range(11))  # skip drawing face keypoints

def landmarks_to_dict(landmarks) -> dict:
    kp = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        lm = landmarks[i]
        kp[f"{name}_x"] = lm.x
        kp[f"{name}_y"] = lm.y
    return kp


# generic feedback per exercise shown on bad form
GENERIC_FEEDBACK = {
    "squats":  "Check depth, knees, and torso position",
    "pushups": "Straighten back, keep hips level, bend elbows to 90 degrees",
    "lunges":  "Step far enough for 90 degree bends, keeping torso upright",
    "pullups": "Engage shoulders and avoid swinging",
}

def get_feedback(exercise) -> str:
    return GENERIC_FEEDBACK.get(exercise, "Check your form")


# Load model, label encoder, and scaler
with open(args.le,     "rb") as f: le     = pickle.load(f)
with open(args.scaler, "rb") as f: scaler = pickle.load(f)
with open(args.model,  "rb") as f: model  = pickle.load(f)

print(f"Loaded model  |  classes: {list(le.classes_)}")

# smoothing buffer
pred_history = collections.deque(maxlen=5)

# color coding, good is green, bad is red, info is gray
GOOD_COLOR = (0, 220, 0)
BAD_COLOR  = (0, 0, 220)
INFO_COLOR = (200, 200, 200)


def draw_skeleton(frame, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)
    for i, (x, y) in enumerate(pts):
        if i not in FACE_INDICES:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)


def overlay_prediction(frame, label, confidence, fps, feedback=""):
    h, w     = frame.shape[:2]
    parts    = label.rsplit("_", 1)
    exercise = parts[0].replace("_", " ").title() if len(parts) == 2 else label
    form     = parts[1].upper()                   if len(parts) == 2 else ""
    color    = GOOD_COLOR if form == "GOOD" else BAD_COLOR

    banner_h = 90 if feedback else 70
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, exercise,
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INFO_COLOR, 2)
    cv2.putText(frame, f"{form}  {confidence*100:.0f}%",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if feedback:
        cv2.putText(frame, feedback,
                    (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BAD_COLOR, 1)

    cv2.putText(frame, f"{fps:.1f} fps",
                (w - 100, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INFO_COLOR, 1)


# Main loop
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
)

cap    = cv2.VideoCapture(args.camera)
t_prev = time.time()

current_label      = "Waiting for pose..."
current_confidence = 0.0
current_feedback   = ""

print("Starting webcam — press Q to quit")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w   = frame.shape[:2]
        t_now  = time.time()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # Pose detection
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = landmarker.detect(mp_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            draw_skeleton(frame, landmarks, h, w)

            kp   = landmarks_to_dict(landmarks)
            feat = extract_features(kp)

            # scale and predict on single frames
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            probs       = model.predict_proba(feat_scaled)[0]
            pred_idx    = int(probs.argmax())
            pred_label  = le.inverse_transform([pred_idx])[0]
            pred_conf   = float(probs[pred_idx])

            # smooth over last 5 predictions to reduce flickering
            pred_history.append(pred_label)
            smoothed = collections.Counter(pred_history).most_common(1)[0][0]

            current_label      = smoothed
            current_confidence = pred_conf

            parts    = smoothed.rsplit("_", 1)
            exercise = parts[0] if len(parts) == 2 else ""
            form     = parts[1] if len(parts) == 2 else ""
            current_feedback = get_feedback(exercise) if form == "bad" else ""
        else:
            current_label      = "No pose detected"
            current_confidence = 0.0
            current_feedback   = ""

        overlay_prediction(frame, current_label, current_confidence, fps, current_feedback)
        cv2.imshow("Exercise Form Classifier — XGBoost", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()