"""
finger_counter.py
-----------------
Finger counting using MediaPipe Hands Task API with auto model download.
Downloads hand_landmarker.task model file on first run if not present.
"""
 
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
 
# Hand skeleton connections (hardcoded — 21 landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),(5,17),
]
 
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_PIPS  = [3, 6, 10, 14, 18]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
 
LANDMARK_COLOR   = (0, 212, 255)
CONNECTION_COLOR = (200, 200, 200)
 
# Model file — downloaded automatically on first run (~8 MB)
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
 
 
def _ensure_model():
    """Download the hand landmarker .task model file if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("[FingerCounter] Downloading hand_landmarker.task (~8 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[FingerCounter] Model saved to:", MODEL_PATH)
 
 
class FingerCounter:
    def __init__(self):
        _ensure_model()   # ensure model file exists before creating detector
 
        base_options = mp_python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._ts = 0   # incrementing timestamp (ms)
 
    def process(self, frame: np.ndarray):
        """Returns: annotated_frame, count (0-5), list of raised finger names."""
        self._ts += 33  # ~30 fps
 
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._detector.detect_for_video(mp_image, self._ts)
 
        annotated  = frame.copy()
        count      = 0
        up_fingers = []
 
        if result.hand_landmarks:
            lm         = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name  # "Left" / "Right"
 
            self._draw_landmarks(annotated, lm)
 
            # Thumb: compare tip.x vs IP joint x (mirrored webcam feed)
            if handedness == "Right":
                thumb_up = lm[4].x < lm[3].x
            else:
                thumb_up = lm[4].x > lm[3].x
 
            if thumb_up:
                count += 1
                up_fingers.append("Thumb")
 
            # Index to Pinky: tip above PIP joint (y=0 at top of frame)
            for i in range(1, 5):
                if lm[FINGER_TIPS[i]].y < lm[FINGER_PIPS[i]].y:
                    count += 1
                    up_fingers.append(FINGER_NAMES[i])
 
        return annotated, count, up_fingers
 
    def release(self):
        self._detector.close()
 
    def _draw_landmarks(self, frame, landmarks):
        h, w = frame.shape[:2]
        for a, b in HAND_CONNECTIONS:
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), CONNECTION_COLOR, 2, cv2.LINE_AA)
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, LANDMARK_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)