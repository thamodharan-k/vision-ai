"""
object_detector.py
------------------
Uses YOLOv8x-oiv7 (600+ classes) as PRIMARY detector.
Custom model only fills in classes NOT detected by oiv7.
This prevents custom model from wrongly overriding correct detections
(e.g. cat being called fox due to similar orange color).
"""
 
import cv2
import numpy as np
import os
 
CONF_THRESHOLD = 0.30
FONT = cv2.FONT_HERSHEY_SIMPLEX
 
BASE = os.path.dirname(os.path.abspath(__file__))
 
CUSTOM_MODELS = [
    os.path.join(BASE, "runs", "detect", "animals_full",      "weights", "best.pt"),
    os.path.join(BASE, "runs", "detect", "animal_model",      "weights", "best.pt"),
    os.path.join(BASE, "runs", "detect", "animal_bird_model", "weights", "best.pt"),
]
 
# Classes that ONLY exist in custom model (not in OpenImages)
# Only use custom model results for these specific classes
CUSTOM_ONLY_CLASSES = {"fox", "goat", "chicken", "raccoon", "skunk", "cow", "horse"}
 
# Classes that OpenImages knows well — always trust oiv7 for these
OIV7_TRUSTED = {
    "cat", "dog", "bird", "lion", "tiger", "leopard", "elephant",
    "giraffe", "zebra", "bear", "person", "horse", "sheep", "cow",
    "eagle", "owl", "parrot", "flamingo", "peacock", "penguin",
    "crocodile", "snake", "dolphin", "whale", "shark", "rabbit",
    "deer", "wolf", "monkey", "gorilla", "hippopotamus", "rhinoceros",
}
 
def get_color(label):
    label_lower = label.lower()
    big_cats = {"lion","tiger","leopard","cheetah","jaguar","cat","lynx"}
    birds    = {"bird","eagle","owl","parrot","flamingo","peacock","penguin",
                "hawk","falcon","duck","swan","goose","chicken","turkey","ostrich"}
    marine   = {"dolphin","whale","shark","seal","fish","crab","jellyfish"}
    reptiles = {"crocodile","alligator","snake","lizard","turtle","tortoise"}
 
    if label_lower in big_cats:
        return (0, 140, 255)     # orange
    elif label_lower in birds:
        return (255, 100, 200)   # pink
    elif label_lower in marine:
        return (255, 200, 0)     # yellow
    elif label_lower in reptiles:
        return (0, 200, 80)      # green
    elif label_lower in {"fox","raccoon","skunk","wolf","coyote"}:
        return (0, 130, 255)     # orange-red
    elif label_lower == "person":
        return (0, 212, 255)     # cyan
    else:
        return (0, 230, 100)     # green for other animals
 
 
class ObjectDetector:
    def __init__(self, conf=CONF_THRESHOLD):
        from ultralytics import YOLO
 
        # Load custom model if available
        custom_path = None
        for p in CUSTOM_MODELS:
            if os.path.exists(p):
                custom_path = p
                break
 
        if custom_path:
            print(f"[ObjectDetector] Custom model: {custom_path}")
            self.custom_model = YOLO(custom_path)
        else:
            self.custom_model = None
            print("[ObjectDetector] No custom model found")
 
        # Load OpenImages V7 — primary detector (600+ classes)
        print("[ObjectDetector] Loading YOLOv8x-oiv7 (600+ classes)...")
        print("[ObjectDetector] First run downloads ~136MB — please wait...")
        self.oiv7_model = YOLO("yolov8x-oiv7.pt")
 
        self.conf = conf
        self._last_detections = []
        print("[ObjectDetector] Ready! Detects ALL animals and birds correctly.")
 
    def process(self, frame):
        annotated  = frame.copy()
        top_label  = "---"
        top_conf   = 0.0
        best_conf  = -1.0
        detections = []
 
        # ── Step 1: Run OpenImages model (primary, most accurate) ─────────
        oiv7_labels = set()
        results = self.oiv7_model(
            frame, stream=True, conf=self.conf, verbose=False)
 
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf_val = float(box.conf[0])
                cls_id   = int(box.cls[0])
                cls_name = result.names[cls_id]
 
                oiv7_labels.add(cls_name.lower())
                detections.append((cls_name, conf_val))
 
                if conf_val > best_conf:
                    best_conf = conf_val
                    top_label = cls_name
                    top_conf  = conf_val
 
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                self._draw_box(annotated, x1, y1, x2, y2, cls_name, conf_val)
 
        # ── Step 2: Run custom model ONLY for its unique classes ──────────
        # Skip if oiv7 already found something at the same location
        if self.custom_model is not None:
            results2 = self.custom_model(
                frame, stream=True, conf=self.conf, verbose=False)
 
            for result in results2:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf_val = float(box.conf[0])
                    cls_id   = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    cls_lower = cls_name.lower()
 
                    # ONLY add custom result if:
                    # 1. It's a class unique to custom model (fox, goat etc.)
                    # 2. AND oiv7 didn't already detect it
                    # This prevents fox overriding cat, etc.
                    if (cls_lower in CUSTOM_ONLY_CLASSES and
                            cls_lower not in oiv7_labels):
                        detections.append((cls_name, conf_val))
                        if conf_val > best_conf:
                            best_conf = conf_val
                            top_label = cls_name
                            top_conf  = conf_val
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        self._draw_box(
                            annotated, x1, y1, x2, y2, cls_name, conf_val)
 
        self._last_detections = detections
        return annotated, top_label, top_conf
 
    def _draw_box(self, frame, x1, y1, x2, y2, label, conf):
        color = get_color(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
        pad = 5
        label_top = max(y1 - th - 2 * pad, 0)
        cv2.rectangle(frame, (x1, label_top),
                      (x1 + tw + 2 * pad, y1), color, cv2.FILLED)
        cv2.putText(frame, text, (x1 + pad, y1 - pad),
                    FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)