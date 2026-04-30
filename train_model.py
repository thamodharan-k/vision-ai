"""
train_model.py
--------------
Fine-tunes YOLOv8m on your custom animal dataset.
Classes: dog, elephant, giraffe, cat, horse

Run with:  python train_model.py
Trained model saved to: runs/detect/animal_model/weights/best.pt
"""

import os
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML  = os.path.join(os.path.dirname(__file__), "animal_dataset", "data.yaml")
BASE_MODEL    = "yolov8m.pt"   # pretrained weights (downloaded auto if missing)
OUTPUT_NAME   = "animal_model"
EPOCHS        = 50             # increase to 100 for better accuracy
IMAGE_SIZE    = 640
BATCH_SIZE    = 8              # lower to 4 if you get memory errors

# ── Train ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  YOLOv8m Animal Detection — Custom Training")
print("=" * 55)
print(f"  Dataset : {DATASET_YAML}")
print(f"  Epochs  : {EPOCHS}")
print(f"  Batch   : {BATCH_SIZE}")
print("=" * 55)

model = YOLO(BASE_MODEL)

results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    name=OUTPUT_NAME,
    patience=15,          # stop early if no improvement for 15 epochs
    augment=True,         # data augmentation for better generalisation
    mosaic=1.0,           # mosaic augmentation
    degrees=10,           # random rotation
    flipud=0.3,           # vertical flip
    fliplr=0.5,           # horizontal flip
    hsv_h=0.015,          # hue variation
    hsv_s=0.7,            # saturation variation
    verbose=True,
)

print("\n✅ Training complete!")
print(f"📁 Best model saved to: runs/detect/{OUTPUT_NAME}/weights/best.pt")
print("\nNow update object_detector.py to use your trained model.")
print('Change: self.model = YOLO("runs/detect/animal_model/weights/best.pt")')
