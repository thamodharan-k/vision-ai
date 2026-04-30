"""
setup_and_train.py
------------------
Optimized for CPU training — completes in 1-2 hours instead of 60+ hours.
"""
 
import os, sys, shutil, yaml, glob
from pathlib import Path
 
# ── YOUR ROBOFLOW PRIVATE API KEY ─────────────────────────────────────────────
API_KEY = "tJxir2OYm7xzqmVG5wq3"
# ─────────────────────────────────────────────────────────────────────────────
 
try:
    from roboflow import Roboflow
except ImportError:
    os.system("pip install roboflow")
    from roboflow import Roboflow
 
from ultralytics import YOLO
 
BASE = os.path.dirname(os.path.abspath(__file__))
 
print("=" * 55)
print("  Vision AI — Animal Dataset Training (CPU Optimized)")
print("=" * 55)
 
rf = Roboflow(api_key=API_KEY)
 
# ── Download datasets ─────────────────────────────────────────────────────────
DATASETS = [
    ("roboflow-100", "animals-ij5d2", 1),   # cat,dog,cow,horse,fox,goat,chicken
]
 
downloaded_dirs = []
 
print("\n── Step 1: Downloading Dataset ───────────────────────────")
for ws, proj, ver in DATASETS:
    save_to = os.path.join(BASE, f"ds_{proj}")
    try:
        print(f"  Downloading: {proj} ...")
        p = rf.workspace(ws).project(proj)
        d = p.version(ver).download("yolov8", location=save_to)
        downloaded_dirs.append(save_to)
        print(f"  Done: {save_to}")
    except Exception as e:
        print(f"  Skipped {proj}: {e}")
 
# Include your uploaded animal_dataset if present
user_ds = os.path.join(BASE, "animal_dataset")
if os.path.exists(user_ds):
    downloaded_dirs.append(user_ds)
    print(f"  Including: animal_dataset")
 
if not downloaded_dirs:
    print("No datasets found!")
    sys.exit(1)
 
# ── Merge datasets ────────────────────────────────────────────────────────────
print("\n── Step 2: Merging Datasets ──────────────────────────────")
 
MERGED = os.path.join(BASE, "merged_animals")
for split in ["train", "valid"]:
    os.makedirs(f"{MERGED}/{split}/images", exist_ok=True)
    os.makedirs(f"{MERGED}/{split}/labels", exist_ok=True)
 
all_classes = []
 
for d in downloaded_dirs:
    yf = os.path.join(d, "data.yaml")
    if not os.path.exists(yf):
        continue
    with open(yf) as f:
        cfg = yaml.safe_load(f)
    for name in cfg.get("names", []):
        name = name.lower().strip()
        if name not in all_classes:
            all_classes.append(name)
 
print(f"  Classes ({len(all_classes)}): {all_classes}")
 
def remap_and_copy(src_dir, split, src_classes):
    img_src = os.path.join(src_dir, split, "images")
    lbl_src = os.path.join(src_dir, split, "labels")
    if not os.path.exists(img_src):
        img_src = os.path.join(src_dir, "train", "images")
        lbl_src = os.path.join(src_dir, "train", "labels")
    if not os.path.exists(img_src):
        return 0
    prefix = Path(src_dir).name + "_"
    count  = 0
    for img_path in glob.glob(f"{img_src}/*"):
        ext  = Path(img_path).suffix
        stem = Path(img_path).stem
        shutil.copy2(img_path, f"{MERGED}/{split}/images/{prefix}{stem}{ext}")
        lbl_path = f"{lbl_src}/{stem}.txt"
        dst_lbl  = f"{MERGED}/{split}/labels/{prefix}{stem}.txt"
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                lines = f.readlines()
            remapped = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_id   = int(parts[0])
                old_name = src_classes[old_id].lower().strip() if old_id < len(src_classes) else ""
                new_id   = all_classes.index(old_name) if old_name in all_classes else old_id
                remapped.append(f"{new_id} " + " ".join(parts[1:]) + "\n")
            with open(dst_lbl, "w") as f:
                f.writelines(remapped)
        else:
            open(dst_lbl, "w").close()
        count += 1
    return count
 
total_imgs = 0
for d in downloaded_dirs:
    yf = os.path.join(d, "data.yaml")
    if not os.path.exists(yf):
        continue
    with open(yf) as f:
        cfg = yaml.safe_load(f)
    src_classes = [n.lower().strip() for n in cfg.get("names", [])]
    n = remap_and_copy(d, "train", src_classes)
    remap_and_copy(d, "valid", src_classes)
    total_imgs += n
    print(f"  Merged {n} images from {Path(d).name}")
 
merged_yaml = os.path.join(MERGED, "data.yaml")
with open(merged_yaml, "w") as f:
    yaml.dump({
        "train": "train/images",
        "val":   "valid/images",
        "nc":    len(all_classes),
        "names": all_classes,
    }, f, default_flow_style=False)
 
print(f"  Total images: {total_imgs}")
 
# ── Train — CPU optimized settings ───────────────────────────────────────────
print("\n── Step 3: Training (CPU Optimized) ─────────────────────")
print(f"  Model   : YOLOv8n (nano — fastest on CPU)")
print(f"  Epochs  : 20  (was 80 — 4x faster)")
print(f"  ImgSize : 416 (was 640 — 2x faster)")
print(f"  Batch   : 16")
print(f"  Est.Time: ~45-90 minutes on CPU")
print("──────────────────────────────────────────────────────────\n")
 
# Use YOLOv8n (nano) instead of medium — 10x faster on CPU, still accurate
model = YOLO("yolov8n.pt")
 
model.train(
    data=merged_yaml,
    epochs=20,          # was 80 — good enough for fine-tuning
    imgsz=416,          # was 640 — major speed boost
    batch=16,           # larger batch = faster
    name="animals_full",
    patience=10,
    augment=True,
    mosaic=0.5,
    flipud=0.1,
    fliplr=0.5,
    degrees=10,
    hsv_h=0.015,
    hsv_s=0.5,
    workers=0,          # Windows fix — prevents dataloader hangs
    verbose=True,
    exist_ok=True,      # overwrite previous run
)
 
print("""
╔══════════════════════════════════════════════════════╗
║   Training Complete!                                 ║
║                                                      ║
║   Model: runs/detect/animals_full/weights/best.pt    ║
║                                                      ║
║   Now run:  python main.py                           ║
╚══════════════════════════════════════════════════════╝
""")