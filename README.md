# 🤖 Vision AI — Real-Time Hand Gesture & Object Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A real-time AI desktop application with cute voice output, live webcam feed, image upload, and 600+ object classes.**

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| ✋ **Finger Counter** | Detects 21 hand landmarks via MediaPipe Task API, counts extended fingers (0–5), draws skeleton overlay |
| 🎯 **Object Detector** | Detects 600+ classes using YOLOv8x-oiv7 (OpenImages V7) — animals, birds, vehicles, food and more |
| 🔊 **Cute Voice Output** | Speaks detected results in a clear female voice using Windows TTS — announces only when detection changes |
| 📁 **Image Upload** | Upload any JPG/PNG image for instant object detection without needing a camera |
| ↩ **Replay Button** | Replay the last spoken result anytime with one click |
| 🖥️ **Resizable UI** | Fully resizable dark-themed window — starts maximized, camera feed scales to fill screen |
| 🎨 **Kawaii Edition** | Pink accent theme with cute ♡ icons and kawaii-style voice phrases |

---

## 🐾 What It Can Detect

### Animals & Birds (600+ classes via OpenImages V7)
| Category | Classes |
|---|---|
| 🦁 Big Cats | Lion, Tiger, Leopard, Cheetah, Jaguar, Cat |
| 🐶 Canine | Dog, Wolf, Fox, Coyote |
| 🐘 Large Animals | Elephant, Giraffe, Zebra, Hippo, Rhino, Bear |
| 🐎 Farm Animals | Horse, Cow, Sheep, Goat, Chicken, Pig |
| 🦅 Birds | Eagle, Owl, Parrot, Flamingo, Peacock, Penguin, Duck, Swan |
| 🐊 Reptiles | Crocodile, Snake, Lizard, Turtle |
| 🐬 Marine | Dolphin, Whale, Shark, Seal |
| 🐒 Primates | Monkey, Gorilla, Chimpanzee |
| 🚗 Vehicles | Car, Bus, Truck, Bicycle, Airplane, Boat |
| 🍕 Food | Pizza, Burger, Banana, Apple and more |
| 👤 People | Person |

> Custom trained model also included for: **fox, goat, chicken, raccoon, skunk** with higher accuracy.

---

## 🖥️ Screenshots

```
┌─────────────────────────────────────────────────────────────────┐
│ ⬡ VISION AI  kawaii edition ♡   [✋ Finger Counter] [🎯 Object] │
├──────────────────────────────────────────┬──────────────────────┤
│                                          │  RESULT ♡            │
│                                          │  ┌────────────────┐  │
│         Live Camera / Image              │  │   Chicken      │  │
│                                          │  │  Chicken ♡     │  │
│     [Bounding boxes with labels]         │  └────────────────┘  │
│                                          │  🔊 Voice ON  ↩Replay│
│                                          │  CONFIDENCE    96%   │
│                                          │  ━━━━━━━━━━━━━━━━━   │
│                                          │  ALL DETECTIONS      │
│                                          │  ♡ Chicken     96%   │
└──────────────────────────────────────────┴──────────────────────┘
│ FPS: 12.3  Mode: Object Detector  ♡ Voice: ON    ● Camera On   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
vision_app/
│
├── main.py                  ← Entry point — run this to launch
├── gui.py                   ← Full Tkinter UI (dark/pink theme, voice, panels)
├── finger_counter.py        ← MediaPipe Task API hand tracking & finger logic
├── object_detector.py       ← YOLOv8 dual-model inference & bounding boxes
├── train_model.py           ← Train custom animal model on your dataset
├── setup_and_train.py       ← Download datasets from Roboflow + merge + train
├── requirements.txt         ← All Python dependencies
├── README.md                ← This file
│
├── animal_dataset/          ← Your uploaded animal dataset (if present)
│   ├── data.yaml
│   └── train/
│       ├── images/
│       └── labels/
│
└── runs/                    ← Created after training
    └── detect/
        └── animals_full/
            └── weights/
                └── best.pt  ← Your trained model (auto-used by app)
```

---

## ⚙️ Requirements

- **OS:** Windows 10/11 (voice uses Windows TTS built-in)
- **Python:** 3.10, 3.11, or 3.12 (NOT 3.13 — MediaPipe incompatible)
- **Webcam:** Any USB or built-in camera
- **RAM:** 4GB minimum, 8GB recommended
- **Disk:** ~500 MB free (for model weights)
- **Internet:** Required on first run to download model weights

---

## 🚀 Setup & Installation

### Step 1 — Download the project
```bash
git clone https://github.com/YOUR_USERNAME/vision-ai.git
cd vision-ai
```

### Step 2 — Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
python main.py
```

> **First run note:** The app will automatically download:
> - `hand_landmarker.task` (~8 MB) — MediaPipe hand model
> - `yolov8x-oiv7.pt` (~136 MB) — OpenImages object detection model
>
> This only happens once. After that, the app starts instantly.

---

## 🎮 How to Use

### Finger Counter Mode
1. Click **▶ Start Camera**
2. Click **✋ Finger Counter** button
3. Hold your hand in front of the camera
4. The app counts your raised fingers (0–5) and speaks the result
5. The hand skeleton is drawn in real time

### Object Detector — Live Camera
1. Click **▶ Start Camera**
2. Click **🎯 Object Detector** button
3. Point camera at any animal, bird, or object
4. The app draws a bounding box, shows the name and confidence
5. Voice announces the detection when something new appears

### Object Detector — Upload Image
1. Click **🎯 Object Detector** button
2. Click **📁 Upload Image**
3. Select any JPG or PNG file from your computer
4. The app instantly detects all objects and speaks the result

### Voice Controls
| Control | Action |
|---|---|
| 🔊 **Voice ON** icon | Click to toggle voice on/off |
| **↩ Replay** button | Repeat the last spoken result |
| Voice speaks automatically | Only when detection changes — not every frame |

---

## 🧠 How It Works

### Finger Counter (`finger_counter.py`)
- Uses **MediaPipe HandLandmarker Task API** (works with mediapipe 0.10.30+)
- Detects **21 hand landmarks** per frame in VIDEO mode
- Downloads `hand_landmarker.task` model automatically on first run
- **Thumb logic:** `tip.x < ip.x` for right hand, `tip.x > ip.x` for left hand
- **Index–Pinky logic:** `tip.y < pip.y` means finger is raised
- Draws skeleton using raw OpenCV (no mediapipe drawing utils needed)

### Object Detector (`object_detector.py`)
- **Primary model:** `yolov8x-oiv7.pt` — YOLOv8 Extra Large trained on Google OpenImages V7 (600+ classes)
- **Secondary model:** Custom trained `animals_full/weights/best.pt` (if present) — adds fox, goat, chicken, raccoon, skunk
- Smart conflict resolution: OpenImages results take priority; custom model only adds classes not already detected
- Confidence threshold: **30%** minimum for detection
- Colour-coded bounding boxes by category (orange = big cats, pink = birds, green = animals)

### Voice System (`gui.py`)
- Uses **Windows PowerShell** built-in TTS — no extra install needed
- Female voice selected automatically
- Rate set to `-1` (slow and clear)
- Speaks **only when detection changes** — prevents repeated/overlapping voices
- `_kill_current()` terminates previous speech before starting new one

---

## 🏋️ Custom Training

Want to train on your own animal dataset?

### Using Roboflow Dataset
```bash
# 1. Get free API key from https://roboflow.com
# 2. Edit setup_and_train.py — paste your API key
# 3. Run:
python setup_and_train.py
```

### Using Your Own Dataset (YOLOv8 format)
```
your_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```
```bash
python train_model.py
```

After training, the app **automatically detects and uses** your trained model — no code changes needed.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open camera` | Close Zoom/Teams/other apps using webcam. Try `cv2.VideoCapture(1)` in `gui.py` |
| `mediapipe has no attribute solutions` | You have mediapipe 0.10.30+ — this is fixed in the current code |
| `ExternalFile must specify file_content` | `hand_landmarker.task` missing — check internet and re-run |
| Voice not working | Make sure you are on Windows. PowerShell must be available |
| Voice too fast/slow | Change `$s.Rate = -1` in `gui.py` — range is `-10` (slow) to `10` (fast) |
| Lion/Tiger detected wrong | These need the OpenImages model (`yolov8x-oiv7.pt`) — wait for it to download |
| Low FPS | Normal on CPU — object detector is heavy. Close other apps |
| `No matching distribution for mediapipe==0.10.9` | Run `pip install mediapipe` without version pin |
| Training takes too long | Use `setup_and_train.py` — it uses YOLOv8n + 20 epochs optimized for CPU |
| Python 3.13 error | MediaPipe does not support Python 3.13. Use Python 3.10, 3.11 or 3.12 |

---

## 📦 Dependencies

```
opencv-python       — webcam capture and image drawing
mediapipe           — hand landmark detection (Task API)
ultralytics         — YOLOv8 object detection
numpy               — array operations
Pillow              — image conversion for Tkinter display
roboflow            — dataset download (optional, for training)
pyttsx3             — TTS fallback (optional)
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🗺️ Roadmap

- [ ] Add more animal species via additional Roboflow datasets
- [ ] Add emotion detection mode
- [ ] Add scene description (multiple objects summarized)
- [ ] Support macOS/Linux voice (espeak / say)
- [ ] Export detection results to CSV
- [ ] Add detection history log panel

---

## 🤝 Contributing

Pull requests are welcome! For major changes please open an issue first.

1. Fork the repo
2. Create your branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify and distribute.

---

## 👤 Author

Thamodharan k

---

<div align="center">

**⭐ If you found this useful, please give it a star on GitHub! ⭐**

</div>
