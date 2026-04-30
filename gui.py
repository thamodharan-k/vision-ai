"""
gui.py - Fixed:
  1. Voice speaks only ONCE — kills previous voice before speaking new one
  2. Finger counter voice only speaks when count CHANGES (not every frame)
  3. Object detector voice only speaks when label CHANGES
  4. Replay button
  5. Voice ON/OFF toggle icon under result
"""
 
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
import numpy as np
import random
import subprocess
 
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_CARD   = "#1c2128"
ACCENT    = "#00d4ff"
GREEN     = "#00ff88"
RED       = "#ff4455"
PINK      = "#ff69b4"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
BTN_INACT = "#21262d"
BTN_TXT   = "#0d1117"
 
PANEL_W = 260
 
CUTE_PREFIXES = [
    "I can see a", "That is a", "It looks like a",
    "I found a", "This is a",
]
CUTE_SUFFIXES = [
    "How cute!", "Amazing!", "Wonderful!", "I love it!", "",
]
NOTHING_PHRASES = [
    "I cannot see anything. Please try again.",
    "Nothing detected. Show me something clearer.",
    "I do not recognize this. Try another image.",
]
FINGER_PHRASES = {
    0: "No fingers detected.",
    1: "One finger.",
    2: "Two fingers. Peace sign!",
    3: "Three fingers.",
    4: "Four fingers.",
    5: "Five fingers. Perfect!",
}
 
 
class CuteVoiceEngine:
    """
    Speaks using Windows PowerShell TTS.
    Only ONE voice plays at a time — kills previous before starting new.
    Only speaks when the detected label actually CHANGES.
    """
 
    def __init__(self):
        self._enabled      = True
        self._last_label   = ""      # last spoken label
        self._last_text    = ""      # full last spoken sentence (for replay)
        self._proc         = None    # current PowerShell process
        self._lock         = threading.Lock()
        print("[Voice] Ready — using Windows PowerShell TTS")
 
    def _kill_current(self):
        """Stop any currently playing voice."""
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass
        self._proc = None
 
    def _speak(self, text: str):
        """Internal: kill previous, then speak new text."""
        with self._lock:
            self._kill_current()
            clean = (text.replace("'", "")
                         .replace('"', "")
                         .replace("`", "")
                         .replace("\\", ""))
            cmd = (
                'powershell -Command "'
                'Add-Type -AssemblyName System.Speech; '
                '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                '$s.Rate = -1; '        # -1 = slow and clear
                '$s.Volume = 100; '
                '$s.SelectVoiceByHints('
                '[System.Speech.Synthesis.VoiceGender]::Female); '
                f'$s.Speak(\\"{clean}\\");"'
            )
            try:
                self._proc = subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[Voice] Error: {e}")
 
    def speak_detection(self, label: str, conf: float, force=False):
        """Speak only when label changes or force=True."""
        if not self._enabled:
            return
        if not force and label == self._last_label:
            return   # same label — don't repeat
        self._last_label = label
        prefix = random.choice(CUTE_PREFIXES)
        suffix = random.choice(CUTE_SUFFIXES)
        pct    = int(conf * 100)
        text   = f"{prefix} {label}. Confidence {pct} percent. {suffix}"
        self._last_text = text
        t = threading.Thread(target=self._speak, args=(text,), daemon=True)
        t.start()
 
    def speak_fingers(self, count: int, force=False):
        """Speak only when finger count changes."""
        if not self._enabled:
            return
        key = f"fingers_{count}"
        if not force and key == self._last_label:
            return   # same count — don't repeat
        self._last_label = key
        text = FINGER_PHRASES.get(count, f"{count} fingers.")
        self._last_text = text
        t = threading.Thread(target=self._speak, args=(text,), daemon=True)
        t.start()
 
    def speak_nothing(self):
        if not self._enabled:
            return
        text = random.choice(NOTHING_PHRASES)
        self._last_text = text
        self._last_label = "nothing"
        t = threading.Thread(target=self._speak, args=(text,), daemon=True)
        t.start()
 
    def speak_raw(self, text: str):
        """Speak any text immediately (for greetings etc.)."""
        if not self._enabled:
            return
        self._last_text = text
        t = threading.Thread(target=self._speak, args=(text,), daemon=True)
        t.start()
 
    def replay(self):
        """Replay last spoken sentence."""
        if not self._last_text:
            return
        t = threading.Thread(
            target=self._speak, args=(self._last_text,), daemon=True)
        t.start()
 
    def stop(self):
        """Stop speaking immediately."""
        with self._lock:
            self._kill_current()
 
    def reset_label(self):
        """Reset so next detection speaks even if same label."""
        self._last_label = ""
 
    def toggle(self):
        self._enabled = not self._enabled
        if not self._enabled:
            self.stop()
        return self._enabled
 
 
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vision AI — Hand & Object Recognition")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(900, 600)
        self.state("zoomed")
 
        self._mode        = "finger"
        self._running     = False
        self._cap         = None
        self._thread      = None
        self._photo       = None
        self._fps         = 0.0
        self._frame_time  = time.time()
        self._finger_det  = None
        self._object_det  = None
        self._uploaded    = False
        self._voice       = CuteVoiceEngine()
 
        # Track previous values to detect changes
        self._prev_count  = -1
        self._prev_label  = ""
 
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Configure>", self._on_resize)
 
        self.after(1200, lambda: self._voice.speak_raw(
            "Hello! I am Vision AI. Show me something and I will tell you what it is."))
 
    # ── Build UI ──────────────────────────────────────────────────────────────
 
    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
 
        # Top bar
        top = tk.Frame(self, bg=BG_PANEL, pady=10)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)
 
        logo = tk.Frame(top, bg=BG_PANEL)
        logo.grid(row=0, column=0, padx=20)
        tk.Label(logo, text="⬡", bg=BG_PANEL, fg=ACCENT,
                 font=("Courier", 20, "bold")).pack(side="left")
        tk.Label(logo, text=" VISION AI", bg=BG_PANEL, fg=ACCENT,
                 font=("Courier", 16, "bold")).pack(side="left")
        tk.Label(logo, text="  kawaii edition ♡",
                 bg=BG_PANEL, fg=PINK,
                 font=("Courier", 9)).pack(side="left", padx=8)
 
        center = tk.Frame(top, bg=BG_PANEL)
        center.grid(row=0, column=1)
 
        self._btn_finger = tk.Button(
            center, text="✋  Finger Counter",
            font=("Courier", 11, "bold"), relief="flat",
            padx=18, pady=8, cursor="hand2",
            command=lambda: self._switch_mode("finger"))
        self._btn_finger.pack(side="left", padx=6)
 
        self._btn_object = tk.Button(
            center, text="🎯  Object Detector",
            font=("Courier", 11, "bold"), relief="flat",
            padx=18, pady=8, cursor="hand2",
            command=lambda: self._switch_mode("object"))
        self._btn_object.pack(side="left", padx=6)
 
        right = tk.Frame(top, bg=BG_PANEL)
        right.grid(row=0, column=2, padx=20)
 
        self._upload_btn = tk.Button(
            right, text="📁  Upload Image",
            bg=BTN_INACT, fg=TEXT_SEC,
            font=("Courier", 10, "bold"), relief="flat",
            padx=12, pady=8, cursor="hand2",
            command=self._upload_image)
        self._upload_btn.pack(side="left", padx=4)
 
        self._cam_btn = tk.Button(
            right, text="▶  Start Camera",
            bg=ACCENT, fg=BTN_TXT,
            font=("Courier", 10, "bold"), relief="flat",
            padx=12, pady=8, cursor="hand2",
            command=self._toggle_camera)
        self._cam_btn.pack(side="left", padx=4)
 
        self._refresh_mode_buttons()
 
        # Body
        body = tk.Frame(self, bg=BG_DARK)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
 
        cam_outer = tk.Frame(body, bg=PINK, padx=2, pady=2)
        cam_outer.grid(row=0, column=0, sticky="nsew")
        cam_outer.grid_rowconfigure(0, weight=1)
        cam_outer.grid_columnconfigure(0, weight=1)
 
        self._canvas = tk.Canvas(cam_outer, bg="#000000",
                                 highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._draw_placeholder()
 
        panel = tk.Frame(body, bg=BG_PANEL, width=PANEL_W)
        panel.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        panel.grid_propagate(False)
        self._build_panel(panel)
 
        # Status bar
        status = tk.Frame(self, bg=BG_PANEL, pady=5)
        status.grid(row=2, column=0, sticky="ew")
 
        self._fps_lbl = tk.Label(status, text="FPS: —",
                                 bg=BG_PANEL, fg=ACCENT,
                                 font=("Courier", 9))
        self._fps_lbl.pack(side="left", padx=16)
 
        self._mode_lbl = tk.Label(status, text="Mode: Finger Counter",
                                  bg=BG_PANEL, fg=TEXT_SEC,
                                  font=("Courier", 9))
        self._mode_lbl.pack(side="left", padx=8)
 
        self._status_lbl = tk.Label(status, text="● Camera Off",
                                    bg=BG_PANEL, fg=TEXT_SEC,
                                    font=("Courier", 9))
        self._status_lbl.pack(side="right", padx=16)
 
    def _build_panel(self, panel):
        inner = tk.Frame(panel, bg=BG_PANEL)
        inner.pack(fill="both", expand=True, padx=12, pady=12)
 
        tk.Label(inner, text="RESULT ♡", bg=BG_PANEL, fg=PINK,
                 font=("Courier", 9, "bold")).pack(anchor="w", pady=(0, 6))
 
        card = tk.Frame(inner, bg=BG_CARD, pady=16, padx=16)
        card.pack(fill="x")
 
        self._big_text = tk.Label(card, text="—", bg=BG_CARD, fg=ACCENT,
                                  font=("Courier", 48, "bold"))
        self._big_text.pack()
 
        self._sub_text = tk.Label(card, text="show me something!",
                                  bg=BG_CARD, fg=PINK,
                                  font=("Courier", 10),
                                  wraplength=200, justify="center")
        self._sub_text.pack(pady=(4, 0))
 
        # Voice controls row
        vrow = tk.Frame(inner, bg=BG_PANEL)
        vrow.pack(fill="x", pady=(10, 2))
 
        self._voice_icon = tk.Label(vrow, text="🔊",
                                    bg=BG_PANEL, font=("Arial", 15),
                                    cursor="hand2")
        self._voice_icon.pack(side="left")
        self._voice_icon.bind("<Button-1>", lambda e: self._toggle_voice())
 
        self._voice_lbl = tk.Label(vrow, text="Voice  ON",
                                   bg=BG_PANEL, fg=PINK,
                                   font=("Courier", 9, "bold"),
                                   cursor="hand2")
        self._voice_lbl.pack(side="left", padx=(4, 8))
        self._voice_lbl.bind("<Button-1>", lambda e: self._toggle_voice())
 
        self._replay_btn = tk.Button(
            vrow, text="↩ Replay",
            bg=BG_CARD, fg=PINK,
            font=("Courier", 9, "bold"),
            relief="flat", padx=8, pady=3,
            cursor="hand2",
            command=self._replay_voice)
        self._replay_btn.pack(side="right")
 
        self._fingers_lbl = tk.Label(inner, text="",
                                     bg=BG_PANEL, fg=GREEN,
                                     font=("Courier", 10),
                                     wraplength=220, justify="center")
        self._fingers_lbl.pack(pady=4)
 
        tk.Frame(inner, bg=BG_CARD, height=1).pack(fill="x", pady=4)
 
        tk.Label(inner, text="CONFIDENCE", bg=BG_PANEL, fg=TEXT_SEC,
                 font=("Courier", 8, "bold")).pack(anchor="w", pady=(4, 4))
 
        bar_bg = tk.Frame(inner, bg=BG_CARD, height=10)
        bar_bg.pack(fill="x")
        bar_bg.pack_propagate(False)
 
        self._conf_bar = tk.Frame(bar_bg, bg=PINK, height=10)
        self._conf_bar.place(x=0, y=0, relheight=1.0, width=0)
 
        self._conf_pct = tk.Label(inner, text="", bg=BG_PANEL, fg=TEXT_PRI,
                                  font=("Courier", 12, "bold"))
        self._conf_pct.pack(anchor="e", pady=(4, 0))
 
        tk.Frame(inner, bg=BG_CARD, height=1).pack(fill="x", pady=6)
 
        tk.Label(inner, text="ALL DETECTIONS", bg=BG_PANEL, fg=TEXT_SEC,
                 font=("Courier", 8, "bold")).pack(anchor="w", pady=(0, 4))
 
        self._det_rows = []
        for _ in range(6):
            row = tk.Frame(inner, bg=BG_PANEL)
            row.pack(fill="x", pady=2)
            dot = tk.Label(row, text="♡", bg=BG_PANEL, fg=BG_CARD,
                           font=("Courier", 8), width=2)
            dot.pack(side="left")
            name_lbl = tk.Label(row, text="", bg=BG_PANEL, fg=TEXT_PRI,
                                font=("Courier", 9), anchor="w")
            name_lbl.pack(side="left", fill="x", expand=True)
            pct_lbl = tk.Label(row, text="", bg=BG_PANEL, fg=PINK,
                               font=("Courier", 9, "bold"),
                               width=5, anchor="e")
            pct_lbl.pack(side="right")
            self._det_rows.append((dot, name_lbl, pct_lbl))
 
    # ── Voice ─────────────────────────────────────────────────────────────────
 
    def _toggle_voice(self):
        enabled = self._voice.toggle()
        if enabled:
            self._voice_icon.config(text="🔊")
            self._voice_lbl.config(text="Voice  ON", fg=PINK)
            self._voice.speak_raw("Voice is on!")
        else:
            self._voice_icon.config(text="🔇")
            self._voice_lbl.config(text="Voice  OFF", fg=TEXT_SEC)
 
    def _replay_voice(self):
        self._voice.replay()
        self._replay_btn.config(bg=PINK, fg=BTN_TXT)
        self.after(400, lambda: self._replay_btn.config(
            bg=BG_CARD, fg=PINK))
 
    # ── Camera ────────────────────────────────────────────────────────────────
 
    def _toggle_camera(self):
        if self._running:
            self.stop_camera()
        else:
            self.start_camera()
 
    def start_camera(self):
        if self._running:
            return
        self._uploaded   = False
        self._prev_count = -1
        self._prev_label = ""
        self._voice.reset_label()
 
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open camera!")
            return
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._load_detectors()
        self._running = True
        self._cam_btn.config(text="⏹  Stop Camera", bg=RED, fg=TEXT_PRI)
        self._status_lbl.config(text="● Camera On", fg=GREEN)
        self._voice.speak_raw("Camera is on. Show me something!")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
 
    def stop_camera(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._voice.stop()
        self._cam_btn.config(text="▶  Start Camera", bg=ACCENT, fg=BTN_TXT)
        self._status_lbl.config(text="● Camera Off", fg=TEXT_SEC)
        self._draw_placeholder()
        self._reset_panel()
 
    def _load_detectors(self):
        if self._finger_det is None:
            from finger_counter import FingerCounter
            self._finger_det = FingerCounter()
        if self._object_det is None:
            from object_detector import ObjectDetector
            self._object_det = ObjectDetector()
 
    # ── Upload ────────────────────────────────────────────────────────────────
 
    def _upload_image(self):
        if self._mode != "object":
            messagebox.showinfo("Switch Mode",
                "Please click Object Detector first!")
            return
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        if self._running:
            self.stop_camera()
        self._load_detectors()
        self._uploaded   = True
        self._prev_label = ""
        self._voice.reset_label()
 
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read image!")
            return
 
        annotated, top_label, top_conf = self._object_det.process(img)
        all_det = self._object_det._last_detections
 
        self._show_frame(annotated)
        self._update_result_object(top_label, top_conf, all_det)
        self._fps_lbl.config(text="FPS: —")
        self._mode_lbl.config(text="Mode: Object Detector")
        self._status_lbl.config(text="● Image Loaded", fg=ACCENT)
 
        if top_label != "---" and top_conf > 0:
            self._voice.speak_detection(top_label, top_conf, force=True)
        else:
            self._voice.speak_nothing()
 
    # ── Frame loop ────────────────────────────────────────────────────────────
 
    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
 
            if self._mode == "finger":
                try:
                    annotated, count, names = self._finger_det.process(frame)
                except Exception as e:
                    print(f"[FingerCounter] Error: {e}")
                    annotated, count, names = frame, 0, []
                payload = ("finger", annotated, count, names)
            else:
                try:
                    annotated, label, conf = self._object_det.process(frame)
                    all_det = self._object_det._last_detections
                except Exception as e:
                    print(f"[ObjectDetector] Error: {e}")
                    annotated, label, conf, all_det = frame, "---", 0.0, []
                payload = ("object", annotated, label, conf, all_det)
 
            now = time.time()
            self._fps = 1.0 / max(now - self._frame_time, 1e-6)
            self._frame_time = now
            self.after(0, self._update_ui, payload)
 
    # ── UI updates ────────────────────────────────────────────────────────────
 
    def _update_ui(self, payload):
        mode      = payload[0]
        annotated = payload[1]
        self._show_frame(annotated)
 
        if mode == "finger":
            _, _, count, names = payload
            self._update_result_finger(count, names)
            # Only speak when count CHANGES
            if count != self._prev_count:
                self._prev_count = count
                self._voice.speak_fingers(count, force=True)
 
        else:
            _, _, label, conf, all_det = payload
            self._update_result_object(label, conf, all_det)
            # Only speak when label CHANGES
            if label != self._prev_label and label != "---":
                self._prev_label = label
                self._voice.speak_detection(label, conf, force=True)
 
        self._fps_lbl.config(text=f"FPS: {self._fps:.1f}")
        mode_name = ("Finger Counter" if self._mode == "finger"
                     else "Object Detector")
        self._mode_lbl.config(text=f"Mode: {mode_name}")
 
    def _show_frame(self, frame):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        fh, fw = frame.shape[:2]
        scale   = min(cw / fw, ch / fh)
        nw, nh  = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo   = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self._canvas.delete("all")
        self._canvas.create_image(x, y, anchor="nw", image=photo)
        self._photo = photo
 
    def _update_result_finger(self, count, names):
        self._big_text.config(text=str(count), fg=ACCENT,
                              font=("Courier", 48, "bold"))
        plural = "s" if count != 1 else ""
        self._sub_text.config(text=f"{count} finger{plural} up! ♡")
        self._fingers_lbl.config(
            text="  ".join(names) if names else "no hand detected")
        self._conf_bar.place_configure(width=0)
        self._conf_pct.config(text="")
        self._clear_det_rows()
 
    def _update_result_object(self, label, conf, all_det):
        if label == "---" or conf == 0:
            self._big_text.config(text="?", fg=RED,
                                  font=("Courier", 48, "bold"))
            self._sub_text.config(
                text="I cannot see anything!\nShow me something!")
            self._conf_bar.place_configure(width=0)
            self._conf_pct.config(text="0%", fg=RED)
            self._clear_det_rows()
            return
 
        display = label if len(label) <= 8 else label[:7] + "…"
        self._big_text.config(text=display, fg=ACCENT,
                              font=("Courier", 32, "bold"))
        self._sub_text.config(text=f"{label} ♡")
        self._fingers_lbl.config(text="")
 
        bar_w     = self._conf_bar.master.winfo_width()
        fill      = int(bar_w * conf) if bar_w > 1 else 0
        bar_color = GREEN if conf >= 0.65 else (PINK if conf >= 0.40 else RED)
        self._conf_bar.config(bg=bar_color)
        self._conf_bar.place_configure(width=fill)
        self._conf_pct.config(text=f"{conf:.0%}", fg=bar_color)
 
        self._clear_det_rows()
        sorted_det = sorted(all_det, key=lambda x: x[1], reverse=True)
        for i, (dot, name_lbl, pct_lbl) in enumerate(self._det_rows):
            if i < len(sorted_det):
                n, c = sorted_det[i]
                color = GREEN if c >= 0.65 else (PINK if c >= 0.40 else RED)
                dot.config(fg=color)
                name_lbl.config(text=n)
                pct_lbl.config(text=f"{c:.0%}", fg=color)
 
    def _reset_panel(self):
        self._big_text.config(text="—", font=("Courier", 48, "bold"),
                              fg=ACCENT)
        self._sub_text.config(text="show me something!")
        self._fingers_lbl.config(text="")
        self._conf_bar.place_configure(width=0)
        self._conf_pct.config(text="")
        self._clear_det_rows()
        self._prev_count = -1
        self._prev_label = ""
 
    def _clear_det_rows(self):
        for dot, name_lbl, pct_lbl in self._det_rows:
            dot.config(fg=BG_CARD)
            name_lbl.config(text="")
            pct_lbl.config(text="")
 
    def _switch_mode(self, mode):
        self._mode = mode
        self._voice.stop()
        self._voice.reset_label()
        self._prev_count = -1
        self._prev_label = ""
        self._refresh_mode_buttons()
        self._reset_panel()
        mode_name = ("Finger Counter" if mode == "finger"
                     else "Object Detector")
        self._mode_lbl.config(text=f"Mode: {mode_name}")
        self._upload_btn.config(
            fg=TEXT_PRI if mode == "object" else TEXT_SEC)
 
    def _refresh_mode_buttons(self):
        for btn, m in [(self._btn_finger, "finger"),
                       (self._btn_object, "object")]:
            btn.config(bg=ACCENT if m == self._mode else BTN_INACT,
                       fg=BTN_TXT if m == self._mode else TEXT_PRI)
 
    def _on_resize(self, event=None):
        if not self._running and not self._uploaded:
            self.after(100, self._draw_placeholder)
 
    def _draw_placeholder(self):
        self._canvas.delete("all")
        cw = self._canvas.winfo_width()  or 640
        ch = self._canvas.winfo_height() or 480
        self._canvas.create_rectangle(0, 0, cw, ch,
                                      fill="#000000", outline="")
        self._canvas.create_text(cw // 2, ch // 2 - 40,
                                 text="♡", fill=PINK,
                                 font=("Courier", 52))
        self._canvas.create_text(cw // 2, ch // 2 + 20,
                                 text="▶  Start Camera  —  or  —  📁 Upload Image",
                                 fill=TEXT_SEC, font=("Courier", 13))
        self._canvas.create_text(cw // 2, ch // 2 + 55,
                                 text="Show me something and I will tell you what it is! ♡",
                                 fill=PINK, font=("Courier", 10))
 
    def _on_close(self):
        self._voice.stop()
        self._voice.speak_raw("Bye bye! See you next time!")
        time.sleep(1.5)
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        if self._finger_det:
            self._finger_det.release()
        self.destroy()