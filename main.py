import os
import time
import csv
import json
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

APP_NAME = "Facial-Expression-Reading (GUI)"

# -----------------------------
# Data folder (created next to this script)
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Facial-Expression-Reading-data")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(DATA_DIR, "fer_model.joblib")
BASELINE_PATH = os.path.join(DATA_DIR, "fer_baseline.json")
PRED_LOG_PATH = os.path.join(DATA_DIR, "fer_predictions_log.csv")
LABELED_PATH = os.path.join(DATA_DIR, "fer_labeled_samples.csv")

# Expanded emotion classes
CLASSES = np.array([
    "Neutral",
    "Happy",
    "Surprised",
    "AngryFocused",
    "Sad",
    "Disgust",
    "Fear",
])

LABEL_TO_PRETTY = {
    "Neutral": "Neutral 😐",
    "Happy": "Happy 🙂",
    "Surprised": "Surprised 😮",
    "AngryFocused": "Angry/Focused 😠",
    "Sad": "Sad 😔",
    "Disgust": "Disgust 🤢",
    "Fear": "Fear 😨",
}

# -----------------------------
# Helpers
# -----------------------------
def now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_csv_header(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_csv_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        return None
    try:
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_baseline(baseline: dict):
    try:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
    except Exception:
        pass


def dist(a, b):
    return float(np.linalg.norm(a - b))


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def ema(prev, new, alpha=0.25):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)


# -----------------------------
# FaceMesh landmark indices (MediaPipe)
# -----------------------------
IDX = {
    # Eyes (key points)
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "left_eye_left": 33,
    "left_eye_right": 133,

    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "right_eye_left": 362,
    "right_eye_right": 263,

    # Eyelids (upper/lower contours)
    "left_upper_lid": 158,
    "left_lower_lid": 153,
    "right_upper_lid": 385,
    "right_lower_lid": 380,

    # Mouth / lips
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,

    # Full lips outline
    "upper_lip_left": 78,
    "upper_lip_center": 13,
    "upper_lip_right": 308,
    "lower_lip_left": 95,
    "lower_lip_center": 14,
    "lower_lip_right": 324,

    # Brows
    "left_brow": 105,
    "right_brow": 334,

    # Nose / nostrils
    "nose_tip": 1,
    "nose_bridge": 6,
    "nose_left": 97,
    "nose_right": 326,
    "nostril_left": 94,
    "nostril_right": 331,

    # Cheeks / smile lines
    "left_cheek": 50,
    "right_cheek": 280,
    "left_smile_line": 62,
    "right_smile_line": 292,

    # Jawline / chin contour
    "jaw_left": 234,
    "jaw_left_mid": 172,
    "chin": 152,
    "jaw_right_mid": 397,
    "jaw_right": 454,

    # Face oval (anchor points)
    "face_top": 10,
    "face_left_top": 338,
    "face_left_bottom": 234,
    "face_right_top": 109,
    "face_right_bottom": 454,
}

FEAT_KEYS = ["eye_open", "mouth_open", "smile", "brow_raise"]


def extract_features(pts):
    face_h = dist(pts[IDX["nose_tip"]], pts[IDX["chin"]])
    face_h = max(face_h, 1e-6)

    l_eye_open = dist(pts[IDX["left_eye_top"]], pts[IDX["left_eye_bottom"]])
    l_eye_w = dist(pts[IDX["left_eye_left"]], pts[IDX["left_eye_right"]])
    left_eye_ratio = safe_div(l_eye_open, l_eye_w)

    r_eye_open = dist(pts[IDX["right_eye_top"]], pts[IDX["right_eye_bottom"]])
    r_eye_w = dist(pts[IDX["right_eye_left"]], pts[IDX["right_eye_right"]])
    right_eye_ratio = safe_div(r_eye_open, r_eye_w)

    eye_open = (left_eye_ratio + right_eye_ratio) / 2.0

    mouth_open = dist(pts[IDX["mouth_top"]], pts[IDX["mouth_bottom"]])
    mouth_w = dist(pts[IDX["mouth_left"]], pts[IDX["mouth_right"]])
    mouth_open_ratio = safe_div(mouth_open, mouth_w)

    smile_ratio = safe_div(mouth_w, face_h)

    left_brow_raise = safe_div(dist(pts[IDX["left_brow"]], pts[IDX["left_eye_top"]]), face_h)
    right_brow_raise = safe_div(dist(pts[IDX["right_brow"]], pts[IDX["right_eye_top"]]), face_h)
    brow_raise = (left_brow_raise + right_brow_raise) / 2.0

    return {
        "eye_open": eye_open,
        "mouth_open": mouth_open_ratio,
        "smile": smile_ratio,
        "brow_raise": brow_raise,
    }


def mean_dict(dicts, keys):
    out = {}
    for k in keys:
        out[k] = float(np.mean([d[k] for d in dicts])) if dicts else 0.0
    return out


def baseline_deltas(s_feats, baseline):
    return np.array([
        s_feats["eye_open"] - baseline["eye_open"],
        s_feats["mouth_open"] - baseline["mouth_open"],
        s_feats["smile"] - baseline["smile"],
        s_feats["brow_raise"] - baseline["brow_raise"],
    ], dtype=np.float32)


# -----------------------------
# Model utilities (auto-retrain on startup)
# -----------------------------
def make_model():
    clf = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42)
    return make_pipeline(StandardScaler(with_mean=True, with_std=True), clf)


def save_model(model, initialized: bool):
    payload = {"model": model, "initialized": initialized, "classes": CLASSES.tolist()}
    joblib.dump(payload, MODEL_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, False
    try:
        payload = joblib.load(MODEL_PATH)
        return payload.get("model"), bool(payload.get("initialized"))
    except Exception:
        return None, False


def load_labeled_dataset():
    if not os.path.exists(LABELED_PATH):
        return None, None

    X, y = [], []
    try:
        with open(LABELED_PATH, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                lbl = row.get("true_label", "")
                if lbl not in CLASSES:
                    continue
                try:
                    X.append([
                        float(row["d_eye"]),
                        float(row["d_mouth"]),
                        float(row["d_smile"]),
                        float(row["d_brow"]),
                    ])
                    y.append(lbl)
                except Exception:
                    continue
    except Exception:
        return None, None

    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=object)


def retrain_on_startup():
    X, y = load_labeled_dataset()
    if X is None or y is None:
        model, initialized = load_model()
        return model, initialized, 0

    model = make_model()
    scaler = model.steps[0][1]
    clf = model.steps[-1][1]

    scaler.fit(X)
    Xs = scaler.transform(X)

    clf.partial_fit(Xs[:1], y[:1], classes=CLASSES)
    if len(Xs) > 1:
        clf.partial_fit(Xs[1:], y[1:])

    initialized = True
    save_model(model, initialized)
    return model, initialized, int(len(X))


def model_predict(model, x):
    if model is None:
        return None, None
    try:
        probs = model.predict_proba(x)[0]
        pred = CLASSES[int(np.argmax(probs))]
        conf = float(np.max(probs))
        return pred, conf
    except Exception:
        try:
            pred = model.predict(x)[0]
            return pred, None
        except Exception:
            return None, None


def model_online_update(model, x, y, initialized):
    if model is None:
        model = make_model()
    clf = model.steps[-1][1]

    # x comes in unscaled; pipeline handles scaling on predict, but for partial_fit we need to scale ourselves
    scaler = model.steps[0][1]
    try:
        _ = scaler.mean_
        fitted = True
    except Exception:
        fitted = False

    if not fitted:
        scaler.fit(x)
    xs = scaler.transform(x)

    if not initialized:
        clf.partial_fit(xs, np.array([y]), classes=CLASSES)
        initialized = True
    else:
        clf.partial_fit(xs, np.array([y]))
    return model, initialized


# -----------------------------
# GUI App
# -----------------------------
class FERApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_NAME)

        # CSV headers
        ensure_csv_header(PRED_LOG_PATH, ["timestamp", "pred_label", "confidence", "d_eye", "d_mouth", "d_smile", "d_brow"])
        ensure_csv_header(LABELED_PATH, ["timestamp", "true_label", "d_eye", "d_mouth", "d_smile", "d_brow"])

        # Auto-retrain model
        self.model, self.model_initialized, self.trained_n = retrain_on_startup()

        # Baseline
        self.baseline = load_baseline()

        # Mediapipe FaceMesh
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam. Try changing camera index in code.")
            raise RuntimeError("Could not open webcam.")

        # State
        self.smooth = {k: None for k in FEAT_KEYS}
        self.emotion_hist = deque(maxlen=12)
        self.last_x = None
        self.last_d = None

        # Calibration
        self.calibrating = False
        self.calib_samples = []
        self.calib_start = 0.0
        self.CALIB_SECONDS = 2.0
        self.auto_calib_armed = (self.baseline is None)

        # Logging interval
        self.LOG_INTERVAL_SEC = 0.5
        self.last_log_time = 0.0

        # UI layout
        self._build_ui()

        # Start loop
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_frame()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        # Left: video
        video_frame = ttk.Frame(self.root, padding=8)
        video_frame.grid(row=0, column=0, sticky="nsew")
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Right: controls + status
        side = ttk.Frame(self.root, padding=8)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)

        title = ttk.Label(side, text="Controls", font=("Segoe UI", 14, "bold"))
        title.grid(row=0, column=0, sticky="w")

        info = ttk.Label(
            side,
            text=f"Data folder:\n{DATA_DIR}\n\nAuto-train samples loaded: {self.trained_n}",
            justify="left",
        )
        info.grid(row=1, column=0, sticky="w", pady=(6, 10))

        # Buttons
        btn_frame = ttk.LabelFrame(side, text="Label current expression", padding=8)
        btn_frame.grid(row=2, column=0, sticky="ew")
        for i in range(2):
            btn_frame.columnconfigure(i, weight=1)

        buttons = [
            ("Neutral (1)", "Neutral"),
            ("Happy (2)", "Happy"),
            ("Surprised (3)", "Surprised"),
            ("Angry (4)", "AngryFocused"),
            ("Sad (5)", "Sad"),
            ("Disgust (6)", "Disgust"),
            ("Fear (7)", "Fear"),
        ]
        for idx, (txt, lbl) in enumerate(buttons):
            r = 0 + idx // 2
            c = idx % 2
            b = ttk.Button(btn_frame, text=txt, command=lambda L=lbl: self.label_emotion(L))
            b.grid(row=r, column=c, sticky="ew", padx=4, pady=4)

        actions = ttk.LabelFrame(side, text="Actions", padding=8)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        ttk.Button(actions, text="Calibrate (neutral)", command=self.start_calibration).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(actions, text="Save model", command=self.save_model_clicked).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(actions, text="Open data folder", command=self.open_data_folder).grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=4)

        # Status
        self.status_var = tk.StringVar(value="Starting…")
        self.pred_var = tk.StringVar(value="Prediction: —")
        self.conf_var = tk.StringVar(value="Confidence: —")

        ttk.Label(side, textvariable=self.pred_var, font=("Segoe UI", 12, "bold")).grid(row=4, column=0, sticky="w", pady=(12, 0))
        ttk.Label(side, textvariable=self.conf_var).grid(row=5, column=0, sticky="w")
        ttk.Label(side, textvariable=self.status_var, wraplength=320, justify="left").grid(row=6, column=0, sticky="w", pady=(10, 0))

        # Keyboard shortcuts
        self.root.bind("<Key>", self.on_key)

    def on_key(self, event):
        k = event.char
        mapping = {
            "1": "Neutral",
            "2": "Happy",
            "3": "Surprised",
            "4": "AngryFocused",
            "5": "Sad",
            "6": "Disgust",
            "7": "Fear",
        }
        if k in mapping:
            self.label_emotion(mapping[k])
        elif k.lower() == "c":
            self.start_calibration()
        elif k.lower() == "s":
            self.save_model_clicked()
        elif event.keysym == "Escape":
            self.on_close()

    def start_calibration(self):
        self.calibrating = True
        self.calib_samples = []
        self.calib_start = time.time()
        self.baseline = None
        self.emotion_hist.clear()
        self.last_x = None
        self.last_d = None
        self.status_var.set("Calibrating… Hold a neutral face for ~2 seconds.")

    def save_model_clicked(self):
        if self.model is None:
            self.status_var.set("No model to save yet. Add some labels first.")
            return
        save_model(self.model, self.model_initialized)
        self.status_var.set(f"Saved model to: {MODEL_PATH}")

    def open_data_folder(self):
        # Opens in File Explorer on Windows
        try:
            os.startfile(DATA_DIR)  # type: ignore[attr-defined]
        except Exception:
            self.status_var.set(f"Data folder: {DATA_DIR}")

    def label_emotion(self, lbl: str):
        if self.baseline is None or self.last_x is None or self.last_d is None:
            self.status_var.set("Can’t label yet: wait for face + calibration.")
            return

        # Save labeled sample
        append_csv_row(LABELED_PATH, [
            now_iso(), lbl,
            f"{self.last_d[0]:+.6f}", f"{self.last_d[1]:+.6f}", f"{self.last_d[2]:+.6f}", f"{self.last_d[3]:+.6f}"
        ])

        # Online update
        self.model, self.model_initialized = model_online_update(self.model, self.last_x, lbl, self.model_initialized)
        self.status_var.set(f"Labeled as {lbl} and trained on it. (Saved to fer_labeled_samples.csv)")

    def draw_overlay(self, frame_bgr, pts):
        # Points
        draw_points = [
            # Eyes + eyelids
            "left_eye_top", "left_eye_bottom", "left_upper_lid", "left_lower_lid",
            "right_eye_top", "right_eye_bottom", "right_upper_lid", "right_lower_lid",

            # Brows
            "left_brow", "right_brow",

            # Nose + nostrils
            "nose_bridge", "nose_tip", "nose_left", "nose_right",
            "nostril_left", "nostril_right",

            # Mouth / lips
            "mouth_left", "mouth_right", "mouth_top", "mouth_bottom",
            "upper_lip_left", "upper_lip_center", "upper_lip_right",
            "lower_lip_left", "lower_lip_center", "lower_lip_right",

            # Cheeks / smile lines
            "left_cheek", "right_cheek",
            "left_smile_line", "right_smile_line",

            # Jaw / chin
            "jaw_left", "jaw_left_mid", "chin", "jaw_right_mid", "jaw_right",

            # Face oval anchors
            "face_top", "face_left_top", "face_left_bottom",
            "face_right_top", "face_right_bottom",
        ]

        for key in draw_points:
            x0, y0 = pts[IDX[key]].astype(int)
            cv2.circle(frame_bgr, (x0, y0), 2, (0, 255, 0), -1)

        # Contours
        jaw_chain = ["jaw_left", "jaw_left_mid", "chin", "jaw_right_mid", "jaw_right"]
        for a, b in zip(jaw_chain, jaw_chain[1:]):
            ax, ay = pts[IDX[a]].astype(int)
            bx, by = pts[IDX[b]].astype(int)
            cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 255, 0), 1)

        lip_chain = [
            "upper_lip_left", "upper_lip_center", "upper_lip_right",
            "lower_lip_right", "lower_lip_center", "lower_lip_left",
            "upper_lip_left",
        ]
        for a, b in zip(lip_chain, lip_chain[1:]):
            ax, ay = pts[IDX[a]].astype(int)
            bx, by = pts[IDX[b]].astype(int)
            cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 255, 0), 1)

        oval_chain = [
            "face_top", "face_left_top", "face_left_bottom",
            "jaw_left", "chin", "jaw_right",
            "face_right_bottom", "face_right_top", "face_top",
        ]
        for a, b in zip(oval_chain, oval_chain[1:]):
            ax, ay = pts[IDX[a]].astype(int)
            bx, by = pts[IDX[b]].astype(int)
            cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 255, 0), 1)

    def update_frame(self):
        if not self.running:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Camera read failed.")
            self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        face_present = bool(res.multi_face_landmarks)

        # Auto calibration once if baseline missing
        if self.baseline is None and self.auto_calib_armed and face_present and not self.calibrating:
            self.calibrating = True
            self.calib_samples = []
            self.calib_start = time.time()
            self.auto_calib_armed = False
            self.status_var.set("Auto-calibrating… hold neutral for ~2 seconds.")

        pred_label = None
        conf = None

        if face_present:
            h, w = frame.shape[:2]
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

            feats = extract_features(pts)

            # Calibration sampling
            if self.calibrating:
                self.calib_samples.append(feats)
                if (time.time() - self.calib_start) >= self.CALIB_SECONDS:
                    self.baseline = mean_dict(self.calib_samples, FEAT_KEYS)
                    save_baseline(self.baseline)
                    self.calibrating = False
                    self.calib_samples = []
                    self.emotion_hist.clear()
                    self.status_var.set("Calibrated ✅ (baseline saved)")

            # Smooth features
            for k in FEAT_KEYS:
                self.smooth[k] = ema(self.smooth[k], feats[k], alpha=0.25)
            s_feats = {**feats, **{k: self.smooth[k] for k in FEAT_KEYS if self.smooth[k] is not None}}

            if self.baseline is not None:
                d = baseline_deltas(s_feats, self.baseline)
                self.last_d = d
                x = d.reshape(1, -1)
                self.last_x = x

                pred_label, conf = model_predict(self.model, x)
                if pred_label is not None:
                    pretty = LABEL_TO_PRETTY.get(pred_label, pred_label)
                    self.emotion_hist.append(pretty)
                    stable = max(set(self.emotion_hist), key=self.emotion_hist.count)
                    self.pred_var.set(f"Prediction: {stable}")
                else:
                    self.pred_var.set("Prediction: (untrained) label with buttons/keys")

                if conf is not None:
                    self.conf_var.set(f"Confidence: {conf:.2f}")
                else:
                    self.conf_var.set("Confidence: —")

                # Log predictions periodically
                t = time.time()
                if pred_label is not None and (t - self.last_log_time) >= self.LOG_INTERVAL_SEC:
                    append_csv_row(PRED_LOG_PATH, [
                        now_iso(), pred_label, f"{conf:.3f}" if conf is not None else "",
                        f"{d[0]:+.6f}", f"{d[1]:+.6f}", f"{d[2]:+.6f}", f"{d[3]:+.6f}"
                    ])
                    self.last_log_time = t
            else:
                self.pred_var.set("Prediction: waiting for calibration…")
                self.conf_var.set("Confidence: —")

            # Overlay landmarks
            self.draw_overlay(frame, pts)

        else:
            self.pred_var.set("Prediction: (no face detected)")
            self.conf_var.set("Confidence: —")
            if self.baseline is None and not self.calibrating:
                self.status_var.set("Show your face to the camera to calibrate.")

        # Draw a small HUD text
        cv2.putText(frame, APP_NAME, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Esc to quit | C calibrate | S save model | 1-7 label", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Convert frame to Tk image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_frame)

    def on_close(self):
        self.running = False
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    # A slightly nicer default size
    root.geometry("1100x700")
    # Use ttk theme if available
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    app = FERApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
