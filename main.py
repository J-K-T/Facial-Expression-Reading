import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import os
import csv
import json

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

APP_NAME = "Facial-Expression-Reading"

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

KEY_TO_LABEL = {
    ord('1'): "Neutral",
    ord('2'): "Happy",
    ord('3'): "Surprised",
    ord('4'): "AngryFocused",
    ord('5'): "Sad",
    ord('6'): "Disgust",
    ord('7'): "Fear",
}

# -----------------------------
# Helpers
# -----------------------------
def dist(a, b):
    return float(np.linalg.norm(a - b))


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def ema(prev, new, alpha=0.2):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)


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
            data = json.load(f)
        # Expect keys: eye_open, mouth_open, smile, brow_raise
        return data
    except Exception:
        return None


def save_baseline(baseline: dict):
    try:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
    except Exception:
        pass


# -----------------------------
# FaceMesh landmark indices (MediaPipe)
# -----------------------------
IDX = {
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "left_eye_left": 33,
    "left_eye_right": 133,

    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "right_eye_left": 362,
    "right_eye_right": 263,

    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,

    "left_brow": 105,
    "right_brow": 334,

    "nose_tip": 1,
    "chin": 152,
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
        "face_h": face_h,
    }


def mean_dict(dicts, keys):
    arr = {k: [] for k in keys}
    for d in dicts:
        for k in keys:
            arr[k].append(d[k])
    return {k: float(np.mean(arr[k])) if arr[k] else 0.0 for k in keys}


def baseline_deltas(s_feats, baseline):
    return np.array([
        s_feats["eye_open"] - baseline["eye_open"],
        s_feats["mouth_open"] - baseline["mouth_open"],
        s_feats["smile"] - baseline["smile"],
        s_feats["brow_raise"] - baseline["brow_raise"],
    ], dtype=np.float32)


def pretty_label(lbl):
    return {
        "Neutral": "Neutral 😐",
        "Happy": "Happy 🙂",
        "Surprised": "Surprised 😮",
        "AngryFocused": "Angry/Focused 😠",
        "Sad": "Sad 😔",
        "Disgust": "Disgust 🤢",
        "Fear": "Fear 😨",
    }.get(lbl, lbl)


# -----------------------------
# Model utilities
# -----------------------------
def make_model():
    clf = SGDClassifier(loss="log_loss", alpha=0.0005, random_state=42)
    return make_pipeline(StandardScaler(with_mean=True, with_std=True), clf)


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
    if not initialized:
        clf.partial_fit(x, np.array([y]), classes=CLASSES)
        initialized = True
    else:
        clf.partial_fit(x, np.array([y]))
    return model, initialized


def save_model(model, initialized):
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
    """Load X,y from LABELED_PATH; X is (n,4), y is (n,)"""
    if not os.path.exists(LABELED_PATH):
        return None, None

    X = []
    y = []
    try:
        with open(LABELED_PATH, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                lbl = row.get("true_label", "")
                if lbl not in CLASSES:
                    continue
                try:
                    d_eye = float(row["d_eye"])
                    d_mouth = float(row["d_mouth"])
                    d_smile = float(row["d_smile"])
                    d_brow = float(row["d_brow"])
                except Exception:
                    continue
                X.append([d_eye, d_mouth, d_smile, d_brow])
                y.append(lbl)
    except Exception:
        return None, None

    if not X:
        return None, None

    return np.array(X, dtype=np.float32), np.array(y, dtype=object)


def retrain_on_startup():
    """
    Auto-retrain model from fer_labeled_samples.csv if available.
    Returns (model, initialized, n_samples_used).
    """
    X, y = load_labeled_dataset()
    if X is None or y is None:
        # Nothing to train on
        model, initialized = load_model()
        return model, initialized, 0

    # Train fresh model (batch-like using partial_fit in chunks)
    model = make_model()
    clf = model.steps[-1][1]

    # We need at least one partial_fit call with classes.
    # Fit scaler once on whole X
    scaler = model.steps[0][1]
    scaler.fit(X)
    Xs = scaler.transform(X)

    # Initialize and then train
    clf.partial_fit(Xs[:1], y[:1], classes=CLASSES)
    if len(Xs) > 1:
        clf.partial_fit(Xs[1:], y[1:])

    initialized = True
    # Save the trained model so next start is instant
    save_model(model, initialized)
    return model, initialized, int(len(X))


# -----------------------------
# Main
# -----------------------------
def main():
    # Create CSVs in the data folder
    ensure_csv_header(PRED_LOG_PATH, ["timestamp", "pred_label", "confidence", "d_eye", "d_mouth", "d_smile", "d_brow"])
    ensure_csv_header(LABELED_PATH, ["timestamp", "true_label", "d_eye", "d_mouth", "d_smile", "d_brow"])

    # Auto-retrain model from labeled dataset
    model, model_initialized, trained_n = retrain_on_startup()

    # Load baseline if you have one
    baseline = load_baseline()

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different camera index (0, 1, 2...).")

    # Smoothing + stability
    smooth = {k: None for k in FEAT_KEYS}
    emotion_hist = deque(maxlen=12)

    # Calibration (auto if baseline missing)
    calibrating = False
    calib_samples = []
    calib_start = 0.0
    CALIB_SECONDS = 2.0
    auto_calib_armed = (baseline is None)

    # Logging throttle
    LOG_INTERVAL_SEC = 0.5
    last_log_time = 0.0

    # Last vector for labeling
    last_x = None
    last_d = None

    last_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        h, w = frame.shape[:2]
        face_present = bool(res.multi_face_landmarks)

        # Auto-start calibration when face appears (if baseline missing)
        if baseline is None and auto_calib_armed and face_present and not calibrating:
            calibrating = True
            calib_samples = []
            calib_start = time.time()
            auto_calib_armed = False

        status_lines = [
            f"Data folder: {DATA_DIR}",
            f"Startup auto-train samples: {trained_n}",
            "Keys: 1 N 2 H 3 S 4 Angry 5 Sad 6 Disgust 7 Fear | c recal | s save model | q/ESC quit",
        ]

        if baseline is None and not calibrating:
            status_lines.append("Auto-calibration: show neutral face…")
        elif baseline is not None and not calibrating:
            status_lines.append("Calibrated ✅ (press 'c' to recalibrate)")

        if calibrating:
            remaining = max(0.0, CALIB_SECONDS - (time.time() - calib_start))
            status_lines.append(f"Calibrating... hold neutral ({remaining:.1f}s)")

        stable_guess = "Waiting for face…"
        conf_text = ""

        if face_present:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

            feats = extract_features(pts)

            # Calibration sampling
            if calibrating:
                calib_samples.append(feats)
                if (time.time() - calib_start) >= CALIB_SECONDS:
                    baseline = mean_dict(calib_samples, FEAT_KEYS)
                    save_baseline(baseline)
                    calibrating = False
                    calib_samples = []
                    emotion_hist.clear()

            # Smooth features
            for k in FEAT_KEYS:
                smooth[k] = ema(smooth[k], feats[k], alpha=0.25)

            s_feats = {**feats, **{k: smooth[k] for k in FEAT_KEYS if smooth[k] is not None}}

            if baseline is not None:
                d = baseline_deltas(s_feats, baseline)
                last_d = d
                x = d.reshape(1, -1)
                last_x = x

                pred, conf = model_predict(model, x)
                if pred is not None:
                    guess = pretty_label(pred)
                    if conf is not None:
                        conf_text = f"ML confidence: {conf:.2f}"
                else:
                    guess = "Untrained ML (label with keys) 😐"

                emotion_hist.append(guess)
                stable_guess = max(set(emotion_hist), key=emotion_hist.count)

                # Log predictions periodically (if model gives a label)
                t = time.time()
                if pred is not None and (t - last_log_time) >= LOG_INTERVAL_SEC:
                    append_csv_row(PRED_LOG_PATH, [
                        now_iso(), pred, f"{conf:.3f}" if conf is not None else "",
                        f"{d[0]:+.6f}", f"{d[1]:+.6f}", f"{d[2]:+.6f}", f"{d[3]:+.6f}"
                    ])
                    last_log_time = t
            else:
                stable_guess = "Waiting for calibration…"

            # Draw a few landmarks
            for key in ["mouth_left", "mouth_right", "mouth_top", "mouth_bottom",
                        "left_eye_top", "left_eye_bottom", "right_eye_top", "right_eye_bottom",
                        "left_brow", "right_brow"]:
                x0, y0 = pts[IDX[key]].astype(int)
                cv2.circle(frame, (x0, y0), 2, (0, 255, 0), -1)

            # Overlay
            y0 = 30
            cv2.putText(frame, APP_NAME, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            y0 += 30
            cv2.putText(frame, f"Emotion guess: {stable_guess}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            y0 += 26
            if conf_text:
                cv2.putText(frame, conf_text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # Status
        sy = frame.shape[0] - 110
        for line in status_lines:
            cv2.putText(frame, line, (10, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            sy += 20

        # FPS
        nowt = time.time()
        dt = nowt - last_time
        last_time = nowt
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(APP_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key in (27, ord('q')):
            break

        # Recalibrate baseline
        if key == ord('c'):
            calibrating = True
            calib_samples = []
            calib_start = time.time()
            baseline = None
            emotion_hist.clear()
            last_x = None
            last_d = None

        # Save model manually (optional, but handy)
        if key == ord('s'):
            if model is not None:
                save_model(model, model_initialized)

        # Labeling: write to dataset + online update immediately
        if key in KEY_TO_LABEL and last_x is not None and baseline is not None and last_d is not None:
            lbl = KEY_TO_LABEL[key]

            # Save labeled row (persists across sessions)
            append_csv_row(LABELED_PATH, [
                now_iso(), lbl,
                f"{last_d[0]:+.6f}", f"{last_d[1]:+.6f}", f"{last_d[2]:+.6f}", f"{last_d[3]:+.6f}"
            ])

            # Online update so it improves immediately this session
            model, model_initialized = model_online_update(model, last_x, lbl, model_initialized)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
