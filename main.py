import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

APP_NAME = "Facial-Expression-Reading"

# -----------------------------
# Helpers
# -----------------------------
def dist(a, b):
    return float(np.linalg.norm(a - b))


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def ema(prev, new, alpha=0.2):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)


# -----------------------------
# FaceMesh landmark indices (MediaPipe)
# -----------------------------
IDX = {
    # Eyes (approx)
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "left_eye_left": 33,
    "left_eye_right": 133,

    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "right_eye_left": 362,
    "right_eye_right": 263,

    # Mouth
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,

    # Brows (approx)
    "left_brow": 105,
    "right_brow": 334,

    # Nose / face reference
    "nose_tip": 1,
    "chin": 152,
}


# -----------------------------
# Feature extraction (FACS-inspired, not true AU detection)
# -----------------------------
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


# -----------------------------
# Emotion guess using baseline deltas
# -----------------------------
def guess_emotion(features, baseline):
    eye = features["eye_open"]
    mouth = features["mouth_open"]
    smile = features["smile"]
    brow = features["brow_raise"]

    if baseline:
        d_eye = eye - baseline["eye_open"]
        d_mouth = mouth - baseline["mouth_open"]
        d_smile = smile - baseline["smile"]
        d_brow = brow - baseline["brow_raise"]

        # Conservative starter thresholds (deltas)
        if d_smile > 0.03 and mouth < baseline["mouth_open"] + 0.10:
            return "Happy 🙂"
        if d_mouth > 0.12 and d_eye > 0.03 and d_brow > 0.02:
            return "Surprised 😮"
        if d_brow < -0.015 and d_mouth < 0.06:
            return "Angry/Focused 😠"
        return "Neutral 😐"

    # Fallback if not calibrated yet
    if smile > 0.42 and mouth < 0.35:
        return "Happy 🙂"
    if mouth > 0.45 and eye > 0.28 and brow > 0.09:
        return "Surprised 😮"
    if brow < 0.07 and mouth < 0.28:
        return "Angry/Focused 😠"
    return "Neutral 🤖"


# -----------------------------
# Main
# -----------------------------
def main():
    mp_face = mp.solutions.face_mesh  # If your mediapipe build lacks this, tell me and I’ll swap imports.
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

    FEAT_KEYS = ["eye_open", "mouth_open", "smile", "brow_raise"]

    # Smoothing
    smooth = {k: None for k in FEAT_KEYS}
    emotion_hist = deque(maxlen=12)

    # Calibration state
    baseline = None
    calibrating = False
    calib_samples = []
    calib_start_time = 0.0
    CALIB_SECONDS = 2.0

    # Auto-calibration flags
    AUTO_CALIBRATE = True
    auto_calib_armed = True  # start calibrating once a face is detected

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
        status_lines = []

        # Face present?
        face_present = bool(res.multi_face_landmarks)

        # Auto-start calibration when face first appears
        if AUTO_CALIBRATE and baseline is None and auto_calib_armed and face_present and not calibrating:
            calibrating = True
            calib_samples = []
            calib_start_time = time.time()
            auto_calib_armed = False  # only auto-calibrate once

        # Status messages
        if baseline is None and not calibrating:
            status_lines.append("Auto-calibration: show neutral face to camera…")
            status_lines.append("Press 'c' anytime to calibrate manually.")
        elif baseline is not None and not calibrating:
            status_lines.append("Calibrated ✅  (press 'c' to recalibrate)")

        if calibrating:
            remaining = max(0.0, CALIB_SECONDS - (time.time() - calib_start_time))
            status_lines.append(f"Calibrating... hold neutral ({remaining:.1f}s)")

        if face_present:
            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            feats = extract_features(pts)

            # Collect calibration samples (raw)
            if calibrating:
                calib_samples.append(feats)
                if (time.time() - calib_start_time) >= CALIB_SECONDS:
                    baseline = mean_dict(calib_samples, FEAT_KEYS)
                    calibrating = False
                    calib_samples = []
                    emotion_hist.clear()

            # Smooth features
            for k in FEAT_KEYS:
                smooth[k] = ema(smooth[k], feats[k], alpha=0.25)

            s_feats = {**feats, **{k: smooth[k] for k in FEAT_KEYS if smooth[k] is not None}}
            label = guess_emotion(s_feats, baseline)
            emotion_hist.append(label)
            stable_label = max(set(emotion_hist), key=emotion_hist.count)

            # Draw key points
            for key in ["mouth_left", "mouth_right", "mouth_top", "mouth_bottom",
                        "left_eye_top", "left_eye_bottom", "right_eye_top", "right_eye_bottom",
                        "left_brow", "right_brow"]:
                x, y = pts[IDX[key]].astype(int)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Overlay
            y0 = 30
            cv2.putText(frame, APP_NAME, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            y0 += 28
            cv2.putText(frame, f"Emotion guess: {stable_label}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            y0 += 30

            if baseline:
                d_eye = s_feats["eye_open"] - baseline["eye_open"]
                d_mouth = s_feats["mouth_open"] - baseline["mouth_open"]
                d_smile = s_feats["smile"] - baseline["smile"]
                d_brow = s_feats["brow_raise"] - baseline["brow_raise"]

                cv2.putText(frame, f"Δ eye_open:   {d_eye:+.3f}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                y0 += 24
                cv2.putText(frame, f"Δ mouth_open: {d_mouth:+.3f}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                y0 += 24
                cv2.putText(frame, f"Δ smile:      {d_smile:+.3f}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                y0 += 24
                cv2.putText(frame, f"Δ brow_raise: {d_brow:+.3f}", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
                y0 += 24
            else:
                cv2.putText(frame, "Waiting for calibration…", (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y0 += 24

        # Status text
        sy = frame.shape[0] - 70
        for line in status_lines:
            cv2.putText(frame, line, (10, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            sy += 22

        # FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        cv2.putText(frame, f"FPS: {fps:.1f}   (q/ESC to quit)", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(APP_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('c'):
            calibrating = True
            calib_samples = []
            calib_start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
