"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         FACS-Based Facial Expression & Deception Detection System           ║
║    Facial Action Coding System (FACS) · Real-Time Emotion Analysis          ║
╚══════════════════════════════════════════════════════════════════════════════╝

DISCLAIMER:
  Lie/deception detection from facial cues alone is NOT scientifically reliable
  as a standalone method. This tool is for educational/research purposes.
  It demonstrates how FACS-inspired AU analysis works and should NOT be used
  for judicial, employment, or security decisions.

Usage:
  pip install opencv-python mediapipe numpy --break-system-packages
  python facial_analysis.py

Controls:
  Q / ESC  — Quit
  S        — Save screenshot
  R        — Reset baseline
  D        — Toggle deception overlay
  H        — Toggle help overlay
"""

import cv2
import numpy as np
import time
import json
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError(
        "MediaPipe is not installed in this environment. Use Python 3.12 and "
        "install the pinned dependencies from requirements.txt."
    ) from exc


def require_mediapipe_solutions():
    if hasattr(mp, "solutions"):
        return mp.solutions

    version = getattr(mp, "__version__", "unknown")
    raise RuntimeError(
        "This script requires MediaPipe's legacy 'solutions' API "
        "(FaceMesh / drawing_utils). The installed package is mediapipe "
        f"{version} on Python {sys.version.split()[0]}, and it does not expose "
        "'mp.solutions'. Install a compatible MediaPipe release such as "
        "mediapipe==0.10.21 and reinstall the dependencies."
    )

# ──────────────────────────────────────────────────────────────────────────────
# FACS ACTION UNIT DATABASE
# ──────────────────────────────────────────────────────────────────────────────

FACS_DATABASE = {
    # AU_ID: (name, muscle, description)
    "AU1":  ("Inner Brow Raise",     "Frontalis (medial)",      "Oblique pull of inner brow upward"),
    "AU2":  ("Outer Brow Raise",     "Frontalis (lateral)",     "Lateral pull of outer brow upward"),
    "AU4":  ("Brow Lowerer",         "Corrugator supercilii",   "Draws brows down and together"),
    "AU5":  ("Upper Lid Raiser",     "Levator palpebrae",       "Raises upper eyelid, widens eye"),
    "AU6":  ("Cheek Raiser",         "Orbicularis oculi",       "Pushes cheeks up — genuine smile marker"),
    "AU7":  ("Lid Tightener",        "Orbicularis oculi",       "Tightens lower lid, narrows eye"),
    "AU9":  ("Nose Wrinkler",        "Levator labii superioris","Wrinkles nose bridge — disgust"),
    "AU10": ("Upper Lip Raiser",     "Levator labii superioris","Raises upper lip"),
    "AU12": ("Lip Corner Puller",    "Zygomaticus major",       "Pulls lip corners up and back"),
    "AU13": ("Cheek Puffer",         "Levator anguli oris",     "Raises cheek, creates dimples"),
    "AU14": ("Dimpler",              "Buccinator",              "Pulls corners inward — contempt marker"),
    "AU15": ("Lip Corner Depressor", "Depressor anguli oris",   "Pulls corners down — sadness"),
    "AU16": ("Lower Lip Depressor",  "Depressor labii",         "Pulls lower lip down"),
    "AU17": ("Chin Raiser",          "Mentalis",                "Pushes chin skin up — doubt/sadness"),
    "AU20": ("Lip Stretcher",        "Risorius",                "Pulls lips wide — fear"),
    "AU23": ("Lip Tightener",        "Orbicularis oris",        "Tightens lip corners — anger"),
    "AU24": ("Lip Pressor",          "Orbicularis oris",        "Presses lips together"),
    "AU25": ("Lips Part",            "Depressor labii",         "Separates lips"),
    "AU26": ("Jaw Drop",             "Masseter relaxation",     "Drops jaw — surprise/fear"),
    "AU28": ("Lip Suck",             "Incisivii labii",         "Sucks lips inward"),
    "AU41": ("Lid Droop",            "Relaxed orbicularis",     "Upper lid falls — fatigue/sadness"),
    "AU42": ("Slit",                 "Orbicularis oculi",       "Narrows eye aperture"),
    "AU43": ("Eyes Closed",          "Orbicularis oculi",       "Gently closes eyes"),
    "AU45": ("Blink",                "Orbicularis oculi",       "Rapid eye closure"),
    "AU46": ("Wink",                 "Orbicularis oculi",       "Unilateral blink"),
}

EMOTION_RULES = {
    "😊 Happy":     {"required": ["AU12"], "supporting": ["AU6", "AU25"],       "inhibiting": ["AU4", "AU15"], "threshold": 0.45},
    "😢 Sad":       {"required": ["AU4"],  "supporting": ["AU1", "AU15", "AU17"],"inhibiting": ["AU12"],        "threshold": 0.35},
    "😠 Angry":     {"required": ["AU4"],  "supporting": ["AU5", "AU7", "AU23", "AU24"], "inhibiting": ["AU12", "AU25"], "threshold": 0.40},
    "😱 Surprised": {"required": ["AU5"],  "supporting": ["AU1", "AU2", "AU25", "AU26"], "inhibiting": ["AU4"],  "threshold": 0.40},
    "😨 Fearful":   {"required": ["AU20"], "supporting": ["AU1", "AU2", "AU4", "AU5", "AU26"], "inhibiting": [], "threshold": 0.38},
    "🤢 Disgusted": {"required": ["AU9"],  "supporting": ["AU15", "AU16", "AU25"],"inhibiting": ["AU12"],        "threshold": 0.35},
    "😏 Contempt":  {"required": ["AU14"], "supporting": ["AU12"],              "inhibiting": [],               "threshold": 0.40},
    "😐 Neutral":   {"required": [],       "supporting": [],                    "inhibiting": [],               "threshold": 0.0},
}

DECEPTION_INDICATORS = {
    "Asymmetric Expression": {
        "description": "Genuine emotions are symmetric; deceptive ones are often lopsided",
        "weight": 0.25,
    },
    "Forced/Masked Smile": {
        "description": "Smile (AU12) without Duchenne marker (AU6) — mouth smiles but eyes don't",
        "weight": 0.30,
    },
    "Microexpression Leak": {
        "description": "Brief (<250ms) involuntary expression followed by suppression",
        "weight": 0.40,
    },
    "Emotion–Eye Incongruence": {
        "description": "Emotion expressed in lower face doesn't match upper face / eye region",
        "weight": 0.35,
    },
    "Rapid Neutralization": {
        "description": "Expression disappears too quickly (< 500ms) — not natural fade",
        "weight": 0.30,
    },
    "High Blink Rate": {
        "description": "Significantly elevated blink frequency — cognitive load / stress",
        "weight": 0.20,
    },
    "Gaze Aversion": {
        "description": "Sustained gaze away during key moments",
        "weight": 0.15,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES
# ──────────────────────────────────────────────────────────────────────────────

# Eyes
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eyebrows
LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LEFT_BROW_INNER  = [55, 65]
RIGHT_BROW_INNER = [285, 295]
LEFT_BROW_OUTER  = [46]
RIGHT_BROW_OUTER = [276]

# Mouth
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
               409, 270, 269, 267, 0, 37, 39, 40, 185]
MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
               324, 318, 402, 317, 14, 87, 178, 88, 95]
LIP_LEFT_CORNER  = 61
LIP_RIGHT_CORNER = 291
LIP_TOP_CENTER   = 13
LIP_BOTTOM_CENTER = 14
LIP_TOP_OUTER    = 0
LIP_BOTTOM_OUTER = 17

# Nose
NOSE_TIP     = 1
NOSE_BRIDGE  = 168
NOSE_LEFT    = 129
NOSE_RIGHT   = 358

# Jaw / Face
JAW_LINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
            288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
            150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
CHIN_TIP = 152
FOREHEAD = 10
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
             288, 397, 365, 379, 378, 400, 377, 152]

# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionUnit:
    id: str
    intensity: float = 0.0     # 0–1 normalized
    active: bool = False
    history: deque = field(default_factory=lambda: deque(maxlen=30))

@dataclass
class FaceMetrics:
    # Brow
    left_brow_height:  float = 0.0
    right_brow_height: float = 0.0
    brow_furrow_dist:  float = 0.0
    brow_asymmetry:    float = 0.0
    # Eyes
    left_ear:          float = 0.0   # Eye Aspect Ratio
    right_ear:         float = 0.0
    eye_asymmetry:     float = 0.0
    blink_rate:        float = 0.0
    # Mouth
    mouth_open:        float = 0.0
    smile_left:        float = 0.0
    smile_right:       float = 0.0
    mouth_width:       float = 0.0
    mouth_asymmetry:   float = 0.0
    lip_compression:   float = 0.0
    # Nose
    nose_wrinkle:      float = 0.0
    # Cheek
    cheek_raise:       float = 0.0
    chin_raise:        float = 0.0
    jaw_drop:          float = 0.0
    # Overall
    face_width:        float = 1.0   # normalization reference


# ──────────────────────────────────────────────────────────────────────────────
# CORE ANALYZER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class FACSSAnalyzer:
    def __init__(self):
        mp_solutions = require_mediapipe_solutions()
        self.mp_face_mesh = mp_solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp_solutions.drawing_utils
        self.mp_drawing_styles = mp_solutions.drawing_styles

        # Action Units
        self.aus: Dict[str, ActionUnit] = {k: ActionUnit(id=k) for k in FACS_DATABASE}

        # Calibration baseline
        self.baseline: Optional[FaceMetrics] = None
        self.calibrating = True
        self.calib_frames: List[FaceMetrics] = []
        self.calib_needed = 30

        # History for temporal analysis
        self.emotion_history: deque = deque(maxlen=90)          # 3s at 30fps
        self.expression_history: deque = deque(maxlen=15)       # 0.5s window
        self.blink_times: deque = deque(maxlen=30)
        self.was_eye_closed = False

        # Deception scores
        self.deception_scores: Dict[str, float] = {k: 0.0 for k in DECEPTION_INDICATORS}
        self.deception_history: deque = deque(maxlen=60)

        # UI state
        self.show_deception = True
        self.show_help = False
        self.show_mesh = True
        self.fps_timer = time.time()
        self.fps = 0.0
        self.frame_count = 0
        self.current_emotion = "😐 Neutral"
        self.current_confidence = 0.0

        # Color palette
        self.colors = {
            "bg":       (15,  15,  20),
            "panel":    (25,  28,  35),
            "accent":   (0,   200, 150),
            "warn":     (0,   140, 255),
            "danger":   (60,  60,  220),
            "text":     (220, 220, 230),
            "dim":      (100, 105, 115),
            "happy":    (50,  220, 120),
            "sad":      (200, 100,  50),
            "angry":    (60,   60, 230),
            "surprise": (50,  200, 255),
            "fear":     (180,  80, 220),
            "disgust":  (80,  180,  60),
            "contempt": (200, 150,  50),
            "neutral":  (150, 150, 160),
        }

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _lm(self, lms, idx) -> np.ndarray:
        """Get landmark as numpy array [x, y, z]."""
        l = lms.landmark[idx]
        return np.array([l.x, l.y, l.z])

    def _dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _eye_aspect_ratio(self, lms, eye_indices) -> float:
        """EAR = (vertical distances) / (2 * horizontal distance)."""
        pts = [self._lm(lms, i) for i in eye_indices]
        v1 = self._dist(pts[1], pts[5])
        v2 = self._dist(pts[2], pts[4])
        h  = self._dist(pts[0], pts[3])
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def _midpoint(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b) / 2.0

    # ── Metrics extraction ────────────────────────────────────────────────────

    def extract_metrics(self, lms, frame_shape) -> FaceMetrics:
        m = FaceMetrics()
        h, w = frame_shape[:2]

        def lm2d(idx):
            l = lms.landmark[idx]
            return np.array([l.x * w, l.y * h])

        # Face width for normalization
        left_face  = lm2d(234)
        right_face = lm2d(454)
        m.face_width = max(self._dist(left_face, right_face), 1e-6)
        norm = m.face_width

        # ── Eyes ──
        m.left_ear  = self._eye_aspect_ratio(lms, LEFT_EYE)
        m.right_ear = self._eye_aspect_ratio(lms, RIGHT_EYE)
        m.eye_asymmetry = abs(m.left_ear - m.right_ear) / max(m.left_ear + m.right_ear, 1e-6)

        # ── Brows ──
        # Height = distance from brow center to eye center, normalized
        left_brow_pts  = np.array([lm2d(i) for i in LEFT_BROW])
        right_brow_pts = np.array([lm2d(i) for i in RIGHT_BROW])
        left_eye_pts   = np.array([lm2d(i) for i in LEFT_EYE])
        right_eye_pts  = np.array([lm2d(i) for i in RIGHT_EYE])

        left_brow_center  = left_brow_pts.mean(axis=0)
        right_brow_center = right_brow_pts.mean(axis=0)
        left_eye_center   = left_eye_pts.mean(axis=0)
        right_eye_center  = right_eye_pts.mean(axis=0)

        m.left_brow_height  = (left_eye_center[1]  - left_brow_center[1])  / norm
        m.right_brow_height = (right_eye_center[1] - right_brow_center[1]) / norm
        m.brow_asymmetry = abs(m.left_brow_height - m.right_brow_height) / max(m.left_brow_height + m.right_brow_height, 1e-6)

        # Furrow = inner brow distance
        left_inner  = lm2d(LEFT_BROW_INNER[0])
        right_inner = lm2d(RIGHT_BROW_INNER[0])
        m.brow_furrow_dist = self._dist(left_inner, right_inner) / norm

        # ── Mouth ──
        top_lip    = lm2d(LIP_TOP_CENTER)
        bot_lip    = lm2d(LIP_BOTTOM_CENTER)
        left_corn  = lm2d(LIP_LEFT_CORNER)
        right_corn = lm2d(LIP_RIGHT_CORNER)
        mouth_mid  = self._midpoint(left_corn, right_corn)

        m.mouth_open   = self._dist(top_lip, bot_lip) / norm
        m.mouth_width  = self._dist(left_corn, right_corn) / norm

        # Smile = how much corners are raised relative to mouth center
        m.smile_left  = (mouth_mid[1] - left_corn[1])  / norm
        m.smile_right = (mouth_mid[1] - right_corn[1]) / norm
        m.mouth_asymmetry = abs(m.smile_left - m.smile_right) / max(abs(m.smile_left) + abs(m.smile_right), 1e-6)

        # Lip compression = thinness of lips
        top_outer = lm2d(LIP_TOP_OUTER)
        bot_outer = lm2d(LIP_BOTTOM_OUTER)
        m.lip_compression = 1.0 - min(self._dist(top_outer, bot_outer) / (norm * 0.12), 1.0)

        # ── Jaw ──
        chin    = lm2d(CHIN_TIP)
        nose_tip = lm2d(NOSE_TIP)
        m.jaw_drop = self._dist(nose_tip, chin) / norm

        # ── Nose ──
        nose_l = lm2d(NOSE_LEFT)
        nose_r = lm2d(NOSE_RIGHT)
        nose_b = lm2d(NOSE_BRIDGE)
        nose_mid = self._midpoint(nose_l, nose_r)
        m.nose_wrinkle = max(0.0, (lm2d(NOSE_BRIDGE)[1] - nose_mid[1]) / norm * 10)

        # ── Cheek raise ──
        # Approximate: compare cheek point height vs neutral
        cheek_l = lm2d(50)
        cheek_r = lm2d(280)
        mouth_y = mouth_mid[1]
        m.cheek_raise = max(0.0, (mouth_y - (cheek_l[1] + cheek_r[1]) / 2) / norm)

        # ── Chin raise ──
        lower_lip = lm2d(LIP_BOTTOM_OUTER)
        m.chin_raise = max(0.0, (lower_lip[1] - chin[1]) / norm)

        return m

    # ── Action Unit computation ────────────────────────────────────────────────

    def compute_aus(self, m: FaceMetrics):
        """Map FaceMetrics → Action Unit intensities using baseline-relative values."""
        b = self.baseline if self.baseline else FaceMetrics()

        def delta(current, baseline, scale=1.0):
            return max(0.0, (current - baseline) * scale)

        def rdelta(current, baseline, scale=1.0):
            """Reverse delta — active when LESS than baseline."""
            return max(0.0, (baseline - current) * scale)

        def sigmoid(x, center=0.5, steepness=8):
            return 1.0 / (1.0 + np.exp(-steepness * (x - center)))

        # AU1 — Inner Brow Raise (brows up, mainly medial)
        au1 = delta(m.left_brow_height + m.right_brow_height,
                    b.left_brow_height + b.right_brow_height, scale=3.0)
        self.aus["AU1"].intensity = min(1.0, au1)

        # AU2 — Outer Brow Raise
        self.aus["AU2"].intensity = self.aus["AU1"].intensity * 0.85  # correlates

        # AU4 — Brow Lowerer (furrow narrowing + brow drop)
        au4 = rdelta(m.brow_furrow_dist, b.brow_furrow_dist, scale=4.0)
        au4 += rdelta(m.left_brow_height + m.right_brow_height,
                      b.left_brow_height + b.right_brow_height, scale=2.0)
        self.aus["AU4"].intensity = min(1.0, au4 / 2.0)

        # AU5 — Upper Lid Raiser (high EAR)
        au5 = delta(m.left_ear + m.right_ear, b.left_ear + b.right_ear, scale=4.0)
        self.aus["AU5"].intensity = min(1.0, au5)

        # AU6 — Cheek Raiser (genuine smile marker)
        au6 = delta(m.cheek_raise, b.cheek_raise, scale=5.0)
        self.aus["AU6"].intensity = min(1.0, au6)

        # AU7 — Lid Tightener (EAR LOWER than baseline)
        au7 = rdelta(m.left_ear + m.right_ear, b.left_ear + b.right_ear, scale=3.0)
        self.aus["AU7"].intensity = min(1.0, au7)

        # AU9 — Nose Wrinkler
        au9 = delta(m.nose_wrinkle, b.nose_wrinkle, scale=2.0)
        self.aus["AU9"].intensity = min(1.0, au9)

        # AU12 — Lip Corner Puller (smile)
        au12_l = delta(m.smile_left,  b.smile_left,  scale=8.0)
        au12_r = delta(m.smile_right, b.smile_right, scale=8.0)
        self.aus["AU12"].intensity = min(1.0, (au12_l + au12_r) / 2.0)

        # AU14 — Dimpler / Contempt (asymmetric corner pull inward)
        au14 = m.mouth_asymmetry * 2.0
        self.aus["AU14"].intensity = min(1.0, au14)

        # AU15 — Lip Corner Depressor
        au15_l = rdelta(m.smile_left,  b.smile_left,  scale=8.0)
        au15_r = rdelta(m.smile_right, b.smile_right, scale=8.0)
        self.aus["AU15"].intensity = min(1.0, (au15_l + au15_r) / 2.0)

        # AU17 — Chin Raiser
        au17 = delta(m.chin_raise, b.chin_raise, scale=5.0)
        self.aus["AU17"].intensity = min(1.0, au17)

        # AU20 — Lip Stretcher (wide mouth, fear grimace)
        au20 = delta(m.mouth_width, b.mouth_width, scale=4.0)
        self.aus["AU20"].intensity = min(1.0, au20)

        # AU23/AU24 — Lip Tightener / Pressor
        self.aus["AU23"].intensity = min(1.0, m.lip_compression * 1.2)
        self.aus["AU24"].intensity = self.aus["AU23"].intensity * 0.8

        # AU25 — Lips Part
        au25 = delta(m.mouth_open, b.mouth_open, scale=5.0)
        self.aus["AU25"].intensity = min(1.0, au25)

        # AU26 — Jaw Drop
        au26 = delta(m.jaw_drop, b.jaw_drop, scale=3.0)
        self.aus["AU26"].intensity = min(1.0, au26)

        # AU41/42/43 — Eye closure variants
        avg_ear = (m.left_ear + m.right_ear) / 2.0
        if avg_ear < 0.15:
            self.aus["AU43"].intensity = 1.0
        elif avg_ear < 0.22:
            self.aus["AU41"].intensity = rdelta(avg_ear, 0.28, scale=8.0)
        else:
            self.aus["AU41"].intensity = 0.0
            self.aus["AU43"].intensity = 0.0

        # AU45 — Blink detection
        avg_ear = (m.left_ear + m.right_ear) / 2.0
        if avg_ear < 0.20 and not self.was_eye_closed:
            self.blink_times.append(time.time())
            self.was_eye_closed = True
        elif avg_ear >= 0.22:
            self.was_eye_closed = False

        # Update active flags (threshold 0.25)
        for au in self.aus.values():
            au.history.append(au.intensity)
            au.active = au.intensity > 0.25

    # ── Emotion classification ────────────────────────────────────────────────

    def classify_emotion(self) -> Tuple[str, float, Dict[str, float]]:
        scores: Dict[str, float] = {}

        for emotion, rule in EMOTION_RULES.items():
            if emotion == "😐 Neutral":
                continue
            score = 0.0
            req_met = True

            for au_id in rule["required"]:
                intensity = self.aus.get(au_id, ActionUnit(id=au_id)).intensity
                if intensity < 0.2:
                    req_met = False
                    break
                score += intensity * 0.4

            if not req_met:
                scores[emotion] = 0.0
                continue

            for au_id in rule["supporting"]:
                score += self.aus.get(au_id, ActionUnit(id=au_id)).intensity * 0.2

            for au_id in rule["inhibiting"]:
                score -= self.aus.get(au_id, ActionUnit(id=au_id)).intensity * 0.3

            scores[emotion] = max(0.0, score)

        if not scores or max(scores.values()) < 0.15:
            return "😐 Neutral", 1.0, scores

        best = max(scores, key=scores.get)
        conf = min(1.0, scores[best] / (EMOTION_RULES[best]["threshold"] * 2))
        return best, conf, scores

    # ── Deception analysis ────────────────────────────────────────────────────

    def analyze_deception(self, m: FaceMetrics) -> Dict[str, float]:
        scores = {}

        # 1. Expression asymmetry
        asym = (m.brow_asymmetry * 0.3 + m.mouth_asymmetry * 0.5 + m.eye_asymmetry * 0.2)
        scores["Asymmetric Expression"] = min(1.0, asym * 2.5)

        # 2. Forced/Masked Smile — AU12 without AU6
        au12 = self.aus["AU12"].intensity
        au6  = self.aus["AU6"].intensity
        if au12 > 0.3:
            duchenne_ratio = au6 / max(au12, 0.01)
            scores["Forced/Masked Smile"] = max(0.0, 1.0 - duchenne_ratio)
        else:
            scores["Forced/Masked Smile"] = 0.0

        # 3. Microexpression leak — rapid intensity spike then drop in history
        micro = 0.0
        for au in ["AU12", "AU4", "AU5", "AU20", "AU9"]:
            hist = list(self.aus[au].history)
            if len(hist) >= 10:
                peak = max(hist[-10:])
                recent = np.mean(hist[-3:]) if len(hist) >= 3 else hist[-1]
                if peak > 0.4 and recent < 0.15:
                    micro = max(micro, peak - recent)
        scores["Microexpression Leak"] = min(1.0, micro * 1.5)

        # 4. Emotion–Eye incongruence
        mouth_expr = max(self.aus["AU12"].intensity, self.aus["AU15"].intensity,
                         self.aus["AU20"].intensity, self.aus["AU25"].intensity)
        eye_expr   = max(self.aus["AU5"].intensity, self.aus["AU7"].intensity,
                         self.aus["AU43"].intensity)
        scores["Emotion–Eye Incongruence"] = min(1.0, abs(mouth_expr - eye_expr) * 1.5)

        # 5. Rapid neutralization
        rapid = 0.0
        for au in ["AU12", "AU4", "AU5"]:
            hist = list(self.aus[au].history)
            if len(hist) >= 8:
                # Was high 8 frames ago but low last 2 frames?
                if np.mean(hist[-8:-4]) > 0.35 and np.mean(hist[-2:]) < 0.1:
                    rapid = max(rapid, np.mean(hist[-8:-4]))
        scores["Rapid Neutralization"] = min(1.0, rapid * 1.2)

        # 6. High blink rate
        now = time.time()
        recent_blinks = sum(1 for t in self.blink_times if now - t < 10.0)
        norm_blink_rate = recent_blinks / 10.0  # blinks per second
        scores["High Blink Rate"] = min(1.0, max(0.0, (norm_blink_rate - 0.3) / 0.7))

        # 7. Gaze aversion (simple iris position approximation)
        scores["Gaze Aversion"] = 0.0  # Requires iris tracking (extended)

        # Smooth scores
        for k in scores:
            prev = self.deception_scores.get(k, 0.0)
            self.deception_scores[k] = 0.7 * prev + 0.3 * scores[k]

        return self.deception_scores

    def overall_deception_score(self) -> float:
        total_weight = sum(v["weight"] for v in DECEPTION_INDICATORS.values())
        score = sum(
            self.deception_scores.get(k, 0.0) * v["weight"]
            for k, v in DECEPTION_INDICATORS.items()
        )
        return min(1.0, score / total_weight * 3.0)

    # ── Calibration ───────────────────────────────────────────────────────────

    def update_calibration(self, m: FaceMetrics) -> bool:
        """Collect baseline frames. Returns True when done."""
        self.calib_frames.append(m)
        if len(self.calib_frames) >= self.calib_needed:
            # Average all fields
            b = FaceMetrics()
            n = len(self.calib_frames)
            for field_name in b.__dataclass_fields__:
                vals = [getattr(f, field_name) for f in self.calib_frames]
                setattr(b, field_name, float(np.mean(vals)))
            self.baseline = b
            self.calibrating = False
            return True
        return False

    def reset_baseline(self):
        self.calibrating = True
        self.calib_frames = []
        self.baseline = None

    # ── FPS ───────────────────────────────────────────────────────────────────

    def update_fps(self):
        self.frame_count += 1
        now = time.time()
        dt = now - self.fps_timer
        if dt >= 1.0:
            self.fps = self.frame_count / dt
            self.frame_count = 0
            self.fps_timer = now

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def draw_bar(self, img, x, y, w, h, value, color, bg_color=(40, 42, 50)):
        cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
        fill = int(w * max(0.0, min(1.0, value)))
        if fill > 0:
            cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)

    def draw_rounded_rect(self, img, x, y, w, h, r, color, thickness=-1, alpha=0.85):
        overlay = img.copy()
        cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, thickness)
        cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, thickness)
        for cx, cy in [(x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)]:
            cv2.circle(overlay, (cx, cy), r, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def draw_text(self, img, text, x, y, scale=0.5, color=None, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
        if color is None:
            color = self.colors["text"]
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # ── Face mesh overlay ─────────────────────────────────────────────────────

    def draw_face_mesh(self, img, results):
        if results.multi_face_landmarks:
            for face_lms in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_lms,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_lms,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                )

    # ── HUD rendering ─────────────────────────────────────────────────────────

    def render_hud(self, frame, m: FaceMetrics, emotion: str, conf: float,
                   all_scores: Dict[str, float], dec_scores: Dict[str, float]):

        H, W = frame.shape[:2]
        overlay = frame.copy()

        # ── Header bar ────────────────────────────────────────────────────────
        cv2.rectangle(overlay, (0, 0), (W, 40), (15, 15, 22), -1)
        self.draw_text(frame, "FACS Facial Analysis System  v1.0", 10, 26,
                       scale=0.55, color=self.colors["accent"], thickness=1)
        fps_txt = f"FPS: {self.fps:.1f}"
        self.draw_text(frame, fps_txt, W - 90, 26, scale=0.5, color=self.colors["dim"])

        # Calibrating overlay
        if self.calibrating:
            pct = len(self.calib_frames) / self.calib_needed
            self.draw_text(frame, "CALIBRATING BASELINE...", W // 2 - 120, H // 2 - 20,
                           scale=0.7, color=(80, 220, 255), thickness=2)
            prog_x = W // 2 - 120
            self.draw_bar(frame, prog_x, H // 2, 240, 10, pct, (80, 220, 255))
            self.draw_text(frame, "Hold a neutral expression", W // 2 - 100, H // 2 + 35,
                           scale=0.5, color=self.colors["dim"])
            return

        # ── Left panel — Emotion ───────────────────────────────────────────────
        px, py, pw, ph = 8, 50, 210, 200
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (20, 22, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + 1), self.colors["accent"], -1)
        cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)

        self.draw_text(frame, "DETECTED EMOTION", px + 8, py + 16,
                       scale=0.38, color=self.colors["dim"])

        # Emotion name (large)
        em_color_map = {
            "Happy": self.colors["happy"], "Sad": self.colors["sad"],
            "Angry": self.colors["angry"], "Surprised": self.colors["surprise"],
            "Fearful": self.colors["fear"], "Disgusted": self.colors["disgust"],
            "Contempt": self.colors["contempt"], "Neutral": self.colors["neutral"],
        }
        em_key = next((k for k in em_color_map if k.lower() in emotion.lower()), "Neutral")
        em_color = em_color_map[em_key]

        self.draw_text(frame, emotion, px + 8, py + 38,
                       scale=0.65, color=em_color, thickness=2)

        # Confidence bar
        self.draw_text(frame, f"Confidence: {conf*100:.0f}%", px + 8, py + 58,
                       scale=0.38, color=self.colors["dim"])
        self.draw_bar(frame, px + 8, py + 64, pw - 16, 6, conf, em_color)

        # Top emotion scores
        self.draw_text(frame, "ALL EMOTIONS", px + 8, py + 88,
                       scale=0.36, color=self.colors["dim"])
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (em, sc) in enumerate(sorted_scores[:6]):
            ey = py + 102 + i * 16
            em_short = em.split(" ", 1)[1] if " " in em else em
            bar_color = em_color_map.get(next((k for k in em_color_map if k.lower() in em_short.lower()), "Neutral"), self.colors["neutral"])
            self.draw_text(frame, em_short[:11], px + 8, ey + 9, scale=0.33, color=self.colors["text"])
            self.draw_bar(frame, px + 80, ey, 100, 8, min(1.0, sc * 1.5), bar_color)
            pct_txt = f"{sc*100:.0f}%"
            self.draw_text(frame, pct_txt, px + 184, ey + 8, scale=0.3, color=self.colors["dim"])

        # ── Right panel — Active Action Units ─────────────────────────────────
        rx = W - 220
        ry, rw, rh = 50, 212, 320
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (20, 22, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + 1), self.colors["accent"], -1)
        cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)

        self.draw_text(frame, "ACTIVE ACTION UNITS (FACS)", rx + 8, ry + 16,
                       scale=0.38, color=self.colors["dim"])

        active_aus = [(k, v) for k, v in self.aus.items() if v.intensity > 0.15]
        active_aus.sort(key=lambda x: x[1].intensity, reverse=True)

        for i, (au_id, au) in enumerate(active_aus[:14]):
            ay = ry + 26 + i * 21
            if ay > ry + rh - 8:
                break
            name = FACS_DATABASE.get(au_id, (au_id, "", ""))[0][:20]
            intensity_color = (
                (100, 220, 100) if au.intensity < 0.4 else
                (100, 200, 255) if au.intensity < 0.7 else
                (80,  80,  255)
            )
            self.draw_text(frame, au_id, rx + 8, ay + 12,
                           scale=0.38, color=intensity_color, thickness=1)
            self.draw_text(frame, name, rx + 44, ay + 12,
                           scale=0.33, color=self.colors["text"])
            self.draw_bar(frame, rx + 152, ay + 4, 50, 8, au.intensity, intensity_color)
            self.draw_text(frame, f"{au.intensity:.2f}", rx + 206, ay + 12,
                           scale=0.3, color=self.colors["dim"])

        if not active_aus:
            self.draw_text(frame, "No AUs above threshold", rx + 8, ry + 50,
                           scale=0.38, color=self.colors["dim"])

        # ── Bottom panel — Deception Indicators ───────────────────────────────
        if self.show_deception:
            dec_score = self.overall_deception_score()
            bx, by = 8, H - 185
            bw, bh = 310, 178

            overlay = frame.copy()
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (22, 18, 28), -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
            overlay = frame.copy()
            border_col = (60, 60, 220) if dec_score > 0.5 else (60, 180, 220) if dec_score > 0.25 else self.colors["accent"]
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + 1), border_col, -1)
            cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)

            self.draw_text(frame, "DECEPTION INDICATOR ANALYSIS", bx + 8, by + 16,
                           scale=0.38, color=self.colors["dim"])

            # Overall score meter
            score_label = (
                "HIGH RISK" if dec_score > 0.6 else
                "MODERATE"  if dec_score > 0.35 else
                "LOW"
            )
            score_color = (
                (60, 60, 220) if dec_score > 0.6 else
                (60, 140, 255) if dec_score > 0.35 else
                (50, 200, 100)
            )
            self.draw_text(frame, f"Overall: {score_label} ({dec_score*100:.0f}%)",
                           bx + 8, by + 34, scale=0.45, color=score_color, thickness=1)
            self.draw_bar(frame, bx + 8, by + 40, bw - 16, 8, dec_score, score_color)

            disclaimer_y = by + 58
            self.draw_text(frame, "* Research tool only - not a lie detector *",
                           bx + 8, disclaimer_y, scale=0.31, color=(80, 80, 100))

            for i, (ind, info) in enumerate(DECEPTION_INDICATORS.items()):
                dy = by + 70 + i * 16
                if dy > by + bh - 8:
                    break
                sc = dec_scores.get(ind, 0.0)
                ind_color = (
                    (60, 60, 220) if sc > 0.6 else
                    (60, 140, 255) if sc > 0.3 else
                    self.colors["dim"]
                )
                self.draw_text(frame, ind[:28], bx + 8, dy + 10,
                               scale=0.31, color=self.colors["text"])
                self.draw_bar(frame, bx + 195, dy + 2, 80, 8, sc, ind_color)
                self.draw_text(frame, f"{sc*100:.0f}%", bx + 280, dy + 10,
                               scale=0.30, color=ind_color)

        # ── Bottom-right — Blink rate / face metrics ───────────────────────────
        mrx = W - 220
        mry = H - 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (mrx, mry), (mrx + 212, mry + 102), (20, 22, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        now = time.time()
        recent_blinks = sum(1 for t in self.blink_times if now - t < 10.0)
        blink_rate = recent_blinks / 10.0

        metrics_display = [
            ("L-EAR", m.left_ear, 0.0, 0.4),
            ("R-EAR", m.right_ear, 0.0, 0.4),
            ("Brow H", (m.left_brow_height + m.right_brow_height) / 2, 0.05, 0.25),
            ("Smile", (m.smile_left + m.smile_right) / 2 * 20, 0.0, 1.0),
            ("Blink/s", blink_rate, 0.0, 0.8),
        ]
        self.draw_text(frame, "FACE METRICS", mrx + 8, mry + 14,
                       scale=0.36, color=self.colors["dim"])
        for i, (label, val, lo, hi) in enumerate(metrics_display):
            my = mry + 24 + i * 16
            norm_val = (val - lo) / max(hi - lo, 1e-6)
            self.draw_text(frame, f"{label}:", mrx + 8, my + 10,
                           scale=0.32, color=self.colors["text"])
            self.draw_bar(frame, mrx + 62, my + 2, 80, 8, min(1.0, max(0.0, norm_val)),
                          self.colors["accent"])
            self.draw_text(frame, f"{val:.3f}", mrx + 148, my + 10,
                           scale=0.30, color=self.colors["dim"])

        # ── Key hints ─────────────────────────────────────────────────────────
        hints = "[R] Reset  [D] Deception  [M] Mesh  [S] Save  [Q] Quit"
        self.draw_text(frame, hints, 8, H - 8, scale=0.35, color=self.colors["dim"])

    def render_help(self, frame):
        H, W = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (W // 4, H // 6), (3 * W // 4, 5 * H // 6), (15, 15, 22), -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        hx, hy = W // 4 + 12, H // 6 + 22
        self.draw_text(frame, "FACS ANALYZER — HELP", hx, hy, scale=0.65,
                       color=self.colors["accent"], thickness=2)
        lines = [
            ("Q / ESC", "Quit application"),
            ("S",       "Save screenshot"),
            ("R",       "Reset calibration baseline"),
            ("D",       "Toggle deception panel"),
            ("M",       "Toggle face mesh overlay"),
            ("H",       "Toggle this help screen"),
            ("", ""),
            ("About FACS:", "Facial Action Coding System maps 46 AUs"),
            ("",          "to specific muscle movements, enabling"),
            ("",          "objective emotion classification."),
            ("", ""),
            ("Deception:", "7 indicators derived from asymmetry,"),
            ("",          "microexpressions & temporal patterns."),
            ("",          "NOT a lie detector — research only."),
        ]
        for i, (k, v) in enumerate(lines):
            ly = hy + 28 + i * 20
            if k:
                self.draw_text(frame, k, hx, ly, scale=0.42, color=self.colors["warn"])
                self.draw_text(frame, v, hx + 110, ly, scale=0.38, color=self.colors["text"])
            else:
                self.draw_text(frame, v, hx, ly, scale=0.35, color=self.colors["dim"])


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(__doc__)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check your camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {W}x{H}")

    analyzer = FACSSAnalyzer()
    screenshot_dir = os.path.expanduser("~/facs_screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    window_name = "FACS Facial Analysis System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, W, H)

    # State
    show_mesh = True

    print("[INFO] Starting. Hold a neutral expression for baseline calibration...")
    print("[INFO] Controls: Q=Quit, R=Reset baseline, D=Deception panel, M=Mesh, H=Help, S=Save")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed — retrying...")
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)  # Mirror
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = analyzer.face_mesh.process(rgb)
        analyzer.update_fps()

        # Dark background tint
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        if results.multi_face_landmarks:
            face_lms = results.multi_face_landmarks[0]

            if show_mesh:
                analyzer.draw_face_mesh(frame, results)

            m = analyzer.extract_metrics(face_lms, frame.shape)

            if analyzer.calibrating:
                done = analyzer.update_calibration(m)
                analyzer.render_hud(frame, m, "😐 Neutral", 1.0, {}, {})
            else:
                analyzer.compute_aus(m)
                emotion, conf, all_scores = analyzer.classify_emotion()
                analyzer.current_emotion    = emotion
                analyzer.current_confidence = conf
                dec_scores = analyzer.analyze_deception(m)
                analyzer.render_hud(frame, m, emotion, conf, all_scores, dec_scores)

        else:
            # No face detected
            cy, cx = H // 2, W // 2
            self_text = "No face detected — center your face in frame"
            tw, _ = cv2.getTextSize(self_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(frame, self_text, (cx - tw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 100), 1, cv2.LINE_AA)

        if analyzer.show_help:
            analyzer.render_help(frame)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("[INFO] Exiting...")
            break
        elif key in (ord('r'), ord('R')):
            analyzer.reset_baseline()
            print("[INFO] Resetting baseline...")
        elif key in (ord('d'), ord('D')):
            analyzer.show_deception = not analyzer.show_deception
            print(f"[INFO] Deception panel: {'ON' if analyzer.show_deception else 'OFF'}")
        elif key in (ord('m'), ord('M')):
            show_mesh = not show_mesh
            print(f"[INFO] Face mesh: {'ON' if show_mesh else 'OFF'}")
        elif key in (ord('h'), ord('H')):
            analyzer.show_help = not analyzer.show_help
        elif key in (ord('s'), ord('S')):
            fname = os.path.join(screenshot_dir, f"facs_{int(time.time())}.png")
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
