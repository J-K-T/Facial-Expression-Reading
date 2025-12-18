 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/main.py b/main.py
new file mode 100644
index 0000000000000000000000000000000000000000..483beffbf47a720d704371e1e86af5b922172d8d
--- /dev/null
+++ b/main.py
@@ -0,0 +1,256 @@
+"""
+Webcam-based facial expression tracker with simple FACS-inspired heuristics.
+
+The script uses MediaPipe FaceMesh landmarks to estimate action unit intensities,
+infers likely emotions, and provides a rough deception/stress indicator derived
+from mismatched facial cues. It is not a medical or security tool and should not
+be used without informed consent.
+"""
+from __future__ import annotations
+
+from dataclasses import dataclass, field
+from typing import Dict, Iterable, List, Optional, Tuple
+
+import cv2
+import mediapipe as mp
+import numpy as np
+
+
+@dataclass(frozen=True)
+class ActionUnit:
+    """Minimal record describing a FACS action unit."""
+
+    code: str
+    description: str
+    related_emotions: Iterable[str] = field(default_factory=tuple)
+
+
+@dataclass
+class Inference:
+    """Container for results from one frame."""
+
+    action_units: Dict[str, float]
+    emotions: Dict[str, float]
+    deception_score: float
+
+
+class ExpressionAnalyzer:
+    """Estimate action units, emotions, and deception signals from landmarks."""
+
+    def __init__(self) -> None:
+        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
+            min_detection_confidence=0.6,
+            min_tracking_confidence=0.6,
+            refine_landmarks=True,
+        )
+        self._facs_db = self._build_facs_database()
+
+    @staticmethod
+    def _build_facs_database() -> Dict[str, ActionUnit]:
+        return {
+            "AU1": ActionUnit("AU1", "Inner brow raise", ("surprise", "fear", "sadness")),
+            "AU2": ActionUnit("AU2", "Outer brow raise", ("surprise", "fear")),
+            "AU4": ActionUnit("AU4", "Brow lowerer", ("anger", "sadness")),
+            "AU5": ActionUnit("AU5", "Upper lid raise", ("surprise", "fear")),
+            "AU12": ActionUnit("AU12", "Lip corner puller", ("happiness")),
+            "AU15": ActionUnit("AU15", "Lip corner depressor", ("sadness")),
+            "AU20": ActionUnit("AU20", "Lip stretch", ("fear")),
+            "AU23": ActionUnit("AU23", "Lip tightener", ("anger", "disgust")),
+            "AU26": ActionUnit("AU26", "Jaw drop", ("surprise", "fear")),
+        }
+
+    def analyze(self, frame: np.ndarray) -> Optional[Inference]:
+        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
+        results = self.face_mesh.process(rgb)
+        if not results.multi_face_landmarks:
+            return None
+
+        landmarks = np.array(
+            [
+                (lm.x, lm.y, lm.z)
+                for lm in results.multi_face_landmarks[0].landmark
+            ]
+        )
+        metrics = self._compute_metrics(landmarks, frame.shape)
+        action_units = self._estimate_action_units(metrics)
+        emotions = self._estimate_emotions(action_units)
+        deception_score = self._estimate_deception(metrics, action_units, emotions)
+        return Inference(action_units=action_units, emotions=emotions, deception_score=deception_score)
+
+    @staticmethod
+    def _normalized_distance(p1: np.ndarray, p2: np.ndarray, denom: float) -> float:
+        return float(np.linalg.norm(p1 - p2) / (denom + 1e-6))
+
+    def _compute_metrics(self, landmarks: np.ndarray, frame_shape: Tuple[int, int, int]) -> Dict[str, float]:
+        h, w, _ = frame_shape
+        points = landmarks[:, :2] * np.array([w, h])
+
+        mouth_top = points[13]
+        mouth_bottom = points[14]
+        mouth_left = points[61]
+        mouth_right = points[291]
+
+        brow_left_inner = points[65]
+        brow_right_inner = points[295]
+        brow_left_outer = points[52]
+        brow_right_outer = points[282]
+
+        eye_right_top = points[159]
+        eye_right_bottom = points[145]
+        eye_left_top = points[386]
+        eye_left_bottom = points[374]
+        eye_right_outer = points[33]
+        eye_right_inner = points[133]
+        eye_left_inner = points[362]
+        eye_left_outer = points[263]
+
+        nose_tip = points[1]
+        chin = points[152]
+
+        face_height = self._normalized_distance(nose_tip, chin, 1.0) + np.linalg.norm(nose_tip - chin)
+        inter_ocular = self._normalized_distance(eye_left_inner, eye_right_inner, 1.0) + np.linalg.norm(
+            eye_left_inner - eye_right_inner
+        )
+
+        mouth_open = self._normalized_distance(mouth_top, mouth_bottom, face_height)
+        mouth_width = self._normalized_distance(mouth_left, mouth_right, face_height)
+        smile_curve = (mouth_width * 0.6) + max(0.0, self._normalized_distance(mouth_left, mouth_top, face_height))
+
+        brow_raise_left = (brow_left_inner[1] - eye_left_top[1]) / (face_height + 1e-6)
+        brow_raise_right = (brow_right_inner[1] - eye_right_top[1]) / (face_height + 1e-6)
+        brow_raise = max(0.0, -0.5 * (brow_raise_left + brow_raise_right))
+        brow_lower = max(0.0, (brow_left_outer[1] + brow_right_outer[1]) / 2 - (eye_left_top[1] + eye_right_top[1]) / 2)
+        brow_lower /= (face_height + 1e-6)
+
+        eye_open = (
+            self._normalized_distance(eye_right_top, eye_right_bottom, inter_ocular)
+            + self._normalized_distance(eye_left_top, eye_left_bottom, inter_ocular)
+        ) / 2
+
+        lip_tightness = max(0.0, (mouth_width - mouth_open))
+        lip_depress = max(0.0, (mouth_bottom[1] - mouth_top[1])) / (face_height + 1e-6)
+
+        metrics = {
+            "mouth_open": float(mouth_open),
+            "mouth_width": float(mouth_width),
+            "smile_curve": float(smile_curve),
+            "brow_raise": float(brow_raise),
+            "brow_lower": float(brow_lower),
+            "eye_open": float(eye_open),
+            "lip_tightness": float(lip_tightness),
+            "lip_depress": float(lip_depress),
+        }
+        return metrics
+
+    def _estimate_action_units(self, metrics: Dict[str, float]) -> Dict[str, float]:
+        au = {
+            "AU1": self._clamp(metrics["brow_raise"] * 1.2),
+            "AU2": self._clamp(metrics["brow_raise"] * 0.9),
+            "AU4": self._clamp(metrics["brow_lower"] * 2.2),
+            "AU5": self._clamp(metrics["eye_open"] * 1.5),
+            "AU12": self._clamp(metrics["smile_curve"] * 1.8),
+            "AU15": self._clamp(metrics["lip_depress"] * 0.8),
+            "AU20": self._clamp(metrics["mouth_width"] * 1.6),
+            "AU23": self._clamp(metrics["lip_tightness"] * 1.5),
+            "AU26": self._clamp(metrics["mouth_open"] * 2.0),
+        }
+        return au
+
+    @staticmethod
+    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
+        return float(max(low, min(high, value)))
+
+    def _estimate_emotions(self, au: Dict[str, float]) -> Dict[str, float]:
+        emotions = {
+            "happiness": self._clamp(0.7 * au["AU12"] + 0.2 * (1 - au["AU23"])),
+            "surprise": self._clamp(0.5 * au["AU26"] + 0.3 * au["AU5"] + 0.2 * au["AU1"]),
+            "sadness": self._clamp(0.6 * au["AU15"] + 0.3 * au["AU1"] + 0.1 * au["AU4"]),
+            "anger": self._clamp(0.5 * au["AU4"] + 0.3 * au["AU23"]),
+            "fear": self._clamp(0.4 * au["AU26"] + 0.3 * au["AU5"] + 0.3 * au["AU2"]),
+            "disgust": self._clamp(0.6 * au["AU23"] + 0.2 * au["AU20"]),
+        }
+        return emotions
+
+    def _estimate_deception(
+        self,
+        metrics: Dict[str, float],
+        au: Dict[str, float],
+        emotions: Dict[str, float],
+    ) -> float:
+        micro_stress = max(au["AU4"], au["AU23"])
+        weak_affect = 1 - max(emotions.values())
+        conflict = abs(emotions["happiness"] - emotions["sadness"]) + abs(emotions["anger"] - emotions["fear"])
+        mouth_tension = self._clamp(metrics["lip_tightness"] * 0.8 + (1 - metrics["mouth_open"]))
+        deception_score = self._clamp(0.3 * micro_stress + 0.3 * mouth_tension + 0.2 * weak_affect + 0.2 * conflict)
+        return deception_score
+
+
+def draw_overlay(frame: np.ndarray, inference: Inference) -> None:
+    emotions_sorted = sorted(inference.emotions.items(), key=lambda kv: kv[1], reverse=True)
+    x, y = 20, 30
+    cv2.rectangle(frame, (10, 10), (320, 200), (0, 0, 0), 1)
+    cv2.putText(frame, "Top emotions", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
+    for idx, (label, score) in enumerate(emotions_sorted[:3], start=1):
+        cv2.putText(
+            frame,
+            f"{label.title():<10}: {score:.2f}",
+            (x, y + idx * 25),
+            cv2.FONT_HERSHEY_SIMPLEX,
+            0.55,
+            (0, 255, 0),
+            1,
+        )
+
+    cv2.putText(
+        frame,
+        f"Deception/Stress: {inference.deception_score:.2f}",
+        (x, y + 4 * 25),
+        cv2.FONT_HERSHEY_SIMPLEX,
+        0.55,
+        (0, 215, 255),
+        1,
+    )
+
+    cv2.putText(
+        frame,
+        "Heuristic only. Use with consent.",
+        (x, y + 6 * 25),
+        cv2.FONT_HERSHEY_SIMPLEX,
+        0.5,
+        (200, 200, 200),
+        1,
+    )
+
+
+def main() -> None:
+    print("Starting camera. Press 'q' to quit.")
+    print(
+        "This demo uses simplified FACS-inspired rules and cannot reliably detect deception.\n"
+        "Always obtain informed consent before recording video."
+    )
+    analyzer = ExpressionAnalyzer()
+    cap = cv2.VideoCapture(0)
+    if not cap.isOpened():
+        raise RuntimeError("Cannot open webcam. Ensure a camera is available and accessible.")
+
+    try:
+        while True:
+            ret, frame = cap.read()
+            if not ret:
+                break
+
+            inference = analyzer.analyze(frame)
+            if inference:
+                draw_overlay(frame, inference)
+
+            cv2.imshow("FACS-inspired expression tracker", frame)
+            key = cv2.waitKey(1) & 0xFF
+            if key == ord("q"):
+                break
+    finally:
+        cap.release()
+        cv2.destroyAllWindows()
+
+
+if __name__ == "__main__":
+    main()
 
EOF
)
