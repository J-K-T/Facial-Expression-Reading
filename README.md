# FACS Facial Expression & Deception Analysis System

A real-time facial analysis tool inspired by the **Facial Action Coding System (FACS)**
developed by Paul Ekman and Wallace Friesen.

---

## What It Does

| Feature | Detail |
|---|---|
| **Face Mesh** | 468 landmarks via MediaPipe Face Mesh |
| **Action Units** | 24 FACS AUs computed from landmark geometry |
| **Emotions** | 7 basic emotions + Neutral (Ekman's universals + Contempt) |
| **Deception Indicators** | 7 behavioral cues derived from temporal/asymmetry analysis |
| **Baseline Calibration** | Personalized neutral-expression baseline for relative measurement |

---

## Installation

Use Python 3.12 for this project and pin MediaPipe to `0.10.21`. The script
depends on the legacy MediaPipe `solutions` API (`FaceMesh`), and MediaPipe
`0.10.31+` no longer exposes `mp.solutions`.

```bash
pip install opencv-python mediapipe numpy
# or if using system Python:
pip install opencv-python mediapipe numpy --break-system-packages
```

## Running

```bash
python facial_analysis.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `R` | Reset calibration baseline |
| `D` | Toggle deception indicator panel |
| `M` | Toggle face mesh overlay |
| `S` | Save screenshot (saved to `~/facs_screenshots/`) |
| `H` | Toggle help overlay |

---

## FACS Action Units Tracked

| AU | Name | Primary Emotion |
|----|------|-----------------|
| AU1 | Inner Brow Raise | Sadness, Fear |
| AU2 | Outer Brow Raise | Surprise |
| AU4 | Brow Lowerer | Anger, Sadness, Fear |
| AU5 | Upper Lid Raiser | Surprise, Fear |
| AU6 | Cheek Raiser (Duchenne) | Genuine Happiness |
| AU7 | Lid Tightener | Anger, Disgust |
| AU9 | Nose Wrinkler | Disgust |
| AU12 | Lip Corner Puller | Happiness |
| AU14 | Dimpler | Contempt |
| AU15 | Lip Corner Depressor | Sadness |
| AU17 | Chin Raiser | Sadness, Disgust |
| AU20 | Lip Stretcher | Fear |
| AU23 | Lip Tightener | Anger |
| AU24 | Lip Pressor | Anger |
| AU25 | Lips Part | Various |
| AU26 | Jaw Drop | Surprise, Fear |
| AU41-43 | Eye closure variants | Fatigue, Sadness |

---

## Deception Indicators

> ⚠️ **IMPORTANT DISCLAIMER**: These are behavioral *correlates* studied in deception research.
> No facial analysis system is a reliable lie detector. This is for **educational and research
> purposes only** and should never be used for legal, employment, or security decisions.

| Indicator | Basis |
|-----------|-------|
| Asymmetric Expression | Genuine emotions are bilateral; deceptive ones often unilateral |
| Forced/Masked Smile | Smile (AU12) without Duchenne marker (AU6) — "mouth only" smile |
| Microexpression Leak | Brief involuntary expression flash < 250ms before suppression |
| Emotion–Eye Incongruence | Lower and upper face express different things |
| Rapid Neutralization | Expression disappears unnaturally fast (< 500ms) |
| High Blink Rate | Elevated blink frequency — cognitive load / stress signal |
| Gaze Aversion | Sustained look-away (requires iris tracking extension) |

---

## Architecture

```
Camera Feed
    │
    ▼
MediaPipe Face Mesh (468 landmarks)
    │
    ▼
FaceMetrics extraction (brow height, EAR, smile vectors, jaw drop, etc.)
    │
    ├──► Baseline calibration (30 neutral frames)
    │
    ▼
Action Unit computation (delta from baseline, sigmoid-scaled)
    │
    ├──► Emotion classification (rule-based AU combinations)
    │
    └──► Deception analysis (asymmetry + temporal pattern analysis)
         │
         ▼
    HUD rendering (OpenCV overlay panels)
```

---

## References

- Ekman, P. & Friesen, W.V. (1978). *Facial Action Coding System*
- Ekman, P. (2003). *Emotions Revealed*
- Lucey et al. (2010). The Extended Cohn-Kanade Dataset (CK+)
- MediaPipe Face Mesh: https://mediapipe.dev/solutions/face_mesh
