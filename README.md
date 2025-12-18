# Facial-Expression-Reading

Simple webcam demo that uses MediaPipe FaceMesh landmarks plus heuristic, FACS-inspired rules to visualize likely emotions and a coarse deception/stress score. The estimates are only illustrative—never use them for medical, security, or hiring decisions.

## Setup
1. Create/activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run
Start the camera loop (press `q` to quit):
```bash
python main.py
```

## Notes and limitations
- The Facial Action Coding System (FACS) mapping here is simplified and uses manually tuned thresholds.
- Deception estimation is speculative and should not be treated as reliable; it is intended only to highlight potential stress/incongruence cues.
- Always obtain informed consent before recording or analyzing anyone's face.
