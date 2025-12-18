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

## Run in VS Code
1. Install the **Python** extension and open this folder in VS Code.
2. Create/activate a virtual environment (e.g., `python -m venv .venv` and `source .venv/bin/activate` on macOS/Linux or `.venv\\Scripts\\activate` on Windows).
3. Select the `.venv` interpreter from the VS Code status bar.
4. Install dependencies in the built-in terminal:
   ```bash
   pip install -r requirements.txt
   ```
5. Use the **Run and Debug** panel and choose **Run webcam analyzer** (from `.vscode/launch.json`) to start the script, or run `python main.py` in the terminal. Press `q` in the preview window to exit.

## Notes and limitations
- The Facial Action Coding System (FACS) mapping here is simplified and uses manually tuned thresholds.
- Deception estimation is speculative and should not be treated as reliable; it is intended only to highlight potential stress/incongruence cues.
- Always obtain informed consent before recording or analyzing anyone's face.
