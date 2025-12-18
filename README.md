 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index bdde8816bd3adc305ad9914282e78bb467ef3d81..b6bd83cbcb5f4cdb7d41dfbeef156844fe46b4da 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,21 @@
-# Facial-Expression-Reading
\ No newline at end of file
+# Facial-Expression-Reading
+
+Simple webcam demo that uses MediaPipe FaceMesh landmarks plus heuristic, FACS-inspired rules to visualize likely emotions and a coarse deception/stress score. The estimates are only illustrative—never use them for medical, security, or hiring decisions.
+
+## Setup
+1. Create/activate a virtual environment.
+2. Install dependencies:
+   ```bash
+   pip install -r requirements.txt
+   ```
+
+## Run
+Start the camera loop (press `q` to quit):
+```bash
+python main.py
+```
+
+## Notes and limitations
+- The Facial Action Coding System (FACS) mapping here is simplified and uses manually tuned thresholds.
+- Deception estimation is speculative and should not be treated as reliable; it is intended only to highlight potential stress/incongruence cues.
+- Always obtain informed consent before recording or analyzing anyone's face.
 
EOF
)
