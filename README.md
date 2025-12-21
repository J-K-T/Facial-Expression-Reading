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

## Run in Git Bash (Windows)
1. Open **Git Bash** from the Start menu.
2. Go to the project folder (adjust the path to wherever you cloned it):
   ```bash
   cd /c/Users/<your-username>/path/to/Facial-Expression-Reading
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the script:
   ```bash
   python main.py
   ```
   Press `q` in the preview window to exit.

## Run in VS Code
1. Install the **Python** extension and open this folder in VS Code.
2. Create/activate a virtual environment (e.g., `python -m venv .venv` and `source .venv/bin/activate` on macOS/Linux or `.venv\\Scripts\\activate` on Windows).
3. Select the `.venv` interpreter from the VS Code status bar.
4. Install dependencies in the built-in terminal:
   ```bash
   pip install -r requirements.txt
   ```
5. Use the **Run and Debug** panel and choose **Run webcam analyzer** (from `.vscode/launch.json`) to start the script, or run `python main.py` in the terminal. Press `q` in the preview window to exit.

## Connect VS Code to GitHub
1. Make sure Git is installed (`git --version` should work in the VS Code terminal). Install Git from https://git-scm.com if needed.
2. Open VS Code, click the **Accounts** icon (top-right) and choose **Sign in with GitHub**. Complete the browser sign-in and allow VS Code to access your GitHub account.
3. If you already cloned this repo manually, set the remote to your GitHub fork:
   ```bash
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git fetch origin
   ```
4. To clone directly from VS Code instead, use **View > Command Palette…** → **Git: Clone** and paste your repo URL. When prompted, pick a folder and open it.
5. After signing in, the **Source Control** panel will show your changes. Use it to stage, commit, and push. VS Code will reuse your GitHub sign-in for push/pull over HTTPS (no extra PAT needed unless your org requires it).
6. If you prefer SSH, set up an SSH key with GitHub and add the SSH remote URL, then choose it in **Git: Clone** or update the existing `origin` remote.

## Notes and limitations
- The Facial Action Coding System (FACS) mapping here is simplified and uses manually tuned thresholds.
- Deception estimation is speculative and should not be treated as reliable; it is intended only to highlight potential stress/incongruence cues.
- Always obtain informed consent before recording or analyzing anyone's face.
