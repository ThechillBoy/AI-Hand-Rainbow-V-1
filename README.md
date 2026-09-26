# Neon Rainbow Laser Hands

Real-time webcam experiment that tracks both of your hands with MediaPipe and
draws an animated neon-rainbow laser beam — plus glowing spark particles —
between your two index fingertips.

Everything except the machine-learning inference is rendered with raw OpenGL
(GLFW + PyOpenGL) as a shader-lit overlay on the live camera feed, so the beam,
the glow and the sparks all composite on top of the video in a single window.

---

## Features

- **Two-hand tracking** with MediaPipe's `HandLandmarker` Tasks API (up to 2 hands).
- **Live webcam background** drawn as a full-screen OpenGL texture, mirrored for a natural feel.
- **Animated neon rainbow beam** between the two index fingertips, built from a GPU quad and
  coloured by a fragment shader with a hot white plasma core, a soft halo, and a hue that
  scrolls along the beam over time.
- **Glowing spark particles** scattered along the beam with additive blending, per-particle
  rainbow colour, gentle gravity, and a size/alpha fade as they die.
- **Fingertip emission bursts** at each index fingertip that mark where the beam originates.
- **Beam pulse** — a subtle sine-wave width breathing effect.
- **Live HUD** showing FPS, the number of hands currently detected, and the key bindings.
- **Adjustable beam width** and a particle on/off toggle at runtime.
- VSync-enabled window sized to the actual camera resolution.

---

## Requirements / Prerequisites

| Requirement | Notes |
| --- | --- |
| **Windows** (or Linux/macOS) | Run instructions below are written for Windows. |
| **Python 3** | A 64-bit CPython install on your `PATH` as `python`. Verified against Python 3.13 on Windows. |
| **A webcam** | Opened as camera index `0`. |
| **A GPU/driver with OpenGL 3.3 Core Profile** | Required by the shaders the script compiles. |
| **Internet connection on first run** | Only to download the ~8 MB MediaPipe hand model. |

Python packages required (see `requirements.txt`):

```
opencv-python
mediapipe
glfw
PyOpenGL
PyOpenGL_accelerate
numpy
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ThechillBoy/AI-Hand-Rainbow.git
cd AI-Hand-Rainbow
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it — **Windows Command Prompt**:

```cmd
.venv\Scripts\activate
```

Activation prompt (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your prompt. To leave the environment later,
run `deactivate`.

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the script (Windows)

```bash
python "neon_laser_hands.py"
```

A window titled **Neon Rainbow Laser Hands** opens showing your mirrored webcam feed.
Hold both hands up to the camera so both index fingertips are visible — the rainbow beam
appears between them.

> The file name is quoted in the command so it works regardless of the folder it is run
> from; always run the command from the repository root, i.e. the folder that contains
> `neon_laser_hands.py`.

---

## Controls

| Key | Action |
| --- | --- |
| `ESC` or `Q` | Quit |
| `B` | Increase beam width |
| `N` | Decrease beam width |
| `P` | Toggle particle sparks on / off |

The same bindings are printed in the on-screen HUD.

---

## First run: the hand-tracking model

The script needs MediaPipe's hand-landmark model bundle. On first run it checks for a file
named `hand_landmarker.task` next to `neon_laser_hands.py` and, if it is missing, downloads
it automatically (about 8 MB, one time only) from:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

You will see:

```
Hand-tracking model not found locally - downloading (~8 MB, one-time)...
Model saved to: <path>\hand_landmarker.task
```

After that, every later run uses the local copy and needs no download.

`hand_landmarker.task` is listed in `.gitignore`, so it is never committed to the
repository. If the automatic download fails (for example, no internet or a proxy block),
the script prints the URL above — download the file manually and place it in the same
folder as the script under the exact name `hand_landmarker.task`.

---

## Troubleshooting

### Webcam problems

- **`Could not open webcam (index 0).`**
  - Another application (Zoom, Teams, OBS, the Camera app) is probably using the webcam.
    Close it and try again.
  - The script always uses camera index `0`. If you have several cameras, that may not be
    the one you expect — check it in Windows *Settings → Bluetooth & devices → Cameras*.
  - Reinstall/repair the camera driver in Device Manager.
- **`Could not read a frame from the webcam.`**
  - The device opened but produced no image. Close other apps using it, or try a different
    USB port (prefer USB 3.0 ports for higher-resolution webcams).
- **The feed is mirrored.** This is intentional — `cv2.flip(frame, 1)` is applied so moving
  your hand right moves the beam right.
- **The window is not resizable.** The script creates a fixed-size window matched to the
  camera's actual resolution; it prints `Camera resolution: <w>x<h>` at startup.

### OpenGL / window problems

- **`Failed to initialize GLFW.`**
  - GLFW could not start. Make sure you are not running inside a headless/remote session
    without GPU access, and that your graphics drivers are up to date.
- **`Failed to create GLFW window.`**
  - Your GPU or driver does not expose an **OpenGL 3.3 Core Profile** context, which the
    three shader programs in this project require (`#version 330 core`). Update the GPU
    driver (NVIDIA / AMD / Intel) and reboot. Old or virtual/remote-display drivers often
    cap OpenGL at 2.1.
- **`Shader compile error: ...` / `Program link error: ...`**
  - Reported by the script when a shader fails to build; the message contains the driver's
    compiler log. Updating the graphics driver resolves this in almost every case.
- **A black window or an immediate exit**
  - Confirm the startup log printed `Camera resolution: ...` and no `GLFW error [...]`
    lines. GLFW errors are written to stderr with the driver's own description.

### Dependency problems

- **`ModuleNotFoundError: No module named 'cv2'` / `'mediapipe'` / `'glfw'` / `'OpenGL'` / `'numpy'`**
  - The virtual environment is not active, or the packages were installed into a different
    Python. Activate `.venv` and re-run `pip install -r requirements.txt`, then start the
    script with the same interpreter: `python -m pip list` should show the packages.
- **`pip` installs into the wrong Python**
  - Always install with `python -m pip install -r requirements.txt` so the currently active
    interpreter's pip is used.
- **`mediapipe` fails to install**
  - MediaPipe ships prebuilt wheels only for specific Python versions and platforms. Use a
    supported 64-bit CPython version (see *Requirements*), and upgrade pip first with
    `python -m pip install --upgrade pip`.
- **`PyOpenGL_accelerate` fails to build or import**
  - It is an optional accelerator. If it causes trouble, remove that single line from
    `requirements.txt` and reinstall — the script still runs, just slightly slower.
- **Antivirus or corporate proxy blocks the model download**
  - Download `hand_landmarker.task` manually from the URL in the *First run* section and put
    it beside the script.

---

## Project structure

```
AI-Hand-Rainbow/
├── neon_laser_hands.py      # Main script: tracking, OpenGL rendering, main loop
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignores venvs, caches and the downloaded model
├── README.md                # This file
└── hand_landmarker.task     # MediaPipe model - auto-downloaded, NOT committed
```

`hand_landmarker.task` is generated on first run and excluded from version control.

Inside `neon_laser_hands.py` the code is organised as:

| Section | Purpose |
| --- | --- |
| Configuration | Camera resolution, particle counts, beam width, model URL/path |
| `ensure_model()` | Downloads the hand model if it is missing |
| Shader sources | GLSL for the background, beam and particle passes |
| `BackgroundRenderer` | Draws the webcam frame as a full-screen textured quad |
| `BeamRenderer` | Builds and draws the beam quad between the fingertips |
| `ParticleSystem` | CPU-side spark simulation streamed to a point-sprite VBO |
| `HandTracker` | MediaPipe `HandLandmarker` wrapper returning NDC fingertip positions |
| `main()` | Window/camera setup, key callbacks and the render loop |

---

## How it works

1. OpenCV grabs a frame from the webcam and converts it BGR → RGB.
2. MediaPipe `HandLandmarker` runs in video mode and returns up to two sets of hand
   landmarks; landmark index `8` (index fingertip) of each hand is converted from
   normalised image coordinates to OpenGL NDC space.
3. When at least two fingertips are detected, a quad is built along that segment and the
   beam fragment shader paints the scrolling rainbow, white core and halo.
4. Spark particles are spawned along the segment and at each fingertip, integrated with
   simple gravity, and drawn as additive point sprites.
5. The webcam texture is drawn first without blending, then the beam and particles are
   drawn with `GL_SRC_ALPHA, GL_ONE` for an additive glow, and the buffers are swapped.

---

## License

This repository currently contains no license file. Add one before redistributing.
