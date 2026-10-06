# FacePong

> 🕹️ Pong game controlled by your face using computer vision and adaptive AI. Built for Raspberry Pi.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204%20%2F%205%20%7C%20Linux%20%7C%20macOS%20%7C%20Windows-orange.svg)]()
[![Research Lab](https://img.shields.io/badge/Research%20Center-AudacIA-cyan.svg)](https://audacia.unisimon.edu.co/)

**FacePong** is an interactive, computer-vision-powered arcade experience designed for public demonstration and interactive exhibition at **AudacIA** (Robotics and Artificial Intelligence Research Center at Universidad Simón Bolívar, Barranquilla, Colombia).

The player navigates their paddle simply by tilting and moving their head in front of a camera. Opposite them is an **Adaptive AI agent** that dynamically gauges the player's skill level and game score, balancing reaction velocity and shot prediction in real time so matches always remain exhilarating, close, and accessible to visitors of all ages.

---

## Key Features

- **Hands-Free Facial Landmark Control**: Real-time facial landmark tracking using MediaPipe Face Mesh. Tracks nasal bridge movements with an Exponential Moving Average (EMA) filter to eradicate camera sensor jitter.
- **Dynamic Difficulty Adjustment (DDA)**: The AI calculates ball trajectories with multi-bounce raycasting, injecting human-like reaction latency and intentional targeting deviations based on the live score difference.
- **Autonomous Exhibition Flow**: Complete unattended kiosk state machine (`ATTRACT` &rarr; `CALIBRATION` &rarr; `MATCH` &rarr; `GAME OVER`). Includes an **Inactivity Watchdog**: if a player steps away mid-game, the system automatically aborts the match and returns to attract mode.
- **Retro-Futuristic Neon Aesthetic**: Glowing paddles, comet trails, burst particle physics on collisions and goals, cybernetic picture-in-picture (PIP) camera HUD, and real-time facial wireframe overlay.
- **Zero-Asset Procedural Audio**: Integrated real-time audio synthesizer using NumPy wave generation. Runs out-of-the-box without requiring external audio asset files or download mirrors.
- **Dual Control & Robust Fallback**: Instant fallback to keyboard controls (`W`/`S` or `UP`/`DOWN`) if no face is detected or if running without a webcam. Also includes an OpenCV Haar Cascade fallback if MediaPipe is not supported on a specific embedded OS.
- **Optimized for Embedded Devices**: Fully decoupled multithreaded architecture separating the 60 FPS Pygame rendering loop from the camera capture and inference pipeline. Runs smoothly on Raspberry Pi 4 and Raspberry Pi 5.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Webcam / Camera Module    |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Threaded Camera Worker    |  (Zero video buffer lag)
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Face Mesh / Landmark Engine |  (MediaPipe + Haar fallback)
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |   EMA Filter & Calibrator   |  (Jitter dampening)
                      +-----------------------------+
                                     |  (Thread-Safe Tracking State)
                                     v
+-------------------------------------------------------------------------+
|                              MAIN THREAD                                |
|                                                                         |
|   +-------------------+    +--------------------+    +--------------+   |
|   |  State Machine    |--->| Physics & Entities |--->|  Adaptive AI |   |
|   |  (Presence Watch) |    |  (Ball, Paddles)   |    | (Trajectory) |   |
|   +-------------------+    +--------------------+    +--------------+   |
|                                     |                                   |
|                                     v                                   |
|                        +------------------------+                       |
|                        |  Neon Arcade Renderer  | (60 FPS Display)      |
|                        | (HUD, PIP, Particles)  |                       |
|                        +------------------------+                       |
+-------------------------------------------------------------------------+
```

---

## Hardware & System Requirements

### Recommended Hardware
- **Single-Board Computer**: Raspberry Pi 4 (4GB/8GB) or Raspberry Pi 5.
- **Display**: Any HDMI monitor or TV (720p / 1080p).
- **Camera**: Standard USB webcam (Logitech C270, C920, etc.) or Raspberry Pi Camera Module v2 / v3.
- **Audio**: HDMI audio output or 3.5mm analog audio jack.

*Note: FacePong also runs identically on standard PC hardware (Linux, macOS, and Windows).*

### Software Prerequisites
- Python 3.10 or 3.11
- Raspberry Pi OS (64-bit Bookworm recommended) or Ubuntu Linux

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AudacIA/FacePong.git
cd FacePong
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Launching the Game
To run FacePong with standard default parameters:
```bash
python3 main.py
```

### Exhibition Kiosk Mode (Full Screen)
For public display booths or kiosks on a Raspberry Pi:
```bash
python3 main.py --fullscreen
```

### Command-Line Options
```
usage: main.py [-h] [--width WIDTH] [--height HEIGHT] [--fullscreen]
               [--camera CAMERA] [--cam-width CAM_WIDTH]
               [--cam-height CAM_HEIGHT] [--alpha ALPHA]
               [--timeout TIMEOUT] [--win-score WIN_SCORE]
               [--no-sound] [--debug] [--log-level {DEBUG,INFO,WARNING,ERROR}]

optional arguments:
  -h, --help            Show this help message and exit
  --width WIDTH         Display resolution width (default: 1280)
  --height HEIGHT       Display resolution height (default: 720)
  --fullscreen          Launch in fullscreen mode for exhibitions and kiosks
  --camera CAMERA       Webcam device index (default: 0)
  --cam-width CAM_WIDTH Camera capture width for CV pipeline (default: 320)
  --cam-height CAM_HEIGHT
                        Camera capture height for CV pipeline (default: 240)
  --alpha ALPHA         EMA smoothing factor (0.05-0.5, default: 0.22)
  --timeout TIMEOUT     Player absence watchdog timeout in seconds (default: 5.0)
  --win-score WIN_SCORE Points required to win a match (default: 5)
  --no-sound            Disable procedural audio synthesis
  --debug               Display real-time diagnostic overlay (FPS, AI telemetry)
  --log-level           Logging verbosity (INFO, DEBUG, WARNING, ERROR)
```

---

## Controls & Mechanics

| Input | Action |
| :--- | :--- |
| **Head Movement (Up / Down)** | Controls the left player paddle via facial tracking |
| **W / S** or **Up / Down Arrows** | Keyboard paddle control override / manual fallback |
| **Spacebar** | Quick start from Attract mode or restart from Game Over |
| **D** | Toggle real-time debug telemetry overlay |
| **Escape** | Gracefully quit the application |

### Exhibition Flow
1. **Attract Mode (IDLE)**: Prominently displayed arcade title, cyber visuals, and pulsing instructions inviting visitors to step forward.
2. **Presence Detection & Calibration**: When a player steps in front of the camera, the system detects their face and begins a 3-second neutral position calibration.
3. **Competitive Match**: Fast-paced Pong match (first to 5 points by default). The dynamic AI constantly modulates its difficulty to maintain a thrilling, close scoreline.
4. **Presence Watchdog**: If the player walks away during a match, a brief 5-second countdown triggers before resetting the station back to Attract mode.
5. **Game Over & Return**: Match statistics (duration, rally records, scores) are shown for 7 seconds before cycling back to welcome the next player.

---

## Raspberry Pi Optimization Tips

To achieve the best possible performance on a Raspberry Pi 4 or 5:

1. **Lower CV Resolution**: The computer vision capture defaults to `320x240`. This is the sweet spot for MediaPipe Face Mesh inference, preserving high FPS while keeping CPU usage low, while the Pygame display renders at crisp 720p.
2. **GPU Memory Allocation**: Ensure your Raspberry Pi has at least 128 MB allocated to the GPU:
   ```bash
   sudo raspi-config
   # Performance Options -> GPU Memory -> 128
   ```
3. **Kiosk Auto-Start (Optional)**: To launch FacePong automatically on boot on Raspberry Pi OS:
   ```bash
   mkdir -p ~/.config/autostart
   cat << 'EOF' > ~/.config/autostart/facepong.desktop
   [Desktop Entry]
   Type=Application
   Name=FacePong
   Exec=/bin/bash -c "source /home/pi/FacePong/.venv/bin/activate && python3 /home/pi/FacePong/main.py --fullscreen"
   Terminal=false
   EOF
   ```

---

## Running Automated Tests

FacePong includes an automated test suite covering game physics, trajectory prediction, dynamic difficulty adjustment, and presence watchdog behavior:

```bash
python3 -m unittest discover tests
```

---

## Project Information

- **Developer**: Andrés
- **Institution**: AudacIA &mdash; Centro de Investigación en Robótica e Inteligencia Artificial
- **University**: Universidad Simón Bolívar, Barranquilla, Colombia
- **License**: [Apache 2.0](LICENSE)
