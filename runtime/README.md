# CouchMo Runtime

Laptop-side inference surface. Loads an `.onnx` model exported by `training/`,
captures stereo frames from two USB webcams, and drives the IRL robot over a
USB serial link (see `CouchMo/serial_controller.py` for the protocol).

- **Runs on:** Windows laptop (no GPU required), macOS / Linux supported
- **Dependencies:** `onnxruntime`, `opencv-python`, `pyserial`, `numpy`
- **Deliberately does NOT require:** PyTorch, ROS, Docker, or CUDA

All machine-specific values (serial port, camera indices, model path) are
passed in as CLI args — **nothing is hardcoded**.

## Install

### Windows laptop (PowerShell)

```powershell
cd runtime
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux (bash or zsh)

```bash
cd runtime
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Discover your serial (COM) port

### Windows (PowerShell, no extra deps)

```powershell
Get-WmiObject Win32_SerialPort | Select-Object Name, DeviceID, Description
```

### Cross-platform (inside the venv)

```bash
python -m serial.tools.list_ports
```

Use the resulting device id (e.g. `COM5` on Windows, `/dev/tty.usbserial-1410`
on macOS, `/dev/ttyUSB0` on Linux) as the `--port` argument to the runtime
scripts.

## Discover your camera indices

Plug in both USB webcams, then run:

```python
import cv2
for i in range(6):
    cap = cv2.VideoCapture(i)
    ok, _ = cap.read()
    print(f"index {i}: {'opened' if ok else 'no signal'}")
    cap.release()
```

The two indices that return `opened` are your left/right eyes. Pass them via
`--left-cam` and `--right-cam`.

## Sanity check (no hardware required)

Run the policy against a recorded episode shard to confirm the install is
sound. This reads the shard, runs every frame through the ONNX policy, and
prints `predicted` vs `recorded` actions side by side. No serial is opened,
no cameras are touched:

```bash
python -m runtime.drive --source replay \
    --episode samples/tiny_episode.npz \
    --model checkpoints/bc.onnx
```

If this exits 0 and prints one `step NNN  predicted=(..) recorded=(..)` line
per frame, your laptop is ready to drive. Plug in the cameras and ESP32, then
switch to `--source live` with the appropriate `--left-cam`, `--right-cam`,
and `--port` values discovered above.
