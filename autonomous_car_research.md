# Autonomous Car — Architecture Research Report

**Project:** CouchMo Autonomous Campus Vehicle  
**Author:** Coding Team  
**Date:** March 2026 (updated April 2026)  
**Status:** Firmware Implementation In Progress

---

## 1. Overview

This report covers three key research areas the coding team needed to resolve before building:

1. How to handle compute, camera input, and on-chip processing
2. Whether a laptop is necessary on the car
3. How data flows between each system component
4. How to establish a serial connection between the laptop and ESP32
5. ESP32 firmware — motor control, PWM output, and manual driving via PS4 controller

---

## 2. Compute & Camera Input Architecture

### 2.1 The Core Question

The team needed to understand how camera signals come in, how the computer processes them, and whether the hardware can keep up with real-time inference at 10Hz (one decision every 100ms).

### 2.2 Camera Setup

The team has decided on two wide-angle USB webcams capturing in grayscale. Each frame is processed down to 84x84 pixels and four frames are stacked per camera, producing a final input tensor of shape (8, 84, 84) fed into the CNN.

Key specs:
- Resolution after preprocessing: 84x84 grayscale
- Frame stacking: 4 frames per camera
- Final tensor: (8, 84, 84) — 2 cams × 4 frames
- Control frequency: 10Hz (one inference cycle per 100ms)
- Exposure: locked manually to prevent lighting variation breaking the model

### 2.3 Processing Budget Per Cycle

At 10Hz the laptop has 100ms per control cycle. The actual processing breakdown is:

| Step | Time |
|---|---|
| Camera capture + resize | ~5ms |
| Grayscale + normalization | ~2ms |
| Frame stack update | ~1ms |
| CNN forward pass (CPU) | ~10–15ms |
| Serial command send | ~1ms |
| **Total** | **~20ms** |

This leaves ~80ms of headroom per cycle, meaning a mid-range Intel/AMD laptop is sufficient for inference without a GPU.

### 2.4 On-Chip Processing Considerations

Research into autonomous driving edge compute confirms that all safety-critical inference must happen onboard the vehicle — cloud processing introduces unacceptable latency for real-time control. The two viable options for the team are:

**Option A — Laptop/Mini PC (chosen)**  
A standard Intel/AMD laptop running Python handles camera capture, the OpenCV perception pipeline, CNN inference via TorchScript, and serial output to the ESP32. No GPU required for inference. Simple to set up and the team already has access to one.

**Option B — Jetson Nano/Orin (alternative)**  
A dedicated edge AI board with a built-in GPU designed for inference workloads. More power-efficient and easier to mount on a small vehicle, but adds cost ($100–$500) and setup complexity. Worth revisiting if the laptop proves too bulky or thermally unstable during long runs.

The team will proceed with Option A (laptop) and revisit Option B if hardware issues arise.

---

## 3. Do We Need the Laptop?

Yes. Something must run inference on the car — the trained model does not run in the cloud during operation. The laptop serves as the onboard compute unit responsible for:

- Reading frames from both USB webcams
- Running the perception pipeline (OpenCV, scikit-image, NumPy)
- Executing CNN + PPO inference (TorchScript)
- Sending steering and throttle commands to the ESP32 over serial

The trained model is exported from Google Colab as a TorchScript file, transferred to the laptop via Google Drive, and loaded once at startup. After that, Colab is not involved during driving.

### 3.1 Practical Concerns for Laptop on Vehicle

| Concern | Mitigation |
|---|---|
| USB webcam latency | Use USB 3.0 ports, lock exposure |
| Power supply stability | Dedicated power bank, not car battery directly |
| Overheating on long runs | Ensure ventilation, monitor CPU temp |
| Camera driver compatibility | Test OpenCV capture before full integration |

---

## 4. System Data Flow

The full data flow from sensors to wheels is as follows:

```
Left webcam  ──USB──┐
                    ├──► Laptop (perception + inference) ──USB serial (115200 baud)──► ESP32 ──PWM──► Level Shifter (3.3→5V) ──► RC Filter ──► Motor Controller
Right webcam ──USB──┘                                                                   ↑
                                                                                  ACK / ERR back
```

The ESP32 outputs 3.3 V PWM at 5 kHz on GPIO 2 (left) and GPIO 13 (right). A level shifter scales the signal to 5 V, and an RC low-pass filter smooths the PWM into a DC voltage that mimics a hall-effect throttle signal. See section 5.5 for the voltage mapping.

**Manual driving mode (current):**

```
PS4 Controller ──Bluetooth──► ESP32 (Bluepad32) ──PWM──► Level Shifter ──► RC Filter ──► Motor Controller
```

**Training flow (offline, separate from above):**

```
MetaDrive sim ──► PPO training on Colab ──► checkpoint ──► Google Drive ──► Laptop (loaded at startup)
```

---

## 5. Serial Connection — Laptop to ESP32

### 5.1 Protocol Decisions

| Decision | Choice | Reason |
|---|---|---|
| Library | PySerial | Standard Python serial library |
| Transport | UART over USB | Simple, no extra wiring needed |
| Baud rate | 115200 | Matches ESP32 firmware (`SERIAL_BAUD`) |
| Message format | CSV (`"steer,throttle\n"`) | Human-readable, easy to parse |
| Framing | Newline `\n` terminated | Easy to split on both ends |
| Return signal | `ACK\n` or `ERR\n` | Safety validation |
| Safety mechanism | Timeout watchdog | No ACK within 500ms → emergency stop |

### 5.2 Message Format

The laptop sends one command per control cycle (every 100ms):

```
"0.50,0.30\n"
 ↑     ↑
 steer throttle
```

Steering range: -1.0 (full left) to 1.0 (full right)  
Throttle range: 0.0 (stop) to 1.0 (full speed)

The ESP32 replies with:
- `ACK\n` — command received and applied successfully
- `ERR\n` — parse failure or value out of range

### 5.3 Why CSV Over Alternatives

JSON was considered but rejected — it is overkill for two float values and adds parsing overhead on the ESP32. Binary packed bytes are faster but not human-readable, making debugging difficult at this stage. CSV can be read directly in the Arduino Serial Monitor during testing, which is a significant advantage during development.

### 5.4 Responsibilities

The serial connection spans two codebases. Both sides are now implemented.

**Python side (`serial_controller.py`) — implemented:**
- `SerialController` class wraps PySerial at 115200 baud
- `send(steer, throttle)` formats `"steer,throttle\n"`, writes, reads ACK/ERR
- Filters out `[LOG]` debug lines from the ESP32 while waiting for protocol responses
- `stop()` convenience method sends `"0.0,0.0\n"`
- Context manager support (`with SerialController(...) as esc:`)
- Standalone test (`python serial_controller.py /dev/ttyUSB0`) sends a sequence of commands and prints results

**ESP32 firmware side (`Couchmo.ino`) — implemented:**
- `handleUARTInput()` reads UART until newline, splits on comma, parses two floats
- Validates steer ∈ [-1.0, 1.0] and throttle ∈ [0.0, 1.0]
- Maps to 0–255 scale, applies differential mixing via `applyMix()`
- Replies `ACK\n` on success, `ERR\n` on parse failure or out-of-range
- 500 ms UART watchdog — if no valid command arrives, throttle drops to `THROTTLE_REST`

### 5.5 ESP32 Firmware — Motor Control via PWM

The ESP32 firmware (`Couchmo.ino`) is implemented and uses the Bluepad32 library for PS4 controller input over Bluetooth. Motor speed is controlled by generating a filtered PWM signal that emulates a hall-effect throttle.

**Output chain:** ESP32 PWM (3.3 V, 5 kHz) → 3.3→5 V level shifter → RC low-pass filter → motor controller throttle input

**Pin assignment:**

| Channel | GPIO | LEDC Channel |
|---|---|---|
| Left motor | D2 | 0 |
| Right motor | D13 | 1 |

**PWM configuration:**

| Parameter | Value |
|---|---|
| Frequency | 5 kHz |
| Resolution | 8-bit (0–255 duty) |
| Level shifter output | 5 V |

**Three-tier throttle voltage model:**

The motor controller interprets a voltage range as speed. The firmware maps to three distinct voltage levels to ensure clean brake/idle behaviour:

| Constant | Duty | Voltage | Purpose |
|---|---|---|---|
| `THROTTLE_REST` | 41 | ~0.8 V | Rest / brake — below the controller's "go" threshold |
| `THROTTLE_MIN` | 56 | ~1.1 V | Minimum speed the controller recognises as movement |
| `THROTTLE_MAX` | 214 | ~4.2 V | Full speed |

When speed is 0 (brake, idle, disconnect, or boot), the output drops to `THROTTLE_REST` (0.8 V), which sits safely below the 1.1 V "go" threshold. Any non-zero speed (1–255) maps linearly across `THROTTLE_MIN` to `THROTTLE_MAX`. These values should be verified with a multimeter on the actual hardware.

**Why PWM instead of DAC:**

The ESP32 has only two DAC channels (GPIO 25 and 26) with 8-bit resolution. PWM via the LEDC peripheral offers the same resolution on any GPIO, freeing up pin choices and avoiding contention with other DAC uses. After RC filtering, the PWM output is indistinguishable from a true analog voltage to the motor controller.

### 5.6 Dual-Mode Input — Controller / UART

The firmware supports two input modes, toggled at runtime by holding the **Triangle** button on the PS4 controller for 5 seconds. The controller stays connected via Bluetooth in both modes so Triangle and Circle are always available.

| Mode | Source | When |
|---|---|---|
| `MODE_CONTROLLER` (default) | PS4 sticks via Bluepad32 | Manual driving / testing |
| `MODE_UART` | Laptop serial (`"steer,throttle\n"`) | Autonomous driving |

**Mode switching:**
- Hold Triangle for 5 continuous seconds → mode toggles
- Motors are stopped immediately on switch as a safety measure
- Current mode is logged (`[LOG] Mode: CONTROLLER` / `[LOG] Mode: UART`)
- Releasing Triangle before 5 s has no effect

**Controls (always active regardless of mode):**

| Input | Mapping |
|---|---|
| Triangle (y) — hold 5 s | Toggle between CONTROLLER and UART mode |
| Circle (b) | Universal brake — drops both channels to `THROTTLE_REST` in any mode |

**Controller mode controls (only active in MODE_CONTROLLER):**

| Input | Mapping |
|---|---|
| Left stick Y (axisY) | Forward throttle (pushed forward = go, reverse ignored) |
| Right stick X (axisRX) | Differential turning |

**Differential mixing:**

Both modes share the same `applyMix()` function for arcade-style differential skid-steer:
- `leftSpeed = throttle + steer`
- `rightSpeed = throttle - steer`

If either side exceeds 255, both sides are scaled proportionally to preserve the turn ratio without clipping.

In controller mode, a deadzone of 20 (out of ±511) is applied to both axes. The controller disconnection callback drops throttle to rest as a safety measure (in controller mode only — UART mode is unaffected by controller disconnect since the laptop is driving).

### 5.7 Safety Note

Multiple safety layers are now implemented:

1. **UART watchdog (implemented):** In `MODE_UART`, if no valid command arrives within 500 ms, motors drop to `THROTTLE_REST`. This fires if the laptop crashes, serial disconnects, or inference stalls.
2. **Universal brake (implemented):** Circle button on the PS4 controller immediately stops motors in both modes. The operator always has physical override authority.
3. **Controller disconnect (implemented):** If the PS4 controller disconnects while in `MODE_CONTROLLER`, motors drop to rest.
4. **Mode switch safety (implemented):** Switching modes always stops motors first.
5. **Hardware kill switch (open):** A physical emergency stop is still recommended — see open questions.

---

## 6. Open Questions for Team

The following items need team alignment before coding continues:

1. ~~Who owns the ESP32 firmware?~~ ✅ Resolved — firmware is in `Couchmo.ino`, uses Arduino framework + Bluepad32
2. What ESC and motor controller hardware is being used? — Need to confirm exact model and verify the three-tier voltage thresholds (`THROTTLE_REST`/`MIN`/`MAX`) with a multimeter
3. Is there a hardware emergency stop (physical kill switch)?
4. What USB port will be dedicated to each webcam — confirm device IDs before coding camera capture
5. Is there a mounting plan for the laptop on the vehicle?
6. RC filter component values — confirm resistor/capacitor values for the PWM→DC low-pass filter (target: 5 kHz cutoff rejection)
7. Level shifter model — confirm the 3.3→5 V level shifter part number and verify it handles 5 kHz PWM cleanly

---

## 7. What Gets Built Next (Coding Team)

With this research complete, the coding order is:

1. ~~Perception pipeline~~ ✅ Done
2. ~~ESP32 firmware — PWM motor control + PS4 manual driving~~ ✅ Done (`Couchmo.ino`)
3. ~~Serial communication module (PySerial)~~ ✅ Done (`serial_controller.py`) — 115200 baud, ACK/ERR protocol
4. ~~ESP32 dual-mode input — controller / UART with Triangle toggle~~ ✅ Done (`Couchmo.ino`)
5. MetaDrive environment wrapper
6. CNN encoder (PyTorch)
7. PPO training loop (Colab)
8. Model export + inference script
