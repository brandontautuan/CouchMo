# Autonomous Car — Architecture Research Report

**Project:** CouchMo Autonomous Campus Vehicle  
**Author:** Coding Team  
**Date:** March 2026  
**Status:** Planning Phase

---

## 1. Overview

This report covers three key research areas the coding team needed to resolve before building:

1. How to handle compute, camera input, and on-chip processing
2. Whether a laptop is necessary on the car
3. How data flows between each system component
4. How to establish a serial connection between the laptop and ESP32

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
                    ├──► Laptop (perception + inference) ──USB serial──► ESP32 ──PWM──► ESC ──► Motors
Right webcam ──USB──┘                                                      ↑
                                                                     ACK / ERR back
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
| Baud rate | 9600 | Safe and stable for 10Hz control |
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

The serial connection spans two codebases:

**Python side (coding team):**
- Open serial port to ESP32 at startup
- Send `"steer,throttle\n"` every 100ms
- Wait for ACK/ERR response
- Implement timeout watchdog — if no ACK within 500ms, send stop command

**ESP32 firmware side (confirm ownership with team):**
- Read UART until newline
- Parse two floats from CSV string
- Validate ranges
- Write PWM signals to ESC output pins
- Send `ACK\n` or `ERR\n`

### 5.5 Safety Note

The timeout watchdog is critical. If the laptop crashes, loses the serial port, or inference takes too long, the ESP32 must stop the motors automatically rather than hold the last command. This should be implemented on the ESP32 side as a hardware timer that resets on every valid ACK — if it expires, throttle drops to zero.

---

## 6. Open Questions for Team

The following items need team alignment before coding continues:

1. Who owns the ESP32 firmware? (C++ / Arduino framework)
2. What ESC and motor controller hardware is being used?
3. Is there a hardware emergency stop (physical kill switch)?
4. What USB port will be dedicated to each webcam — confirm device IDs before coding camera capture
5. Is there a mounting plan for the laptop on the vehicle?

---

## 7. What Gets Built Next (Coding Team)

With this research complete, the coding order is:

1. ~~Perception pipeline~~ ✅ Done
2. Serial communication module (PySerial)
3. MetaDrive environment wrapper
4. CNN encoder (PyTorch)
5. PPO training loop (Colab)
6. Model export + inference script
