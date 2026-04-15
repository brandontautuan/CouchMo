# Phase 2 — Drivetrain Architecture

## Interior void available: 53.5" × 30.625" (1359 mm × 778 mm)

---

## 1. Drive Configuration: Differential Drive (Recommended)

### Why not skid-steer or omni?

| Config | Pro | Con | Verdict |
|---|---|---|---|
| **Differential drive** | Simple control, efficient, predictable odometry, proven in Nav2 | Can't strafe, zero-radius turn needs both wheels counter-rotating | **Best fit** |
| Skid-steer (4WD) | More traction | Wheel scrub destroys carpet, high lateral forces, odometry drift | Avoid indoors |
| Mecanum/omni | Full holonomic | Complex, expensive, terrible on carpet + outdoor transitions, noisy | Overkill |

### Layout (top-down, within the void)

```
 ←─────────── 1359 mm (53.5") ────────────→
 ┌─────────────────────────────────────────┐  ↑
 │  [Battery]    [ESC/Controller]          │  │
 │                                         │  778 mm
 │ ◎ L-Motor──┤                 ├──R-Motor ◎│  (30.625")
 │  [Drive Wheel]           [Drive Wheel]  │  │
 │     front casters ×2 (passive)          │  ↓
 └─────────────────────────────────────────┘

Drive wheels: rear of void (~250 mm from rear edge)
Wheel track (center-to-center): 1440 mm (56.7") — slightly wider than void via hub extension
```

> **Note:** The drive wheel axle can extend outside the void by ~40mm per side (hidden under the couch skirt fabric), giving a 1440 mm track for stability. The URDF uses ±720 mm from centerline.

---

## 2. Load Analysis

| Component | Weight |
|---|---|
| Couch frame + fabric | ~45 kg (100 lbs) |
| Drivetrain + battery | ~18 kg (40 lbs) |
| Single adult rider | ~90 kg (200 lbs) |
| **Total (worst case)** | **~153 kg (340 lbs)** |

---

## 3. Motor Sizing

### Speed & torque requirements

- Target speed: 5 mph = **2.24 m/s**
- Wheel diameter: 8" = **0.2032 m** → wheel circumference = 0.638 m
- Required RPM at wheel: `(2.24 / 0.638) × 60 ≈ 211 RPM`
- Torque per wheel (flat, static friction + 10% grade reserve):
  - Rolling resistance: `0.02 × 153 kg × 9.81 / 2 ≈ 15 Nm`
  - 10% grade contribution: `sin(5.7°) × 153 × 9.81 / 2 ≈ 74 Nm`
  - **Required torque per wheel: ~90 Nm (66 ft-lbs)**

### Recommended motors

| Motor | Torque | RPM | Voltage | Notes |
|---|---|---|---|---|
| **MY1020 500W brushed DC** | 17.5 Nm × gear ratio | 2750 RPM | 48V | Add 5:1 gearbox → 87 Nm ✓ |
| **Golden Motor BLDC 48V 500W** | 16 Nm × gear | 3000 RPM | 48V | Cleaner, no brush wear |
| **Motenergy ME0708** | 20 Nm continuous | 3000 RPM | 48V | Used in mobility scooters |

**Recommended: 2× MY1020 500W with 5:1 planetary gearbox per side**
- Total system: 1 kW continuous, ~2 kW peak
- Cost: ~$120/motor + $60/gearbox

---

## 4. Battery Pack

### Energy budget

- 2× 500W motors at 60% avg efficiency under load: **~700W draw**
- Target runtime: **90 minutes**
- Energy required: `700W × 1.5h = 1050 Wh`
- With 20% DoD reserve: **~1300 Wh usable**

### Recommended pack

| Option | Capacity | Voltage | Weight | Cost | Verdict |
|---|---|---|---|---|---|
| **4× 12V 50Ah AGM (series)** | 2400 Wh | 48V | ~60 kg | ~$400 | Heavy but cheap, fits void |
| **48V 30Ah LiFePO4** | 1440 Wh | 48V | ~14 kg | ~$650 | Best choice |
| 48V 20Ah Li-ion | 960 Wh | 48V | ~8 kg | ~$400 | Marginal runtime |

**Recommended: 48V 30Ah LiFePO4 pack** (e.g. Chargery / Epoch / RoyPow)
- Dimensions: ~350mm × 200mm × 180mm — fits in void easily
- BMS included, cycle life >2000, safe chemistry
- Estimated runtime: `1440 Wh / 700W ≈ 2.1 hours` ✓

---

## 5. Motor Controllers

| Controller | Protocol | Motors | Current | Notes |
|---|---|---|---|---|
| **RoboClaw 2×30A** | USB/UART/RC | 2 brushed DC | 30A cont | Cheapest, simple, good ROS driver |
| **RoboClaw 2×60A** | USB/UART | 2 brushed DC | 60A cont | Better for peak loads |
| **ODrive 3.6** | USB/CAN | 2 BLDC | 40A cont | Encoder closed-loop, higher performance |
| **VESC 6 (×2)** | CAN/UART | 1 BLDC each | 50A cont | Most flexible, excellent ROS support |

**Recommended path:**
- **Phase 1 (prototype):** RoboClaw 2×60A — simple, has a maintained ROS 2 package (`roboclaw_ros2`)
- **Phase 2 (refined):** 2× VESC 6 on CAN bus — field-oriented control, real-time torque, regenerative braking

### ROS 2 integration

```
# roboclaw_ros2 driver
ros2 run roboclaw_ros2 roboclaw_node --ros-args \
  -p port:=/dev/ttyACM0 \
  -p baud:=115200 \
  -p address:=128
# Subscribes to: /cmd_vel (geometry_msgs/Twist)
# Publishes:     /odom   (nav_msgs/Odometry)
```

---

## 6. Encoder & Odometry

Use quadrature encoders on drive wheels (1000 CPR recommended):
- **Yumo E6B2-CWZ3E 1000PPR** (~$15 each) — direct mount to MY1020 rear shaft
- Or use encoders built into VESC-compatible BLDC motors

---

## 7. Wiring Schematic (simplified)

```
[LiFePO4 48V 30Ah]
     │
  [Main fuse 80A]
     │
  [Main contactor / kill relay]  ←── Safety dead-man switch
     │
  [Motor Controller (RoboClaw 2×60A or 2× VESC)]
     ├── Motor L (left wheel)
     └── Motor R (right wheel)

[48V → 12V DC-DC 30A]  →  [Jetson Orin / RPi CM4]
[48V → 5V DC-DC  5A]   →  [LIDAR, IMU, USB hub]

[E-stop button] → [contactor coil] — cuts all motor power instantly
```

---

## 8. Fit Verification (void: 1359 mm × 778 mm)

| Component | Size (mm) | Placement |
|---|---|---|
| LiFePO4 pack | 350 × 200 × 180 | Center rear |
| RoboClaw 2×60A | 135 × 100 × 38 | Next to battery |
| DC-DC converters | 100 × 60 × 40 each | Near battery |
| Compute (Jetson) | 103 × 90 × 31 | Front center |
| MY1020 motors (×2) | ø100 × 220 L each | Left/right rear |
| Cabling + cable mgmt | — | Along perimeter |

**Total area used: ~35% of void floor** — comfortable fit, room for Phase 3 sensor runs.

---

## 9. Next Steps Checklist

- [ ] Order MY1020 500W motors + 5:1 gearboxes
- [ ] Order 48V 30Ah LiFePO4 pack with BMS
- [ ] Order RoboClaw 2×60A (prototype) or 2× VESC 6 (production)
- [ ] Weld/fabricate steel chassis plate to void dimensions
- [ ] Mount wheel hubs, verify 1440 mm track clears couch skirt
- [ ] Wire e-stop contactor and dead-man relay
- [ ] Flash `roboclaw_ros2` / VESC firmware, test `/cmd_vel` response in sim
