#include <Bluepad32.h>

#define SERIAL_BAUD   115200

// --- Pins and PWM Configuration ---
// D2 for Left, D13 for Right
#define PIN_LEFT      2
#define PIN_RIGHT     13

const int PWM_FREQ = 5000;   // 5 kHz — good for RC low-pass filtering
const int PWM_RES  = 8;      // 8-bit resolution (0–255)
const int CH_LEFT  = 0;
const int CH_RIGHT = 1;

// ---------------------------------------------------------------
// Throttle voltage calibration  (PWM + level-shifter to 5 V)
//   After RC filtering the PWM becomes a DC voltage:
//     duty 0   → 0 V,  duty 255 → 5 V   (with 3.3→5 V shifter)
//   Typical hall-effect throttle idle  ≈ 1.1 V  → duty ~56
//   Typical hall-effect throttle full  ≈ 4.2 V  → duty ~214
//   *** Measure your controller's expected range and adjust! ***
// ---------------------------------------------------------------
#define THROTTLE_ZERO  56     // ~1.1 V — idle / zero speed
#define THROTTLE_FULL  214    // ~4.2 V — full speed

ControllerPtr myController = nullptr;

// ---------------------------------------------------------------
// setThrottle: left/right are 0–255 forward-only values.
//   Negative values (reverse) are clamped to zero because a
//   single-wire potentiometer-style throttle has no reverse lane.
// ---------------------------------------------------------------
void setThrottle(int left, int right) {
  left  = constrain(left,  0, 255);
  right = constrain(right, 0, 255);

  uint8_t dutyLeft  = (uint8_t)map(left,  0, 255, THROTTLE_ZERO, THROTTLE_FULL);
  uint8_t dutyRight = (uint8_t)map(right, 0, 255, THROTTLE_ZERO, THROTTLE_FULL);

  ledcWrite(CH_LEFT,  dutyLeft);
  ledcWrite(CH_RIGHT, dutyRight);
}

void onConnectedController(ControllerPtr ctl) {
  myController = ctl;
  Serial.println("[LOG] PS4 controller connected!");
}

void onDisconnectedController(ControllerPtr ctl) {
  myController = nullptr;
  setThrottle(0, 0);
  Serial.println("[LOG] PS4 controller disconnected — motors stopped.");
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println("[LOG] ESP32 booting...");

  ledcSetup(CH_LEFT,  PWM_FREQ, PWM_RES);
  ledcAttachPin(PIN_LEFT,  CH_LEFT);

  ledcSetup(CH_RIGHT, PWM_FREQ, PWM_RES);
  ledcAttachPin(PIN_RIGHT, CH_RIGHT);

  setThrottle(0, 0);   // Both channels sit at THROTTLE_ZERO volts on boot

  BP32.setup(&onConnectedController, &onDisconnectedController);
  Serial.println("[LOG] Waiting for PS4 controller...");
}

void loop() {
  BP32.update();

  if (myController && myController->isConnected()) {

    int rawThrottle = myController->axisY();    // Left stick Y
    int rawTurn     = myController->axisRX();   // Right stick X
    bool brake      = myController->b();        // Circle button

    if (brake) {
      setThrottle(0, 0);
      Serial.println("[LOG] BRAKE");
      return;
    }

    if (abs(rawThrottle) < 20) rawThrottle = 0;
    if (abs(rawTurn)     < 20) rawTurn     = 0;

    // axisY is negative when pushed forward — flip it
    rawThrottle = -rawThrottle;

    // Forward only — drop any reverse input
    if (rawThrottle < 0) rawThrottle = 0;

    int throttle = map(rawThrottle,    0, 511, 0,    255);
    int turn     = map(rawTurn,     -511, 511, -255, 255);

    // Differential mixing
    int leftSpeed  = throttle + turn;
    int rightSpeed = throttle - turn;

    // Preserve the left/right ratio if either side clips
    int maxVal = max(abs(leftSpeed), abs(rightSpeed));
    if (maxVal > 255) {
      leftSpeed  = leftSpeed  * 255 / maxVal;
      rightSpeed = rightSpeed * 255 / maxVal;
    }

    setThrottle(leftSpeed, rightSpeed);

    Serial.printf("[LOG] Throttle=%d Turn=%d  L=%d R=%d  PWM_L=%d PWM_R=%d\n",
      throttle, turn, leftSpeed, rightSpeed,
      map(constrain(leftSpeed,  0,255), 0,255, THROTTLE_ZERO, THROTTLE_FULL),
      map(constrain(rightSpeed, 0,255), 0,255, THROTTLE_ZERO, THROTTLE_FULL));

  } else {
    setThrottle(0, 0);
    static unsigned long lastWarn = 0;
    if (millis() - lastWarn > 3000) {
      Serial.println("[LOG] Waiting for controller...");
      lastWarn = millis();
    }
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    Serial.print("[LOG] Serial in: ");
    Serial.println(cmd);
  }
}