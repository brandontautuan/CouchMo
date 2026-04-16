#include <Bluepad32.h>

#define SERIAL_BAUD       115200

// --- DAC pin assignments (GPIO 25 & 26 are the two DAC channels on ESP32) ---
#define PIN_LEFT          25   // DAC1
#define PIN_RIGHT         26   // DAC2

// --- brakes may need separate pins since one is Brake High and one is Brake Low
#define BRAKE_PIN_LEFT    14 // Brake Low
#define BRAKE_PIN_RIGHT   14 // Brake High OR Brake Low on controller

// ---------------------------------------------------------------
// Throttle voltage calibration  (DAC direct, no level-shifter)
//   V_out = (dacValue / 255) × 3.3 V
//
//   Controller expects 1.1–4.2 V on its throttle input.
//   DAC caps at 3.3 V, so max reachable throttle ≈ 71 % of full.
//   Add an op-amp / level-shifter stage to reach the full 4.2 V.
//
//   THROTTLE_REST  — 0 V  (DAC 0),   well below 1.1 V "go" threshold
//   THROTTLE_MIN   — 1.1 V            → 1.1 / 3.3 × 255 ≈ 85
//   THROTTLE_MAX   — 3.3 V (DAC max)  → 255
// ---------------------------------------------------------------
#define THROTTLE_REST     0
#define THROTTLE_MIN      85
#define THROTTLE_MAX      255

// --- Mode switching & UART watchdog ---
#define MODE_SWITCH_MS    5000    // Hold triangle for 5 s to toggle
#define UART_TIMEOUT_MS   500    // No valid UART command → rest

enum DriveMode { MODE_CONTROLLER, MODE_UART };

DriveMode         currentMode       = MODE_CONTROLLER;
ControllerPtr     myController      = nullptr;
unsigned long     triangleHoldStart = 0;
bool              triangleSwitched  = false;  // Prevents re-firing while held
unsigned long     lastValidCmd      = 0;

// ---------------------------------------------------------------
void setThrottle(int left, int right) {
  left  = constrain(left,  0, 255);
  right = constrain(right, 0, 255);

  uint8_t dutyLeft  = left  == 0 ? THROTTLE_REST
                                 : (uint8_t)map(left,  1, 255, THROTTLE_MIN, THROTTLE_MAX);
  uint8_t dutyRight = right == 0 ? THROTTLE_REST
                                 : (uint8_t)map(right, 1, 255, THROTTLE_MIN, THROTTLE_MAX);

  dacWrite(PIN_LEFT,  dutyLeft);
  dacWrite(PIN_RIGHT, dutyRight);
}

// ---------------------------------------------------------------
// Differential mixing shared by both input sources.
// throttle255: 0–255,  steer255: -255–255
// ---------------------------------------------------------------
void applyMix(int throttle255, int steer255) {
  int leftSpeed  = throttle255 + steer255;
  int rightSpeed = throttle255 - steer255;

  int maxVal = max(abs(leftSpeed), abs(rightSpeed));
  if (maxVal > 255) {
    leftSpeed  = leftSpeed  * 255 / maxVal;
    rightSpeed = rightSpeed * 255 / maxVal;
  }

  setThrottle(leftSpeed, rightSpeed);
}

// ---------------------------------------------------------------
// UART command parser — reads all buffered lines, applies the
// last valid command.  Protocol: "steer,throttle\n"
//   steer    ∈ [-1.0, 1.0]
//   throttle ∈ [ 0.0, 1.0]
// Replies ACK or ERR per line.
// ---------------------------------------------------------------
void handleUARTInput() {
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;

    int commaIdx = line.indexOf(',');
    if (commaIdx <= 0 || commaIdx >= (int)line.length() - 1) {
      Serial.println("ERR");
      continue;
    }

    float steer    = line.substring(0, commaIdx).toFloat();
    float throttle = line.substring(commaIdx + 1).toFloat();

    if (steer < -1.0f || steer > 1.0f || throttle < 0.0f || throttle > 1.0f) {
      Serial.println("ERR");
      continue;
    }

    int throttle255 = (int)(throttle * 255.0f);
    int steer255    = (int)(steer    * 255.0f);

    applyMix(throttle255, steer255);
    lastValidCmd = millis();
    Serial.println("ACK");
  }
}

// ---------------------------------------------------------------
// Bluepad32 callbacks
// ---------------------------------------------------------------
void onConnectedController(ControllerPtr ctl) {
  myController = ctl;
  Serial.println("[LOG] PS4 controller connected!");
}

void onDisconnectedController(ControllerPtr ctl) {
  myController = nullptr;
  setThrottle(0, 0);
  currentMode = MODE_CONTROLLER;
  Serial.println("[LOG] PS4 controller disconnected — motors disabled, reverted to CONTROLLER mode.");
}

// ---------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println("[LOG] ESP32 booting...");

  dacWrite(PIN_LEFT,  0);
  dacWrite(PIN_RIGHT, 0);

  setThrottle(0, 0);

  BP32.setup(&onConnectedController, &onDisconnectedController);
  Serial.println("[LOG] Mode: CONTROLLER");
  Serial.println("[LOG] Waiting for PS4 controller...");
}

// ---------------------------------------------------------------
void loop() {
  BP32.update();

  // ── Always: triangle mode-switch + universal brake ──────────
  if (myController && myController->isConnected()) {

    // Triangle hold → toggle mode after 5 s
    bool triDown = myController->y();
    if (triDown) {
      if (triangleHoldStart == 0)
        triangleHoldStart = millis();

      if (!triangleSwitched && millis() - triangleHoldStart >= MODE_SWITCH_MS) {
        currentMode = (currentMode == MODE_CONTROLLER) ? MODE_UART : MODE_CONTROLLER;
        setThrottle(0, 0);
        triangleSwitched = true;

        if (currentMode == MODE_UART) {
          lastValidCmd = millis();   // Seed watchdog so it doesn't fire instantly
          Serial.println("[LOG] Mode: UART");
        } else {
          Serial.println("[LOG] Mode: CONTROLLER");
        }
      }
    } else {
      triangleHoldStart = 0;
      triangleSwitched  = false;
    }

    // Circle = universal brake in every mode
    if (myController->b()) {
      setThrottle(0, 0);
      Serial.println("[LOG] BRAKE");
      return;
    }
  }

  // ── Mode-specific input ─────────────────────────────────────
  if (currentMode == MODE_CONTROLLER) {

    if (myController && myController->isConnected()) {
      int rawThrottle = myController->axisY();
      int rawTurn     = myController->axisRX();

      if (abs(rawThrottle) < 20) rawThrottle = 0;
      if (abs(rawTurn)     < 20) rawTurn     = 0;

      rawThrottle = -rawThrottle;
      if (rawThrottle < 0) rawThrottle = 0;

      int throttle = map(rawThrottle, 0, 511, 0,    255);
      int turn     = map(rawTurn,  -511, 511, -255, 255);

      applyMix(throttle, turn);

      int clL = constrain(throttle + turn, 0, 255);
      int clR = constrain(throttle - turn, 0, 255);
      int dacL = clL == 0 ? THROTTLE_REST : (int)map(clL, 1, 255, THROTTLE_MIN, THROTTLE_MAX);
      int dacR = clR == 0 ? THROTTLE_REST : (int)map(clR, 1, 255, THROTTLE_MIN, THROTTLE_MAX);
      Serial.printf("[LOG] T=%d S=%d  L=%d R=%d  DAC_L=%d DAC_R=%d\n",
        throttle, turn, clL, clR, dacL, dacR);

    } else {
      setThrottle(0, 0);
      static unsigned long lastWarn = 0;
      if (millis() - lastWarn > 3000) {
        Serial.println("[LOG] Waiting for controller...");
        lastWarn = millis();
      }
    }

  } else {
    // MODE_UART
    if (Serial.available()) {
      handleUARTInput();
    }

    // Watchdog — no valid command within timeout → rest
    if (millis() - lastValidCmd > UART_TIMEOUT_MS) {
      setThrottle(0, 0);
      static unsigned long lastWdWarn = 0;
      if (millis() - lastWdWarn > 3000) {
        Serial.println("[LOG] UART watchdog — no command, motors at rest.");
        lastWdWarn = millis();
      }
    }
  }
}
