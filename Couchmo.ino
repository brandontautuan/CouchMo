#include <Bluepad32.h>

#define SERIAL_BAUD       115200

// --- Pins and PWM Configuration ---
#define PIN_LEFT          2
#define PIN_RIGHT         13

const int PWM_FREQ = 10000;  // 10kHz to "trick" the controller into thinking it is recieving analog input
const int PWM_RES  = 8;      // 8-bit (0–255) (reasonable for 10kHz)
const int CH_LEFT  = 0;
const int CH_RIGHT = 1;

// ---------------------------------------------------------------
// Throttle voltage calibration  (PWM + level-shifter to 5 V)
//   THROTTLE_REST  — below the "go" threshold.       ~0.8 V → duty 41
//   THROTTLE_MIN   — minimum recognised speed.       ~1.1 V → duty 56
//   THROTTLE_MAX   — full speed.                     ~4.2 V → duty 214
// ---------------------------------------------------------------
#define THROTTLE_REST     41
#define THROTTLE_MIN      56
#define THROTTLE_MAX      214

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

  ledcWrite(CH_LEFT,  dutyLeft);
  ledcWrite(CH_RIGHT, dutyRight);
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
  if (currentMode == MODE_CONTROLLER) {
    setThrottle(0, 0);
  }
  Serial.println("[LOG] PS4 controller disconnected.");
}

// ---------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println("[LOG] ESP32 booting...");

  ledcSetup(CH_LEFT,  PWM_FREQ, PWM_RES);
  ledcAttachPin(PIN_LEFT,  CH_LEFT);

  ledcSetup(CH_RIGHT, PWM_FREQ, PWM_RES);
  ledcAttachPin(PIN_RIGHT, CH_RIGHT);

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
      Serial.printf("[LOG] T=%d S=%d  L=%d R=%d  PWM_L=%d PWM_R=%d\n",
        throttle, turn, clL, clR,
        clL == 0 ? THROTTLE_REST : (int)map(clL, 1, 255, THROTTLE_MIN, THROTTLE_MAX),
        clR == 0 ? THROTTLE_REST : (int)map(clR, 1, 255, THROTTLE_MIN, THROTTLE_MAX));

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
