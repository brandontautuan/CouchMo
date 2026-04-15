#include <Bluepad32.h>

#define SERIAL_BAUD       115200

// --- Pins and PWM Configuration ---
#define PIN_LEFT          2
#define PIN_RIGHT         13

const int PWM_FREQ      = 10000;  // 10 kHz — fast enough for the low-pass to produce a smooth analog voltage
const int PWM_RES       = 8;      // 8-bit resolution (reasonable for 10 kHz)
const int PWM_MAX_DUTY  = (1 << PWM_RES) - 1;  // 255 for 8-bit
const int CH_LEFT       = 0;
const int CH_RIGHT      = 1;

// --- Joystick Configuration ---
#define STICK_DEADZONE    20      // Ignore joystick values within ±20 of centre
#define STICK_MAX         511     // Bluepad32 axis range: -512..511

// ---------------------------------------------------------------
// Throttle voltage calibration
//   PWM (8-bit) → low-pass → level-shifter (0–V_REF from controller)
//   V_out = (duty / PWM_MAX_DUTY) × V_REF
//   duty  = V_target × PWM_MAX_DUTY / V_REF
//
//   V_GO  — Minimum voltage at which the motor controller actually
//           drives the wheels.  If the motor is unresponsive at low
//           throttle, raise this value.  The original 1.1 V estimate
//           was too low; ~3.3 V matches observed behaviour (motor
//           starts at roughly throttle input 180/255).
// ---------------------------------------------------------------
const float V_REF  = 4.856f;   // Controller reference voltage (measured at throttle connector)
const float V_GO   = 3.3f;     // Measured "go" threshold — raise if motor is still unresponsive
const float V_FULL = 4.2f;     // Full-speed voltage

const int THROTTLE_REST = 0;
const int THROTTLE_MIN  = (int)(V_GO   * PWM_MAX_DUTY / V_REF + 0.5f);  // ≈173
const int THROTTLE_MAX  = (int)(V_FULL * PWM_MAX_DUTY / V_REF + 0.5f);  // ≈220

// --- Mode switching & UART watchdog ---
#define MODE_SWITCH_MS    5000    // Hold triangle for 5 s to toggle
#define UART_TIMEOUT_MS   500     // No valid UART command → rest
#define WARN_INTERVAL_MS  3000    // Minimum interval between repeated log warnings

enum DriveMode { MODE_CONTROLLER, MODE_UART };

DriveMode         currentMode       = MODE_CONTROLLER;
ControllerPtr     myController      = nullptr;
unsigned long     triangleHoldStart = 0;
bool              triangleSwitched  = false;  // Prevents re-firing while held
unsigned long     lastValidCmd      = 0;

uint8_t           lastDutyLeft      = 0;      // Cached for debug logging
uint8_t           lastDutyRight     = 0;

// ---------------------------------------------------------------
void setThrottle(int left, int right) {
  left  = constrain(left,  0, PWM_MAX_DUTY);
  right = constrain(right, 0, PWM_MAX_DUTY);

  lastDutyLeft  = left  == 0 ? THROTTLE_REST
                              : (uint8_t)map(left,  1, PWM_MAX_DUTY, THROTTLE_MIN, THROTTLE_MAX);
  lastDutyRight = right == 0 ? THROTTLE_REST
                              : (uint8_t)map(right, 1, PWM_MAX_DUTY, THROTTLE_MIN, THROTTLE_MAX);

  ledcWrite(CH_LEFT,  lastDutyLeft);
  ledcWrite(CH_RIGHT, lastDutyRight);
}

// ---------------------------------------------------------------
// Differential mixing shared by both input sources.
// throttle255: 0–255,  steer255: -255–255
// ---------------------------------------------------------------
void applyMix(int throttle255, int steer255) {
  int leftSpeed  = throttle255 + steer255;
  int rightSpeed = throttle255 - steer255;

  int maxVal = max(abs(leftSpeed), abs(rightSpeed));
  if (maxVal > PWM_MAX_DUTY) {
    leftSpeed  = leftSpeed  * PWM_MAX_DUTY / maxVal;
    rightSpeed = rightSpeed * PWM_MAX_DUTY / maxVal;
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

    int throttle255 = (int)(throttle * PWM_MAX_DUTY);
    int steer255    = (int)(steer    * PWM_MAX_DUTY);

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

      if (abs(rawThrottle) < STICK_DEADZONE) rawThrottle = 0;
      if (abs(rawTurn)     < STICK_DEADZONE) rawTurn     = 0;

      rawThrottle = -rawThrottle;
      if (rawThrottle < 0) rawThrottle = 0;

      int throttle = map(rawThrottle, 0, STICK_MAX, 0,             PWM_MAX_DUTY);
      int turn     = map(rawTurn, -STICK_MAX, STICK_MAX, -PWM_MAX_DUTY, PWM_MAX_DUTY);

      applyMix(throttle, turn);

      Serial.printf("[LOG] T=%d S=%d  PWM_L=%d PWM_R=%d\n",
        throttle, turn, lastDutyLeft, lastDutyRight);

    } else {
      setThrottle(0, 0);
      static unsigned long lastWarn = 0;
      if (millis() - lastWarn > WARN_INTERVAL_MS) {
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
      if (millis() - lastWdWarn > WARN_INTERVAL_MS) {
        Serial.println("[LOG] UART watchdog — no command, motors at rest.");
        lastWdWarn = millis();
      }
    }
  }
}
