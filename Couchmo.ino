#include <Bluepad32.h>
#include <driver/dac.h>

#define SERIAL_BAUD   115200
#define DAC_LEFT      DAC_CHANNEL_1   // GPIO25
#define DAC_RIGHT     DAC_CHANNEL_2   // GPIO26

#define EBRAKE        5 // GPIO5 — held HIGH, pulled LOW when brake pressed

// ---------------------------------------------------------------
// Throttle voltage calibration
//   ESP32 DAC: 0 = 0V, 255 = 3.3V  →  1 LSB ≈ 12.9 mV
//   Typical hall-effect throttle idle  ≈ 0.8 V  → DAC value ~63
//   Typical hall-effect throttle full  ≈ 3.3 V  → DAC value 255
//   *** Measure your controller's expected range and adjust! ***
// ---------------------------------------------------------------
#define THROTTLE_ZERO  85    // ~1.1 V — idle / zero speed
#define THROTTLE_FULL  255   // ~3.3 V — full speed

// ---------------------------------------------------------------
// Dual-mode input: PS4 controller vs laptop UART
// ---------------------------------------------------------------
enum DriveMode { MODE_CONTROLLER, MODE_UART };
DriveMode currentMode = MODE_CONTROLLER;

#define TRIANGLE_HOLD_MS  5000   // Hold Triangle this long to toggle mode
unsigned long trianglePressStart = 0;
bool          triangleWasHeld    = false;

#define UART_WATCHDOG_MS  500    // Stop motors if no UART command within this window
#define UART_DETECT_MS    2000   // Laptop considered "present" if command arrived within this window
unsigned long lastUartCommandMs  = 0;
int           uartThrottle       = 0;   // 0–255 from last UART command
int           uartSteer          = 0;   // -255–255 from last UART command
String        uartBuffer;

ControllerPtr myController = nullptr;

// ---------------------------------------------------------------
// setThrottle: left/right are 0–255 forward-only values.
//   Negative values (reverse) are clamped to zero because a
//   single-wire potentiometer-style throttle has no reverse lane.
// ---------------------------------------------------------------
void setThrottle(int left, int right) {
  left  = constrain(left,  0, 255);
  right = constrain(right, 0, 255);

  // Map drive range (0-255) onto the throttle's voltage window
  uint8_t dacLeft  = (uint8_t)map(left,  0, 255, THROTTLE_ZERO, THROTTLE_FULL);
  uint8_t dacRight = (uint8_t)map(right, 0, 255, THROTTLE_ZERO, THROTTLE_FULL);

  dac_output_voltage(DAC_LEFT,  dacLeft);
  dac_output_voltage(DAC_RIGHT, dacRight);
}

// ---------------------------------------------------------------
// applyMix: arcade-style differential skid-steer.
//   throttle 0–255, steer -255–255 (negative = left, positive = right)
//   Shared by both controller mode and UART mode.
// ---------------------------------------------------------------
void applyMix(int throttle, int steer) {
  int leftSpeed  = throttle + steer;
  int rightSpeed = throttle - steer;

  int maxVal = max(abs(leftSpeed), abs(rightSpeed));
  if (maxVal > 255) {
    leftSpeed  = leftSpeed  * 255 / maxVal;
    rightSpeed = rightSpeed * 255 / maxVal;
  }

  setThrottle(leftSpeed, rightSpeed);
}

void onConnectedController(ControllerPtr ctl) {
  myController = ctl;
  Serial.println("[LOG] PS4 controller connected!");
}

void onDisconnectedController(ControllerPtr ctl) {
  myController = nullptr;
  if (currentMode == MODE_CONTROLLER) {
    setThrottle(0, 0);
  }
  Serial.println("[LOG] PS4 controller disconnected — motors stopped.");
}

// ---------------------------------------------------------------
// handleUARTInput: non-blocking UART parser.
//   Protocol: laptop sends "steer,throttle\n"
//     steer ∈ [-1.0, 1.0], throttle ∈ [0.0, 1.0]
//   ESP32 replies "ACK\n" or "ERR\n".
//   Always parses regardless of mode so the laptop can prove
//   it is connected while the ESP32 is still in controller mode.
// ---------------------------------------------------------------
void handleUARTInput() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      uartBuffer.trim();
      if (uartBuffer.length() == 0) { uartBuffer = ""; continue; }

      int commaIdx = uartBuffer.indexOf(',');
      if (commaIdx < 0) { Serial.println("ERR"); uartBuffer = ""; continue; }

      float steer    = uartBuffer.substring(0, commaIdx).toFloat();
      float throttle = uartBuffer.substring(commaIdx + 1).toFloat();
      uartBuffer = "";

      if (steer < -1.0f || steer > 1.0f || throttle < 0.0f || throttle > 1.0f) {
        Serial.println("ERR");
        continue;
      }

      uartSteer    = (int)(steer    * 255.0f);
      uartThrottle = (int)(throttle * 255.0f);
      lastUartCommandMs = millis();
      Serial.println("ACK");
    } else {
      uartBuffer += c;
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.println("[LOG] ESP32 booting...");

  pinMode(EBRAKE, OUTPUT);
  digitalWrite(EBRAKE, HIGH);  // Brake-low controller: keep released by default

  dac_output_enable(DAC_LEFT);
  dac_output_enable(DAC_RIGHT);
  setThrottle(0, 0);   // Both channels sit at THROTTLE_ZERO volts on boot

  BP32.setup(&onConnectedController, &onDisconnectedController);
  Serial.println("[LOG] Waiting for PS4 controller...");
}

void loop() {
  BP32.update();
  handleUARTInput();

  // ── Controller buttons (always active if connected) ─────────
  if (myController && myController->isConnected()) {
    bool brake    = myController->b();   // Circle
    bool triangle = myController->y();   // Triangle

    // Triangle hold → toggle mode
    if (triangle) {
      if (trianglePressStart == 0) trianglePressStart = millis();

      if (!triangleWasHeld && (millis() - trianglePressStart >= TRIANGLE_HOLD_MS)) {
        triangleWasHeld = true;

        if (currentMode == MODE_CONTROLLER) {
          if (millis() - lastUartCommandMs < UART_DETECT_MS) {
            currentMode = MODE_UART;
            setThrottle(0, 0);
            Serial.println("[LOG] Mode: UART");
          } else {
            Serial.println("[LOG] Mode switch ignored — no UART connection");
          }
        } else {
          currentMode = MODE_CONTROLLER;
          setThrottle(0, 0);
          Serial.println("[LOG] Mode: CONTROLLER");
        }
      }
    } else {
      trianglePressStart = 0;
      triangleWasHeld    = false;
    }

    // Circle → universal brake (both modes)
    if (brake) {
      digitalWrite(EBRAKE, LOW);
      setThrottle(0, 0);
      Serial.println("[LOG] BRAKE");
      return;
    }
    digitalWrite(EBRAKE, HIGH);
  }

  // ── Drive output based on mode ─────────────────────────────
  if (currentMode == MODE_CONTROLLER) {

    if (myController && myController->isConnected()) {
      int rawThrottle = myController->axisY();
      int rawTurn     = myController->axisRX();

      if (abs(rawThrottle) < 20) rawThrottle = 0;
      if (abs(rawTurn)     < 20) rawTurn     = 0;

      rawThrottle = -rawThrottle;                    // axisY negative = forward
      if (rawThrottle < 0) rawThrottle = 0;          // forward only

      int throttle = map(rawThrottle, 0, 511, 0,    255);
      int turn     = map(rawTurn,  -511, 511, -255, 255);

      applyMix(throttle, turn);
    } else {
      digitalWrite(EBRAKE, HIGH);
      setThrottle(0, 0);
      static unsigned long lastWarn = 0;
      if (millis() - lastWarn > 3000) {
        Serial.println("[LOG] Waiting for controller...");
        lastWarn = millis();
      }
    }

  } else {  // MODE_UART

    if (lastUartCommandMs > 0 && (millis() - lastUartCommandMs < UART_WATCHDOG_MS)) {
      applyMix(uartThrottle, uartSteer);
    } else {
      setThrottle(0, 0);
    }
  }
}