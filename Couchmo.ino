#include <Bluepad32.h>
#include <driver/dac.h>

#define SERIAL_BAUD   115200
#define DAC_LEFT      DAC_CHANNEL_1   // GPIO25
#define DAC_RIGHT     DAC_CHANNEL_2   // GPIO26

// ---------------------------------------------------------------
// Throttle voltage calibration
//   ESP32 DAC: 0 = 0V, 255 = 3.3V  →  1 LSB ≈ 12.9 mV
//   Typical hall-effect throttle idle  ≈ 0.8 V  → DAC value ~63
//   Typical hall-effect throttle full  ≈ 3.3 V  → DAC value 255
//   *** Measure your controller's expected range and adjust! ***
// ---------------------------------------------------------------
#define THROTTLE_ZERO  85    // ~1.1 V — idle / zero speed
#define THROTTLE_FULL  255   // ~3.3 V — full speed

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

  dac_output_enable(DAC_LEFT);
  dac_output_enable(DAC_RIGHT);
  setThrottle(0, 0);   // Both channels sit at THROTTLE_ZERO volts on boot

  BP32.setup(&onConnectedController, &onDisconnectedController);
  Serial.println("[LOG] Waiting for PS4 controller...");
}

void loop() {
  BP32.update();

  if (myController && myController->isConnected()) {

    // Bluepad32 axes: -511 … +511
    int rawThrottle = myController->axisY();    // Left stick Y
    int rawTurn     = myController->axisRX();   // Right stick X
    bool brake      = myController->b();        // Circle button

    if (brake) {
      setThrottle(0, 0);
      Serial.println("[LOG] BRAKE");
      return;
    }

    // Deadzone
    if (abs(rawThrottle) < 20) rawThrottle = 0;
    if (abs(rawTurn)     < 20) rawTurn     = 0;

    // axisY is negative when pushed forward — flip it
    rawThrottle = -rawThrottle;

    // Forward only — drop any reverse input
    if (rawThrottle < 0) rawThrottle = 0;

    // Rescale to 0-255 / -255 to 255
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

    Serial.printf("[LOG] Throttle=%d Turn=%d  L=%d R=%d  DAC_L=%d DAC_R=%d\n",
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

  // Optional: commands from Serial monitor
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    Serial.print("[LOG] Serial in: ");
    Serial.println(cmd);
  }
}