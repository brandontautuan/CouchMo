//Scott Change Code
//Gemini PWM example - Updated for Pins D13 (Right) and D2 (Left)

#include <Bluepad32.h>

// --- Pins and PWM Configuration ---
// Updated per request: D2 for Left, D13 for Right
#define PIN_LEFT  2  
#define PIN_RIGHT 13

const int PWM_FREQ = 5000; // 5kHz is good for RC filtering
const int PWM_RES  = 8;    // 8-bit resolution (0-255)
const int CH_LEFT  = 0;
const int CH_RIGHT = 1;

// --- Voltage Mapping (Based on 5V output from Level Shifter) ---
// 1.1V / 5.0V * 255 = 56  (Idle/Zero)
// 4.2V / 5.0V * 255 = 214 (Full Speed)
#define THROTTLE_IDLE 56
#define THROTTLE_MAX  214

ControllerPtr myController = nullptr;

void setThrottle(int left, int right) {
  // Constrain incoming mixer values to 0-255
  left = constrain(left, 0, 255);
  right = constrain(right, 0, 255);

  // Map the 0-255 "speed" to the 1.1V-4.2V PWM range
  uint8_t dutyLeft = (uint8_t)map(left, 0, 255, THROTTLE_IDLE, THROTTLE_MAX);
  uint8_t dutyRight = (uint8_t)map(right, 0, 255, THROTTLE_IDLE, THROTTLE_MAX);

  ledcWrite(CH_LEFT, dutyLeft);
  ledcWrite(CH_RIGHT, dutyRight);
}

void onConnectedController(ControllerPtr ctl) {
  myController = ctl;
  Serial.println("[LOG] PS4 Connected!");
}

void onDisconnectedController(ControllerPtr ctl) {
  myController = nullptr;
  setThrottle(0, 0); // Safety: Set to 1.1V (Idle) if disconnected
  Serial.println("[LOG] Disconnected - Motors to Idle.");
}

void setup() {
  Serial.begin(115200);

  // Setup PWM Channels
  ledcSetup(CH_LEFT, PWM_FREQ, PWM_RES);
  ledcAttachPin(PIN_LEFT, CH_LEFT);

  ledcSetup(CH_RIGHT, PWM_FREQ, PWM_RES);
  ledcAttachPin(PIN_RIGHT, CH_RIGHT);

  // Initialize at Idle (1.1V)
  setThrottle(0, 0);

  BP32.setup(&onConnectedController, &onDisconnectedController);
}

void loop() {
  BP32.update();

  if (myController && myController->isConnected()) {
    // AxisY is negative when pushed forward — flip it
    int rawThrottle = -myController->axisY(); 
    int rawTurn = myController->axisRX();
    bool brake = myController->b(); // Circle button

    if (brake) {
      setThrottle(0, 0);
      return;
    }

    // Deadzone
    if (abs(rawThrottle) < 25) rawThrottle = 0;
    if (abs(rawTurn) < 25) rawTurn = 0;

    // We are currently ignoring reverse (negative Y)
    if (rawThrottle < 0) rawThrottle = 0;

    // Normalize to 0-255 (PS4 axis is usually -511 to 511)
    int throttle = map(rawThrottle, 0, 511, 0, 255);
    int turn = map(rawTurn, -511, 511, -127, 127);

    // Differential Mixing
    int leftSpeed = throttle + turn;
    int rightSpeed = throttle - turn;

    setThrottle(leftSpeed, rightSpeed);
  } else {
    setThrottle(0, 0); // Keep at 1.1V if no controller
  }
}