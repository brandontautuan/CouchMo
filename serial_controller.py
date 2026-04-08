"""
serial_controller.py
--------------------
PySerial wrapper for laptop ↔ ESP32 UART communication.

Protocol (matches autonomous_car_research.md §5.2):
    Outbound:  "steer,throttle\n"   steer ∈ [-1.0, 1.0], throttle ∈ [0.0, 1.0]
    Inbound:   "ACK\n"  or  "ERR\n"
    Lines starting with "[LOG]" are debug output from the ESP32 and are skipped
    when waiting for a protocol response.
"""

import time
import serial


class SerialController:
    """
    Manages a serial link to the ESP32 motor controller.

    Usage::

        with SerialController("/dev/ttyUSB0") as esc:
            ok = esc.send(steer=0.0, throttle=0.3)
            esc.stop()
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.1):
        self.port    = port
        self.baud    = baud
        self.timeout = timeout
        self._ser: serial.Serial | None = None

    # ── lifecycle ─────────────────────────────────────────────

    def connect(self) -> None:
        """Open the serial port and wait for the ESP32 to finish booting."""
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        time.sleep(2)  # ESP32 prints boot logs; let them flush

    def disconnect(self) -> None:
        """Send a stop command, then close the port."""
        if self._ser and self._ser.is_open:
            self.stop()
            self._ser.close()
        self._ser = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    # ── commands ──────────────────────────────────────────────

    def send(self, steer: float, throttle: float) -> bool:
        """
        Send a drive command to the ESP32.

        Args:
            steer:    -1.0 (full left) to 1.0 (full right)
            throttle:  0.0 (stop) to 1.0 (full speed)

        Returns:
            True if the ESP32 replied ACK, False on ERR or timeout.
        """
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("Serial port is not open — call connect() first")

        steer    = max(-1.0, min(1.0, steer))
        throttle = max( 0.0, min(1.0, throttle))

        msg = f"{steer:.2f},{throttle:.2f}\n"
        self._ser.write(msg.encode("ascii"))
        return self._read_ack()

    def stop(self) -> bool:
        """Convenience: send zero steer and zero throttle."""
        return self.send(0.0, 0.0)

    # ── internal ──────────────────────────────────────────────

    def _read_ack(self) -> bool:
        """
        Read lines until ACK, ERR, or timeout.
        Skips any [LOG] debug lines from the ESP32.
        """
        deadline = time.monotonic() + 0.5  # 500 ms max wait

        while time.monotonic() < deadline:
            raw = self._ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("[LOG]"):
                continue
            if line == "ACK":
                return True
            if line == "ERR":
                return False

        return False  # timed out


# ── standalone test ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
    print(f"Connecting to ESP32 on {port} …")

    with SerialController(port) as esc:
        tests = [
            ("zero",      0.0,  0.0),
            ("low fwd",   0.0,  0.2),
            ("turn left", -0.5, 0.3),
            ("turn right", 0.5, 0.3),
            ("full fwd",  0.0,  1.0),
            ("stop",      0.0,  0.0),
        ]

        for label, steer, throttle in tests:
            ok = esc.send(steer, throttle)
            status = "ACK" if ok else "ERR/timeout"
            print(f"  {label:12s}  steer={steer:+.1f}  throttle={throttle:.1f}  → {status}")
            time.sleep(0.5)

    print("Done — port closed.")
