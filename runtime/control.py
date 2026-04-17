"""Serial control wrapper for the CouchMo laptop runtime.

Thin adapter around the repo-root :class:`serial_controller.SerialController`.
The wrapper exists only to (a) add a ``runtime``-local import path so callers
don't need to know about the ``sys.path`` shim, and (b) present a small,
runtime-friendly surface (``send``, ``stop``, context manager).

The underlying serial protocol — ``"steer,throttle\\n"`` with ``ACK``/``ERR``
responses at 115200 baud — is defined once in ``serial_controller.py`` and
must not be duplicated here.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The repo root holds ``serial_controller.py`` as a bare module (no package
# __init__.py), so we need the shim for ``from serial_controller import ...``
# to succeed when this file is imported as ``runtime.control``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from serial_controller import SerialController  # noqa: E402 — after sys.path shim


class Controller:
    """Context-manager wrapper around :class:`SerialController`.

    Usage::

        with Controller(port="/dev/tty.usbserial-1410") as ctl:
            ctl.send(steer=0.0, throttle=0.2)
            ctl.stop()
    """

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.1) -> None:
        """Store connection params.  The port is not opened until :meth:`connect`.

        Args:
            port: Serial device string.  Examples: ``"COM5"`` (Windows),
                ``"/dev/tty.usbserial-1410"`` (macOS),
                ``"/dev/ttyUSB0"`` (Linux).
            baud: UART baud rate — must match the ESP32 firmware.
            timeout: Per-readline timeout in seconds.
        """
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._controller: SerialController | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the serial port and wait for the ESP32 boot banner to flush."""
        self._controller = SerialController(
            port=self.port, baud=self.baud, timeout=self.timeout
        )
        self._controller.connect()

    def disconnect(self) -> None:
        """Send a stop command then close the port.  Safe to call twice."""
        if self._controller is not None:
            self._controller.disconnect()
            self._controller = None

    def __enter__(self) -> "Controller":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def send(self, steer: float, throttle: float) -> bool:
        """Send a ``(steer, throttle)`` command and wait for ACK.

        Args:
            steer: Desired steer in ``[-1.0, 1.0]``.  Positive = RIGHT.  The
                underlying :class:`SerialController` also clamps.
            throttle: Desired throttle in ``[0.0, 1.0]``.

        Returns:
            ``True`` if the ESP32 replied ACK, ``False`` on ERR or timeout.
        """
        if self._controller is None:
            raise RuntimeError("Controller is not connected; call connect() first")
        return self._controller.send(steer=steer, throttle=throttle)

    def stop(self) -> bool:
        """Send ``(steer=0, throttle=0)`` — the idle/safe command."""
        if self._controller is None:
            raise RuntimeError("Controller is not connected; call connect() first")
        return self._controller.stop()
