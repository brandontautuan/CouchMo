#!/usr/bin/env python3
"""Cross-platform preflight check for the CouchMo sim bringup.

Runs before `docker compose up` to surface common misconfigurations with
actionable fix instructions. Uses ONLY the Python standard library so it
can run on any host that has `python3` without extra pip installs.

Exit codes:
    0  all checks passed, or only non-fatal warnings
    1  Docker is unreachable (fatal)
"""
from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

NOVNC_PORT = 6081
MIN_RAM_GB = 8


def _print_ok(msg: str) -> None:
    print(f"[ OK ]  {msg}")


def _print_warn(msg: str) -> None:
    print(f"[WARN]  {msg}")


def _print_err(msg: str) -> None:
    print(f"[FAIL]  {msg}")


def check_docker() -> bool:
    """Return True if `docker version` exits 0."""
    docker = shutil.which("docker")
    if docker is None:
        _print_err("Docker CLI not found on PATH.")
        print("        Install Docker Desktop:")
        print("          Windows/macOS: https://www.docker.com/products/docker-desktop/")
        print("          Linux:         https://docs.docker.com/engine/install/")
        return False

    try:
        result = subprocess.run(
            [docker, "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        _print_err(f"Could not execute `docker version`: {exc}")
        print("        Start Docker Desktop (or the docker daemon) and retry.")
        return False

    if result.returncode != 0:
        _print_err("`docker version` returned a non-zero exit code.")
        stderr = (result.stderr or "").strip()
        if stderr:
            for line in stderr.splitlines():
                print(f"        {line}")
        print("        Fix: start Docker Desktop (or `sudo systemctl start docker`)")
        print("             then re-run this script.")
        return False

    _print_ok("Docker CLI reachable (`docker version` ok).")
    return True


def check_port_free(port: int) -> None:
    """Warn (but do not fail) if the noVNC port is already bound."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        _print_warn(
            f"Port {port} is already in use. If this is a previous "
            "couchmo_sim or couchmo_preview container, that is fine."
        )
        if platform.system() == "Windows":
            print(f"        Inspect: netstat -ano | findstr :{port}")
        else:
            print(f"        Inspect: lsof -i :{port}")
        print("        Stop the offending process if it is not ours:")
        print("          docker compose down")
        return
    finally:
        sock.close()

    _print_ok(f"Port {port} appears free on loopback.")


def _ram_gb_windows() -> float | None:
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullTotalPhys / (1024 ** 3)
    except Exception:
        return None


def _ram_gb_linux() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                kb = int(parts[1])
                return kb / (1024 ** 2)
    except (OSError, ValueError):
        return None
    return None


def _ram_gb_macos() -> float | None:
    sysctl = shutil.which("sysctl")
    if sysctl is None:
        return None
    try:
        result = subprocess.run(
            [sysctl, "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.strip()) / (1024 ** 3)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def check_ram() -> None:
    """Warn (but never fail) if we can't confirm >= MIN_RAM_GB of RAM."""
    system = platform.system()
    ram_gb: float | None
    if system == "Windows":
        ram_gb = _ram_gb_windows()
    elif system == "Linux":
        ram_gb = _ram_gb_linux()
    elif system == "Darwin":
        ram_gb = _ram_gb_macos()
    else:
        ram_gb = None

    if ram_gb is None:
        _print_warn(
            f"Could not determine host RAM on {system}. "
            f"Sim recommends >= {MIN_RAM_GB} GB. Proceeding anyway."
        )
        return

    if ram_gb < MIN_RAM_GB:
        _print_warn(
            f"Host has {ram_gb:.1f} GB RAM, below the recommended "
            f"{MIN_RAM_GB} GB. Gazebo may be sluggish."
        )
        return

    _print_ok(f"Host RAM: {ram_gb:.1f} GB (>= {MIN_RAM_GB} GB).")


def main() -> int:
    print(f"CouchMo sim preflight check ({platform.system()} {platform.release()})")
    print("-" * 60)

    docker_ok = check_docker()
    check_port_free(NOVNC_PORT)
    check_ram()

    print("-" * 60)
    if not docker_ok:
        print("Preflight FAILED: Docker is unreachable. Fix the error above.")
        return 1

    print("Preflight OK. Launch the sim with your platform wrapper:")
    print("  Windows:     .\\run.ps1 sim")
    print("  macOS/Linux: ./run.sh sim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
