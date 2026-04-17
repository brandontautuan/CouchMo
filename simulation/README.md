# CouchMo Simulation

ROS 2 + Gazebo simulation, fully containerized. The only host requirements are
**Docker Desktop** (or Docker Engine on Linux) and a web browser — the sim UI
is served via **noVNC** on port 6081, so there is nothing to install for X11,
XQuartz, VcXsrv, etc.

Open `http://localhost:6081/vnc.html` in any browser after the container
starts.

## Bringup

| OS          | Command          |
| ----------- | ---------------- |
| Windows     | `.\run.ps1 sim`  |
| macOS/Linux | `./run.sh sim`   |

Both wrappers run `scripts/bringup_check.py` first (a stdlib-only preflight
check for Docker, the noVNC port, and host RAM) and then `docker compose up`.

## Modes

| Mode       | What it does                                                  |
| ---------- | ------------------------------------------------------------- |
| `sim`      | Full Gazebo + ROS 2 + noVNC. Open `http://localhost:6081/vnc.html`. |
| `preview`  | URDF / RViz preview only (faster boot). Same noVNC URL.       |
| `shell`    | Interactive `bash` inside the sim image — no services started. |
| `headless` | Runs `sim_headless` (no noVNC port) for batch dataset gen.    |

Examples:

```powershell
# Windows
.\run.ps1 sim
.\run.ps1 preview
.\run.ps1 headless
.\run.ps1 shell
```

```bash
# macOS / Linux
./run.sh sim
./run.sh preview
./run.sh headless
./run.sh shell
```

## Dataset bind mount

The `sim` and `sim_headless` services mount `../training/data` (repo-relative,
i.e. `CouchMo/training/data`) into the container at `/workspace/data`. That
directory is created on the host on first run and is **gitignored** (see the
repo-root `.gitignore`). Anything the sim writes to `/workspace/data` shows up
directly under `training/data/` and is what the training pipeline consumes.

## Failure modes

### Docker daemon not running

`docker version` fails with "Cannot connect to the Docker daemon" or similar.

- **Windows / macOS:** install or start [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- **Linux:** follow the [Docker Engine install guide](https://docs.docker.com/engine/install/) and `sudo systemctl start docker`.

### Port 6081 already in use

The preflight check will warn you. To find the offender:

- **Windows:** `netstat -ano | findstr :6081`
- **macOS / Linux:** `lsof -i :6081`

If it is a stale CouchMo container, `docker compose down` will release the port.

### WSL2 not enabled (Windows)

Docker Desktop on Windows requires WSL2. If Docker Desktop complains about
"WSL 2 installation is incomplete", follow the
[Microsoft WSL install guide](https://learn.microsoft.com/en-us/windows/wsl/install).

### Insufficient RAM

Gazebo + ROS 2 + SLAM want at least **8 GB** of system RAM. The preflight
check will warn (but not fail) if less is available; expect a sluggish sim.

## Why noVNC instead of X11 forwarding

The previous `run.sh` forwarded X11 via XQuartz on macOS only. That approach
did not work on Windows and was fragile even on macOS. noVNC works the same
on every host: the container runs its own display server and streams it to a
browser tab on `localhost:6081`.
