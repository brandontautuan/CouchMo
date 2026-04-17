# run.ps1 — one-command launcher for the CouchMo sim on Windows.
# The sim UI is served via noVNC, so no X server is needed on the host.

param(
    [string]$Mode = "sim"   # sim | preview | shell | headless
)

$ErrorActionPreference = "Stop"

Push-Location -Path $PSScriptRoot
try {
    Write-Host "Running preflight check..."
    python scripts/bringup_check.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Preflight check failed. Aborting." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    $NoVncUrlSim = "http://localhost:6081/vnc.html"
    $NoVncUrlPreview = "http://localhost:6080/vnc.html"

    switch ($Mode) {
        "sim" {
            Write-Host "Starting Gazebo sim. Open: $NoVncUrlSim"
            docker compose up sim
        }
        "preview" {
            Write-Host "Starting URDF preview. Open: $NoVncUrlPreview"
            docker compose up preview
        }
        "headless" {
            Write-Host "Starting headless sim (no noVNC, no GUI)."
            docker compose up sim_headless
        }
        "shell" {
            Write-Host "Opening interactive shell inside the sim container..."
            docker compose run --rm sim bash
        }
        default {
            Write-Host "Usage: .\run.ps1 [sim|preview|shell|headless]"
            Write-Host "  sim      - full Gazebo sim, view in browser at $NoVncUrlSim"
            Write-Host "  preview  - URDF/RViz preview, view in browser at $NoVncUrlPreview"
            Write-Host "  headless - sim without noVNC (batch dataset generation)"
            Write-Host "  shell    - interactive bash inside the sim container"
            exit 2
        }
    }
}
finally {
    Pop-Location
}
