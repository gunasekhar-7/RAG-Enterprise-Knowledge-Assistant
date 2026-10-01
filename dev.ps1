# ==============================================================================
# RAG Enterprise Knowledge Assistant — Windows PowerShell Developer CLI
# ==============================================================================
# Usage:
#   .\dev.ps1 setup          - Initialize Python venv, backend & frontend dependencies
#   .\dev.ps1 infra-up       - Start Docker containers (Postgres, Redis, MinIO)
#   .\dev.ps1 infra-down     - Stop Docker containers
#   .\dev.ps1 infra-status   - Check Docker container health
#   .\dev.ps1 run-backend    - Start FastAPI with hot reload (http://localhost:8000)
#   .\dev.ps1 run-frontend   - Start Vite dev server (http://localhost:5173)
#   .\dev.ps1 test           - Run Pytest test suite
#   .\dev.ps1 lint           - Run Ruff and Mypy checks
#   .\dev.ps1 check          - Run comprehensive pre-flight verification checklist
# ==============================================================================

param (
    [Parameter(Position=0, Mandatory=$true)]
    [ValidateSet("setup", "infra-up", "infra-down", "infra-status", "run-backend", "run-frontend", "test", "lint", "format", "check", "help")]
    [string]$Command
)

$ErrorActionPreference = "Stop"
$RootDir = $PSScriptRoot

function Write-Header {
    param([string]$Message)
    Write-Host "`n>>> $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host " [PASS] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host " [WARN] $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host " [FAIL] $Message" -ForegroundColor Red
}

switch ($Command) {
    "setup" {
        Write-Header "Setting up development environment..."
        
        # 1. Initialize .env if missing
        if (-not (Test-Path "$RootDir\.env")) {
            Write-Host "Creating .env from .env.example..."
            Copy-Item "$RootDir\.env.example" "$RootDir\.env"
            Write-Success ".env created."
        }

        # 2. Python Virtual Environment
        $VenvPython = "$RootDir\.venv\Scripts\python.exe"
        if (-not (Test-Path $VenvPython)) {
            Write-Host "Creating Python 3.11 virtual environment at .venv..."
            python -m venv "$RootDir\.venv"
            Write-Success "Virtual environment created."
        }

        Write-Host "Upgrading pip and installing backend dependencies..."
        & "$RootDir\.venv\Scripts\python.exe" -m pip install --upgrade pip
        & "$RootDir\.venv\Scripts\pip.exe" install -r "$RootDir\backend\requirements-dev.txt"
        Write-Success "Backend dependencies installed."

        # 3. Node.js Frontend Dependencies
        Write-Host "Installing frontend dependencies (npm install)..."
        Push-Location "$RootDir\frontend"
        & npm.cmd install
        Pop-Location
        Write-Success "Frontend dependencies installed."

        Write-Host "`n Setup completed successfully! Run '.\dev.ps1 infra-up' to start local infrastructure." -ForegroundColor Green
    }

    "infra-up" {
        Write-Header "Starting local infrastructure (Postgres + pgvector, Redis, MinIO)..."
        docker compose up -d
        Write-Host "Waiting for containers to report healthy..."
        Start-Sleep -Seconds 5
        docker compose ps
        Write-Success "Infrastructure running."
    }

    "infra-down" {
        Write-Header "Stopping local infrastructure containers..."
        docker compose down
        Write-Success "Infrastructure stopped."
    }

    "infra-status" {
        Write-Header "Container Health Status:"
        docker compose ps
    }

    "run-backend" {
        Write-Header "Starting FastAPI Backend Server on http://localhost:8000..."
        $env:PYTHONPATH = "$RootDir\backend"
        & "$RootDir\.venv\Scripts\uvicorn.exe" app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir "$RootDir\backend"
    }

    "run-frontend" {
        Write-Header "Starting React Frontend on http://localhost:5173..."
        Push-Location "$RootDir\frontend"
        & npm.cmd run dev
        Pop-Location
    }

    "test" {
        Write-Header "Running Pytest Test Suite..."
        $env:PYTHONPATH = "$RootDir\backend"
        & "$RootDir\.venv\Scripts\pytest.exe" "$RootDir\backend\tests" -v
    }

    "lint" {
        Write-Header "Running Ruff Linter..."
        & "$RootDir\.venv\Scripts\ruff.exe" check "$RootDir\backend"
        Write-Header "Running Mypy Type Checker..."
        & "$RootDir\.venv\Scripts\mypy.exe" --config-file "$RootDir\backend\pyproject.toml" "$RootDir\backend\app"
    }

    "format" {
        Write-Header "Formatting codebase with Ruff..."
        & "$RootDir\.venv\Scripts\ruff.exe" format "$RootDir\backend"
        & "$RootDir\.venv\Scripts\ruff.exe" check --fix "$RootDir\backend"
    }

    "check" {
        Write-Header "Running Pre-Flight Verification Checklist..."

        # Check Python
        try {
            $pyVer = & python --version
            Write-Success "Python is available: $pyVer"
        } catch {
            Write-Fail "Python is not in PATH."
        }

        # Check Virtualenv
        if (Test-Path "$RootDir\.venv\Scripts\python.exe") {
            Write-Success "Python venv found at .venv"
        } else {
            Write-Warn "Python venv missing. Run '.\dev.ps1 setup'."
        }

        # Check Node
        try {
            $nodeVer = & node --version
            Write-Success "Node.js is available: $nodeVer"
        } catch {
            Write-Fail "Node.js is not in PATH."
        }

        # Check Docker
        try {
            $dockerVer = & docker --version
            Write-Success "Docker is available: $dockerVer"
        } catch {
            Write-Fail "Docker is not in PATH."
        }

        # Check Git
        try {
            $gitVer = & git --version
            Write-Success "Git is available: $gitVer"
        } catch {
            Write-Fail "Git is not in PATH."
        }

        # Check .env
        if (Test-Path "$RootDir\.env") {
            Write-Success ".env configuration file is present"
        } else {
            Write-Fail ".env missing. Run '.\dev.ps1 setup'."
        }

        # Check Docker Containers
        try {
            $running = (docker ps --format "{{.Names}}" | Select-String "rag_postgres")
            if ($running) {
                Write-Success "PostgreSQL (rag_postgres) container is running"
            } else {
                Write-Warn "Docker containers not running. Start with '.\dev.ps1 infra-up'."
            }
        } catch {}
    }

    "help" {
        Write-Host "Available commands: setup, infra-up, infra-down, infra-status, run-backend, run-frontend, test, lint, format, check"
    }
}
