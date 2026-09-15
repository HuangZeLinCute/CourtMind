#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# ``bash`` from Windows CMD starts WSL on this machine. WSL cannot directly
# execute the Windows Conda python from a script, so hand startup back to the
# native CMD launcher through WSL's interop host.
if [[ -x /init && -f /mnt/c/Windows/System32/cmd.exe ]]; then
  cd "$PROJECT_DIR"
  # A fresh native CMD session is required: starting npm inside WSL's interop
  # process can print "Vite ready" while its localhost socket remains isolated.
  nohup /init /mnt/c/Windows/System32/cmd.exe /d /c start "" /min cmd.exe /d /c start.bat \
    >/dev/null 2>&1 &
  # Keep WSL alive briefly so the native launcher is created before the
  # interop host tears down this shell session.
  sleep 3
  disown || true
  echo "Good-Badminton startup was handed to Windows (a minimized CMD window)."
  echo "Open http://127.0.0.1:5173 after a few seconds. Run stop.bat in CMD to stop."
  exit 0
fi

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm was not found. Install Node.js and add npm to PATH." >&2
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Error: frontend dependencies are missing. Run: cd frontend && npm install" >&2
  exit 1
fi

port_is_open() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

BACKEND_OPEN=false
FRONTEND_OPEN=false
port_is_open 8000 && BACKEND_OPEN=true
port_is_open 5173 && FRONTEND_OPEN=true

if $BACKEND_OPEN && $FRONTEND_OPEN && command -v curl >/dev/null 2>&1 \
  && curl --noproxy '*' -fsS http://127.0.0.1:8000/api/health | grep -q 'Good-Badminton' \
  && curl --noproxy '*' -fsS http://127.0.0.1:5173/ | grep -q 'Good-Badminton'; then
  echo "Good-Badminton is already running:"
  echo "  Backend:  http://127.0.0.1:8000"
  echo "  Frontend: http://127.0.0.1:5173"
  echo "Run stop.bat from CMD to stop both services."
  exit 0
fi

if $BACKEND_OPEN; then
  echo "Error: port 8000 is already in use by another or unhealthy service." >&2
  echo "If this is a leftover Good-Badminton backend, run 'stop.bat' directly in CMD" >&2
  echo "(do not put 'bash' before it), then start again." >&2
  exit 1
fi
if $FRONTEND_OPEN; then
  echo "Error: port 5173 is already in use by another or unhealthy service." >&2
  echo "Run 'stop.bat' directly in CMD (without 'bash'), then start again." >&2
  exit 1
fi

# Prefer Conda by environment name. The absolute paths keep the launcher usable
# from Git Bash or WSL when Conda has not initialized shell integration.
if command -v conda >/dev/null 2>&1; then
  BACKEND_COMMAND=(conda run --no-capture-output -n badminton python)
elif [[ -f "C:/Users/14181/.conda/envs/badminton/python.exe" ]]; then
  BACKEND_COMMAND=("C:/Users/14181/.conda/envs/badminton/python.exe")
elif [[ -f "/mnt/c/Users/14181/.conda/envs/badminton/python.exe" ]]; then
  BACKEND_COMMAND=("/mnt/c/Users/14181/.conda/envs/badminton/python.exe")
else
  echo "Error: Conda environment 'badminton' was not found." >&2
  echo "Set up the environment or add conda to PATH, then run this script again." >&2
  exit 1
fi

BACKEND_PID=""
FRONTEND_PID=""

stop_services() {
  trap - EXIT INT TERM
  for pid in "$FRONTEND_PID" "$BACKEND_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$FRONTEND_PID" "$BACKEND_PID"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}

trap stop_services EXIT INT TERM

echo "Starting Good-Badminton..."
echo "  Backend:  http://127.0.0.1:8000"
echo "  Frontend: http://127.0.0.1:5173"
echo "Press Ctrl+C to stop both services."

(
  cd "$PROJECT_DIR"
  "${BACKEND_COMMAND[@]}" -m uvicorn backend.main:app \
    --host 127.0.0.1 --port 8000 --reload
) &
BACKEND_PID=$!

(
  cd "$FRONTEND_DIR"
  npm run dev -- --host 127.0.0.1 --strictPort
) &
FRONTEND_PID=$!

# Stop both services if either one exits.
set +e
wait -n "$BACKEND_PID" "$FRONTEND_PID"
EXIT_CODE=$?
set -e
exit "$EXIT_CODE"
