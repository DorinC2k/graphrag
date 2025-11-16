#!/usr/bin/env bash
# start-azurite-cross.sh
# Windows (Git Bash/MSYS/Cygwin): run npx
# Ubuntu/WSL: run the Linux azurite shim from $(npm bin -g)

set -euo pipefail

# Config
DATA_DIR="${AZURITE_DATA_DIR:-./temp_azurite}"
LOG_FILE="${AZURITE_LOG_FILE:-$DATA_DIR/debug.log}"

BLOB_HOST="${AZURITE_BLOB_HOST:-0.0.0.0}"
QUEUE_HOST="${AZURITE_QUEUE_HOST:-0.0.0.0}"
TABLE_HOST="${AZURITE_TABLE_HOST:-0.0.0.0}"

BLOB_PORT="${AZURITE_BLOB_PORT:-10000}"
QUEUE_PORT="${AZURITE_QUEUE_PORT:-10001}"
TABLE_PORT="${AZURITE_TABLE_PORT:-10002}"

is_windows_shell() {
  case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) return 0 ;;
    *) return 1 ;;
  esac
}

is_windows_path() {
  [[ "$1" == /mnt/c/* ]]
}

run_windows() {
  echo "[azurite] Detected Windows shell - running with npx"
  mkdir -p "$DATA_DIR"
  exec npx --yes azurite -L -l "$DATA_DIR" -d "$LOG_FILE"
}

run_ubuntu_like() {
  echo "[azurite] Detected Ubuntu/WSL - using Linux npm shim"
  mkdir -p "$DATA_DIR"

  local node_bin npm_bin
  node_bin="$(command -v node || true)"
  npm_bin="$(command -v npm  || true)"
  if [[ -z "$node_bin" || -z "$npm_bin" ]] || is_windows_path "$node_bin" || is_windows_path "$npm_bin"; then
    echo "[azurite] No Linux Node or npm found, or they point to Windows."
    echo "[azurite] Install Node via nvm inside WSL and re-run:"
    echo "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash"
    echo "  . \"\$HOME/.nvm/nvm.sh\" && nvm install --lts && nvm use --lts"
    exit 1
  fi

  # ensure azurite is installed, then use the global shim
  local AZ_SHIM
  AZ_SHIM="$(command -v azurite || true)"
  if [[ -z "$AZ_SHIM" ]]; then
    echo "[azurite] Installing azurite globally in Linux npm..."
    npm i -g azurite
    AZ_SHIM="$(command -v azurite || true)"
  fi
  if [[ -z "$AZ_SHIM" ]]; then
    # fallback to npm bin -g if PATH didn’t pick it up yet
    AZ_SHIM="$(npm bin -g)/azurite"
  fi

  if [[ ! -x "$AZ_SHIM" ]]; then
    echo "[azurite] Could not locate the azurite executable shim."
    echo "[azurite] Check: npm bin -g; ls \"$(npm bin -g)\""
    exit 1
  fi

  echo "[azurite] Starting Azurite..."
  exec "$AZ_SHIM" \
    -L \
    --blobHost "$BLOB_HOST"   --blobPort "$BLOB_PORT" \
    --queueHost "$QUEUE_HOST" --queuePort "$QUEUE_PORT" \
    --tableHost "$TABLE_HOST" --tablePort "$TABLE_PORT" \
    -l "$DATA_DIR" \
    -d "$LOG_FILE"
}

main() {
  if is_windows_shell; then
    run_windows
  else
    run_ubuntu_like
  fi
}
main "$@"
