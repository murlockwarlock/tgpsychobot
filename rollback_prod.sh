#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env.deploy" ]]; then
    set -a
    source "$ROOT_DIR/.env.deploy"
    set +a
fi

HOST="${PROD_HOST:-}"
USER_NAME="${PROD_USER:-root}"
PASSWORD="${PROD_PASSWORD:-}"
SSH_KEY="${PROD_SSH_KEY:-$HOME/.ssh/yc_key}"
REMOTE_DIR="${PROD_REMOTE_DIR:-/root/telegram_bots/newbots}"
REMOTE_PY="${PROD_REMOTE_PY:-/root/telegram_bots/venv/bin/python}"
REMOTE_RUNTIME_ENV="${PROD_RUNTIME_ENV:-/root/telegram_bots/runtime.env}"
PM2_CONFIG="${PROD_PM2_CONFIG:-ecosystem.config.js}"
REQUESTED_PM2_NAMES="${PROD_PM2_NAMES:-}"
EXPLICIT_ARCHIVE="${PROD_ROLLBACK_ARCHIVE:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            cat <<'EOF'
Usage: ./rollback_prod.sh [OPTIONS]

Safely rolls back production bots to a pre-deploy backup archive.
Reconciles target directory, removes orphan files from bad releases without
touching runtime secrets or database assets, recompiles bytecode, and reloads
targeted PM2 processes.

Options:
  --archive PATH   Specific pre-deploy tar.gz backup archive to restore.
                   Defaults to the path in /root/telegram_bots/backups/LATEST_PRE_DEPLOY_ARCHIVE.
  -h, --help       Show this help message.

Environment variables:
  PROD_HOST              Required. Target VPS host (or set in .env.deploy).
  PROD_USER              Optional. Defaults to root.
  PROD_PASSWORD          Optional. SSH password.
  PROD_SSH_KEY           Optional. Defaults to ~/.ssh/yc_key.
  PROD_REMOTE_DIR        Optional. Defaults to /root/telegram_bots/newbots.
  PROD_REMOTE_PY         Optional. Defaults to /root/telegram_bots/venv/bin/python.
  PROD_RUNTIME_ENV       Optional. Defaults to /root/telegram_bots/runtime.env.
  PROD_PM2_CONFIG        Optional. Defaults to ecosystem.config.js.
  PROD_PM2_NAMES         Optional. Comma-separated list of PM2 processes to reload.
  PROD_ROLLBACK_ARCHIVE  Optional. Explicit archive path on remote host.
EOF
            exit 0
            ;;
        --archive)
            EXPLICIT_ARCHIVE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

PM2_NAMES="${REQUESTED_PM2_NAMES:-$(python3 scripts/verify_bot_instances.py --print-pm2-names)}"
if [[ ! "$PM2_NAMES" =~ ^[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*$ ]]; then
    echo "Refusing to rollback: PROD_PM2_NAMES contains invalid process names." >&2
    exit 1
fi

if [[ -z "$HOST" ]]; then
    echo "Set PROD_HOST in .env.deploy or the environment." >&2
    exit 1
fi

SSH_CMD=(ssh -o StrictHostKeyChecking=no)
if [[ -f "$SSH_KEY" ]]; then
    SSH_CMD+=(-i "$SSH_KEY")
fi
if [[ -n "$PASSWORD" ]]; then
    export SSHPASS="$PASSWORD"
    SSH_CMD=(sshpass -e "${SSH_CMD[@]}")
fi

echo "Initiating production rollback on ${USER_NAME}@${HOST}:${REMOTE_DIR}"

"${SSH_CMD[@]}" "${USER_NAME}@${HOST}" \
    "set -euo pipefail && \
     cd '${REMOTE_DIR}' && \
     ARCHIVE='${EXPLICIT_ARCHIVE}' && \
     if [[ -z \"\$ARCHIVE\" ]]; then \
         POINTER_FILE='/root/telegram_bots/backups/LATEST_PRE_DEPLOY_ARCHIVE'; \
         if [[ ! -f \"\$POINTER_FILE\" ]]; then \
             echo \"Refusing rollback: no explicit archive specified and \$POINTER_FILE does not exist.\" >&2; \
             exit 1; \
         fi; \
         ARCHIVE=\$(cat \"\$POINTER_FILE\"); \
     fi && \
     if [[ ! -f \"\$ARCHIVE\" ]]; then \
         echo \"Refusing rollback: backup archive '\$ARCHIVE' not found.\" >&2; \
         exit 1; \
     fi && \
     echo \"Validating backup archive '\$ARCHIVE'...\" && \
     tar -tzf \"\$ARCHIVE\" >/dev/null && \
     echo \"Reconciling orphan files in '${REMOTE_DIR}' against archive...\" && \
     '${REMOTE_PY}' 'scripts/reconcile_rollback.py' --archive \"\$ARCHIVE\" --target-dir '${REMOTE_DIR}' && \
     echo \"Extracting restored files from archive...\" && \
     tar -xzf \"\$ARCHIVE\" -C '${REMOTE_DIR}' && \
     if [[ -f '${REMOTE_RUNTIME_ENV}' ]]; then set -a; source '${REMOTE_RUNTIME_ENV}'; set +a; fi && \
     find . -type f -name '*.py' ! -name '._*' -print0 | xargs -0 '${REMOTE_PY}' -m py_compile && \
     '${REMOTE_PY}' 'scripts/verify_bot_instances.py' \
         --validate-runtime-env \
         --runtime-env '${REMOTE_RUNTIME_ENV}' \
         --pm2-names '${PM2_NAMES}' && \
     echo \"Reloading targeted PM2 processes (${PM2_NAMES})...\" && \
     pm2 startOrReload '${PM2_CONFIG}' --only '${PM2_NAMES}' --update-env && \
     sleep 5 && \
     pm2 status && \
     '${REMOTE_PY}' 'scripts/verify_bot_instances.py' \
         --runtime \
         --runtime-env '${REMOTE_RUNTIME_ENV}' \
         --pm2-names '${PM2_NAMES}' \
         --steady-state && \
     RESTORED_REV=\$(cat REVISION 2>/dev/null || echo 'unknown') && \
     '${REMOTE_PY}' 'scripts/verify_prod_runtime.py' \
         --revision \"\$RESTORED_REV\" \
         --pm2-names '${PM2_NAMES}' \
         --root '${REMOTE_DIR}' \
         --startup-settle-seconds 10"

echo "Rollback complete."
