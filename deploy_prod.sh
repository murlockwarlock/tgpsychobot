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
PM2_NAMES="${PROD_PM2_NAMES:-psy5d_new,veraveda_new,someone01_new,someone02_new,veraveda_legacy,someone02_legacy,someone01_legacy,psy5d_legacy,test01_legacy,test02_legacy,someone03_new,someone04_new,someone05_new,someone06_new,someone07_new,yourself_way_bot,max_veraveda_legacy,max_yourself_way}"

ALLOW_DIRTY=0

for arg in "$@"; do
    case "$arg" in
        --allow-dirty)
            ALLOW_DIRTY=1
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./deploy_prod.sh [--allow-dirty]

Environment variables:
  PROD_PASSWORD    Optional. SSH password; without it an SSH key is used.
  PROD_SSH_KEY     Optional. Defaults to ~/.ssh/yc_key.
  PROD_HOST        Required. It may be set in the local .env.deploy.
  PROD_USER        Optional. Defaults to root.
  PROD_REMOTE_DIR  Optional. Defaults to /root/telegram_bots/newbots.
  PROD_REMOTE_PY   Optional. Defaults to /root/telegram_bots/venv/bin/python.
  PROD_RUNTIME_ENV Optional. Remote untracked runtime environment file.
  PROD_PM2_NAMES   Optional. Comma-separated PM2 process names to reload.
  PROD_BLOCKED_HOSTS Optional comma-separated denylist for accidental deploy protection.
  PROD_ALLOW_BLOCKED_HOST=1 overrides that protection for an intentional rollback.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Git is not initialized in $ROOT_DIR" >&2
    exit 1
fi

if [[ -z "$HOST" ]]; then
    echo "Set PROD_HOST in .env.deploy or the environment." >&2
    exit 1
fi

if [[ ",${PROD_BLOCKED_HOSTS:-}," == *",${HOST},"* && "${PROD_ALLOW_BLOCKED_HOST:-0}" != "1" ]]; then
    echo "Refusing to deploy to blocked host $HOST." >&2
    exit 1
fi

if [[ "$ALLOW_DIRTY" -ne 1 ]]; then
    if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
        echo "Refusing to deploy a dirty worktree." >&2
        echo "Commit your changes first or rerun with --allow-dirty." >&2
        exit 1
    fi
fi

TRACKED_FILES=()
while IFS= read -r -d '' file; do
    TRACKED_FILES+=("$file")
done < <(git ls-files -z)
if [[ "${#TRACKED_FILES[@]}" -eq 0 ]]; then
    echo "No tracked files to deploy." >&2
    exit 1
fi

REVISION="$(git rev-parse --short HEAD)"
echo "Deploying revision $REVISION to ${USER_NAME}@${HOST}:${REMOTE_DIR}"

SSH_CMD=(ssh -o StrictHostKeyChecking=no)
if [[ -f "$SSH_KEY" ]]; then
    SSH_CMD+=(-i "$SSH_KEY")
fi
if [[ -n "$PASSWORD" ]]; then
    export SSHPASS="$PASSWORD"
    SSH_CMD=(sshpass -e "${SSH_CMD[@]}")
fi

tar czf - -- "${TRACKED_FILES[@]}" | \
"${SSH_CMD[@]}" "${USER_NAME}@${HOST}" \
    "cd '${REMOTE_DIR}' && tar xzf -"

"${SSH_CMD[@]}" "${USER_NAME}@${HOST}" \
    "cd '${REMOTE_DIR}' && \
     if [[ -f '${REMOTE_RUNTIME_ENV}' ]]; then set -a; source '${REMOTE_RUNTIME_ENV}'; set +a; fi && \
     printf '%s\n' '${REVISION}' > REVISION && \
     find . -type f -name '*.py' ! -name '._*' -print0 | xargs -0 '${REMOTE_PY}' -m py_compile && \
     echo 'Checking for ghost processes on ports 8080-8100...' && \
     pm2_pids=\$(pm2 jlist | grep -o '\"pid\":[0-9]*' | cut -d: -f2 | tr '\n' ' ') && \
     for port in {8080..8100}; do \
         for pid in \$(command -v lsof >/dev/null && lsof -t -i :\$port || true); do \
             if [[ ! \" \$pm2_pids \" =~ \" \$pid \" ]]; then \
                 echo \"Killing ghost process \$pid holding port \$port\" && \
                 kill -9 \"\$pid\" || true; \
             fi; \
         done; \
     done && \
     pm2 reload ecosystem.config.js --only ${PM2_NAMES} --update-env && \
     sleep 10 && \
     pm2 status"

echo "Deploy complete."
