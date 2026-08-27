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
PM2_NAMES="${PROD_PM2_NAMES:-$(python3 scripts/verify_bot_instances.py --print-pm2-names)}"
ALLOW_BOT_INSTANCE_PM2_RENAME="${PROD_ALLOW_PM2_RENAME:-0}"

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            cat <<'EOF'
Usage: ./deploy_prod.sh

Environment variables:
  PROD_PASSWORD    Optional. SSH password; without it an SSH key is used.
  PROD_SSH_KEY     Optional. Defaults to ~/.ssh/yc_key.
  PROD_HOST        Required. It may be set in the local .env.deploy.
  PROD_USER        Optional. Defaults to root.
  PROD_REMOTE_DIR  Optional. Defaults to /root/telegram_bots/newbots.
  PROD_REMOTE_PY   Optional. Defaults to /root/telegram_bots/venv/bin/python.
  PROD_RUNTIME_ENV Optional. Remote untracked runtime environment file.
  PROD_PM2_NAMES   Optional. Comma-separated PM2 process names to reload.
  PROD_ALLOW_PM2_RENAME=1 required when replacing a legacy PM2 name.
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

if [[ ! "$PM2_NAMES" =~ ^[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*$ ]]; then
    echo "Refusing to deploy: PROD_PM2_NAMES contains invalid process names." >&2
    exit 1
fi
if [[ "$ALLOW_BOT_INSTANCE_PM2_RENAME" != "0" && "$ALLOW_BOT_INSTANCE_PM2_RENAME" != "1" ]]; then
    echo "Refusing to deploy: PROD_ALLOW_PM2_RENAME must be 0 or 1." >&2
    exit 1
fi
LEGACY_PM2_NAMES="$(python3 scripts/verify_bot_instances.py --print-legacy-pm2-names --pm2-names "$PM2_NAMES")"
if [[ -n "$LEGACY_PM2_NAMES" && ! "$LEGACY_PM2_NAMES" =~ ^[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*$ ]]; then
    echo "Refusing to deploy: registry contains invalid legacy process names." >&2
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Git is not initialized in $ROOT_DIR" >&2
    exit 1
fi

CURRENT_BRANCH="$(git symbolic-ref --quiet --short HEAD || true)"
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    if [[ -z "$CURRENT_BRANCH" ]]; then
        echo "Refusing to deploy: detached HEAD; production deploys require branch 'main'." >&2
    else
        echo "Refusing to deploy from branch '$CURRENT_BRANCH'; production deploys require 'main'." >&2
    fi
    exit 1
fi

if ! git fetch origin main; then
    echo "Refusing to deploy: git fetch origin main failed." >&2
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo "Refusing to deploy a dirty worktree." >&2
    exit 1
fi

if ! LOCAL_HEAD="$(git rev-parse --verify HEAD)"; then
    echo "Refusing to deploy: unable to resolve local HEAD." >&2
    exit 1
fi
if ! REMOTE_MAIN="$(git rev-parse --verify refs/remotes/origin/main^{commit})"; then
    echo "Refusing to deploy: unable to resolve origin/main after fetch." >&2
    exit 1
fi

DIVERGENCE="$(git rev-list --left-right --count HEAD...refs/remotes/origin/main)"
read -r LOCAL_ONLY BEHIND <<<"$DIVERGENCE"
if [[ "$LOCAL_ONLY" != "0" ]]; then
    echo "Refusing to deploy: local main has $LOCAL_ONLY local-only commit(s)." >&2
    exit 1
fi
if [[ "$BEHIND" != "0" ]]; then
    echo "Refusing to deploy: local main is behind origin/main by $BEHIND commit(s)." >&2
    exit 1
fi
if [[ "$LOCAL_HEAD" != "$REMOTE_MAIN" ]]; then
    echo "Refusing to deploy: local HEAD $LOCAL_HEAD does not equal origin/main $REMOTE_MAIN." >&2
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
     '${REMOTE_PY}' 'scripts/verify_bot_instances.py' \
         --validate-runtime-env \
         --runtime-env '${REMOTE_RUNTIME_ENV}' \
         --pm2-names '${PM2_NAMES}' && \
     baseline_path= && \
     trap 'status=\$?; if [[ -n "\$baseline_path" ]]; then rm -f -- "\$baseline_path" 2>/dev/null || true; fi; trap - EXIT; exit "\$status"' EXIT && \
     baseline_path=\$('${REMOTE_PY}' 'scripts/verify_prod_runtime.py' \
         --create-log-baseline \
         --pm2-names '${PM2_NAMES}' \
         --baseline-source-names '${LEGACY_PM2_NAMES}') && \
     [[ -n "\$baseline_path" ]] && \
     legacy_names='${LEGACY_PM2_NAMES}' && \
     if [[ -n "\$legacy_names" ]]; then \
         '${REMOTE_PY}' 'scripts/verify_bot_instances.py' \
             --runtime \
             --runtime-env '${REMOTE_RUNTIME_ENV}' \
             --pm2-names '${LEGACY_PM2_NAMES}' \
             --allow-legacy \
             --allow-missing; \
     fi && \
     if [[ -n "\$legacy_names" ]]; then \
         IFS=',' read -r -a legacy_name_list <<< "\$legacy_names"; \
         for legacy_name in "\${legacy_name_list[@]}"; do \
             if pm2 describe "\$legacy_name" >/dev/null 2>&1; then \
                 if [[ '${ALLOW_BOT_INSTANCE_PM2_RENAME}' != '1' ]]; then \
                     echo "Refusing to replace legacy PM2 process \$legacy_name; set PROD_ALLOW_PM2_RENAME=1 for the reviewed rename." >&2; \
                     exit 1; \
                 fi; \
                 pm2 delete "\$legacy_name"; \
             fi; \
         done; \
     fi && \
     if [[ -n "\$legacy_names" ]]; then \
         for legacy_name in "\${legacy_name_list[@]}"; do \
             if pm2 describe "\$legacy_name" >/dev/null 2>&1; then \
                 echo "Refusing to start canonical PM2 processes while legacy process \$legacy_name remains." >&2; \
                 exit 1; \
             fi; \
         done; \
     fi && \
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
     pm2 startOrReload '${PM2_CONFIG}' --only '${PM2_NAMES}' --update-env && \
     sleep 10 && \
     pm2 status && \
     '${REMOTE_PY}' 'scripts/verify_bot_instances.py' \
         --runtime \
         --runtime-env '${REMOTE_RUNTIME_ENV}' \
         --pm2-names '${PM2_NAMES}' && \
     '${REMOTE_PY}' 'scripts/verify_prod_runtime.py' \
         --revision '${REVISION}' \
         --pm2-names '${PM2_NAMES}' \
         --root '${REMOTE_DIR}' \
         --log-baseline "\$baseline_path""

echo "Deploy complete."
