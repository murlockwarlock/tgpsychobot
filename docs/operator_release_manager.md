# Operator Guide: Isolated Release Manager

This guide describes how to operate the versioned, isolated release management system for the Psychobot fleet (`scripts/release_manager.py`).

---

## 1. Directory Layout & Architecture

```text
/root/telegram_bots/
├── releases/
│   ├── .staging.lock           # Concurrency staging lock (flock)
│   ├── .prune.lock             # Concurrency release pruning lock (flock)
│   └── <commit_sha_40>/        # Immutable release tree
│       ├── REVISION            # 40-character commit SHA
│       ├── requirements.txt
│       ├── requirements.sha256
│       ├── MANIFEST.sha256     # Cryptographic SHA256 manifest of all staged files
│       └── ...
├── deployments/
│   ├── .state/                 # Per-instance deployment state metadata (fail-closed if corrupted)
│   │   ├── tg_autobusbusbot_new.json
│   │   └── ...
│   ├── .locks/                 # Per-instance deployment & rollback locks (flock)
│   │   ├── tg_autobusbusbot_new.lock
│   │   └── ...
│   ├── tg_autobusbusbot_new -> /root/telegram_bots/releases/<sha_A>
│   ├── max_se13639182_bot_legacy -> /root/telegram_bots/releases/<sha_B>
│   └── ...
├── venvs/
│   ├── <requirements_hash>/    # Hash-addressed isolated virtual environments
│   │   └── bin/python
│   └── ...
├── shared/
│   ├── runtime.env
│   ├── logs/
│   └── backups/
└── newbots/                    # Legacy static directory for unmigrated bots
```

- Each migrated instance points via `/root/telegram_bots/deployments/<instance_name>` directly to its immutable release in `releases/<sha>`.
- Unmigrated instances continue running out of `/root/telegram_bots/newbots`.
- Unmanaged processes (`vitaliy_bot`, `darimiru_bot`, etc.) are protected and never touched.
- Release directories are cryptographically verified against `MANIFEST.sha256` to detect post-staging file modifications or unauthorized file additions.
- Python dependencies are isolated per requirements hash in `/root/telegram_bots/venvs/<hash>/bin/python`, resolved consistently across `ecosystem.config.js` and `release_manager.py`.

---

## 2. Fleet Status Inspection (`status`)

Inspects running processes directly via PM2 and reads the real runtime SHA from the actual running process's working directory / symlink (resolving `/proc/<pid>/cwd` or `pm_cwd` to `releases/<sha>/REVISION`):

```bash
# Print formatted human-readable table
python3 scripts/release_manager.py status

# Verify matching against a target commit SHA
python3 scripts/release_manager.py status --expected-sha e56f707c22917aa2de86bc763ec0918681a5b845

# Output machine-readable JSON
python3 scripts/release_manager.py status --json
```

Example Output:
```text
=== MANAGED BOT INSTANCES ===
INSTANCE                         PLATFORM   PORT   STATUS    PID      MODE       RUNTIME SHA  MATCH   
------------------------------------------------------------------------------------------------------
tg_autobusbusbot_new             telegram   8081   online    14021    isolated   e56f707c     MATCH   
tg_someonewithyou01_bot_new      telegram   8083   online    12845    isolated   e56f707c     MATCH   
max_se13639182_bot_legacy        max        8099   online    14088    isolated   e56f707c     MATCH   
tg_psy5d_bot_legacy              telegram   8087   online    9102     legacy     42682484     MISMATCH

=== UNMANAGED PROCESSES (PROTECTED - NEVER TOUCHED) ===
NAME                         STATUS     PID      CWD                                     
-----------------------------------------------------------------------------------------
vitaliy_bot                  online     8812     /root/telegram_bots/vitaliy_bot         
darimiru_bot                 online     8815     /root/telegram_bots/darimiru_bot        
```

---

## 3. Preflight Validation (`preflight`)

Verifies git cleanliness, dependency compatibility (`requirements.sha256` matching active releases or isolated venv availability), and PR A database schema guards (blocks unapplied migrations):

```bash
python3 scripts/release_manager.py preflight --target-sha <commit_sha>
```

---

## 4. Canary Deployment (`deploy --canary`)

Stages the immutable release directly from Git commit tree (`git archive <sha>`), verifies 40-char SHA integrity and `MANIFEST.sha256`, updates ONLY the targeted instance's symlink atomically via `os.replace`, checks for port conflicts, and reloads only that instance via PM2 (`--only <name> --update-env`).

```bash
python3 scripts/release_manager.py deploy \
    --canary tg_autobusbusbot_new \
    --target-sha <commit_sha> \
    --settle-seconds 3
```

- **Per-Instance Locking**: An exclusive non-blocking lock (`deployments/.locks/<instance>.lock` via `fcntl.flock`) serializes deployments and rollbacks per instance. Concurrent operations on the same bot fail fast with `RuntimeError`. Global operations (staging, pruning) use dedicated flock files (`releases/.staging.lock`, `releases/.prune.lock`).
- **Deep Readiness Verification**:
  - Probes the configured application port (`APP_PORT` / `MAX_APP_PORT`) via TCP `socket.create_connection` with a 3.0s retry window.
  - Verifies that PM2 status is `online` and PID is alive.
  - Verifies `pm2_env.unstable_restarts == 0`.
  - Scans recent PM2 error logs for fatal exceptions (`ModuleNotFoundError`, `ImportError`, `Address already in use`, fatal tracebacks), while safely distinguishing normal graceful shutdown tracebacks (`KeyboardInterrupt`, `SystemExit`).
- **Fail-Safe Automatic Rollback**: If post-reload readiness check or runtime SHA verification fails, the release manager automatically reverts the instance symlink, restores previous state, and restarts the previously running version. If a rollback itself were to fail, the symlink is reverted to the pre-rollback state and an emergency diagnostic is logged.
- **Neighbor Isolation**: Other instances on the same host or sharing the same database remain completely unaffected.

---

## 5. Rollback of an Instance (`rollback`)

Rollback is strictly deterministic and governed by the instance's history stored in `deployments/.state/<instance>.json`:

```bash
# Roll back to the immediately preceding confirmed release recorded in instance state
python3 scripts/release_manager.py rollback \
    --instance tg_autobusbusbot_new

# Roll back to a specific commit SHA
python3 scripts/release_manager.py rollback \
    --instance tg_autobusbusbot_new \
    --target-sha <previous_sha>
```

### Why Rollback Never Relies on Directory `mtime`:
In a multi-tenant or multi-bot architecture, staging release C for Bot 1 updates timestamps or creates new directories in `/releases`. If Bot 2 (running release B) rolls back using directory modification times, it would incorrectly roll to release C or the newest staged release.
Instead:
- Every deployment updates `deployments/.state/<instance>.json`:
  ```json
  {
    "instance_name": "tg_autobusbusbot_new",
    "current_release_sha": "e56f707c22917aa2de86bc763ec0918681a5b845",
    "previous_successful_release_sha": "6150e530aea37ac225babebd79c2702ae4fe83f8",
    "previous_successful_target_dir": "/root/telegram_bots/releases/6150e530aea37ac225babebd79c2702ae4fe83f8",
    "mode": "isolated",
    "updated_at": "2026-11-10T02:00:00Z"
  }
  ```
- When `--target-sha` is not provided, the rollback command reads `previous_successful_release_sha` from the instance state.
- If the state file is corrupted or unparsable, the release manager fails closed with `RuntimeError` rather than silently discarding history.
- If the bot was previously running in `legacy` mode (`newbots`), rolling back removes the symlink, restores legacy execution in PM2, and health-checks the restored legacy process.

---

## 6. Port Conflict & Process Ownership Verification (`verify_process_ownership`)

Before reloading an instance, the release manager inspects the configured port:
- **Fail-Closed Inspection Tools**: Checks strictly TCP LISTEN sockets using `lsof -sTCP:LISTEN`, `ss -ltnp`, or `netstat -ltnp`. If all socket inspection tools are missing, the system fails closed with `RuntimeError`.
- If a process is listening on the port, the release manager checks whether the PID is managed by PM2.
- If not managed by PM2, it checks **authoritative process ownership**:
  - Checks `/proc/<pid>/cmdline` (or `ps -p <pid> -o command=`).
  - Checks `/proc/<pid>/cwd` (or `lsof -p <pid>`).
  - The process is identified as a stale ghost ONLY IF its cmdline and cwd match the instance's deployment or legacy directory and script (`main.py` or `-m max_messenger_bot.app`).
  - **APP_PORT environment alone is NEVER accepted as proof of ownership**, preventing accidental killing of foreign processes that happen to use the same port number.
- **Fail-Closed Semantics**: If the process belongs to another service (e.g. system daemon, unrelated bot, foreign service), it is **NEVER killed**. The release manager aborts with `RuntimeError` describing the foreign PID, process name, and user, requiring manual operator intervention.

---

## 7. Legacy Deploy Protection & Bootstrap Safety (`--check-legacy-safety`)

To prevent accidental legacy deployments (`deploy_prod.sh`) from overwriting code or restarting bots that have already been migrated to isolated releases:

1. `scripts/verify_bot_instances.py --check-legacy-safety --base-dir <root>`:
   - Scans `deployments/*` for active isolated symlinks.
   - If an operator tries to run `deploy_prod.sh --target <instance>` for an instance that is already migrated to isolated deployments, the command fails immediately with an error and instructions to use `scripts/release_manager.py deploy`.
2. **Safe Initial Deploy Bootstrap**:
   - During the first deployment from old `main` to a server where `--check-legacy-safety` is not yet available in the remote script, `deploy_prod.sh` detects this via `--help`.
   - Instead of bypassing safety or crashing, it executes a remote inline Python check inspecting whether `/root/telegram_bots/deployments` exists and contains migrated bots.
   - If no instances are migrated, the initial bootstrap deployment is permitted. If any instances are already migrated, legacy deployment is strictly refused.

---

## 8. Migrating a Legacy Instance (`migrate-instance`)

Transitions an unmigrated bot running from `/newbots` to the isolated layout:

```bash
python3 scripts/release_manager.py migrate-instance \
    --instance tg_autobusbusbot_new \
    --target-sha <commit_sha>
```

When switching from legacy to isolated mode, the release manager performs a fresh PM2 cutover (`pm2 delete <instance>` followed by starting the instance from `ecosystem.config.js`) to guarantee PM2 evaluates the new cwd and interpreter without worker caching.

---

## 9. Pruning Old Inactive Releases (`prune`)

Prunes unreferenced releases while protecting:
- All releases currently pointed to by any `deployments/<instance>` symlink.
- All releases referenced as `previous_successful_release_sha` across all `deployments/.state/*.json` files.
- Even if an operator runs `stage_release --force` or `prune --keep 0`, protected releases cannot be deleted.

```bash
# Dry run to see what would be pruned
python3 scripts/release_manager.py prune --keep 5 --dry-run

# Prune keeping latest 5 releases plus all protected releases
python3 scripts/release_manager.py prune --keep 5
```

---

## 10. Emergency Recovery & Manual Fallback

In the event of an infrastructure or release manager failure:

### A. Manual Rollback of an Isolated Bot
If `release_manager.py` cannot be run:
1. Re-point the instance symlink:
   ```bash
   ln -sfn /root/telegram_bots/releases/<previous_working_sha> /root/telegram_bots/deployments/<instance_name>
   ```
2. Restart the PM2 process:
   ```bash
   pm2 restart <instance_name> --update-env
   ```

### B. Fallback to Legacy Directory
If a migrated bot needs to be urgently restored to the legacy `/newbots` directory:
1. Remove the isolated deployment symlink:
   ```bash
   rm /root/telegram_bots/deployments/<instance_name>
   ```
2. Delete and recreate the PM2 process from `ecosystem.config.js`:
   ```bash
   pm2 delete <instance_name>
   pm2 start /root/telegram_bots/ecosystem.config.js --only <instance_name> --update-env
   ```
3. Update or remove the state file:
   ```bash
   rm /root/telegram_bots/deployments/.state/<instance_name>.json
   ```

### C. Foreign Port Conflict Resolution
If deployment aborts due to `PORT_CONFLICT: Port <port> is occupied by foreign process <pid> (<cmd>)`:
1. Check who owns the port:
   ```bash
   lsof -i :<port>
   ```
2. If it is an orphaned test process or incorrect service, stop it explicitly.
3. Re-run `release_manager.py deploy --canary <instance> --target-sha <sha>`.


