# Operator Guide: Isolated Release Manager

This guide describes how to operate the versioned, isolated release management system for the Psychobot fleet (`scripts/release_manager.py`).

---

## 1. Directory Layout & Architecture

```text
/root/telegram_bots/
├── releases/
│   └── <commit_sha_40>/        # Immutable release tree
│       ├── REVISION            # 40-character commit SHA
│       ├── requirements.txt
│       ├── requirements.sha256
│       └── ...
├── deployments/
│   ├── tg_autobusbusbot_new -> /root/telegram_bots/releases/<sha_A>
│   ├── max_se13639182_bot_legacy -> /root/telegram_bots/releases/<sha_B>
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

Stages the immutable release into `releases/<commit_sha>`, atomically updates ONLY the targeted instance's symlink, checks for ghost processes holding only its configured port, and reloads only that instance via PM2 (`--only <name> --update-env`).

```bash
python3 scripts/release_manager.py deploy \
    --canary tg_autobusbusbot_new \
    --target-sha <commit_sha> \
    --settle-seconds 3
```

- If post-reload healthcheck fails, the tool automatically rolls back the symlink and reloads the previous working version.
- Other instances on the same host or sharing the same database remain completely unaffected.

---

## 5. Rollback of an Instance (`rollback`)

Atomically points the instance's deployment symlink back to a previous release and reloads only that instance:

```bash
# Roll back to the immediately preceding release
python3 scripts/release_manager.py rollback \
    --instance tg_autobusbusbot_new

# Roll back to a specific commit SHA
python3 scripts/release_manager.py rollback \
    --instance tg_autobusbusbot_new \
    --target-sha <previous_sha>
```

---

## 6. Migrating a Legacy Instance (`migrate-instance`)

Transitions an unmigrated bot running from `/newbots` to the isolated layout:

```bash
python3 scripts/release_manager.py migrate-instance \
    --instance tg_autobusbusbot_new \
    --target-sha <commit_sha>
```

---

## 7. Pruning Old Inactive Releases (`prune`)

Prunes unreferenced releases while protecting all releases currently pointed to by any `deployments/*` symlink:

```bash
# Dry run to see what would be pruned
python3 scripts/release_manager.py prune --keep 5 --dry-run

# Prune keeping latest 5 releases plus all active releases
python3 scripts/release_manager.py prune --keep 5
```
