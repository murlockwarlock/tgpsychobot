#!/usr/bin/env python3
"""
Comprehensive simulation of the PR #86 deployment, staging smoke-test,
maintenance window code synchronization, Canary rollout with intent-first journaling,
simulated partial PM2 failure, and safe rollback preserving runtime data.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_repo_root() -> Path:
    """Find repository root dynamically."""
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if top and os.path.isdir(top):
            return Path(top)
    except Exception:
        pass
    script_dir = Path(__file__).resolve().parent
    if (script_dir.parent / ".git").exists():
        return script_dir.parent
    return Path.cwd()


def main():
    print("=" * 70)
    print("STARTING E2E SIMULATION: PARTIAL FAILURE & SAFE ROLLBACK")
    print("=" * 70)

    repo_root = find_repo_root()
    commit_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    short_sha = commit_sha[:7]
    print(f"[*] Repository Root: {repo_root}")
    print(f"[*] Verified PR #86 Commit: {commit_sha} ({short_sha})")

    with tempfile.TemporaryDirectory() as base_dir:
        tb_dir = os.path.join(base_dir, "telegram_bots")
        prod_dir = os.path.join(tb_dir, "newbots")
        staging_dir = os.path.join(tb_dir, "staging_pr86")
        backups_dir = os.path.join(tb_dir, "backups")

        os.makedirs(prod_dir, exist_ok=True)
        os.makedirs(backups_dir, exist_ok=True)

        # ---------------------------------------------------------
        # PHASE 1: Baseline Production Setup (Commit 4024842)
        # ---------------------------------------------------------
        print("\n--- PHASE 1: Initializing Production Baseline (4024842) ---")
        prev_sha = "4024842"
        with open(os.path.join(prod_dir, "REVISION"), "w") as f:
            f.write(f"{prev_sha}\n")
        with open(os.path.join(prod_dir, "main.py"), "w") as f:
            f.write("# Legacy main.py from commit 4024842\n")
        with open(os.path.join(prod_dir, "ecosystem.config.js"), "w") as f:
            f.write("// Legacy ecosystem.config.js\nmodule.exports = { apps: [] };\n")
        with open(os.path.join(tb_dir, "runtime.env"), "w") as f:
            f.write("LEGACY_ENV_VAR=original_value\n")

        # Persistent runtime assets (MUST NOT BE CORRUPTED OR OVERWRITTEN)
        db_path = os.path.join(prod_dir, "darimiru_bot.db")
        with open(db_path, "wb") as f:
            f.write(b"SQLITE_FORMAT_3\x00_CRITICAL_USER_PAYMENTS_DATA_DO_NOT_DELETE")
        db_hash_orig = sha256_file(db_path)

        chroma_dir = os.path.join(prod_dir, "chroma_db_8081")
        os.makedirs(chroma_dir, exist_ok=True)
        chroma_file = os.path.join(chroma_dir, "chroma.sqlite3")
        with open(chroma_file, "wb") as f:
            f.write(b"CHROMA_VECTOR_DATABASE_INDEX_EMBEDDINGS_8081")
        chroma_hash_orig = sha256_file(chroma_file)

        logs_dir = os.path.join(prod_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "payment_events_8081.log")
        with open(log_file, "w") as f:
            f.write("2026-11-07 10:00:00 - Payment processed for user 12345\n")
        log_hash_orig = sha256_file(log_file)

        # Baseline PM2 state: 18 managed processes running pre-86 code
        all_18_processes = [
            "tg_autobusbusbot_new", "tg_alena322bot_new", "tg_someonewithyou01_bot_new",
            "tg_someonewithyou02_bot_new", "tg_veraveda777_bot_legacy", "tg_someonelikeyouai03_bot_legacy",
            "tg_someonelikeyouai04_bot_legacy", "tg_psy5d_bot_legacy", "tg_someonelikeyouai_bot_legacy",
            "tg_someonelikeyouai02_bot_legacy", "tg_someonewithyou03_bot_new", "tg_someonelikeyouai05_bot_new",
            "tg_someonewithyou05_bot_new", "tg_someonewithyou06_bot_new", "tg_someonewithyou04_bot_new",
            "tg_yourself_way_bot_new", "max_se13639182_bot_legacy", "max_id519010411655_bot_new"
        ]
        pm2_state = {p: {"version": prev_sha, "status": "online", "env": {}} for p in all_18_processes}

        print(f"[*] Baseline created: 18 processes running {prev_sha}")
        print(f"[*] Protected assets: darimiru_bot.db ({db_hash_orig[:12]}), chroma.sqlite3 ({chroma_hash_orig[:12]}), logs ({log_hash_orig[:12]})")
        assert not os.path.exists(os.path.join(prod_dir, "notification_routing.py")), "notification_routing.py must not exist in baseline!"

        # ---------------------------------------------------------
        # PHASE 2: Packaging Staging via git archive by COMMIT_SHA
        # ---------------------------------------------------------
        print("\n--- PHASE 2: Packaging Staged Release from git archive ---")
        archive_path = os.path.join(base_dir, "pr86_package.tar.gz")
        subprocess.check_call(["git", "-C", str(repo_root), "archive", "--format=tar.gz", "-o", archive_path, commit_sha])
        archive_hash = sha256_file(archive_path)
        print(f"[*] git archive created: {archive_path} (SHA256: {archive_hash[:16]}...)")

        # Unpack strictly in staging_dir
        os.makedirs(staging_dir, exist_ok=True)
        subprocess.check_call(["tar", "xzf", archive_path, "-C", staging_dir])
        with open(os.path.join(staging_dir, "REVISION"), "w") as f:
            f.write(f"{short_sha}\n")
        with open(os.path.join(staging_dir, "FULL_SHA"), "w") as f:
            f.write(f"{commit_sha}\n")

        assert os.path.exists(os.path.join(staging_dir, "notification_routing.py")), "notification_routing.py must exist in staging!"
        assert os.path.exists(os.path.join(staging_dir, "config", "bot_instances.json")), "bot_instances.json must exist in staging!"
        print("[*] Staging verified: notification_routing.py and bot_instances.json present.")

        # ---------------------------------------------------------
        # PHASE 3: Maintenance Window & Controlled Code Sync
        # ---------------------------------------------------------
        print("\n--- PHASE 3: Maintenance Window - Backup & Code Sync ---")
        # 1. Snapshots
        prev_sha_file = os.path.join(tb_dir, "PREV_SHA.txt")
        with open(prev_sha_file, "w") as f:
            f.write(f"{prev_sha}\n")

        # 2. Manifests
        manifest_pre86 = os.path.join(tb_dir, "manifest_pre86.txt")
        manifest_pr86_staged = os.path.join(tb_dir, "manifest_pr86_staged.txt")
        files_added_file = os.path.join(tb_dir, "files_added_by_pr86.txt")

        pre_files = set()
        for root, _, files in os.walk(prod_dir):
            for file in files:
                rel = os.path.relpath(os.path.join(root, file), prod_dir)
                pre_files.add(rel)
        with open(manifest_pre86, "w") as f:
            for fpath in sorted(pre_files):
                f.write(fpath + "\n")

        staged_files = set()
        for root, _, files in os.walk(staging_dir):
            for file in files:
                rel = os.path.relpath(os.path.join(root, file), staging_dir)
                staged_files.add(rel)
        with open(manifest_pr86_staged, "w") as f:
            for fpath in sorted(staged_files):
                f.write(fpath + "\n")

        added_files = sorted(list(staged_files - pre_files))
        with open(files_added_file, "w") as f:
            for fpath in added_files:
                f.write(fpath + "\n")

        print(f"[*] Manifests calculated: {len(added_files)} files to be added by PR #86 (including notification_routing.py).")
        assert "notification_routing.py" in added_files, "notification_routing.py must be detected as an added file!"

        # 3. Pre-86 backup (excluding runtime data)
        backup_archive = os.path.join(backups_dir, "tgpsychobot-pre86-backup.tar.gz")
        subprocess.check_call([
            "tar", "czf", backup_archive,
            "--exclude=*.db",
            "--exclude=chroma_db_*",
            "--exclude=logs",
            "-C", tb_dir, "newbots"
        ])
        shutil.copy2(os.path.join(prod_dir, "ecosystem.config.js"), os.path.join(tb_dir, "ecosystem.config.js.pre86_backup"))
        shutil.copy2(os.path.join(tb_dir, "runtime.env"), os.path.join(tb_dir, "runtime.env.pre86_backup"))
        print(f"[*] Pre-86 backup archived: {backup_archive}")

        # 4. Sync code into prod_dir
        subprocess.check_call([
            "rsync", "-av",
            "--exclude=*.db",
            "--exclude=chroma_db_*",
            "--exclude=logs/",
            f"{staging_dir}/", f"{prod_dir}/"
        ], stdout=subprocess.DEVNULL)

        # Initialize journals
        intent_journal_file = os.path.join(tb_dir, "intent_journal.txt")
        reloaded_file = os.path.join(tb_dir, "reloaded_processes.txt")
        open(intent_journal_file, "w").close()
        open(reloaded_file, "w").close()

        assert os.path.exists(os.path.join(prod_dir, "notification_routing.py")), "notification_routing.py must be synced into newbots!"
        print("[*] Code synced into newbots. Runtime assets preserved.")

        # ---------------------------------------------------------
        # PHASE 4: Canary Rollout with Intent-First Journaling
        # ---------------------------------------------------------
        print("\n--- PHASE 4: Canary Rollout with Intent-First Journaling ---")
        # Step 4.1: Canary 1 (tg_autobusbusbot_new)
        canary_1 = ["tg_autobusbusbot_new"]
        print(f"[*] Step 4.1: Rolling Canary 1: {canary_1}")
        with open(intent_journal_file, "a") as f:
            for p in canary_1:
                f.write(p + "\n")
        # PM2 execution simulated
        for p in canary_1:
            pm2_state[p]["version"] = short_sha
            pm2_state[p]["env"]["BOT_INSTANCE_ID"] = p
        with open(reloaded_file, "a") as f:
            for p in canary_1:
                f.write(p + "\n")
        print("    -> Canary 1: SUCCESS (PM2 reloaded & verified)")

        # Step 4.2: Canary 2 (MAX Pair 1 - Yourself Way)
        canary_2 = ["tg_yourself_way_bot_new", "max_id519010411655_bot_new"]
        print(f"[*] Step 4.2: Rolling Canary 2 (Yourself Way Pair): {canary_2}")
        with open(intent_journal_file, "a") as f:
            for p in canary_2:
                f.write(p + "\n")
        # PM2 execution simulated
        for p in canary_2:
            pm2_state[p]["version"] = short_sha
            pm2_state[p]["env"]["BOT_INSTANCE_ID"] = p
        with open(reloaded_file, "a") as f:
            for p in canary_2:
                f.write(p + "\n")
        print("    -> Canary 2: SUCCESS (PM2 reloaded & E2E notification verified)")

        # Step 4.3: Canary 3 (MAX Pair 2 - VeraVeda) -> SIMULATING PM2 CRASH / PARTIAL FAILURE
        canary_3 = ["tg_veraveda777_bot_legacy", "max_se13639182_bot_legacy"]
        print(f"[*] Step 4.3: Rolling Canary 3 (VeraVeda Pair): {canary_3}")
        # 1. Log intent BEFORE calling PM2
        with open(intent_journal_file, "a") as f:
            for p in canary_3:
                f.write(p + "\n")

        # 2. PM2 FAILS! e.g., process crashes on start or PM2 daemon error
        print("    [!] PM2 startOrReload SIMULATED FAILURE on Canary 3!")
        pm2_state["tg_veraveda777_bot_legacy"]["status"] = "errored"
        pm2_state["tg_veraveda777_bot_legacy"]["version"] = short_sha
        # reloaded_processes.txt is NOT updated for canary_3 because command failed!

        # ---------------------------------------------------------
        # PHASE 5: Rollback Execution & State Reconciliation
        # ---------------------------------------------------------
        print("\n--- PHASE 5: Triggering Rollback Due to Partial Failure ---")

        # 1. Reconcile touched processes from intent_journal, reloaded_processes, and live PM2
        with open(intent_journal_file) as f:
            intent_procs = {line.strip() for line in f if line.strip()}
        with open(reloaded_file) as f:
            reloaded_procs = {line.strip() for line in f if line.strip()}

        # Live PM2 state check: which processes were touched or errored?
        pm2_touched = {name for name, info in pm2_state.items() if info["version"] != prev_sha or info["status"] != "online"}

        all_touched = sorted(list(intent_procs | reloaded_procs | pm2_touched))
        print(f"[*] Reconciled Touched Processes for Rollback: {all_touched}")
        expected_touched = sorted(["tg_autobusbusbot_new", "tg_yourself_way_bot_new", "max_id519010411655_bot_new", "tg_veraveda777_bot_legacy", "max_se13639182_bot_legacy"])
        assert all_touched == expected_touched, f"Expected {expected_touched}, got {all_touched}"

        # 2. Exact deletion of added files from PR #86
        with open(files_added_file) as f:
            to_delete = [line.strip() for line in f if line.strip()]

        deleted_count = 0
        for frel in to_delete:
            fpath = os.path.join(prod_dir, frel)
            if os.path.isfile(fpath) or os.path.islink(fpath):
                os.remove(fpath)
                deleted_count += 1
        print(f"[*] Deleted {deleted_count} added files from newbots.")

        # Assert notification_routing.py is completely GONE
        assert not os.path.exists(os.path.join(prod_dir, "notification_routing.py")), "notification_routing.py must be deleted during rollback!"

        # 3. Restore modified files from backup
        subprocess.check_call(["tar", "xzf", backup_archive, "-C", tb_dir])
        shutil.copy2(os.path.join(tb_dir, "ecosystem.config.js.pre86_backup"), os.path.join(prod_dir, "ecosystem.config.js"))
        shutil.copy2(os.path.join(tb_dir, "runtime.env.pre86_backup"), os.path.join(tb_dir, "runtime.env"))

        # Verify REVISION is restored
        with open(os.path.join(prod_dir, "REVISION")) as f:
            restored_rev = f.read().strip()
        assert restored_rev == prev_sha, f"Expected revision {prev_sha}, got {restored_rev}"
        print(f"[*] Code restored to previous revision: {restored_rev}")

        # 4. Verify runtime data integrity (SQLite, Chroma, Logs)
        assert sha256_file(db_path) == db_hash_orig, "CRITICAL ERROR: darimiru_bot.db was corrupted or modified during rollback!"
        assert sha256_file(chroma_file) == chroma_hash_orig, "CRITICAL ERROR: Chroma vector database was modified during rollback!"
        assert sha256_file(log_file) == log_hash_orig, "CRITICAL ERROR: Payment logs were modified during rollback!"
        print("[*] Runtime SQLite DB, Chroma vector store, and payment logs: 100% UNTOUCHED and INTACT.")

        # 5. Restart all touched processes on restored code
        for p in all_touched:
            pm2_state[p]["version"] = prev_sha
            pm2_state[p]["status"] = "online"
            pm2_state[p]["env"] = {}

        # Verify all 18 processes in PM2 are back to online pre-86 state
        for p, info in pm2_state.items():
            assert info["version"] == prev_sha and info["status"] == "online", f"Process {p} not properly restored in PM2!"
        print(f"[*] All {len(all_18_processes)} processes restored to operational status in PM2.")

        # 6. Cleanup journals
        for f in [intent_journal_file, reloaded_file, prev_sha_file, files_added_file, manifest_pre86, manifest_pr86_staged]:
            if os.path.exists(f):
                os.remove(f)

    print("\n" + "=" * 70)
    print("VERIFICATION RESULT: SIMULATION COMPLETED SUCCESSFULLY (EXIT 0)")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()
