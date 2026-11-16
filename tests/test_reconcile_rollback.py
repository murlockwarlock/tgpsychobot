import os
from pathlib import Path
import tarfile
import pytest

from scripts.reconcile_rollback import (
    get_archive_members,
    is_protected,
    normalize_relpath,
    reconcile_rollback_directory,
)


def test_normalize_relpath_variants():
    assert normalize_relpath("./main.py") == "main.py"
    assert normalize_relpath("main.py") == "main.py"
    assert normalize_relpath("/main.py") == "main.py"
    assert normalize_relpath("././config/bot_instances.json") == "config/bot_instances.json"
    assert normalize_relpath("config/bot_instances.json") == "config/bot_instances.json"
    assert normalize_relpath("./") == ""
    assert normalize_relpath(".") == ""
    assert normalize_relpath("") == ""


def test_is_protected_assets():
    assert is_protected("runtime.env") is True
    assert is_protected("./runtime.env") is True
    assert is_protected(".env.deploy") is True
    assert is_protected(".env.local") is True
    assert is_protected("REVISION") is True
    assert is_protected("darimiru_bot.db") is True
    assert is_protected("subdir/test.sqlite3") is True
    assert is_protected("chroma_db_8081/index.bin") is True
    assert is_protected("logs/payment_events.log") is True
    assert is_protected("media/image.png") is True

    # Code files must NOT be protected
    assert is_protected("main.py") is False
    assert is_protected("notification_routing.py") is False
    assert is_protected("config/bot_instances.json") is False


def test_reconcile_rollback_preserves_runtime_assets_and_handles_dot_slash(tmp_path: Path):
    target_dir = tmp_path / "newbots"
    target_dir.mkdir()

    # Baseline files in target_dir
    (target_dir / "main.py").write_text("# baseline main.py")
    (target_dir / "config").mkdir()
    (target_dir / "config" / "bot_instances.json").write_text('{"instances": []}')

    # Create pre-deploy backup archive where tar entries have leading './'
    archive_path = tmp_path / "backup_pre.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        # Add files explicitly with leading ./
        tar.add(target_dir / "main.py", arcname="./main.py")
        tar.add(target_dir / "config" / "bot_instances.json", arcname="./config/bot_instances.json")

    # Verify get_archive_members strips leading ./
    members = get_archive_members(archive_path)
    assert members == {"main.py", "config/bot_instances.json"}

    # Now simulate deployment introducing:
    # 1. A new code file
    (target_dir / "new_feature.py").write_text("# new feature from bad release")
    (target_dir / "new_dir").mkdir()
    (target_dir / "new_dir" / "helper.py").write_text("# helper")

    # 2. Runtime secrets and assets (created or modified during runtime)
    (target_dir / "runtime.env").write_text("SECRET_KEY=keep_me")
    (target_dir / "darimiru_bot.db").write_bytes(b"SQLITE_DATA_KEEP_ME")
    chroma_dir = target_dir / "chroma_db_8081"
    chroma_dir.mkdir()
    (chroma_dir / "vector.bin").write_bytes(b"CHROMA_DATA")
    logs_dir = target_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "bot.log").write_text("log line 1\n")
    (target_dir / "REVISION").write_text("target_rev")

    # Reconcile directory against archive
    removed = reconcile_rollback_directory(archive_path, target_dir)

    # Assert newly added code files were deleted
    assert "new_feature.py" in removed
    assert "new_dir/helper.py" in removed
    assert not (target_dir / "new_feature.py").exists()
    assert not (target_dir / "new_dir").exists()

    # Assert baseline code files were NOT deleted (no ./ mismatch bug!)
    assert (target_dir / "main.py").exists()
    assert (target_dir / "config" / "bot_instances.json").exists()

    # Assert runtime assets were NOT deleted
    assert (target_dir / "runtime.env").exists()
    assert (target_dir / "darimiru_bot.db").exists()
    assert (chroma_dir / "vector.bin").exists()
    assert (logs_dir / "bot.log").exists()
    assert (target_dir / "REVISION").exists()
