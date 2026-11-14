"""Unified AI Provider Incident and Cooldown Management.

Enforces the canonical lifecycle for AI provider incidents across Psychobot instances:
- Provider failure classification via provider_failures.py (e.g. insufficient_balance_quota).
- Bot-level isolation: alerts route directly to each bot's administrators from that bot.
- Account fingerprinting via cryptographic hash (never exposing raw API keys).
- 6-hour repeat alert cooldown with persistence across process restarts.
- Confirmed delivery before updating alert state (enabling safe retry on failure).
- Balance recovery confirmed strictly via successful provider balance checks.
- Clean isolation between bot admin alerts and the central autobus infrastructure monitor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Sequence

from aiogram import Bot

logger = logging.getLogger("ai_incident_manager")

DEFAULT_STATE_FILE = "/tmp/psychobots_ai_incidents.json"
DEFAULT_REPEAT_ALERT_SECONDS = 6 * 3600  # 6 hours


def get_ai_incident_state_path() -> str:
    return os.environ.get("PSYCHOBOTS_AI_INCIDENT_STATE", DEFAULT_STATE_FILE)


def get_provider_account_fingerprint(api_key: str | None) -> str:
    """Return a safe opaque fingerprint for any provider API key without exposing secrets."""
    if not api_key or not str(api_key).strip():
        return "unconfigured"
    return hashlib.sha256(str(api_key).strip().encode("utf-8")).hexdigest()[:12]


def resolve_bot_identifier(bot: Any = None, fallback: str | None = None) -> str:
    """Return canonical bot identifier from bot instance, env, or fallback."""
    if bot is not None:
        token = getattr(bot, "token", None)
        if token and isinstance(token, str):
            prefix = token.split(":")[0]
            if prefix.isdigit():
                return f"bot_{prefix}"
        bot_id = getattr(bot, "id", None)
        if bot_id:
            return f"bot_{bot_id}"
        bot_name = (
            getattr(bot, "bot_name", None)
            or getattr(bot, "_bot_name", None)
            or getattr(bot, "name", None)
        )
        if bot_name:
            return str(bot_name)
    return (
        fallback
        or os.getenv("BOT_NAME")
        or os.getenv("BOT_USERNAME")
        or os.getenv("name")
        or "default_bot"
    )


def is_provider_active_for_bot(config: Any, provider: str) -> bool:
    """Check if the given provider is actually active in any functional slot of AIConfig."""
    if not config or not provider:
        return False
    prov = provider.strip().upper()

    def _val(attr: str) -> str:
        if isinstance(config, dict):
            return str(config.get(attr) or "").strip().upper()
        return str(getattr(config, attr, "") or "").strip().upper()

    active_providers = {
        _val("provider"),
        _val("vision_provider"),
        _val("transcription_provider"),
        _val("image_generation_provider"),
        _val("image_edit_provider"),
        _val("fallback_provider"),
        _val("vision_fallback_provider"),
    }
    return prov in active_providers


def get_provider_api_key(config: Any, provider: str) -> str | None:
    """Retrieve API key for the specified provider from AIConfig."""
    if not config or not provider:
        return None
    prov = provider.strip().lower()
    attr_map = {
        "kie": "kie_api_key",
        "openai": "openai_api_key",
        "gemini": "gemini_api_key",
        "claude": "claude_api_key",
        "anthropic": "claude_api_key",
        "deepseek": "deepseek_api_key",
        "openrouter": "openrouter_api_key",
        "perplexity": "perplexity_api_key",
        "deepgram": "deepgram_api_key",
        "yandex": "yandex_api_key",
    }
    attr = attr_map.get(prov, f"{prov}_api_key")
    if isinstance(config, dict):
        return config.get(attr)
    return getattr(config, attr, None)


def get_effective_balance_threshold(config: Any, provider: str) -> float:
    """
    Get the configured threshold for the provider from AIConfig.
    Respects the exact semantics of kie_credit_alert_threshold:
    If threshold > 0, returns the user-configured proactive threshold.
    If threshold <= 0, returns 0.0 (no proactive threshold alerting configured).
    """
    prov = provider.strip().lower()
    if prov == "kie":
        if isinstance(config, dict):
            return float(config.get("kie_credit_alert_threshold") or 0.0)
        return float(getattr(config, "kie_credit_alert_threshold", 0.0) or 0.0)
    return 0.0


def format_provider_balance_alert_message(provider: str, credits: float | None = None) -> str:
    """Human-friendly Russian alert for low provider balance."""
    prov_upper = provider.strip().upper()
    lines = [f"⚠️ <b>Низкий баланс {prov_upper}</b>\n"]
    if credits is not None:
        cr_str = f"{credits:.2f}".replace(".", ",")
        if cr_str.endswith(",00"):
            cr_str = cr_str[:-3]
        lines.append(f"Осталось кредитов: <b>{cr_str}</b>\n")
    if prov_upper == "KIE":
        lines.append("Для генерации изображений требуется больше кредитов.\n")
        lines.append("Генерация и редактирование изображений могут быть недоступны.\n")
    else:
        lines.append(f"Текущего баланса недостаточно для выполнения запросов к {prov_upper}.\n")
        lines.append("Операции AI могут быть приостановлены.\n")
    lines.append(f"<b>Что сделать:</b> пополнить баланс {prov_upper}.")
    return "\n".join(lines)


def format_provider_recovery_message(provider: str, credits: float | None = None) -> str:
    """Human-friendly Russian alert for restored provider balance."""
    prov_upper = provider.strip().upper()
    lines = [f"✅ <b>Баланс {prov_upper} восстановлен</b>\n"]
    if credits is not None:
        cr_str = f"{credits:.2f}".replace(".", ",")
        if cr_str.endswith(",00"):
            cr_str = cr_str[:-3]
        lines.append(f"Осталось кредитов: <b>{cr_str}</b>\n")
    if prov_upper == "KIE":
        lines.append("Генерация и редактирование изображений снова доступны.")
    else:
        lines.append(f"Работа операций AI с провайдером {prov_upper} восстановлена.")
    return "\n".join(lines)


def _make_incident_key(bot_id: str, provider: str, account_fingerprint: str) -> str:
    b = (bot_id or "unknown").strip()
    p = (provider or "unknown").strip().upper()
    a = (account_fingerprint or "default").strip()
    return f"{b}:{p}:{a}"


def load_ai_incident_state() -> dict:
    path = get_ai_incident_state_path()
    if not os.path.exists(path):
        return {"incidents": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "incidents" in data:
                return data
            return {"incidents": {}}
    except Exception as exc:
        logger.warning("Failed to load AI incident state from %s: %s", path, exc)
        return {"incidents": {}}


def save_ai_incident_state(state: dict) -> None:
    path = get_ai_incident_state_path()
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.error("Failed to save AI incident state to %s: %s", path, exc)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def should_alert_provider_incident(
    bot_id: str,
    provider: str,
    account_fingerprint: str,
    cooldown_seconds: int = DEFAULT_REPEAT_ALERT_SECONDS,
    now: int | None = None,
) -> bool:
    cur_time = int(time.time()) if now is None else int(now)
    state = load_ai_incident_state()
    key = _make_incident_key(bot_id, provider, account_fingerprint)
    incident = state.get("incidents", {}).get(key)
    if not incident:
        return True
    if incident.get("status") != "active":
        return True
    # If not yet confirmed delivered by Telegram, retry!
    if not incident.get("alert_delivered", False):
        return True
    last_alerted = int(incident.get("last_alerted", 0))
    if cur_time - last_alerted >= cooldown_seconds:
        return True
    return False


def record_provider_incident_alerted(
    bot_id: str,
    provider: str,
    account_fingerprint: str,
    details: dict[str, Any] | None = None,
    now: int | None = None,
) -> None:
    cur_time = int(time.time()) if now is None else int(now)
    state = load_ai_incident_state()
    incidents = state.setdefault("incidents", {})
    key = _make_incident_key(bot_id, provider, account_fingerprint)
    existing = incidents.get(key, {})
    first_seen = existing.get("first_seen", cur_time) if existing.get("status") == "active" else cur_time
    incidents[key] = {
        "bot_id": bot_id,
        "provider": provider.upper(),
        "account_fingerprint": account_fingerprint,
        "status": "active",
        "first_seen": first_seen,
        "last_seen": cur_time,
        "last_alerted": cur_time,
        "alert_delivered": True,
        "recovery_delivered": False,
        "details": details or {},
    }
    save_ai_incident_state(state)


def is_provider_incident_active(bot_id: str, provider: str, account_fingerprint: str) -> bool:
    state = load_ai_incident_state()
    key = _make_incident_key(bot_id, provider, account_fingerprint)
    incident = state.get("incidents", {}).get(key)
    return bool(incident and incident.get("status") == "active")


def record_provider_incident_recovered(
    bot_id: str,
    provider: str,
    account_fingerprint: str,
    details: dict[str, Any] | None = None,
    now: int | None = None,
) -> None:
    cur_time = int(time.time()) if now is None else int(now)
    state = load_ai_incident_state()
    incidents = state.setdefault("incidents", {})
    key = _make_incident_key(bot_id, provider, account_fingerprint)
    if key in incidents:
        incidents[key]["status"] = "resolved"
        incidents[key]["alert_delivered"] = False
        incidents[key]["recovery_delivered"] = True
        incidents[key]["resolved_at"] = cur_time
        if details:
            incidents[key]["details"] = details
        save_ai_incident_state(state)


def get_active_provider_incidents() -> dict[str, dict]:
    state = load_ai_incident_state()
    return {
        k: v for k, v in state.get("incidents", {}).items()
        if v.get("status") == "active"
    }


def clear_all_provider_incidents() -> None:
    state = {"incidents": {}}
    save_ai_incident_state(state)


async def send_provider_incident_message_to_admins(
    bot: Bot,
    admin_ids: Sequence[int],
    text: str,
) -> bool:
    """
    Deliver provider incident message to bot administrators.
    Returns True ONLY if at least one delivery succeeded.
    """
    any_delivered = False
    for admin_id in admin_ids:
        try:
            await bot.send_message(admin_id, text, parse_mode="HTML")
            any_delivered = True
        except Exception as exc:
            logger.error("Failed to send provider incident notification to admin %s: %s", admin_id, exc)
    return any_delivered
