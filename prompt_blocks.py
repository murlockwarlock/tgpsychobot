import re
import textwrap
from dataclasses import dataclass


DEFAULT_SHARED_PROMPT_BLOCK = ""


DEFAULT_SHORT_RESPONSE_INSTRUCTION = textwrap.dedent("""
ВАЖНО — РЕЖИМ КРАТКИХ ОТВЕТОВ: Пользователь выбрал короткий формат.
Строго ограничивай длину ответа: не более 300-400 слов на весь ответ.
Выделяй только самое главное, убирай детальные объяснения и развёрнутые блоки.
Если тема предполагает несколько блоков — объединяй их в один компактный ответ.
Никакой воды, повторений и длинных вступлений.
""").strip()


DEFAULT_BASE_SERVICE_PROMPT_TEMPLATE = textwrap.dedent("""
ТЕХНИЧЕСКИЕ ПРАВИЛА ОФОРМЛЕНИЯ (СТРОГО):
1. Формат: Используй Markdown. HTML запрещен.
2. ЗАГОЛОВКИ: Используй жирный шрифт (**Заголовок**).
3. ОТСТУПЫ: Перед каждым заголовком и списком — ДВА переноса строки.
4. ЗАПРЕТЫ:
   - НЕ используй вложенное форматирование (**_текст_**).
   - НЕ используй '*' или '***' как разделительную линию (используй '———').
5. СПИСКИ: Используй дефис '-' для списков. Каждый пункт с новой строки.

📷 ВИЗУАЛИЗАЦИЯ (ГЕНЕРАЦИЯ):
Если пользователь просит 'показать', 'визуализировать' что-то новое — используй формат:
GEN_IMG: [подробный промпт на АНГЛИЙСКОМ языке]
""").strip()


DEFAULT_MEDIA_RULES_TEMPLATE = textwrap.dedent("""
🎵 ДОСТУПНЫЙ МЕДИА-КОНТЕНТ В ЭТОЙ ТЕМЕ:
{available_media_text}

ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ:
1. АУДИО: [SEND_AUDIO: имя_файла] — отправить аудиофайл. Точное имя из списка выше.
2. КАРТЫ (ТАРО/МАК): В тегах указывай КАТЕГОРИЮ (не имя файла):
   - [RANDOM_IMG: категория] — одна случайная карта. Интерпретацию дашь после, когда узнаешь какая выпала.
   - [RANDOM_IMG: категория | N] — N случайных карт сразу (до 10). Для раскладов где нужно показать все карты разом.
   - [CHOICE_IMG: категория | N] — выбор одной из N карт (лицом вверх).
   - [CHOICE_IMG: категория | N | R] — расклад из R карт. На каждом этапе пользователь выбирает из N новых карт. Интерпретация после каждого выбора.
   - [CHOICE_IMG_HIDDEN: категория | N] — выбор одной из N карт (рубашкой вверх, вслепую).
   - [CHOICE_IMG_HIDDEN: категория | N | R] — расклад из R карт вслепую. На каждом этапе выбор из N новых закрытых карт.
3. [SHOW_IMG: имя_файла] — показать конкретную карту по имени файла из списка выше перед текстом; для показа после текста используй [SHOW_IMG: имя_файла | position=after].
4. ВАЖНО: После RANDOM_IMG, CHOICE_IMG, CHOICE_IMG_HIDDEN НЕ пиши интерпретацию — ты ещё не знаешь какая выпадет. Вводный текст, интерпретация после.
5. Теги с новой строки. Не выдумывай категории и имена файлов.
""").strip()


DEFAULT_SERVICE_PROMPT_TEMPLATE = textwrap.dedent("""
ТЕХНИЧЕСКИЕ ПРАВИЛА ОФОРМЛЕНИЯ (СТРОГО):
1. Формат: Используй Markdown. HTML запрещен.
2. ЗАГОЛОВКИ: Используй жирный шрифт (**Заголовок**).
3. ОТСТУПЫ: Перед каждым заголовком и списком — ДВА переноса строки.
4. ЗАПРЕТЫ:
   - НЕ используй вложенное форматирование (**_текст_**).
   - НЕ используй '*' или '***' как разделительную линию (используй '———').
5. СПИСКИ: Используй дефис '-' для списков. Каждый пункт с новой строки.

📷 ВИЗУАЛИЗАЦИЯ (ГЕНЕРАЦИЯ):
Если пользователь просит 'показать', 'визуализировать' что-то новое — используй формат:
GEN_IMG: [подробный промпт на АНГЛИЙСКОМ языке]
{media_instruction_block}
""").strip()


_LEGACY_MEDIA_BLOCK_REGEX = re.compile(
    r"(?:\r?\n)*[ \t]*(?:🎵?[ \t]*ДОСТУПНЫЙ МЕДИА-КОНТЕНТ|ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ:).*?"
    r"[Нн]е выдумывай категории и имена файлов[^\n]*",
    re.DOTALL,
)


def build_media_instruction_block(available_media_text: str | None) -> str:
    if not available_media_text or not available_media_text.strip():
        return ""
    text = available_media_text.strip()
    if "НЕ загружены" in text or "не загружены" in text:
        return ""
    return "\n\n" + render_prompt_block(DEFAULT_MEDIA_RULES_TEMPLATE, available_media_text=text)


@dataclass(frozen=True)
class ServiceCapabilities:
    supports_data: bool = True
    supports_image_generation: bool = True
    supports_media_collections: bool = False
    supports_send_audio: bool = False
    supports_card_spreads: bool = False


TELEGRAM_CAPABILITIES = ServiceCapabilities(
    supports_data=True,
    supports_image_generation=True,
    supports_media_collections=True,
    supports_send_audio=True,
    supports_card_spreads=True,
)

MAX_CAPABILITIES = ServiceCapabilities(
    supports_data=True,
    supports_image_generation=True,
    supports_media_collections=False,
    supports_send_audio=False,
    supports_card_spreads=False,
)

_IMAGE_GEN_BLOCK_REGEX = re.compile(
    r"(?:\r?\n)*[ \t]*📷?[ \t]*ВИЗУАЛИЗАЦИЯ\s*\(ГЕНЕРАЦИЯ\):.*?"
    r"GEN_IMG:[^\n]*",
    re.DOTALL,
)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.endswith(":") or s.startswith("#") or (s.startswith("**") and s.endswith("**")):
        return True
    cleaned = re.sub(r"[^\w\s]", "", s)
    if cleaned and cleaned.isupper() and len(cleaned.split()) <= 6:
        return True
    return False


def _sanitize_logical_units(text: str, unsupported_patterns: list[re.Pattern]) -> str:
    if not unsupported_patterns or not text.strip():
        return text

    combined_pat = re.compile("|".join(f"(?:{p.pattern})" for p in unsupported_patterns), re.IGNORECASE)

    paragraphs = re.split(r"\n\s*\n", text)
    cleaned_paragraphs = []

    list_marker_regex = re.compile(r"^(\s*(?:(\d+)[\.\)]|[-*+•])\s+)")

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        lines = para.splitlines()
        has_list_items = any(list_marker_regex.match(line) for line in lines)

        if not has_list_items:
            if not combined_pat.search(para):
                cleaned_paragraphs.append(para)
            continue

        heading_lines = []
        items = []
        curr_marker_match = None
        curr_item_lines = []
        in_items = False

        for line in lines:
            m = list_marker_regex.match(line)
            if m:
                in_items = True
                if curr_item_lines:
                    items.append(("\n".join(curr_item_lines), curr_marker_match))
                    curr_item_lines = []
                curr_marker_match = m
                curr_item_lines.append(line)
            elif in_items:
                curr_item_lines.append(line)
            else:
                heading_lines.append(line)

        if curr_item_lines:
            items.append(("\n".join(curr_item_lines), curr_marker_match))

        kept_items = []
        for item_text, m in items:
            if not combined_pat.search(item_text):
                kept_items.append((item_text, m))

        if not kept_items:
            continue

        renumbered_lines = []
        if heading_lines:
            renumbered_lines.extend(heading_lines)

        is_numbered_list = all(m and m.group(2) for _, m in items)
        num_counter = 1
        for item_text, m in kept_items:
            if is_numbered_list and m and m.group(2):
                old_num_str = m.group(2)
                prefix = m.group(1)
                new_prefix = prefix.replace(old_num_str, str(num_counter), 1)
                first_line, *rest = item_text.splitlines()
                first_line = new_prefix + first_line[len(prefix):]
                renumbered_lines.append("\n".join([first_line, *rest]))
                num_counter += 1
            else:
                renumbered_lines.append(item_text)

        cleaned_paragraphs.append("\n".join(renumbered_lines))

    final_paras = []
    for i, p in enumerate(cleaned_paragraphs):
        p_stripped = p.strip()
        if _is_heading(p_stripped):
            if i == len(cleaned_paragraphs) - 1:
                continue
            next_p = cleaned_paragraphs[i + 1].strip()
            if _is_heading(next_p):
                continue
        final_paras.append(p)

    return "\n\n".join(final_paras).strip()


def render_service_prompt(
    template: str,
    *,
    capabilities: ServiceCapabilities | None = None,
    available_media_text: str = "",
    media_instruction_block: str = "",
    **values: str,
) -> str:
    caps = capabilities or TELEGRAM_CAPABILITIES
    rendered = template or ""

    all_values = {
        "available_media_text": available_media_text or "",
        "media_instruction_block": media_instruction_block or "",
        **values,
    }

    if not caps.supports_media_collections:
        all_values["available_media_text"] = ""
        all_values["media_instruction_block"] = ""
        if "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" in rendered or "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" in rendered:
            rendered = _LEGACY_MEDIA_BLOCK_REGEX.sub("", rendered)
    elif not all_values["available_media_text"].strip() and not all_values["media_instruction_block"].strip():
        if "ДОСТУПНЫЙ МЕДИА-КОНТЕНТ" in rendered or "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ" in rendered:
            rendered = _LEGACY_MEDIA_BLOCK_REGEX.sub("", rendered)

    if not caps.supports_image_generation:
        rendered = _IMAGE_GEN_BLOCK_REGEX.sub("", rendered)

    for key, value in all_values.items():
        rendered = rendered.replace(f"{{{key}}}", value or "")

    unsupported_patterns = []
    if not caps.supports_send_audio:
        unsupported_patterns.append(re.compile(r"\[SEND_AUDIO:[^\]]*\]|\bSEND_AUDIO\b"))
    if not caps.supports_card_spreads:
        unsupported_patterns.append(re.compile(
            r"\[(?:RANDOM_IMG|CHOICE_IMG|CHOICE_IMG_HIDDEN|SHOW_IMG):[^\]]*\]"
            r"|\b(?:RANDOM_IMG|CHOICE_IMG|CHOICE_IMG_HIDDEN|SHOW_IMG)\b"
        ))
    if not caps.supports_image_generation:
        unsupported_patterns.append(re.compile(r"GEN_IMG:[^\n]*|\bGEN_IMG\b"))
    if not caps.supports_data:
        unsupported_patterns.append(re.compile(r"</?DATA>|\b<DATA>\b"))

    rendered = _sanitize_logical_units(rendered, unsupported_patterns)
    return rendered.strip()


def render_prompt_block(
    template: str,
    *,
    capabilities: ServiceCapabilities | None = None,
    **values: str,
) -> str:
    return render_service_prompt(template, capabilities=capabilities, **values)


def build_test_context_injection(
    test_results: str | None,
    secret_answers: str | None,
    *,
    secret_test_enabled: bool = True,
) -> str:
    context_parts = []
    if test_results:
        context_parts.append(f"Результаты основного теста (пройден):\n{test_results}")
    if secret_answers:
        context_parts.append(
            f"Ответы пользователя на СЕКРЕТНЫЙ тест (УЖЕ ПРОЙДЕН):\n{secret_answers}"
        )
        status_instruction = "Пользователь УЖЕ прошел все тесты. Обсуждай результаты."
    elif test_results and secret_test_enabled:
        status_instruction = "Пользователь прошел основной тест. Предложи пройти секретный блок."
    elif test_results:
        status_instruction = (
            "Пользователь завершил основной тест. Продолжи диалог по его результатам. "
            "Не предлагай секретный блок."
        )
    else:
        return ""

    joined_results = "\n\n".join(context_parts)
    return f"\n\n[КОНТЕКСТ ТЕСТА]\n{joined_results}\nИНСТРУКЦИЯ: {status_instruction}"
