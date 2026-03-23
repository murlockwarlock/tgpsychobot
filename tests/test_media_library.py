import html
import re
import unittest
from types import SimpleNamespace


# ══════════════════════════════════════════════════════════════════
#  Вспомогательные функции (копии логики из основного кода,
#  чтобы тесты не зависели от импортов с aiogram/SQLAlchemy)
# ══════════════════════════════════════════════════════════════════

def parse_media_tags(response_text: str):
    """handlers.py → handle_ai_media_content (парсинг тегов)."""
    audio_pattern = r'\[SEND_AUDIO:\s*(.*?)\]'
    random_img_pattern = r'\[RANDOM_IMG:\s*(.*?)\]'
    choice_img_pattern = r'\[CHOICE_IMG:\s*(.*?)\s*\|\s*(\d+)\]'
    choice_hidden_pattern = r'\[CHOICE_IMG_HIDDEN:\s*(.*?)\s*\|\s*(\d+)\]'
    show_img_pattern = r'\[SHOW_IMG:\s*(.*?)\]'

    audios = re.findall(audio_pattern, response_text)
    random_imgs = re.findall(random_img_pattern, response_text)
    choices = re.findall(choice_img_pattern, response_text)
    choices_hidden = re.findall(choice_hidden_pattern, response_text)
    show_imgs = re.findall(show_img_pattern, response_text)

    clean_text = re.sub(audio_pattern, '', response_text)
    clean_text = re.sub(random_img_pattern, '', clean_text)
    clean_text = re.sub(choice_hidden_pattern, '', clean_text)
    clean_text = re.sub(choice_img_pattern, '', clean_text)
    clean_text = re.sub(show_img_pattern, '', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

    return clean_text, audios, random_imgs, choices, choices_hidden, show_imgs


def parse_media_tags_execute(response_text: str):
    """handlers.py → execute_media_commands (альтернативный парсер)."""
    audio_matches = re.findall(r"\[SEND_AUDIO:\s*(.+?)\]", response_text)
    random_matches = re.findall(r"\[RANDOM_IMG:\s*(.+?)\]", response_text)
    choice_matches = re.findall(r"\[CHOICE_IMG:\s*(.+?)\s*\|\s*(\d+)\]", response_text)
    choice_hidden_matches = re.findall(r"\[CHOICE_IMG_HIDDEN:\s*(.+?)\s*\|\s*(\d+)\]", response_text)
    show_matches = re.findall(r"\[SHOW_IMG:\s*(.+?)\]", response_text)
    clean_text = re.sub(r"\[(SEND_AUDIO|RANDOM_IMG|CHOICE_IMG_HIDDEN|CHOICE_IMG|SHOW_IMG):.*?\]", "", response_text).strip()
    return clean_text, audio_matches, random_matches, choice_matches, choice_hidden_matches, show_matches


def build_available_media_text(media_files):
    """ai_integration.py → формирование промпта с медиа."""
    if not media_files:
        return "Доступные медиа-файлы: не загружены.\n"

    categories = {}
    for m in media_files:
        cat = m.category or ''
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(m)

    available_media_text = "Доступные медиа-файлы в этой теме:\n"
    for cat, files in categories.items():
        if cat:
            available_media_text += f'\nКатегория (для тегов RANDOM_IMG/CHOICE_IMG): "{cat}"\n'
        for m in files:
            desc_part = f" — {m.description}" if m.description else ""
            available_media_text += f"  - [{m.media_type.upper()}] {m.file_name}{desc_part}\n"

    return available_media_text


def build_media_instruction(available_media_text: str):
    """ai_integration.py → media_instruction блок промпта."""
    return (
        f"\n\n🎵 ДОСТУПНЫЙ МЕДИА-КОНТЕНТ В ЭТОЙ ТЕМЕ:\n"
        f"{available_media_text}\n"
        "ПРАВИЛА ИСПОЛЬЗОВАНИЯ МЕДИА-ТЕГОВ:\n"
        "1. АУДИО: Чтобы отправить аудиофайл, пиши строго [SEND_AUDIO: имя_файла]. "
        "Используй точное имя файла из списка выше.\n"
        "2. КАРТЫ (ТАРО/МАК): Если в списке выше есть фотографии, ты можешь их использовать. "
        "В тегах указывай значение КАТЕГОРИИ (не имя файла!):\n"
        "   - [RANDOM_IMG: категория] — система выберет одну случайную карту из этой категории и покажет её. "
        "Ты узнаешь, какая карта выпала, в следующем сообщении — тогда сможешь дать интерпретацию.\n"
        "   - [CHOICE_IMG: категория | 3] — предложить пользователю выбрать одну из N карт (картинки показываются лицом вверх). "
        "После выбора ты получишь информацию о выбранной карте и сможешь дать интерпретацию.\n"
        "   - [CHOICE_IMG_HIDDEN: категория | 3] — предложить пользователю выбрать одну из N закрытых карт (показываются рубашкой вверх). "
        "Используй этот тег когда нужен интригующий выбор «вслепую». После выбора карта откроется и ты получишь информацию для интерпретации.\n"
        "3. ПОКАЗ КОНКРЕТНОЙ КАРТЫ: [SHOW_IMG: точное_имя_файла] — показать конкретную карту/изображение по имени файла. "
        "Используй, когда нужно показать определённую карту (например, при обсуждении). Имя файла должно точно совпадать с именем из списка выше.\n"
        "4. ВАЖНО: После тега RANDOM_IMG, CHOICE_IMG или CHOICE_IMG_HIDDEN НЕ пиши интерпретацию конкретной карты — "
        "ты ещё не знаешь, какая выпадет. Напиши вводный текст, а интерпретацию дашь после.\n"
        "5. Теги ставь с новой строки. Не выдумывай категории и имена файлов, которых нет в списке выше.\n"
    )


def markdown_to_html(text: str) -> str:
    """handlers.py → markdown_to_html (полная копия)."""
    if not text:
        return ""

    text = re.sub(r'<b>(.*?)</b>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'<i>(.*?)</i>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'<pre><code>(.*?)</code></pre>', r'```\1```', text, flags=re.DOTALL)
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL)

    text = html.escape(text, quote=False)
    code_blocks = {}

    def save_code_block(match):
        key = f"\x01CODEBLOCK{len(code_blocks)}\x01"
        code_blocks[key] = match.group(1)
        return key

    def save_inline_code(match):
        key = f"\x01INLINE{len(code_blocks)}\x01"
        code_blocks[key] = match.group(1)
        return key

    text = re.sub(r'```(.*?)```', save_code_block, text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', save_inline_code, text)
    text = re.sub(r'^\s*[\*_-]{3,}\s*$', '———', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\s*$', '———', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*#{1,6}\s+(.*)', lambda m: '\n\n<b>' + m.group(1).replace('***', '').replace('**', '').replace('*', '') + '</b>\n', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*\*(?=[^<>]*\*\*\*)((?:(?!\n\n)[^<>])+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'___((?:(?!\n\n)[^<>])+?)___', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*(?=[^<>]*\*\*)((?:(?!\n\n)[^<>])+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(?=[^<>]*__)((?:(?!\n\n)[^<>])+?)__', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\w)\*(?!\s)([^<>\n]+?)(?<!\s)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'(?<!\w)_(?!\s)([^<>\n]+?)(?<!\s)_(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'~~(?=[^<>\n]+~~)([^<>\n]+?)~~', r'<s>\1</s>', text)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r'([^\n])\n(<b>)', r'\1\n\n\2', text)
    text = re.sub(r'([^\n])\n(•)', r'\1\n\n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    text = text.replace('**', '').replace('__', '').replace('~~', '')
    text = text.replace('<b></b>', '').replace('<i></i>', '')
    text = re.sub(r'(?<![\w*])\*(?![\w*])', '', text)

    for key, value in code_blocks.items():
        if "CODEBLOCK" in key:
            replacement = f'<pre><code>{value}</code></pre>'
        else:
            replacement = f'<code>{value}</code>'
        text = text.replace(key, replacement)

    text = text.strip()
    return text


def split_html_text(text: str, max_length: int = 4090) -> list:
    """handlers.py → split_html_text (упрощённая копия для тестов)."""
    if not text:
        return []
    if len(text) <= max_length:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        split_at = text.rfind('\n', 0, max_length)
        if split_at == -1:
            split_at = text.rfind(' ', 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return [c for c in chunks if c.strip()]


def make_media(file_name, category, media_type='photo', description=None, id=None):
    return SimpleNamespace(
        id=id,
        file_name=file_name,
        category=category,
        media_type=media_type,
        description=description,
    )


def normalize_input(raw: str) -> str:
    """Нормализация ввода категории/имени (как в handlers.py)."""
    return raw.strip().lower().replace(" ", "_")


# ═══════════════════════════════════════════════════════════════
#  1. Парсинг тегов — базовые кейсы
# ═══════════════════════════════════════════════════════════════

class ParseMediaTagsTests(unittest.TestCase):

    def test_no_tags_returns_original_text(self):
        text = "Просто текст без тегов."
        clean, audios, imgs, choices, choices_hidden, show_imgs = parse_media_tags(text)
        self.assertEqual(clean, "Просто текст без тегов.")
        self.assertEqual(audios, [])
        self.assertEqual(imgs, [])
        self.assertEqual(choices, [])
        self.assertEqual(choices_hidden, [])
        self.assertEqual(show_imgs, [])

    def test_random_img_parsed(self):
        text = "Вот твоя карта:\n[RANDOM_IMG: tarot]\nПосмотри внимательно."
        clean, _, imgs, _, _, _ = parse_media_tags(text)
        self.assertEqual(imgs, ["tarot"])
        self.assertNotIn("[RANDOM_IMG", clean)
        self.assertIn("Вот твоя карта:", clean)

    def test_random_img_with_spaces(self):
        r"""Пробелы после двоеточия съедаются \s*, trailing остаются."""
        text = "[RANDOM_IMG:   tarot  ]"
        _, _, imgs, _, _, _ = parse_media_tags(text)
        self.assertEqual(imgs, ["tarot  "])

    def test_choice_img_parsed(self):
        text = "Выбери:\n[CHOICE_IMG: tarot | 3]"
        clean, _, _, choices, _, _ = parse_media_tags(text)
        self.assertEqual(choices, [("tarot", "3")])
        self.assertNotIn("[CHOICE_IMG", clean)

    def test_choice_img_different_counts(self):
        _, _, _, choices, _, _ = parse_media_tags("[CHOICE_IMG: mak | 5]")
        self.assertEqual(choices, [("mak", "5")])

    def test_send_audio_parsed(self):
        text = "Послушай:\n[SEND_AUDIO: morning_meditation]"
        clean, audios, _, _, _, _ = parse_media_tags(text)
        self.assertEqual(audios, ["morning_meditation"])
        self.assertNotIn("[SEND_AUDIO", clean)

    def test_multiple_tags_in_one_response(self):
        text = "До\n[SEND_AUDIO: relax]\nМежду\n[RANDOM_IMG: tarot]\n[CHOICE_IMG: oracle | 2]\nПосле"
        clean, audios, imgs, choices, _, _ = parse_media_tags(text)
        self.assertEqual(audios, ["relax"])
        self.assertEqual(imgs, ["tarot"])
        self.assertEqual(choices, [("oracle", "2")])

    def test_multiple_random_imgs(self):
        _, _, imgs, _, _, _ = parse_media_tags("[RANDOM_IMG: tarot]\n[RANDOM_IMG: mak]")
        self.assertEqual(imgs, ["tarot", "mak"])

    def test_only_tag_no_text(self):
        clean, _, imgs, _, _, _ = parse_media_tags("[RANDOM_IMG: tarot]")
        self.assertEqual(clean, "")
        self.assertEqual(imgs, ["tarot"])

    def test_excessive_newlines_collapsed(self):
        clean, _, _, _, _, _ = parse_media_tags("До\n\n\n\n\n[RANDOM_IMG: tarot]\n\n\n\nПосле")
        self.assertNotIn("\n\n\n", clean)

    def test_choice_img_no_space_around_pipe(self):
        _, _, _, choices, _, _ = parse_media_tags("[CHOICE_IMG: tarot|3]")
        self.assertEqual(choices, [("tarot", "3")])

    def test_gen_img_not_consumed(self):
        clean, _, imgs, _, _, _ = parse_media_tags("GEN_IMG: sunset")
        self.assertIn("GEN_IMG:", clean)
        self.assertEqual(imgs, [])


# ═══════════════════════════════════════════════════════════════
#  1b. Парсинг тегов — edge cases
# ═══════════════════════════════════════════════════════════════

class ParseMediaTagsEdgeCasesTests(unittest.TestCase):

    def test_nested_brackets_ignored(self):
        """Вложенные скобки не должны крашить парсер."""
        text = "[RANDOM_IMG: [RANDOM_IMG: tarot]]"
        _, _, imgs, _, _, _ = parse_media_tags(text)
        # (.*?) — non-greedy, поймает до первой ]
        self.assertTrue(len(imgs) >= 1)

    def test_unclosed_tag_not_parsed(self):
        """Незакрытый тег не должен парситься."""
        text = "[RANDOM_IMG: tarot"
        _, _, imgs, _, _, _ = parse_media_tags(text)
        self.assertEqual(imgs, [])

    def test_empty_category(self):
        """Пустая категория: [RANDOM_IMG: ]."""
        text = "[RANDOM_IMG: ]"
        _, _, imgs, _, _, _ = parse_media_tags(text)
        self.assertEqual(imgs, [""])

    def test_choice_img_zero_count(self):
        """CHOICE_IMG с числом 0."""
        _, _, _, choices, _, _ = parse_media_tags("[CHOICE_IMG: tarot | 0]")
        self.assertEqual(choices, [("tarot", "0")])

    def test_unicode_category(self):
        """Кириллица в категории."""
        text = "[RANDOM_IMG: таро]"
        _, _, imgs, _, _, _ = parse_media_tags(text)
        self.assertEqual(imgs, ["таро"])

    def test_tag_inside_text_not_on_new_line(self):
        """Тег посреди строки тоже ловится."""
        text = "Смотри вот [RANDOM_IMG: tarot] карта."
        clean, _, imgs, _, _, _ = parse_media_tags(text)
        self.assertEqual(imgs, ["tarot"])
        self.assertIn("Смотри вот", clean)
        self.assertIn("карта.", clean)

    def test_choice_img_large_count(self):
        _, _, _, choices, _, _ = parse_media_tags("[CHOICE_IMG: tarot | 99]")
        self.assertEqual(choices, [("tarot", "99")])

    def test_multiple_same_tags(self):
        text = "[RANDOM_IMG: tarot]\n[RANDOM_IMG: tarot]\n[RANDOM_IMG: tarot]"
        _, _, imgs, _, _, _ = parse_media_tags(text)
        self.assertEqual(len(imgs), 3)

    def test_show_img_parsed(self):
        text = "Вот эта карта:\n[SHOW_IMG: Шут]\nОбрати внимание."
        clean, _, _, _, _, show_imgs = parse_media_tags(text)
        self.assertEqual(show_imgs, ["Шут"])
        self.assertNotIn("[SHOW_IMG", clean)
        self.assertIn("Вот эта карта:", clean)

    def test_show_img_with_spaces(self):
        text = "[SHOW_IMG:   Королева мечей  ]"
        _, _, _, _, _, show_imgs = parse_media_tags(text)
        self.assertEqual(show_imgs, ["Королева мечей  "])

    def test_show_img_multiple(self):
        text = "[SHOW_IMG: Шут]\n[SHOW_IMG: Маг]"
        _, _, _, _, _, show_imgs = parse_media_tags(text)
        self.assertEqual(show_imgs, ["Шут", "Маг"])

    def test_show_img_with_other_tags(self):
        text = "Текст\n[SHOW_IMG: Шут]\n[RANDOM_IMG: tarot]\n[SEND_AUDIO: relax]"
        clean, audios, imgs, _, _, show_imgs = parse_media_tags(text)
        self.assertEqual(show_imgs, ["Шут"])
        self.assertEqual(imgs, ["tarot"])
        self.assertEqual(audios, ["relax"])
        self.assertNotIn("[SHOW_IMG", clean)

    def test_show_img_execute_parser(self):
        text = "Вот карта [SHOW_IMG: Шут] смотри"
        clean, _, _, _, _, show_imgs = parse_media_tags_execute(text)
        self.assertEqual(show_imgs, ["Шут"])
        self.assertNotIn("[SHOW_IMG", clean)

    def test_show_img_consistency(self):
        text = "[SHOW_IMG: Королева мечей]"
        _, _, _, _, _, s1 = parse_media_tags(text)
        _, _, _, _, _, s2 = parse_media_tags_execute(text)
        self.assertEqual([s.strip() for s in s1], [s.strip() for s in s2])

    def test_choice_hidden_parsed(self):
        text = "Выбери вслепую:\n[CHOICE_IMG_HIDDEN: tarot | 3]"
        clean, _, _, _, ch, _ = parse_media_tags(text)
        self.assertEqual(ch, [("tarot", "3")])
        self.assertNotIn("[CHOICE_IMG_HIDDEN", clean)

    def test_choice_hidden_not_confused_with_choice(self):
        text = "[CHOICE_IMG: tarot | 2]\n[CHOICE_IMG_HIDDEN: mak | 3]"
        _, _, _, choices, choices_hidden, _ = parse_media_tags(text)
        self.assertEqual(choices, [("tarot", "2")])
        self.assertEqual(choices_hidden, [("mak", "3")])

    def test_choice_hidden_execute_parser(self):
        text = "[CHOICE_IMG_HIDDEN: oracle | 5]"
        clean, _, _, _, ch, _ = parse_media_tags_execute(text)
        self.assertEqual(ch, [("oracle", "5")])
        self.assertNotIn("[CHOICE_IMG_HIDDEN", clean)

    def test_choice_hidden_consistency(self):
        text = "[CHOICE_IMG_HIDDEN: tarot | 3]"
        _, _, _, _, ch1, _ = parse_media_tags(text)
        _, _, _, _, ch2, _ = parse_media_tags_execute(text)
        self.assertEqual([(c.strip(), n) for c, n in ch1], [(c.strip(), n) for c, n in ch2])


# ═══════════════════════════════════════════════════════════════
#  2. Консистентность двух парсеров
# ═══════════════════════════════════════════════════════════════

class ParserConsistencyTests(unittest.TestCase):
    """handle_ai_media_content и execute_media_commands должны парсить одинаково."""

    def _assert_same(self, text):
        _, a1, r1, c1, ch1, s1 = parse_media_tags(text)
        _, a2, r2, c2, ch2, s2 = parse_media_tags_execute(text)
        # Оба парсера должны найти те же теги (с точностью до strip)
        self.assertEqual([a.strip() for a in a1], [a.strip() for a in a2])
        self.assertEqual([r.strip() for r in r1], [r.strip() for r in r2])
        self.assertEqual([(c.strip(), n) for c, n in c1], [(c.strip(), n) for c, n in c2])
        self.assertEqual([(c.strip(), n) for c, n in ch1], [(c.strip(), n) for c, n in ch2])
        self.assertEqual([s.strip() for s in s1], [s.strip() for s in s2])

    def test_single_random_img(self):
        self._assert_same("[RANDOM_IMG: tarot]")

    def test_single_choice_img(self):
        self._assert_same("[CHOICE_IMG: mak | 3]")

    def test_single_audio(self):
        self._assert_same("[SEND_AUDIO: relax]")

    def test_mixed_tags(self):
        self._assert_same(
            "Текст\n[SEND_AUDIO: a]\n[RANDOM_IMG: b]\n[CHOICE_IMG: c | 2]\n[CHOICE_IMG_HIDDEN: e | 4]\n[SHOW_IMG: d]\nКонец"
        )

    def test_no_tags(self):
        self._assert_same("Просто текст.")

    def test_both_clean_text_strips_tags(self):
        text = "До [RANDOM_IMG: x] после"
        c1, _, _, _, _, _ = parse_media_tags(text)
        c2, _, _, _, _, _ = parse_media_tags_execute(text)
        self.assertNotIn("[RANDOM_IMG", c1)
        self.assertNotIn("[RANDOM_IMG", c2)


# ═══════════════════════════════════════════════════════════════
#  3. Формирование промпта с медиа
# ═══════════════════════════════════════════════════════════════

class BuildAvailableMediaTextTests(unittest.TestCase):

    def test_empty_list(self):
        self.assertIn("не загружены", build_available_media_text([]))

    def test_single_photo_with_category_and_description(self):
        files = [make_media("tarot_fool", "tarot", "photo", "Дурак — начало пути")]
        result = build_available_media_text(files)
        self.assertIn('"tarot"', result)
        self.assertIn("tarot_fool", result)
        self.assertIn("Дурак — начало пути", result)
        self.assertIn("[PHOTO]", result)

    def test_audio_file(self):
        files = [make_media("morning_meditation", "audio", "audio", "Утренняя медитация")]
        result = build_available_media_text(files)
        self.assertIn("[AUDIO]", result)

    def test_multiple_categories_grouped(self):
        files = [
            make_media("tarot_fool", "tarot", "photo", "Дурак"),
            make_media("tarot_death", "tarot", "photo", "Смерть"),
            make_media("mak_sun", "mak", "photo", "Солнце"),
        ]
        result = build_available_media_text(files)
        self.assertIn('"tarot"', result)
        self.assertIn('"mak"', result)

    def test_file_without_description(self):
        files = [make_media("tarot_fool", "tarot", "photo", None)]
        result = build_available_media_text(files)
        self.assertIn("tarot_fool", result)
        self.assertNotIn(" — ", result)

    def test_file_without_category(self):
        files = [make_media("random_pic", "", "photo", "Просто")]
        result = build_available_media_text(files)
        self.assertIn("random_pic", result)
        self.assertNotIn('Категория', result)

    def test_description_included_for_ai_context(self):
        files = [
            make_media("t1", "tarot", "photo", "конфликт, победа ценой потерь"),
            make_media("t2", "tarot", "photo", "начало нового пути"),
        ]
        result = build_available_media_text(files)
        self.assertIn("конфликт", result)
        self.assertIn("начало нового пути", result)


# ═══════════════════════════════════════════════════════════════
#  4. Промпт-инструкция для AI
# ═══════════════════════════════════════════════════════════════

class MediaInstructionTests(unittest.TestCase):

    def test_instruction_mentions_category_not_filename(self):
        instr = build_media_instruction("anything")
        self.assertIn("КАТЕГОРИИ", instr)
        self.assertIn("не имя файла", instr)

    def test_instruction_warns_not_to_interpret_before_draw(self):
        instr = build_media_instruction("anything")
        self.assertIn("НЕ пиши интерпретацию", instr)

    def test_instruction_contains_random_img_example(self):
        instr = build_media_instruction("anything")
        self.assertIn("[RANDOM_IMG:", instr)

    def test_instruction_contains_choice_img_example(self):
        instr = build_media_instruction("anything")
        self.assertIn("[CHOICE_IMG:", instr)

    def test_instruction_with_empty_media(self):
        media_text = build_available_media_text([])
        instr = build_media_instruction(media_text)
        self.assertIn("не загружены", instr)

    def test_instruction_contains_show_img(self):
        instr = build_media_instruction("anything")
        self.assertIn("[SHOW_IMG:", instr)
        self.assertIn("точное_имя_файла", instr)

    def test_instruction_contains_choice_hidden(self):
        instr = build_media_instruction("anything")
        self.assertIn("[CHOICE_IMG_HIDDEN:", instr)
        self.assertIn("рубашкой", instr)


# ═══════════════════════════════════════════════════════════════
#  5. Карточный flow — системные сообщения
# ═══════════════════════════════════════════════════════════════

class CardFlowTests(unittest.TestCase):

    def test_drawn_card_system_message_format(self):
        drawn = ["tarot_5_of_swords: Пятёрка Мечей — конфликт"]
        msg = f"[СИСТЕМА: Случайно выпала карта: {'; '.join(drawn)}. Дай интерпретацию этой карты.]"
        self.assertIn("tarot_5_of_swords", msg)
        self.assertIn("конфликт", msg)
        self.assertIn("интерпретацию", msg)

    def test_chosen_card_system_message_format(self):
        info = "tarot_the_fool: Шут — начало нового пути"
        msg = f"[СИСТЕМА: Пользователь выбрал карту: {info}. Дай интерпретацию этой карты.]"
        self.assertIn("Пользователь выбрал", msg)
        self.assertIn("tarot_the_fool", msg)

    def test_multiple_drawn_cards_joined(self):
        drawn = ["tarot_fool: Шут", "tarot_death: Смерть"]
        text = "; ".join(drawn)
        self.assertIn("; ", text)
        self.assertIn("Шут", text)
        self.assertIn("Смерть", text)

    def test_card_without_description(self):
        drawn = ["tarot_unknown: без описания"]
        msg = f"[СИСТЕМА: Случайно выпала карта: {'; '.join(drawn)}. Дай интерпретацию этой карты.]"
        self.assertIn("без описания", msg)

    def test_interpretation_tags_stripped(self):
        text = "Интерпретация.\n[RANDOM_IMG: tarot]\n[CHOICE_IMG: mak | 2]\n[GEN_IMG: test]\nЕщё."
        clean = re.sub(r"\[(SEND_AUDIO|RANDOM_IMG|CHOICE_IMG|GEN_IMG):.*?\]", "", text).strip()
        self.assertNotIn("[RANDOM_IMG", clean)
        self.assertNotIn("[CHOICE_IMG", clean)
        self.assertNotIn("[GEN_IMG", clean)
        self.assertIn("Интерпретация.", clean)
        self.assertIn("Ещё.", clean)


# ═══════════════════════════════════════════════════════════════
#  6. card_selection_keyboard — формат callback_data
# ═══════════════════════════════════════════════════════════════

class CardSelectionKeyboardTests(unittest.TestCase):
    """Тесты на формат callback_data кнопок выбора карт."""

    def _build_buttons(self, category, card_ids):
        """Эмуляция card_selection_keyboard — генерация callback_data."""
        return [f"card_select_{category}_{cid}" for cid in card_ids]

    def _parse_card_id(self, callback_data):
        """Эмуляция парсинга из process_card_selection (rsplit fix)."""
        return int(callback_data.rsplit("_", 1)[-1])

    def test_callback_data_format(self):
        buttons = self._build_buttons("tarot", [10, 20, 30])
        self.assertEqual(buttons[0], "card_select_tarot_10")
        self.assertEqual(buttons[1], "card_select_tarot_20")
        self.assertEqual(buttons[2], "card_select_tarot_30")

    def test_button_count_matches_card_count(self):
        ids = [1, 2, 3, 4, 5]
        buttons = self._build_buttons("mak", ids)
        self.assertEqual(len(buttons), 5)

    def test_parse_card_id_from_callback(self):
        self.assertEqual(self._parse_card_id("card_select_tarot_42"), 42)
        self.assertEqual(self._parse_card_id("card_select_mak_1"), 1)

    def test_roundtrip_build_parse(self):
        """Генерация → парсинг → тот же id."""
        for card_id in [1, 99, 12345]:
            cb = f"card_select_tarot_{card_id}"
            self.assertEqual(self._parse_card_id(cb), card_id)

    def test_category_with_underscore_in_callback(self):
        """Категория с подчёркиванием: card_select_tarot_extended_42 — rsplit fix."""
        cb = "card_select_tarot_extended_42"
        card_id = self._parse_card_id(cb)
        self.assertEqual(card_id, 42)


# ═══════════════════════════════════════════════════════════════
#  7. Нормализация ввода (категория, имя файла)
# ═══════════════════════════════════════════════════════════════

class NormalizationTests(unittest.TestCase):

    def test_lowercase_and_underscore(self):
        self.assertEqual(normalize_input("Tarot Cards"), "tarot_cards")

    def test_already_normalized(self):
        self.assertEqual(normalize_input("tarot"), "tarot")

    def test_spaces_become_underscores(self):
        self.assertEqual(normalize_input("my cool deck"), "my_cool_deck")

    def test_strip_whitespace(self):
        self.assertEqual(normalize_input("  tarot  "), "tarot")

    def test_cyrillic(self):
        self.assertEqual(normalize_input("Таро Уэйта"), "таро_уэйта")

    def test_mixed_case(self):
        self.assertEqual(normalize_input("TaRoT"), "tarot")

    def test_empty_string(self):
        self.assertEqual(normalize_input("  "), "")


# ═══════════════════════════════════════════════════════════════
#  8. Фильтрация по категории
# ═══════════════════════════════════════════════════════════════

class CategoryFilteringTests(unittest.TestCase):

    def test_exact_match(self):
        media = [make_media("c1", "tarot"), make_media("c2", "tarot"), make_media("c3", "tarot_extended")]
        filtered = [m for m in media if m.category == "tarot"]
        self.assertEqual(len(filtered), 2)

    def test_different_category_excluded(self):
        media = [make_media("m1", "mak"), make_media("t1", "tarot")]
        filtered = [m for m in media if m.category == "tarot"]
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].file_name, "t1")

    def test_empty_category_not_matched(self):
        media = [make_media("p1", ""), make_media("t1", "tarot")]
        self.assertEqual(len([m for m in media if m.category == "tarot"]), 1)

    def test_none_category_not_matched(self):
        media = [make_media("p1", None), make_media("t1", "tarot")]
        self.assertEqual(len([m for m in media if m.category == "tarot"]), 1)

    def test_case_sensitive(self):
        """Категория 'Tarot' != 'tarot' — нормализация должна быть при загрузке."""
        media = [make_media("c1", "Tarot"), make_media("c2", "tarot")]
        self.assertEqual(len([m for m in media if m.category == "tarot"]), 1)


# ═══════════════════════════════════════════════════════════════
#  9. markdown_to_html
# ═══════════════════════════════════════════════════════════════

class MarkdownToHtmlTests(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(markdown_to_html(""), "")

    def test_bold(self):
        result = markdown_to_html("**жирный**")
        self.assertIn("<b>", result)
        self.assertIn("жирный", result)

    def test_italic(self):
        result = markdown_to_html("*курсив*")
        self.assertIn("<i>", result)

    def test_heading_becomes_bold(self):
        result = markdown_to_html("## Заголовок")
        self.assertIn("<b>", result)
        self.assertIn("Заголовок", result)

    def test_list_items(self):
        result = markdown_to_html("- пункт 1\n- пункт 2")
        self.assertIn("•", result)

    def test_html_entities_escaped(self):
        result = markdown_to_html("a < b & c > d")
        self.assertIn("&lt;", result)
        self.assertIn("&amp;", result)
        self.assertIn("&gt;", result)

    def test_code_block_preserved(self):
        result = markdown_to_html("```код```")
        self.assertIn("<code>", result)

    def test_card_description_with_special_chars(self):
        """Описание карты с HTML-спецсимволами не ломает разметку."""
        desc = "**Пятёрка Мечей** — \"конфликт\" & <хаос>"
        result = markdown_to_html(desc)
        self.assertIn("&lt;хаос&gt;", result)
        self.assertIn("&amp;", result)
        self.assertIn("<b>", result)


# ═══════════════════════════════════════════════════════════════
#  10. split_html_text
# ═══════════════════════════════════════════════════════════════

class SplitHtmlTextTests(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(split_html_text(""), [])

    def test_short_text_single_chunk(self):
        result = split_html_text("Короткий текст")
        self.assertEqual(len(result), 1)

    def test_long_text_splits(self):
        text = "Слово " * 1000  # ~6000 символов
        result = split_html_text(text, max_length=500)
        self.assertTrue(len(result) > 1)
        for chunk in result:
            self.assertLessEqual(len(chunk), 500)

    def test_all_content_preserved(self):
        words = ["слово" + str(i) for i in range(200)]
        text = " ".join(words)
        chunks = split_html_text(text, max_length=200)
        joined = " ".join(chunks)
        for w in words:
            self.assertIn(w, joined)

    def test_card_interpretation_split(self):
        """Длинная интерпретация карты корректно делится."""
        text = "**Пятёрка Мечей**\n\n" + ("Описание. " * 500)
        chunks = split_html_text(text, max_length=4090)
        self.assertTrue(len(chunks) >= 1)
        self.assertTrue(all(len(c) <= 4090 for c in chunks))


# ═══════════════════════════════════════════════════════════════
#  11. Платёжный лог — форматы записей
# ═══════════════════════════════════════════════════════════════

class PaymentLogFormatTests(unittest.TestCase):
    """Проверяет формат строк платёжного лога."""

    def _log_payment(self, provider, user_ref, plan_name, amount):
        return f"ОПЛАТА | {provider} | {user_ref} | {plan_name} | {amount:.2f} руб"

    def _log_renewal(self, provider, user_ref, plan_name, amount, inv_id=None):
        s = f"ПРОДЛЕНИЕ | {provider} | {user_ref} | {plan_name} | {amount:.2f} руб"
        if inv_id:
            s += f" | InvId={inv_id}"
        return s

    def _log_error(self, provider, user_ref, attempt, plan_name):
        return f"ОШИБКА_СПИСАНИЯ | {provider} | {user_ref} | попытка {attempt} | {plan_name}"

    def _log_deactivate(self, user_ref, reason, plan_name):
        return f"АВТОПРОДЛ_ОТКЛ | {user_ref} | причина: {reason} | {plan_name}"

    def _log_expire(self, user_ref, plan_name):
        return f"ИСТЕЧЕНИЕ | {user_ref} | {plan_name}"

    def _log_cancel(self, user_ref, plan_name, end_date):
        return f"ОТМЕНА_АВТОПРОДЛ | {user_ref} | {plan_name} | до {end_date}"

    def _log_promo(self, user_ref, code, discount, days):
        return f"ПРОМОКОД | {user_ref} | код={code} | скидка={discount}% | дней={days}"

    def test_payment_yookassa(self):
        line = self._log_payment("Yookassa", "Иван [id=123]", "Базовый", 990)
        self.assertTrue(line.startswith("ОПЛАТА"))
        self.assertIn("Yookassa", line)
        self.assertIn("990.00 руб", line)

    def test_payment_robokassa(self):
        line = self._log_payment("Robokassa", "Иван", "Про", 1490)
        self.assertIn("Robokassa", line)
        self.assertIn("1490.00", line)

    def test_renewal_with_inv_id(self):
        line = self._log_renewal("Robokassa", "Иван", "Базовый", 990, inv_id=42)
        self.assertIn("ПРОДЛЕНИЕ", line)
        self.assertIn("InvId=42", line)

    def test_renewal_without_inv_id(self):
        line = self._log_renewal("Yookassa", "Иван", "Базовый", 990)
        self.assertIn("ПРОДЛЕНИЕ", line)
        self.assertNotIn("InvId", line)

    def test_error_attempt_format(self):
        line = self._log_error("Yookassa", "Иван", 2, "Базовый")
        self.assertIn("ОШИБКА_СПИСАНИЯ", line)
        self.assertIn("попытка 2", line)

    def test_deactivate_3_attempts(self):
        line = self._log_deactivate("Иван", "3 попытки", "Базовый")
        self.assertIn("АВТОПРОДЛ_ОТКЛ", line)
        self.assertIn("3 попытки", line)

    def test_deactivate_provider_reject(self):
        line = self._log_deactivate("Иван", "deactivate", "Базовый")
        self.assertIn("deactivate", line)

    def test_expire(self):
        line = self._log_expire("Иван", "Базовый")
        self.assertIn("ИСТЕЧЕНИЕ", line)

    def test_cancel(self):
        line = self._log_cancel("Иван", "Базовый", "2026-04-01")
        self.assertIn("ОТМЕНА_АВТОПРОДЛ", line)
        self.assertIn("до 2026-04-01", line)

    def test_promo(self):
        line = self._log_promo("Иван", "SPRING2026", 20, 7)
        self.assertIn("ПРОМОКОД", line)
        self.assertIn("код=SPRING2026", line)
        self.assertIn("скидка=20%", line)
        self.assertIn("дней=7", line)

    def test_all_formats_pipe_separated(self):
        """Все форматы используют | как разделитель."""
        lines = [
            self._log_payment("Y", "u", "p", 100),
            self._log_renewal("Y", "u", "p", 100),
            self._log_error("Y", "u", 1, "p"),
            self._log_deactivate("u", "x", "p"),
            self._log_expire("u", "p"),
            self._log_cancel("u", "p", "d"),
            self._log_promo("u", "c", 10, 5),
        ]
        for line in lines:
            self.assertIn(" | ", line)


# ═══════════════════════════════════════════════════════════════
#  12. get_all_admin_ids логика
# ═══════════════════════════════════════════════════════════════

class AdminIdsLogicTests(unittest.TestCase):
    """Тест логики объединения OWNER_IDS + is_admin=True из БД."""

    def _get_all_admin_ids(self, owner_ids, db_admin_ids):
        all_ids = set(owner_ids)
        all_ids.update(db_admin_ids)
        return all_ids

    def test_union_of_owners_and_db_admins(self):
        result = self._get_all_admin_ids([100, 200], [200, 300])
        self.assertEqual(result, {100, 200, 300})

    def test_owners_only(self):
        result = self._get_all_admin_ids([100], [])
        self.assertEqual(result, {100})

    def test_db_admins_only(self):
        result = self._get_all_admin_ids([], [300])
        self.assertEqual(result, {300})

    def test_no_duplicates(self):
        result = self._get_all_admin_ids([100, 100], [100])
        self.assertEqual(result, {100})

    def test_non_admin_filter(self):
        """Правильный запрос 'все НЕ-админы': исключаем is_admin=True И OWNER_IDS."""
        owner_ids = [100, 200]
        db_admin_ids = [200, 300]
        all_admin = self._get_all_admin_ids(owner_ids, db_admin_ids)

        all_users = [100, 200, 300, 400, 500, 600]
        non_admins = [u for u in all_users if u not in all_admin]
        self.assertEqual(non_admins, [400, 500, 600])

    def test_empty_both(self):
        result = self._get_all_admin_ids([], [])
        self.assertEqual(result, set())


if __name__ == "__main__":
    unittest.main()
