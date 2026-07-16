import csv
import io
import json
import os
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from xml.sax.saxutils import escape

os.environ.setdefault("BOT_TOKEN", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from file_parser import parse_formulas_file, parse_questions_file
from universal_tests import (
    answer_callback_data,
    build_prompt_payload,
    build_question_text,
    build_result_handoff_prompt,
    calculate_formulas,
    get_answer_options,
    is_universal_test_report,
    json_dumps,
    make_option_answer_record,
    make_text_answer_record,
    parse_answer_callback,
    validate_test_definition,
)


HEADER = [
    "Номер",
    "Вопрос",
    "Комментарий",
    "Переменная",
    "Свой ответ",
    "Кнопки",
    "Вариант 1",
    "Вариант 2",
    "Вариант 3",
    "Значение 1",
    "Значение 2",
    "Значение 3",
]


def question(
    variable="answer_01",
    options=None,
    *,
    custom=False,
    text="Вопрос",
    comment=None,
    layout="vertical",
):
    return SimpleNamespace(
        text=text,
        comment=comment,
        variable_name=variable,
        allow_custom_answer=custom,
        buttons_layout=layout,
        answer_options_json=json_dumps(options) if options else None,
        category="general",
        is_reverse=False,
    )


def delimited_bytes(rows, delimiter="|", encoding="utf-8"):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode(encoding)


def _column_name(index):
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _worksheet_xml(rows):
    xml_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row):
            if value is None or value == "":
                continue
            reference = f"{_column_name(column_index)}{row_index}"
            cells.append(
                f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'
            )
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
    )


def xlsx_bytes(sheets):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        sheet_entries = []
        relationship_entries = []
        overrides = []
        for index, (title, rows) in enumerate(sheets, start=1):
            sheet_entries.append(
                f'<sheet name="{escape(title)}" sheetId="{index}" r:id="rId{index}"/>'
            )
            relationship_entries.append(
                f'<Relationship Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{index}.xml"/>'
            )
            overrides.append(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(rows))

        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{"".join(sheet_entries)}</sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(relationship_entries)}</Relationships>',
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/></Relationships>',
        )
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            f'{"".join(overrides)}</Types>',
        )
    return stream.getvalue()


class UniversalFileParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_repository_pipe_sample_parses_numeric_labels_once(self):
        with open("docs/universal_test_sample.csv", "rb") as sample:
            result = await parse_questions_file(io.BytesIO(sample.read()), "sample.csv")

        self.assertEqual(len(result["questions"]), 5)
        energy = get_answer_options(question(options=json.loads(result["questions"][1]["answer_options_json"])))
        self.assertEqual([(item.text, item.value) for item in energy], [("1", 1.0), ("2", 2.0), ("3", 3.0)])
        self.assertEqual(len(result["formulas"]), 3)

    async def test_semicolon_csv_preserves_quoted_multiline_comment(self):
        rows = [HEADER, [1, "Как вы себя чувствуете?", "Первая строка\nВторая строка", "mood", "", 1, "Плохо", "Нормально", "Хорошо", 1, 2, 3]]
        result = await parse_questions_file(io.BytesIO(delimited_bytes(rows, ";")), "test.csv")

        self.assertEqual(result["questions"][0]["comment"], "Первая строка\nВторая строка")
        self.assertEqual(json.loads(result["questions"][0]["answer_options_json"])[2]["value"], 3.0)

    async def test_comma_csv_with_bom_and_blank_line_before_header(self):
        rows = [[], HEADER, [1, "Вопрос", "", "score", "", "", "Нет", "Иногда", "Да", 0, 1, 2]]
        raw = b"\xef\xbb\xbf" + delimited_bytes(rows, ",")
        result = await parse_questions_file(io.BytesIO(raw), "excel.csv")

        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["variable_name"], "score")

    async def test_cp1251_text_file_is_supported(self):
        rows = [HEADER, [1, "Самочувствие", "Комментарий", "health", "", "", "Плохо", "Средне", "Хорошо", 1, 2, 3]]
        result = await parse_questions_file(io.BytesIO(delimited_bytes(rows, "|", "cp1251")), "test.txt")
        self.assertEqual(result["questions"][0]["text"], "Самочувствие")

    async def test_xlsx_questions_and_formula_sheet_without_openpyxl(self):
        question_rows = [HEADER, [1, "Энергия", "Выберите", "energy", "", 1, 1, 2, 3, 10, 20, 30]]
        formula_rows = [["Переменная", "Формула"], ["total", "energy * 2"]]
        raw = xlsx_bytes([("формулы", formula_rows), ("questions", question_rows)])

        result = await parse_questions_file(io.BytesIO(raw), "test.xlsx")
        options = json.loads(result["questions"][0]["answer_options_json"])
        self.assertEqual(options, [
            {"text": "1", "value": 10.0},
            {"text": "2", "value": 20.0},
            {"text": "3", "value": 30.0},
        ])
        self.assertEqual(result["formulas"], [{"name": "total", "formula": "energy * 2"}])

    async def test_formula_only_csv_and_xlsx(self):
        csv_raw = delimited_bytes([["Переменная", "Формула"], ["sum", "=a + b"]], ";")
        csv_formulas = await parse_formulas_file(io.BytesIO(csv_raw), "formulas.csv")
        xlsx_formulas = await parse_formulas_file(
            io.BytesIO(xlsx_bytes([("formulas", [["name", "formula"], ["sum", "a + b"]])])),
            "formulas.xlsx",
        )
        self.assertEqual(csv_formulas, [{"name": "sum", "formula": "a + b"}])
        self.assertEqual(xlsx_formulas, csv_formulas)

        parsed_as_questions = await parse_questions_file(io.BytesIO(csv_raw), "formulas.csv")
        self.assertEqual(parsed_as_questions["questions"], [])
        self.assertEqual(parsed_as_questions["formulas"], csv_formulas)

    async def test_missing_numeric_value_stays_missing_and_fails_formula_validation(self):
        rows = [HEADER, [1, "Оценка", "", "score", "", "", "Нет", "Иногда", "Да", 0, "", 2]]
        result = await parse_questions_file(io.BytesIO(delimited_bytes(rows)), "test.csv")
        parsed_question = question(
            variable="score",
            options=json.loads(result["questions"][0]["answer_options_json"]),
        )
        errors = validate_test_definition([parsed_question], [{"name": "total", "formula": "score"}])
        self.assertTrue(any("score" in error for error in errors))

    async def test_numeric_button_labels_are_numeric_without_separate_scores(self):
        rows = [HEADER, [1, "Оценка", "", "score", "", 1, 1, 2, 3, "", "", ""]]
        result = await parse_questions_file(io.BytesIO(delimited_bytes(rows)), "numeric.csv")
        options = json.loads(result["questions"][0]["answer_options_json"])
        self.assertEqual([item["value"] for item in options], [1.0, 2.0, 3.0])


class UniversalAnswerFlowTests(unittest.TestCase):
    def setUp(self):
        self.strict = question(
            variable="score",
            options=[{"text": "Нет", "value": 0}, {"text": "Да", "value": 2}],
        )

    def test_button_payload_records_label_and_numeric_value(self):
        payload = answer_callback_data(3, 1)
        record = make_option_answer_record(self.strict, 3, payload)
        self.assertEqual(record["answer"], "Да")
        self.assertEqual(record["numeric_value"], 2.0)
        self.assertEqual(record["question_number"], 4)

    def test_stale_question_button_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "уже закрыт"):
            make_option_answer_record(self.strict, 2, answer_callback_data(1, 0))

    def test_unknown_option_and_malformed_callback_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "варианта"):
            make_option_answer_record(self.strict, 0, answer_callback_data(0, 9))
        with self.assertRaisesRegex(ValueError, "Некорректный"):
            parse_answer_callback("test_opt_bad", 0)

    def test_legacy_active_button_remains_compatible(self):
        self.assertEqual(parse_answer_callback("test_opt_1", 7), 1)

    def test_strict_question_rejects_text(self):
        with self.assertRaisesRegex(ValueError, "выберите"):
            make_text_answer_record(self.strict, 0, "свой ответ")

    def test_custom_and_open_questions_accept_trimmed_text(self):
        custom = question(options=[{"text": "Готовый", "value": None}], custom=True)
        opened = question(options=None)
        self.assertEqual(make_text_answer_record(custom, 0, "  свой вариант  ")["answer"], "свой вариант")
        self.assertEqual(make_text_answer_record(opened, 0, "текст")["answer"], "текст")

    def test_short_button_label_keeps_full_answer_for_storage(self):
        from keyboards import universal_test_answer_keyboard

        item = question(options=[{"text": "Когда мной командуют и не дают решать самому", "button_text": "Когда командуют"}])
        option = get_answer_options(item)[0]
        markup = universal_test_answer_keyboard([option], question_index=0)
        record = make_option_answer_record(item, 0, answer_callback_data(0, 0))

        self.assertEqual(markup.inline_keyboard[0][0].text, "Когда командуют")
        self.assertEqual(record["answer"], "Когда мной командуют и не дают решать самому")

    def test_question_output_escapes_html_and_keeps_multiline_comment(self):
        item = question(text="2 < 3 & 5 > 4", comment="Строка <1>\nСтрока &2")
        rendered = build_question_text(item, 0, 5, True)
        self.assertIn("2 &lt; 3 &amp; 5 &gt; 4", rendered)
        self.assertIn("Строка &lt;1&gt;\nСтрока &amp;2", rendered)
        self.assertIn("20% (1/5)", rendered)

    def test_telegram_and_max_keyboards_emit_question_bound_callbacks(self):
        from keyboards import universal_test_answer_keyboard
        from max_messenger_bot.keyboards import universal_test_answers_keyboard

        options = get_answer_options(self.strict)
        telegram_markup = universal_test_answer_keyboard(options, True, 4)
        telegram_payloads = [button.callback_data for row in telegram_markup.inline_keyboard for button in row]
        max_markup = universal_test_answers_keyboard(options, True, 4)
        max_payloads = [button["payload"] for row in max_markup[0]["payload"]["buttons"] for button in row]
        option_payloads = [payload for payload in telegram_payloads if payload.startswith("test_opt_")]
        max_option_payloads = [payload for payload in max_payloads if payload.startswith("test_opt_")]
        self.assertEqual(option_payloads, ["test_opt_4_0", "test_opt_4_1"])
        self.assertIn("cancel_test", telegram_payloads)
        self.assertEqual(max_option_payloads, option_payloads)
        self.assertIn("cancel_test", max_payloads)


class MaxAnswerHandlerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from max_messenger_bot.services import tests as tests_service

        self.service = tests_service
        self.questions = [
            question("score", [{"text": "Нет", "value": 0}, {"text": "Да", "value": 2}]),
            question("note", None),
        ]
        self.test_session = SimpleNamespace(current_question_index=0, answers="[]", is_finished=False)
        self.session = MagicMock()
        self.session.get = AsyncMock(return_value=self.test_session)
        result = MagicMock()
        result.scalars.return_value.all.return_value = self.questions
        self.session.execute = AsyncMock(return_value=result)
        self.session.commit = AsyncMock()
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=self.session)
        context.__aexit__ = AsyncMock(return_value=False)
        self.session_context = context
        self.client = SimpleNamespace(send_message=AsyncMock())

    async def test_real_max_callback_handler_records_answer_and_advances(self):
        with (
            patch.object(self.service, "async_session_maker", return_value=self.session_context),
            patch.object(self.service, "_send_question", AsyncMock()) as send_question,
        ):
            await self.service.process_answer(self.client, 10, 20, "test_opt_0_1")

        stored = json.loads(self.test_session.answers)
        self.assertEqual(stored[0]["answer"], "Да")
        self.assertEqual(stored[0]["numeric_value"], 2.0)
        self.assertEqual(self.test_session.current_question_index, 1)
        send_question.assert_awaited_once_with(self.client, 10, 20, 1)

    async def test_real_max_callback_handler_rejects_stale_button(self):
        with patch.object(self.service, "async_session_maker", return_value=self.session_context):
            await self.service.process_answer(self.client, 10, 20, "test_opt_1_0")

        self.assertEqual(self.test_session.current_question_index, 0)
        self.session.commit.assert_not_awaited()
        self.assertIn("уже закрыт", self.client.send_message.await_args.kwargs["text"])

    async def test_real_max_text_handler_rejects_text_for_strict_question(self):
        states = SimpleNamespace(clear=AsyncMock())
        with patch.object(self.service, "async_session_maker", return_value=self.session_context):
            await self.service.process_text_answer(self.client, states, 10, 20, "свой ответ")

        self.assertEqual(self.test_session.current_question_index, 0)
        self.assertIn("выберите", self.client.send_message.await_args.kwargs["text"])


class MaxCompletionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_runs_separate_prompt_then_calling_prompt_and_clears_state(self):
        from max_messenger_bot.services import tests as tests_service

        item = question("score", [{"text": "Да", "value": 2}])
        answer = make_option_answer_record(item, 0, answer_callback_data(0, 0))
        test_session = SimpleNamespace(
            answers=json.dumps([answer], ensure_ascii=False),
            invocation_topic_id=77,
            invocation_dialogue_id=5,
        )
        config = SimpleNamespace(
            formulas_json=None,
            formulas_enabled=False,
            interpretation_input_mode="all",
            interpretation_selected_variables=None,
            separate_result_prompt_enabled=True,
            result_system_prompt="Отдельный промпт",
        )
        user = SimpleNamespace(current_dialogue_id=9, current_topic_id=88)

        async def get_model(model, _key, **_kwargs):
            return {
                "TestSession": test_session,
                "TestConfig": config,
                "User": user,
            }[model.__name__]

        session = MagicMock()
        session.get = AsyncMock(side_effect=get_model)
        result = MagicMock()
        result.scalars.return_value.all.return_value = [item]
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        session.add = MagicMock()
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=session)
        context.__aexit__ = AsyncMock(return_value=False)
        client = SimpleNamespace(send_message=AsyncMock())
        states = SimpleNamespace(clear=AsyncMock())

        with (
            patch.object(tests_service, "async_session_maker", return_value=context),
            patch.object(tests_service, "get_ai_response_direct", AsyncMock(return_value="Предварительный вывод")) as direct,
            patch.object(tests_service, "get_ai_response", AsyncMock(return_value="Финальный ответ")) as calling_prompt,
        ):
            await tests_service._finish_universal_test(client, 10, 20, states)

        direct.assert_awaited_once()
        calling_prompt.assert_awaited_once()
        self.assertIn("Предварительный вывод", calling_prompt.await_args.args[1])
        self.assertEqual(calling_prompt.await_args.kwargs["topic_id_override"], 77)
        self.assertEqual(calling_prompt.await_args.kwargs["dialogue_id_override"], 5)
        states.clear.assert_awaited_once_with(20)
        self.assertEqual(client.send_message.await_args.kwargs["text"], "Финальный ответ")

    async def test_disabled_max_test_does_not_start_for_regular_user(self):
        from max_messenger_bot.services import tests as tests_service

        user = SimpleNamespace(is_admin=False)
        config = SimpleNamespace(is_enabled=False)

        async def get_model(model, _key, **_kwargs):
            return {"User": user, "TestConfig": config}[model.__name__]

        session = MagicMock()
        session.get = AsyncMock(side_effect=get_model)
        session.execute = AsyncMock()
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=session)
        context.__aexit__ = AsyncMock(return_value=False)
        client = SimpleNamespace(send_message=AsyncMock())

        with (
            patch.object(tests_service, "async_session_maker", return_value=context),
            patch("max_messenger_bot.services.common.is_admin", new=AsyncMock(return_value=False)),
        ):
            await tests_service.start_test(client, 10, 20)

        session.execute.assert_not_awaited()
        self.assertIn("отключено", client.send_message.await_args.kwargs["text"])

    async def test_max_continue_does_not_send_internal_report_to_client(self):
        from max_messenger_bot.services import tests as tests_service

        internal_report = "Результаты теста:\n\nОтвет: Да\n\nРасчётные показатели:\ntotal_score: 8"
        test_session = SimpleNamespace(answers=internal_report)
        config = SimpleNamespace(marathon_url="https://example.test")
        user = SimpleNamespace(current_dialogue_id=1, current_topic_id=None)

        async def get_model(model, _key, **_kwargs):
            return {
                "TestSession": test_session,
                "TestConfig": config,
                "Content": None,
                "User": user,
            }[model.__name__]

        session = MagicMock()
        session.get = AsyncMock(side_effect=get_model)
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=session)
        context.__aexit__ = AsyncMock(return_value=False)
        client = SimpleNamespace(send_message=AsyncMock())

        with patch.object(tests_service, "async_session_maker", return_value=context):
            await tests_service.show_results(client, 10, 20)

        visible_texts = [call.kwargs.get("text", "") for call in client.send_message.await_args_list]
        self.assertTrue(visible_texts)
        self.assertFalse(any("total_score" in text for text in visible_texts))
        self.assertFalse(any("Результаты теста:" in text for text in visible_texts))


class UniversalFormulaAndPromptTests(unittest.TestCase):
    def test_duplicate_variables_and_formula_names_are_rejected(self):
        questions = [question("same"), question("same")]
        errors = validate_test_definition(
            questions,
            [{"name": "result", "formula": "1"}, {"name": "result", "formula": "2"}],
        )
        self.assertTrue(any("уже используется" in error for error in errors))
        self.assertTrue(any("уже задан" in error for error in errors))

    def test_custom_answer_question_cannot_be_used_in_formula(self):
        item = question("score", [{"text": "Да", "value": 1}], custom=True)
        errors = validate_test_definition([item], [{"name": "total", "formula": "score"}])
        self.assertTrue(any("строгих числовых" in error for error in errors))

    def test_calculation_and_all_selected_formula_payloads(self):
        questions = [
            question("a", [{"text": "Один", "value": 1}], text="Первый"),
            question("b", [{"text": "Два", "value": 2}], text="Второй"),
        ]
        answers = [
            make_option_answer_record(questions[0], 0, answer_callback_data(0, 0)),
            make_option_answer_record(questions[1], 1, answer_callback_data(1, 0)),
        ]
        formulas = calculate_formulas(answers, [{"name": "sum", "formula": "a + b"}])
        self.assertEqual(formulas, {"sum": 3.0})
        self.assertIn("Первый", build_prompt_payload(questions, answers, formulas, "all"))
        selected = build_prompt_payload(questions, answers, formulas, "selected", ["b"])
        self.assertNotIn("Первый", selected)
        self.assertIn("Второй", selected)
        self.assertNotIn("sum: 3", selected)
        self.assertEqual(build_prompt_payload(questions, answers, formulas, "formulas"), "sum: 3")

    def test_selected_legacy_auto_variable_keeps_original_question_number(self):
        questions = [question(variable=None, text="Первый"), question(variable=None, text="Второй")]
        answers = [
            {"question_number": 1, "variable": "answer_01", "question": "Первый", "answer": "A", "numeric_value": None},
            {"question_number": 2, "variable": "answer_02", "question": "Второй", "answer": "B", "numeric_value": None},
        ]
        selected = build_prompt_payload(questions, answers, {}, "selected", ["answer_02"])
        self.assertNotIn("Первый", selected)
        self.assertIn("2. Второй", selected)

    def test_separate_interpretation_is_handed_to_calling_prompt(self):
        prompt = build_result_handoff_prompt("score: 3", "Предварительный вывод", "Анна")
        self.assertIn("Предварительная интерпретация", prompt)
        self.assertIn("Предварительный вывод", prompt)
        self.assertIn("текущего системного промпта", prompt)
        self.assertIn("Не придумывай максимальные баллы", prompt)
        self.assertIn("Имя пользователя из профиля: Анна", prompt)
        self.assertIn("total_score", prompt)

    def test_internal_report_is_detected_for_client_side_hiding(self):
        self.assertTrue(is_universal_test_report("\nРезультаты теста:\n\ntotal_score: 8"))
        self.assertFalse(is_universal_test_report("ОБЩИЙ ИТОГ: 8/10"))


if __name__ == "__main__":
    unittest.main()
