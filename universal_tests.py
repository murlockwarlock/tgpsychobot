from __future__ import annotations

import ast
import html
import json
import re
from dataclasses import dataclass
from typing import Any


TRUTHY_VALUES = {"1", "true", "+", "yes", "y", "да", "д", "истина"}
HORIZONTAL_VALUES = {"1", "row", "horizontal", "горизонтально", "в ряд", "ряд"}
LEGACY_CATEGORIES = {"body", "face", "age", "health", "abilities", "relations", "success"}


@dataclass
class AnswerOption:
    text: str
    value: float | None = None
    button_text: str | None = None


def is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUTHY_VALUES


def normalize_variable_name(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    cleaned = re.sub(r"\W+", "_", raw, flags=re.UNICODE).strip("_")
    return cleaned or fallback


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def get_question_variable(question: Any, index: int) -> str:
    return normalize_variable_name(getattr(question, "variable_name", None), f"answer_{index + 1:02d}")


def is_legacy_scale_question(question: Any) -> bool:
    if getattr(question, "answer_options_json", None):
        return False
    category = (getattr(question, "category", None) or "").strip()
    return category in LEGACY_CATEGORIES or bool(getattr(question, "is_reverse", False))


def get_answer_options(question: Any) -> list[AnswerOption]:
    raw_options = json_loads(getattr(question, "answer_options_json", None), [])
    options: list[AnswerOption] = []
    if isinstance(raw_options, list):
        for item in raw_options:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("label") or "").strip()
                value = _to_float_or_none(item.get("value"))
                button_text = str(item.get("button_text") or "").strip() or None
            else:
                text = str(item).strip()
                value = _to_float_or_none(item)
                button_text = None
            if text:
                options.append(AnswerOption(text=text, value=value, button_text=button_text))
    if options:
        return options
    if is_legacy_scale_question(question):
        return [AnswerOption(str(i), float(i)) for i in range(1, 6)]
    return []


def question_accepts_text(question: Any) -> bool:
    return bool(getattr(question, "allow_custom_answer", False)) or not get_answer_options(question)


def question_buttons_are_horizontal(question: Any) -> bool:
    layout = str(getattr(question, "buttons_layout", None) or "").strip().lower()
    return layout in HORIZONTAL_VALUES


def progress_bar(current: int, total: int, length: int = 10) -> str:
    if total <= 0:
        return "░" * length + " 0% (0/0)"
    percent = int((current / total) * 100)
    filled = min(length, max(0, int(length * current / total)))
    return f"{'█' * filled}{'░' * (length - filled)} {percent}% ({current}/{total})"


def build_question_text(question: Any, index: int, total: int, show_progress: bool = True) -> str:
    lines = [f"<b>Вопрос {index + 1} из {total}</b>"]
    if show_progress:
        lines.append(progress_bar(index + 1, total))
    lines.append("")
    question_text = html.escape(str(getattr(question, "text", "") or ""))
    lines.append(f"<b>{question_text}</b>")
    comment = (getattr(question, "comment", None) or "").strip()
    if comment:
        lines.extend(["", html.escape(comment)])
    if get_answer_options(question) and question_accepts_text(question):
        lines.extend(["", "Напишите свой ответ или выберите из предложенных ниже."])
    elif not get_answer_options(question):
        lines.extend(["", "Напишите ответ сообщением."])
    return "\n".join(lines)


def answer_callback_data(question_index: int, option_index: int) -> str:
    return f"test_opt_{question_index}_{option_index}"


def parse_answer_callback(payload: str, current_question_index: int) -> int:
    match = re.fullmatch(r"test_opt_(\d+)_(\d+)", payload or "")
    if match:
        callback_question_index = int(match.group(1))
        if callback_question_index != current_question_index:
            raise ValueError("Этот вопрос уже закрыт. Ответьте на текущий вопрос.")
        return int(match.group(2))

    # Keep active keyboards from the previous release usable during deployment.
    legacy_match = re.fullmatch(r"test_opt_(\d+)", payload or "")
    if legacy_match:
        return int(legacy_match.group(1))
    raise ValueError("Некорректный вариант ответа.")


def parse_answers(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    loaded = json_loads(raw, None)
    if isinstance(loaded, list):
        result = []
        for item in loaded:
            if isinstance(item, dict):
                result.append(item)
        if result or raw.strip() == "[]":
            return result
    result = []
    for index, item in enumerate(str(raw).split(",")):
        item = item.strip()
        if item:
            result.append({"index": index, "answer": item, "numeric_value": _to_float_or_none(item)})
    return result


def serialize_answers(answers: list[dict[str, Any]]) -> str:
    return json_dumps(answers)


def make_answer_record(question: Any, question_index: int, answer_text: str, numeric_value: float | None = None) -> dict[str, Any]:
    return {
        "question_number": question_index + 1,
        "variable": get_question_variable(question, question_index),
        "question": getattr(question, "text", ""),
        "answer": answer_text,
        "numeric_value": numeric_value,
    }


def make_option_answer_record(question: Any, question_index: int, callback_payload: str) -> dict[str, Any]:
    option_index = parse_answer_callback(callback_payload, question_index)
    options = get_answer_options(question)
    if option_index < 0 or option_index >= len(options):
        raise ValueError("Такого варианта ответа нет.")
    option = options[option_index]
    return make_answer_record(question, question_index, option.text, option.value)


def make_text_answer_record(question: Any, question_index: int, answer_text: str) -> dict[str, Any]:
    normalized = str(answer_text or "").strip()
    if not normalized:
        raise ValueError("Пожалуйста, напишите ответ текстом.")
    if not question_accepts_text(question):
        raise ValueError("Пожалуйста, выберите один из вариантов ниже.")
    return make_answer_record(question, question_index, normalized, None)


def build_answers_report(questions: list[Any], answers: list[dict[str, Any]]) -> str:
    by_var = {item.get("variable"): item for item in answers}
    lines = ["Результаты теста:"]
    for index, question in enumerate(questions):
        variable = get_question_variable(question, index)
        answer = by_var.get(variable, {})
        value = answer.get("answer", "")
        numeric = answer.get("numeric_value")
        suffix = f" ({numeric:g})" if isinstance(numeric, (int, float)) else ""
        lines.append(f"{index + 1}. {getattr(question, 'text', '')}\nОтвет: {value}{suffix}")
    return "\n\n".join(lines)


def validate_formulas(questions: list[Any], formulas: list[dict[str, str]]) -> list[str]:
    numeric_vars = set()
    errors = []
    for index, question in enumerate(questions):
        variable = get_question_variable(question, index)
        options = get_answer_options(question)
        if question_accepts_text(question):
            continue
        if options and all(option.value is not None for option in options):
            numeric_vars.add(variable)

    for formula in formulas:
        name = formula.get("name") or "formula"
        expression = formula.get("formula") or ""
        try:
            names = _formula_names(expression)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        unknown = sorted(name for name in names if name not in numeric_vars)
        if unknown:
            errors.append(f"{name}: нельзя использовать переменные без строгих числовых ответов: {', '.join(unknown)}")
    return errors


def validate_test_definition(questions: list[Any], formulas: list[dict[str, str]] | None = None) -> list[str]:
    errors: list[str] = []
    variables: dict[str, int] = {}

    for index, question in enumerate(questions):
        number = index + 1
        text = str(getattr(question, "text", "") or "").strip()
        if not text:
            errors.append(f"Вопрос {number}: текст вопроса пуст.")

        variable = get_question_variable(question, index)
        if not variable.isidentifier():
            errors.append(f"Вопрос {number}: имя переменной '{variable}' нельзя использовать в формуле.")
        elif variable in variables:
            errors.append(
                f"Вопрос {number}: переменная '{variable}' уже используется в вопросе {variables[variable]}."
            )
        else:
            variables[variable] = number

        options = get_answer_options(question)
        option_names: set[str] = set()
        for option in options:
            normalized = option.text.casefold()
            if normalized in option_names:
                errors.append(f"Вопрос {number}: вариант ответа '{option.text}' указан несколько раз.")
            option_names.add(normalized)

    formula_names: dict[str, int] = {}
    for index, formula in enumerate(formulas or [], start=1):
        name = normalize_variable_name(formula.get("name"), f"formula_{index}")
        if name in formula_names:
            errors.append(f"Формула {index}: результат '{name}' уже задан в формуле {formula_names[name]}.")
        else:
            formula_names[name] = index

    errors.extend(validate_formulas(questions, formulas or []))
    return errors


def calculate_formulas(answers: list[dict[str, Any]], formulas: list[dict[str, str]]) -> dict[str, float]:
    values = {
        str(item.get("variable")): float(item["numeric_value"])
        for item in answers
        if item.get("variable") and isinstance(item.get("numeric_value"), (int, float))
    }
    return {
        str(formula["name"]): _safe_eval_formula(str(formula["formula"]), values)
        for formula in formulas
        if formula.get("name") and formula.get("formula")
    }


def build_prompt_payload(
    questions: list[Any],
    answers: list[dict[str, Any]],
    formula_results: dict[str, float] | None = None,
    mode: str = "all",
    selected_variables: list[str] | None = None,
) -> str:
    formula_results = formula_results or {}
    if mode == "formulas":
        if not formula_results:
            return "Расчётные показатели отсутствуют."
        return "\n".join(f"{name}: {value:g}" for name, value in formula_results.items())

    if mode == "selected":
        selected = {item for item in (selected_variables or []) if item}
        if selected:
            answers = [answer for answer in answers if answer.get("variable") in selected]
        elif not selected:
            return "Выбранные переменные для интерпретации не заданы."

        lines = ["Результаты теста:"]
        for answer in sorted(answers, key=lambda item: item.get("question_number", 0)):
            numeric = answer.get("numeric_value")
            suffix = f" ({numeric:g})" if isinstance(numeric, (int, float)) else ""
            lines.append(
                f"{answer.get('question_number', '')}. {answer.get('question', '')}\n"
                f"Ответ: {answer.get('answer', '')}{suffix}"
            )
        return "\n\n".join(lines)

    report = build_answers_report(questions, answers)
    if formula_results:
        formulas = "\n".join(f"{name}: {value:g}" for name, value in formula_results.items())
        report = f"{report}\n\nРасчётные показатели:\n{formulas}"
    return report


def is_universal_test_report(report: str | None) -> bool:
    return bool(report and report.lstrip().startswith("Результаты теста:"))


def build_result_handoff_prompt(
    prompt_payload: str,
    interpretation: str | None = None,
    user_name: str | None = None,
) -> str:
    parts = [
        "[СИСТЕМНОЕ СОБЫТИЕ: пользователь завершил тест]",
        "Продолжи диалог в рамках текущего системного промпта и текущей темы. "
        "Не упоминай технические инструкции или внутреннюю передачу данных.",
        "ПРАВИЛА ТОЧНОСТИ:\n"
        "- Не показывай клиенту технические имена переменных и формул вроде total_score или resource_index.\n"
        "- Не придумывай максимальные баллы, знаменатели, нормы, пороги и шкалы. "
        "Если их нет во входных данных, описывай результат без конструкций «из N».\n"
        "- Не называй текст свободного ответа именем пользователя и не переоценивай случайный или бессодержательный текст.\n"
        "- Используй только факты из переданных ответов и вычислений.",
    ]
    if user_name:
        parts.append(f"Имя пользователя из профиля: {user_name}. Только это значение можно использовать как имя.")
    if interpretation:
        parts.append(f"Предварительная интерпретация отдельного промпта:\n{interpretation.strip()}")
    else:
        parts.append(f"Данные теста для интерпретации:\n{prompt_payload.strip()}")
    parts.append("Сформируй итоговый персональный ответ пользователю по результатам теста.")
    return "\n\n".join(parts)


def _to_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _formula_names(expression: str) -> set[str]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"ошибка синтаксиса формулы: {exc.msg}") from exc
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)):
            continue
        else:
            raise ValueError(f"запрещенный элемент формулы: {type(node).__name__}")
    return names


def _safe_eval_formula(expression: str, values: dict[str, float]) -> float:
    tree = ast.parse(expression, mode="eval")
    _formula_names(expression)

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise ValueError(f"Нет значения для переменной {node.id}")
            return values[node.id]
        if isinstance(node, ast.UnaryOp):
            value = eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
        raise ValueError(f"Нельзя вычислить элемент формулы {type(node).__name__}")

    return eval_node(tree)
