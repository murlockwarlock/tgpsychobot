import io
import csv
try:
    from docx import Document
except ModuleNotFoundError:
    Document = None
try:
    from pypdf import PdfReader
except ModuleNotFoundError:
    PdfReader = None
try:
    import openpyxl
except ModuleNotFoundError:
    openpyxl = None

from universal_tests import HORIZONTAL_VALUES, is_truthy, json_dumps, normalize_variable_name


async def parse_file(file_bytes: io.BytesIO, filename: str) -> str:
    import asyncio
    text = ""
    file_extension = filename.split('.')[-1].lower()

    if file_extension == 'txt' or file_extension == 'md':
        text = file_bytes.read().decode('utf-8')
    elif file_extension == 'pdf':
        if PdfReader is None:
            raise RuntimeError("Для чтения PDF нужен пакет pypdf.")
        def parse_pdf():
            nonlocal text
            reader = PdfReader(file_bytes)
            for page in reader.pages:
                text += page.extract_text() or ""
        await asyncio.to_thread(parse_pdf)
    elif file_extension == 'docx':
        if Document is None:
            raise RuntimeError("Для чтения DOCX нужен пакет python-docx.")
        def parse_docx():
            nonlocal text
            doc = Document(file_bytes)
            for para in doc.paragraphs:
                text += para.text + '\n'
        await asyncio.to_thread(parse_docx)
    elif file_extension == 'xlsx':
        if openpyxl is None:
            raise RuntimeError("Для чтения XLSX нужен пакет openpyxl.")
        def parse_xlsx():
            nonlocal text
            workbook = openpyxl.load_workbook(file_bytes)
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.value:
                            text += str(cell.value) + ' '
                    text += '\n'
        await asyncio.to_thread(parse_xlsx)

    return text.strip()


async def parse_questions_file(file_bytes: io.BytesIO, filename: str) -> list[dict]:
    import asyncio
    questions = []
    formulas = []
    file_extension = filename.split('.')[-1].lower()

    if file_extension == 'xlsx':
        if openpyxl is None:
            raise RuntimeError("Для чтения XLSX нужен пакет openpyxl.")
        def parse_xlsx_questions():
            workbook = openpyxl.load_workbook(file_bytes)
            sheet = workbook.active
            rows = list(sheet.iter_rows(min_row=1, values_only=True))
            question_rows, inline_formula_rows = _split_formula_section(rows)
            questions.extend(_parse_question_rows(question_rows))
            formulas.extend(_parse_formula_rows(inline_formula_rows))
            formula_sheet = next((ws for ws in workbook.worksheets if ws.title.strip().lower() in {"formulas", "формулы"}), None)
            if formula_sheet:
                formulas.extend(_parse_formula_rows(formula_sheet.iter_rows(min_row=1, values_only=True)))

        await asyncio.to_thread(parse_xlsx_questions)

    elif file_extension in {'txt', 'csv'}:
        content = file_bytes.read().decode('utf-8')
        delimiter = '|' if '|' in content else ';' if ';' in content else ','
        rows = list(csv.reader(content.splitlines(), delimiter=delimiter))
        question_rows, formula_rows = _split_formula_section(rows)
        questions.extend(_parse_question_rows(question_rows))
        formulas.extend(_parse_formula_rows(formula_rows))

    return {"questions": questions, "formulas": formulas}


def _parse_question_rows(rows) -> list[dict]:
    rows = [tuple(row or []) for row in rows]
    first_real = next((row for row in rows if any(_cell_text(cell) for cell in row)), ())
    new_format = _looks_like_new_test_format(first_real)
    result = []
    for row_index, row in enumerate(rows, start=1):
        if not any(_cell_text(cell) for cell in row):
            continue
        lowered = [_cell_text(cell).lower() for cell in row[:6]]
        if row_index == 1 and any("вопрос" == item or item.startswith("номер") for item in lowered):
            continue
        if new_format:
            parsed = _parse_new_question_row(row, len(result))
        else:
            parsed = _parse_legacy_question_row(row)
        if parsed:
            result.append(parsed)
    return result


def _split_formula_section(rows) -> tuple[list, list]:
    question_rows = []
    formula_rows = []
    target = question_rows
    for row in rows:
        first = str(row[0]).strip().lower() if row else ""
        if first in {"[formulas]", "formulas", "формулы", "[формулы]"}:
            target = formula_rows
            continue
        target.append(row)
    return question_rows, formula_rows


def _parse_legacy_question_row(row) -> dict | None:
    if not _cell_text(row[0] if len(row) > 0 else None):
        return None
    if _cell_text(row[0]).lower().startswith('вопрос') or _cell_text(row[1] if len(row) > 1 else None).lower().startswith('категория'):
        return None
    return {
        'text': _cell_text(row[0]),
        'category': _cell_text(row[1] if len(row) > 1 else None) or 'general',
        'is_reverse': is_truthy(row[2] if len(row) > 2 else None),
    }


def _parse_new_question_row(row, index: int) -> dict | None:
    question_text = _cell_text(row[1] if len(row) > 1 else None)
    if not question_text:
        return None
    options = _parse_answer_options(row[6:])
    return {
        'text': question_text,
        'category': 'general',
        'is_reverse': False,
        'comment': _cell_text(row[2] if len(row) > 2 else None) or None,
        'variable_name': normalize_variable_name(row[3] if len(row) > 3 else None, f"answer_{index + 1:02d}"),
        'allow_custom_answer': is_truthy(row[4] if len(row) > 4 else None),
        'buttons_layout': 'horizontal' if _cell_text(row[5] if len(row) > 5 else None).lower() in HORIZONTAL_VALUES else 'vertical',
        'answer_options_json': json_dumps(options) if options else None,
    }


def _parse_answer_options(cells) -> list[dict]:
    values = [_cell_text(cell) for cell in cells if _cell_text(cell)]
    if not values:
        return []

    numbers = [_to_float_or_none(value) for value in values]
    if all(item is not None for item in numbers):
        return [{"text": value, "value": number} for value, number in zip(values, numbers)]

    split_at = None
    for candidate in range(1, len(values)):
        labels = values[:candidate]
        numeric_tail = values[candidate:]
        if len(numeric_tail) == len(labels) and all(_to_float_or_none(item) is not None for item in numeric_tail):
            split_at = candidate
            break

    if split_at is None:
        return [{"text": value, "value": _to_float_or_none(value)} for value in values]

    labels = values[:split_at]
    numeric_tail = values[split_at:]
    return [{"text": label, "value": _to_float_or_none(numeric_tail[index])} for index, label in enumerate(labels)]


def _parse_formula_rows(rows) -> list[dict]:
    formulas = []
    for row in rows:
        if not row or not any(_cell_text(cell) for cell in row):
            continue
        first = _cell_text(row[0])
        second = _cell_text(row[1] if len(row) > 1 else None)
        if first.lower() in {"переменная", "variable", "name", "название"}:
            continue
        if first.lower() == "formula" and len(row) > 2:
            first = _cell_text(row[1])
            second = _cell_text(row[2])
        if first and second:
            formulas.append({"name": normalize_variable_name(first, f"formula_{len(formulas) + 1}"), "formula": second})
    return formulas


def _looks_like_new_test_format(row) -> bool:
    cells = [_cell_text(cell).lower() for cell in row[:6]]
    if any("свой" in cell or "комментар" in cell or "переменн" in cell for cell in cells):
        return True
    first = _cell_text(row[0] if len(row) > 0 else None)
    second = _cell_text(row[1] if len(row) > 1 else None)
    return first.isdigit() and bool(second)


def _cell_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_float_or_none(value):
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
