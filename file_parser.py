import io
from docx import Document
from pypdf import PdfReader
import openpyxl


async def parse_file(file_bytes: io.BytesIO, filename: str) -> str:
    import asyncio
    text = ""
    file_extension = filename.split('.')[-1].lower()

    if file_extension == 'txt' or file_extension == 'md':
        text = file_bytes.read().decode('utf-8')
    elif file_extension == 'pdf':
        def parse_pdf():
            nonlocal text
            reader = PdfReader(file_bytes)
            for page in reader.pages:
                text += page.extract_text() or ""
        await asyncio.to_thread(parse_pdf)
    elif file_extension == 'docx':
        def parse_docx():
            nonlocal text
            doc = Document(file_bytes)
            for para in doc.paragraphs:
                text += para.text + '\n'
        await asyncio.to_thread(parse_docx)
    elif file_extension == 'xlsx':
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
    file_extension = filename.split('.')[-1].lower()

    if file_extension == 'xlsx':
        def parse_xlsx_questions():
            workbook = openpyxl.load_workbook(file_bytes)
            sheet = workbook.active
            for row in sheet.iter_rows(min_row=1, values_only=True):
                if row[0]:
                    if str(row[0]).lower().startswith('вопрос') or str(row[1]).lower().startswith('категория'):
                        continue

                    is_rev = False
                    if len(row) > 2 and row[2]:
                        val = str(row[2]).strip().lower()
                        if val in ['1', 'true', '+', 'да', 'yes']:
                            is_rev = True

                    questions.append({
                        'text': str(row[0]).strip(),
                        'category': str(row[1]).strip() if len(row) > 1 and row[1] else 'general',
                        'is_reverse': is_rev
                    })

        await asyncio.to_thread(parse_xlsx_questions)

    elif file_extension == 'txt':
        content = file_bytes.read().decode('utf-8')
        lines = content.split('\n')
        for line in lines:
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    is_rev = False
                    if len(parts) > 2:
                        val = parts[2].strip().lower()
                        if val in ['1', 'true', '+']:
                            is_rev = True

                    questions.append({
                        'text': parts[0].strip(),
                        'category': parts[1].strip(),
                        'is_reverse': is_rev
                    })

    return questions