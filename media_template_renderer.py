import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


class TemplateRenderError(ValueError):
    pass


class TemplateRenderInvariantError(TemplateRenderError):
    pass


FONT_FILES = {
    'certificate_regular': Path(__file__).parent / 'assets/fonts/DejaVuSans.ttf',
    'certificate_bold': Path(__file__).parent / 'assets/fonts/DejaVuSans-Bold.ttf',
}
MAX_FIELDS = 16
MAX_VALUE_LENGTH = 256
FIELD_KEYS = {'label', 'bbox', 'font_key', 'min_font_size', 'max_font_size', 'align', 'valign', 'color', 'uppercase', 'max_lines', 'required', 'sample_value'}


def validate_fonts():
    for path in FONT_FILES.values():
        font = ImageFont.truetype(str(path), 20)
        missing = bytes(font.getmask('\uffff'))
        if any(bytes(font.getmask(char)) == missing for char in 'ИВАНСОЛДАТОВЁё'):
            raise RuntimeError('Template font is unavailable')


def plain_text(value, *, limit=MAX_VALUE_LENGTH):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or any(unicodedata.category(c).startswith('C') for c in value):
        raise TemplateRenderError('Некорректное значение поля шаблона.')
    return value.strip()


def normalized_source(content):
    if not isinstance(content, bytes) or not 0 < len(content) <= 20 * 1024 * 1024:
        raise TemplateRenderError('Не удалось прочитать изображение шаблона.')
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 20_000_000 or getattr(image, 'n_frames', 1) != 1:
                raise TemplateRenderError('Изображение шаблона слишком большое или анимировано.')
            return ImageOps.exif_transpose(image).convert('RGBA')
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise TemplateRenderError('Не удалось прочитать изображение шаблона.') from exc


def bind_source(content, fields=None):
    image = normalized_source(content)
    return {'version': 1, 'source_sha256': hashlib.sha256(content).hexdigest(), 'canvas': {'width': image.width, 'height': image.height}, 'output_format': 'png', 'delivery': 'document', 'fields': fields or {}}


def _validate_config(config, *, allow_empty=False):
    def invalid():
        raise TemplateRenderError('Некорректная настройка шаблона изображения.')
    if not isinstance(config, dict) or len(json.dumps(config)) > 65536 or set(config) != {'version', 'source_sha256', 'canvas', 'output_format', 'delivery', 'fields'}:
        invalid()
    if type(config['version']) is not int or config['version'] != 1 or config['output_format'] != 'png' or config['delivery'] != 'document':
        invalid()
    if not isinstance(config['source_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', config['source_sha256']):
        invalid()
    canvas = config['canvas']
    if not isinstance(canvas, dict) or set(canvas) != {'width', 'height'} or any(type(v) is not int or not 0 < v <= 10000 for v in canvas.values()) or canvas['width'] * canvas['height'] > 20_000_000:
        invalid()
    fields = config['fields']
    if not isinstance(fields, dict) or not (0 if allow_empty else 1) <= len(fields) <= MAX_FIELDS:
        invalid()
    for key, field in fields.items():
        if not re.fullmatch('[a-z][a-z0-9_]{0,31}', key) or not isinstance(field, dict) or set(field) - FIELD_KEYS:
            invalid()
        required = FIELD_KEYS - {'sample_value'}
        if not required <= set(field):
            invalid()
        plain_text(field['label'], limit=80)
        box = field['bbox']
        if not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box) or not (0 <= box[0] < box[2] <= canvas['width'] and 0 <= box[1] < box[3] <= canvas['height']):
            invalid()
        if field['font_key'] not in FONT_FILES or field['align'] not in ('left', 'center', 'right') or field['valign'] not in ('top', 'middle', 'bottom'):
            invalid()
        if any(type(field[k]) is not int for k in ('min_font_size', 'max_font_size', 'max_lines')) or not 8 <= field['min_font_size'] <= field['max_font_size'] <= 256 or not 1 <= field['max_lines'] <= 4:
            invalid()
        if not isinstance(field['color'], str) or not re.fullmatch('#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?', field['color']) or any(type(field[k]) is not bool for k in ('required', 'uppercase')):
            invalid()
        if 'sample_value' in field:
            plain_text(field['sample_value'])
    return config


def validate_config(config, *, allow_empty=False):
    try:
        return _validate_config(config, allow_empty=allow_empty)
    except TemplateRenderError:
        raise
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise TemplateRenderError('Некорректная настройка шаблона изображения.') from exc


def validate_values(config, values):
    if not isinstance(values, dict) or len(values) > MAX_FIELDS or set(values) - set(config['fields']):
        raise TemplateRenderError('Неизвестное поле шаблона.')
    result = {key: plain_text(value) for key, value in values.items()}
    if any(field['required'] and key not in result for key, field in config['fields'].items()):
        raise TemplateRenderError('Не заполнены обязательные поля шаблона.')
    return result


def fitted_text(text, field):
    if field['uppercase']:
        text = text.upper()
    probe = ImageFont.truetype(str(FONT_FILES[field['font_key']]), field['max_font_size'], layout_engine=ImageFont.Layout.BASIC)
    missing = bytes(probe.getmask('\uffff'))
    if any(bytes(probe.getmask(char)) == missing for char in set(text) if not char.isspace()):
        raise TemplateRenderError('Шрифт не поддерживает символы поля шаблона.')
    width, height = field['bbox'][2] - field['bbox'][0], field['bbox'][3] - field['bbox'][1]
    for size in range(field['max_font_size'], field['min_font_size'] - 1, -1):
        font = ImageFont.truetype(str(FONT_FILES[field['font_key']]), size, layout_engine=ImageFont.Layout.BASIC)
        lines = ['']
        for word in text.split(' '):
            candidate = (lines[-1] + ' ' + word).strip()
            if font.getlength(candidate) <= width:
                lines[-1] = candidate
            elif lines[-1] and len(lines) < field['max_lines']:
                lines.append(word)
            else:
                break
        else:
            rendered = '\n'.join(lines)
            draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
            box = draw.multiline_textbbox((0, 0), rendered, font=font, spacing=4, align=field['align'])
            if box[2] - box[0] <= width and box[3] - box[1] <= height and all(font.getlength(line) <= width for line in lines):
                return rendered, font, box
    raise TemplateRenderError('Текст не помещается в поле шаблона.')


def pixels_differ(first, second):
    return any(band.getbbox() is not None for band in ImageChops.difference(first, second).split())


def validate_pixels(source, result, boxes):
    if source.size != result.size or source.mode != result.mode:
        raise TemplateRenderInvariantError('Нарушена геометрия шаблона. Результат не отправлен.')
    difference = ImageChops.difference(source, result)
    for box in boxes:
        difference.paste((0, 0, 0, 0), tuple(box))
    if any(band.getbbox() is not None for band in difference.split()):
        raise TemplateRenderInvariantError('Изменены пиксели вне полей шаблона. Результат не отправлен.')


@dataclass(frozen=True)
class RenderedTemplate:
    image_bytes: bytes
    width: int
    height: int
    source_sha256: str
    fields: tuple[str, ...]


def render_template(content, config, values):
    validate_config(config)
    source = normalized_source(content)
    if hashlib.sha256(content).hexdigest() != config['source_sha256'] or source.size != (config['canvas']['width'], config['canvas']['height']):
        raise TemplateRenderError('Шаблон изображения был изменён и требует повторной настройки.')
    values = validate_values(config, values)
    if not values:
        raise TemplateRenderError('Не заполнены поля шаблона.')
    result = source.copy()
    for key, text in values.items():
        field = config['fields'][key]
        rendered, font, bounds = fitted_text(text, field)
        left, top, right, bottom = field['bbox']
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = left + {'left': 0, 'center': (right - left - width) // 2, 'right': right - left - width}[field['align']] - bounds[0]
        y = top + {'top': 0, 'middle': (bottom - top - height) // 2, 'bottom': bottom - top - height}[field['valign']] - bounds[1]
        layer = Image.new('RGBA', source.size)
        ImageDraw.Draw(layer).multiline_text((x, y), rendered, font=font, fill=field['color'], spacing=4, align=field['align'])
        result.alpha_composite(layer)
    output = io.BytesIO()
    result.save(output, format='PNG')
    final = normalized_source(output.getvalue())
    validate_pixels(source, final, [field['bbox'] for field in config['fields'].values()])
    if any(not pixels_differ(source.crop(config['fields'][key]['bbox']), final.crop(config['fields'][key]['bbox'])) for key in values):
        raise TemplateRenderInvariantError('Поле шаблона не изменилось. Результат не отправлен.')
    return RenderedTemplate(output.getvalue(), final.width, final.height, config['source_sha256'], tuple(values))


validate_fonts()
