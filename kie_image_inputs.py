import io
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from PIL import Image


SEEDREAM_RATIOS = ('1:1', '3:4', '2:3', '9:16', '4:3', '3:2', '16:9', '21:9')
IMAGEN_RATIOS = ('1:1', '3:4', '9:16', '4:3', '16:9')
NANO_BANANA_RATIOS = (*SEEDREAM_RATIOS, '4:5', '5:4')
SEEDREAM_SIZES = {
    '1:1': 'square_hd',
    '3:4': 'portrait_4_3',
    '2:3': 'portrait_3_2',
    '9:16': 'portrait_16_9',
    '4:3': 'landscape_4_3',
    '3:2': 'landscape_3_2',
    '16:9': 'landscape_16_9',
    '21:9': 'landscape_21_9',
}
QUOTED_TEXT = re.compile(r'''"(?:\\.|[^"\\])*"|“[^”]*”|«[^»]*»|(?<!\w)'(?:\\.|[^'\\])*'(?!\w)|`[^`]*`''')


class ImageGeometryError(ValueError):
    pass


class ModelGeometryLimitation(ImageGeometryError):
    pass


@dataclass(frozen=True)
class EditGeometry:
    mode: Literal['preserve', 'resize']
    source_size: tuple[int, int]
    target_ratio: Fraction
    target_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class EditRequest:
    payload: dict
    geometry: EditGeometry


def shape_prompt(prompt):
    prompt = (prompt or '').lower().strip()
    return prompt[1:-1] if QUOTED_TEXT.fullmatch(prompt) else prompt


def without_literals(prompt):
    return QUOTED_TEXT.sub(lambda match: ' ' * len(match[0]), prompt)


def ratio_value(ratio):
    width, height = ratio.split(':')
    return int(width) / int(height)


def closest_ratio(value, supported):
    return min(supported, key=lambda ratio: abs(math.log(value / ratio_value(ratio))))


def is_negated(prompt, position):
    match = re.search(r"\b(?:not|never|don't|не)\b[^.;!?\n]{0,80}$", prompt[:position], re.I)
    return bool(match and not re.search(r'\b(?:but|instead|но|а)\b', match[0], re.I))


def has_geometry_request(prefix):
    return bool(re.search(
        r'\b(?:aspect[ -]?ratio|ratio|format|canvas|dimensions?|resolution|size|resize|crop|формат\w*|соотношен\w*|размер\w*|обреж\w*|холст\w*)\b[^.;!?\n]{0,40}$'
        r'|\b(?:make|convert|change|сделай|сделать|измени|изменить)(?:\s+(?:the|this|image|canvas|photo|it|to|a|an|изображение|фото|в|формат)){0,5}\s*$',
        prefix,
        re.I,
    ))


def preservation_context(prefix):
    clause = re.split(r'[.!?;\n]', without_literals(prefix))[-1]
    actions = list(re.finditer(r'\b(keep|preserve|retain|maintain|остав\w*|сохран\w*|resize|crop|change|convert|make|сделай|измени|обрежь)\b', clause, re.I))
    if actions:
        return bool(re.fullmatch(r'keep|preserve|retain|maintain|остав\w*|сохран\w*', actions[-1][0], re.I))
    return bool(re.search(r'\b(?:same|original|source|current|исходн\w*|текущ\w*)\b', clause, re.I))


def has_edit_geometry_request(prefix):
    field = r'(?:aspect[ _-]?ratio|ratio|format|dimensions?|size|соотношение\s+сторон|формат\w*|размер\w*|пропорц\w*)'
    image = r'(?:image|canvas|photo|picture|frame|it|изображение|фото|холст)'
    article = r'(?:(?:the|this|original|source)\s+)?'
    target = r'\s*(?:(?:to|into|in|at|в|на|до)\s*)?(?:[:=]\s*)?$'
    clause = re.split(r'[.!?;\n]', prefix)[-1].strip()
    return bool(
        re.search(rf'\b(?:resize|crop|обрежь)\s+{article}(?:{image}\s+)?(?:{field}\s*)?{target}', prefix, re.I)
        or re.search(rf'\b(?:make|change|convert|set|use|сделай|сделать|измени|изменить)\s+{article}(?:{image}|{field})(?:\s+{field})?{target}', prefix, re.I)
        or re.fullmatch(rf'(?:(?:target|output|requested|canvas)\s+)?{field}{target}', clause, re.I)
    )


def explicit_geometry(prompt, *, editing=False):
    requested = None
    literals = list(QUOTED_TEXT.finditer(prompt))
    for match in re.finditer(r'(?<![\d.])([1-9]\d{0,3})\s*([:x×])\s*([1-9]\d{0,3})(?!\d|\.\d)', prompt, re.I):
        width, separator, height = match.groups()
        literal = next((item for item in literals if item.start() < match.start() and match.end() < item.end()), None)
        prefix = prompt[:literal.start() if literal else match.start()]
        geometry_request = has_geometry_request(prefix)
        if literal and (literal[0][1:-1].strip() != match[0] or not geometry_request):
            continue
        if is_negated(prompt, match.start()):
            continue
        if re.search(r'\b(?:at|time|starts|в|время)\s*$', prefix, re.I) and not geometry_request:
            continue
        if editing and (preservation_context(prefix) or not has_edit_geometry_request(prefix)) and prompt.strip() != match[0]:
            continue
        requested = (int(width), int(height), separator.lower())
    return requested


def explicit_ratio(prompt, *, editing=False):
    geometry = explicit_geometry(prompt, editing=editing)
    return geometry[0] / geometry[1] if geometry else None


def generation_ratio(prompt):
    prompt = shape_prompt(prompt)
    ratio = explicit_ratio(prompt)
    if ratio is not None:
        return ratio
    prompt = without_literals(prompt)
    direction = r'(?:vertical|horizontal|square|landscape|portrait|вертикаль\w*|горизонталь\w*|квадрат\w*)'
    canvas = r'(?:image|format|canvas|poster|composition|orientation|banner|photo|view|frame|illustration|wallpaper|certificate|изображен\w*|формат\w*|плакат\w*|постер\w*|баннер\w*|холст\w*|фото|сертификат\w*)'
    adjectives = r'(?:(?!(?:of|on|with|and|featuring|containing|depicting|с|на|в|и|для|из)\b)\w+\s+){0,2}'
    orientations = re.finditer(rf'\b{direction}\s+{adjectives}{canvas}\b|\b{canvas}\s+(?:is\s+)?{direction}\b', prompt)
    for match in orientations:
        if is_negated(prompt, match.start()):
            continue
        if re.search(r'square|квадрат', match[0]):
            return 1.0
        return 4 / 3 if re.search(r'horizontal|landscape|горизонталь', match[0]) else 3 / 4
    if re.search(r'\b(?:horizontal|landscape|widescreen|panoramic|banner|горизонталь\w*|широкоформат\w*|баннер\w*)\b|wide shot|cinematic wide', prompt):
        return 4 / 3
    if re.search(r'\b(?:vertical|portrait|вертикаль\w*|портрет\w*)\b|phone wallpaper', prompt):
        return 3 / 4
    if re.search(r'\b(?:tarot|card|oracle|poster|cover|таро|карта|плакат|обложка)\b|full[ -]body', prompt):
        return 3 / 4
    return 1.0


def generation_shape(prompt):
    ratio = closest_ratio(generation_ratio(prompt), SEEDREAM_RATIOS)
    value = ratio_value(ratio)
    size = '1024x1536' if value < 1 else '1536x1024' if value > 1 else '1024x1024'
    return ratio, size


def source_size(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        if image.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
    return width, height


def source_ratio(image_bytes):
    width, height = source_size(image_bytes)
    return width / height


def requested_edit_ratio(prompt):
    prompt = shape_prompt(prompt)
    requested = explicit_ratio(prompt, editing=True)
    if requested is None:
        prompt = without_literals(prompt)
        direction = r'(horizontal|landscape|vertical|portrait|square|горизонталь\w*|вертикаль\w*|квадрат\w*)'
        patterns = (
            r'\b(?:make|convert|crop|resize|change|сделай|сделать|измени|изменить|обрежь)\s+'
            r'(?:(?:the|this)\s+)?(?:image|canvas|photo|picture|frame|format|it|изображение|фото|холст|формат)\s+'
            r'(?:(?:to|into|a|an|в)\s+)*' + direction + r'\b',
            r'\b(?:orientation|format|ориентац\w*|формат\w*)\s*(?:(?:to|into|в|:|=)\s*)?' + direction + r'\b',
            r'\b(?:crop|resize|convert|change|обрежь|измени)\s+(?:to|into|в)\s+(?:(?:a|an)\s+)?' + direction + r'\b',
            r'\b' + direction + r'\s+(?:orientation|format)\b',
        )
        matches = sorted((match for pattern in patterns for match in re.finditer(pattern, prompt)), key=lambda match: match.start())
        for match in matches:
            if not is_negated(prompt, match.start()) and not preservation_context(prompt[:match.end()]):
                requested = generation_ratio(match[1])
    return requested


def edit_geometry(prompt, image_bytes):
    original_size = source_size(image_bytes)
    numeric = explicit_geometry(shape_prompt(prompt), editing=True)
    requested = requested_edit_ratio(prompt)
    if requested is None:
        return EditGeometry('preserve', original_size, Fraction(*original_size), original_size)
    target_ratio = Fraction(numeric[0], numeric[1]) if numeric else Fraction(requested).limit_denominator(10000)
    target_size = (numeric[0], numeric[1]) if numeric and numeric[2] != ':' else None
    return EditGeometry('resize', original_size, target_ratio, target_size)


def generation_input(model, prompt):
    value = generation_ratio(prompt)
    if model in ('google/imagen4-fast', 'google/imagen4-ultra', 'google/imagen4'):
        result = {'prompt': prompt, 'aspect_ratio': closest_ratio(value, IMAGEN_RATIOS)}
        if model == 'google/imagen4-fast':
            result['num_images'] = '1'
        return result
    ratio = closest_ratio(value, SEEDREAM_RATIOS)
    if model == 'bytedance/seedream-v4-text-to-image':
        return {'prompt': prompt, 'image_size': SEEDREAM_SIZES[ratio], 'image_resolution': '1K', 'max_images': 1}
    if model == 'seedream/4.5-text-to-image':
        return {'prompt': prompt, 'aspect_ratio': ratio, 'quality': 'basic'}
    raise ValueError(f'Неподдерживаемая KIE image generation model: {model}')


def prepare_edit(model, prompt, source_url, image_bytes):
    geometry = edit_geometry(prompt, image_bytes)
    supported = NANO_BANANA_RATIOS if model == 'google/nano-banana-edit' else SEEDREAM_RATIOS
    ratio = next((ratio for ratio in supported if Fraction(ratio.replace(':', '/')) == geometry.target_ratio), None)
    if ratio is None:
        width, height = geometry.source_size
        if geometry.mode == 'preserve':
            raise ModelGeometryLimitation(f'Модель {model} не поддерживает точное сохранение исходного формата {width}×{height} в этом API. Редактирование не запущено.')
        raise ModelGeometryLimitation(f'Модель {model} не поддерживает запрошенное соотношение сторон {geometry.target_ratio.numerator}:{geometry.target_ratio.denominator}. Редактирование не запущено.')
    if geometry.mode == 'resize' and geometry.target_size is not None:
        width, height = geometry.target_size
        raise ModelGeometryLimitation(f'API модели {model} не позволяет задать точные размеры {width}×{height}. Редактирование не запущено.')
    result = {'prompt': prompt, 'image_urls': [source_url]}
    if model == 'google/nano-banana-edit':
        result.update(output_format='png', aspect_ratio=ratio)
    elif model == 'bytedance/seedream-v4-edit':
        result.update(image_size=SEEDREAM_SIZES[ratio], image_resolution='1K', max_images=1)
    elif model == 'seedream/4.5-edit':
        result.update(aspect_ratio=ratio, quality='basic')
    else:
        raise ValueError(f'Неподдерживаемая KIE image edit model: {model}')
    return EditRequest(result, geometry)


def edit_input(model, prompt, source_url, image_bytes):
    return prepare_edit(model, prompt, source_url, image_bytes).payload


def validate_result(image_bytes, *, geometry: EditGeometry):
    result_size = source_size(image_bytes)
    if geometry.mode == 'preserve':
        if result_size != geometry.source_size:
            width, height = geometry.source_size
            raise ImageGeometryError(f'Модель не сохранила исходные размеры {width}×{height}. Результат не отправлен.')
    elif geometry.target_size is not None:
        if result_size != geometry.target_size:
            raise ImageGeometryError('Модель не выполнила запрошенное изменение размеров. Результат не отправлен.')
    elif Fraction(*result_size) != geometry.target_ratio:
        raise ImageGeometryError('Модель не выполнила запрошенное изменение пропорций. Результат не отправлен.')
    return image_bytes


def validate_generation_result(image_bytes, payload):
    requested = payload.get('aspect_ratio')
    if requested is None:
        requested = next(ratio for ratio, size in SEEDREAM_SIZES.items() if size == payload['image_size'])
    if abs(math.log(source_ratio(image_bytes) / ratio_value(requested))) > 0.03:
        raise ValueError('Image aspect ratio does not match the request')
    return image_bytes
