import io
import math
import re

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


def ratio_value(ratio):
    width, height = ratio.split(':')
    return int(width) / int(height)


def closest_ratio(value, supported):
    return min(supported, key=lambda ratio: abs(math.log(value / ratio_value(ratio))))


def is_negated(prompt, position):
    match = re.search(r"\b(?:not|never|don't|не)\b[^.;!?\n]{0,80}$", prompt[:position], re.I)
    return bool(match and not re.search(r'\b(?:but|instead|но|а)\b', match[0], re.I))


def explicit_ratio(prompt):
    requested = None
    for match in re.finditer(r'(?<![\d.])([1-9]\d{0,3})\s*([:x×])\s*([1-9]\d{0,3})(?![\d.])', prompt, re.I):
        width, _, height = match.groups()
        if not is_negated(prompt, match.start()):
            requested = int(width) / int(height)
    return requested


def generation_ratio(prompt):
    prompt = (prompt or '').lower()
    ratio = explicit_ratio(prompt)
    if ratio is not None:
        return ratio
    orientations = re.finditer(
        r'\b(?:vertical|horizontal|вертикаль\w*|горизонталь\w*|квадрат\w*)\b'
        r'|\bsquare\s+(?:image|format|canvas|poster|composition)\b'
        r'|\b(?:image|canvas|photo|poster)\s+square\b'
        r'|\b(?:landscape|portrait)\s+(?:orientation|format|canvas|poster)\b',
        prompt,
    )
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


def source_ratio(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.verify()
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
        if image.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
    return width / height


def edit_ratio(prompt, image_bytes):
    source = source_ratio(image_bytes)
    prompt = (prompt or '').lower()
    requested = explicit_ratio(prompt)
    if requested is None:
        match = re.search(
            r'\b(?:make|convert|crop|resize|сделай|сделать|измени|изменить|обрежь)\s+'
            r'(?:(?:the|this|image|canvas|photo|it|to|a|an|изображение|фото|формат)\s+)*'
            r'(horizontal|landscape|vertical|portrait|square|горизонталь\w*|вертикаль\w*|квадрат\w*)\b',
            prompt,
        )
        if match and not is_negated(prompt, match.start()):
            requested = generation_ratio(match[1])
    return source if requested is None else requested


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


def edit_input(model, prompt, source_url, image_bytes):
    value = edit_ratio(prompt, image_bytes)
    result = {'prompt': prompt, 'image_urls': [source_url]}
    if model == 'google/nano-banana-edit':
        result.update(output_format='png', aspect_ratio=closest_ratio(value, NANO_BANANA_RATIOS))
    elif model == 'bytedance/seedream-v4-edit':
        result.update(image_size=SEEDREAM_SIZES[closest_ratio(value, SEEDREAM_RATIOS)], image_resolution='1K', max_images=1)
    elif model == 'seedream/4.5-edit':
        result.update(aspect_ratio=closest_ratio(value, SEEDREAM_RATIOS), quality='basic')
    else:
        raise ValueError(f'Неподдерживаемая KIE image edit model: {model}')
    return result


def validate_result(image_bytes, payload):
    requested = payload.get('aspect_ratio')
    if requested is None:
        requested = next(ratio for ratio, size in SEEDREAM_SIZES.items() if size == payload['image_size'])
    if abs(math.log(source_ratio(image_bytes) / ratio_value(requested))) > 0.03:
        raise ValueError('Image aspect ratio does not match the request')
    return image_bytes
