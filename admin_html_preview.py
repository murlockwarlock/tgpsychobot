import html
from html.parser import HTMLParser


class _HTMLPreviewParser(HTMLParser):
    void_tags = {"br", "hr", "img", "input", "meta", "link"}

    def __init__(self, allowed_tags=None):
        super().__init__(convert_charrefs=False)
        self.allowed_tags = allowed_tags
        self.tokens = []
        self.stack = []

    def _check_tag(self, tag):
        tag = tag.lower()
        if self.allowed_tags is not None and tag not in self.allowed_tags:
            raise ValueError(f"unsupported HTML tag: {tag}")
        return tag

    def handle_starttag(self, tag, attrs):
        tag = self._check_tag(tag)
        raw = self.get_starttag_text() or f"<{tag}>"
        if tag in self.void_tags:
            self.tokens.append(("atomic", tag, raw))
        else:
            self.tokens.append(("start", tag, raw))
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        tag = self._check_tag(tag)
        raw = self.get_starttag_text() or f"<{tag}/>"
        self.tokens.append(("atomic", tag, raw))

    def handle_endtag(self, tag):
        tag = self._check_tag(tag)
        if not self.stack or self.stack[-1] != tag:
            raise ValueError(f"unbalanced HTML tag: {tag}")
        self.stack.pop()
        self.tokens.append(("end", tag, f"</{tag}>"))

    def handle_data(self, data):
        if "<" in data or "&" in data:
            raise ValueError("unescaped HTML character")
        self.tokens.append(("data", "", data))

    def handle_entityref(self, name):
        self.tokens.append(("atomic", "", f"&{name};"))

    def handle_charref(self, name):
        self.tokens.append(("atomic", "", f"&#{name};"))

    def handle_comment(self, data):
        raise ValueError("HTML comments are not supported")

    def handle_decl(self, decl):
        raise ValueError("HTML declarations are not supported")

    def handle_pi(self, data):
        raise ValueError("HTML processing instructions are not supported")

    def finish(self):
        self.close()
        if self.stack or getattr(self, "rawdata", ""):
            raise ValueError("incomplete HTML")


def _escaped_preview(text: str, max_length: int, suffix: str) -> str:
    suffix = suffix[:max_length]
    budget = max(0, max_length - len(suffix))
    low, high = 0, len(text)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        escaped = html.escape(text[:middle])
        if len(escaped) <= budget:
            best = escaped
            low = middle + 1
        else:
            high = middle - 1
    return best + suffix


def truncate_html_preview(
    text: str | None,
    max_length: int,
    *,
    suffix: str = "…",
    allowed_tags: set[str] | None = None,
) -> str:
    source = text or ""
    if max_length <= 0:
        return ""

    parser = _HTMLPreviewParser(allowed_tags=allowed_tags)
    try:
        parser.feed(source)
        parser.finish()
    except (TypeError, ValueError):
        return _escaped_preview(source, max_length, suffix)

    if len(source) <= max_length:
        return source

    suffix = suffix[:max_length]
    output = []
    open_tags = []
    used = 0
    truncated = False

    def closing_size(tags):
        return sum(len(f"</{tag}>") for tag in tags)

    for kind, tag, raw in parser.tokens:
        if kind == "start":
            next_tags = [*open_tags, tag]
            if used + len(raw) + len(suffix) + closing_size(next_tags) > max_length:
                truncated = True
                break
            output.append(raw)
            used += len(raw)
            open_tags.append(tag)
            continue

        if kind == "end":
            next_tags = open_tags[:-1]
            if used + len(raw) + len(suffix) + closing_size(next_tags) > max_length:
                truncated = True
                break
            output.append(raw)
            used += len(raw)
            open_tags.pop()
            continue

        available = max_length - used - len(suffix) - closing_size(open_tags)
        if len(raw) <= available:
            output.append(raw)
            used += len(raw)
            continue

        if kind == "data" and available > 0:
            output.append(raw[:available])
            used += available
        truncated = True
        break

    if not truncated:
        return source

    closing = "".join(f"</{tag}>" for tag in reversed(open_tags))
    result = "".join(output) + suffix + closing
    if len(result) > max_length:
        return _escaped_preview(source, max_length, suffix)
    return result
