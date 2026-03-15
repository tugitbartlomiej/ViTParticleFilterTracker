"""Pure markdown-to-HTML conversion for session content display.

No Qt dependency -- this module works with plain strings only.
"""

import re


def _inline_fmt(text: str) -> str:
    """Apply inline markdown formatting: bold, italic, inline code, links."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Inline code: `code`
    text = re.sub(
        r"`([^`]+)`",
        r'<code style="background-color: #161B22; color: #79C0FF; '
        r'padding: 1px 4px; border-radius: 3px; font-size: 12px;">\1</code>',
        text,
    )

    # Bold: **text**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # Italic: *text* (but not inside bold markers)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)

    # Links: [text](url)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" style="color: #58A6FF;">\1</a>',
        text,
    )

    return text


def _close_open_list(html_lines: list[str], in_ul: bool, in_ol: bool) -> tuple[bool, bool]:
    """Close any open <ul> or <ol> tags and return updated flags."""
    if in_ul:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")
    return False, False


def _render_code_fence(
    line: str,
    html_lines: list[str],
    in_code_block: bool,
    in_ul: bool,
    in_ol: bool,
) -> tuple[bool, bool, bool]:
    """Handle opening/closing of fenced code blocks."""
    if in_code_block:
        html_lines.append("</code></pre>")
        return False, in_ul, in_ol

    in_ul, in_ol = _close_open_list(html_lines, in_ul, in_ol)
    html_lines.append(
        '<pre style="background-color: #0D1117; border: 1px solid #30363D; '
        'border-radius: 6px; padding: 12px; overflow-x: auto;">'
        '<code style="color: #C9D1D9;">'
    )
    return True, in_ul, in_ol


_HEADING_STYLES = {
    4: ('h4', 'color: #E6EDF3; margin: 8px 0 4px 0; font-size: 13px;'),
    3: ('h3', 'color: #E6EDF3; margin: 12px 0 6px 0; font-size: 14px; '
         'border-bottom: 1px solid #21262D; padding-bottom: 4px;'),
    2: ('h2', 'color: #58A6FF; margin: 16px 0 8px 0; font-size: 16px; '
         'border-bottom: 1px solid #21262D; padding-bottom: 4px;'),
    1: ('h1', 'color: #E6EDF3; margin: 16px 0 8px 0; font-size: 20px; '
         'border-bottom: 2px solid #21262D; padding-bottom: 6px;'),
}


def _try_render_heading(stripped: str, html_lines: list[str]) -> bool:
    """Render a markdown heading if the line matches. Returns True if handled."""
    for level in (4, 3, 2, 1):
        prefix = "#" * level + " "
        if stripped.startswith(prefix):
            tag, style = _HEADING_STYLES[level]
            text = _inline_fmt(stripped[len(prefix):])
            html_lines.append(f'<{tag} style="{style}">{text}</{tag}>')
            return True
    return False


def _try_render_checkbox(stripped: str, html_lines: list[str], in_ul: bool) -> tuple[bool, bool]:
    """Render a checkbox list item if the line matches.

    Returns (handled, in_ul).
    """
    checked = stripped.startswith("- [x] ") or stripped.startswith("- [X] ")
    unchecked = stripped.startswith("- [ ] ")

    if not checked and not unchecked:
        return False, in_ul

    if not in_ul:
        html_lines.append('<ul style="list-style: none; padding-left: 8px;">')
        in_ul = True

    text = _inline_fmt(stripped[6:])
    if checked:
        html_lines.append(
            f'<li style="color: #8B949E; text-decoration: line-through;">'
            f'&#9745; {text}</li>'
        )
    else:
        html_lines.append(f"<li>&#9744; {text}</li>")

    return True, in_ul


def markdown_to_html(md: str) -> str:
    """Convert simple markdown to styled HTML for display in QTextBrowser.

    Supports: headings, bold, italic, unordered/ordered lists,
    checkboxes, code blocks, inline code, horizontal rules, links.
    """
    html_lines: list[str] = []
    in_code_block = False
    in_ul = False
    in_ol = False

    for line in md.split("\n"):
        # Fenced code blocks
        if line.strip().startswith("```"):
            in_code_block, in_ul, in_ol = _render_code_fence(
                line, html_lines, in_code_block, in_ul, in_ol,
            )
            continue

        if in_code_block:
            escaped = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            html_lines.append(escaped)
            continue

        stripped = line.strip()

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            in_ul, in_ol = _close_open_list(html_lines, in_ul, in_ol)
            html_lines.append(
                '<hr style="border: 1px solid #30363D; margin: 16px 0;">'
            )
            continue

        # Empty line
        if not stripped:
            in_ul, in_ol = _close_open_list(html_lines, in_ul, in_ol)
            html_lines.append("<br>")
            continue

        # Headings
        if stripped.startswith("#"):
            in_ul, in_ol = _close_open_list(html_lines, in_ul, in_ol)
            if _try_render_heading(stripped, html_lines):
                continue

        # Checkboxes
        handled, in_ul = _try_render_checkbox(stripped, html_lines, in_ul)
        if handled:
            continue

        # Unordered list
        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if not in_ul:
                html_lines.append(
                    '<ul style="padding-left: 20px; margin: 4px 0;">'
                )
                in_ul = True
            text = _inline_fmt(stripped[2:])
            html_lines.append(f"<li>{text}</li>")
            continue

        # Ordered list
        ol_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if ol_match:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if not in_ol:
                html_lines.append(
                    '<ol style="padding-left: 20px; margin: 4px 0;">'
                )
                in_ol = True
            text = _inline_fmt(ol_match.group(2))
            html_lines.append(f"<li>{text}</li>")
            continue

        # Close open lists before a regular paragraph
        in_ul, in_ol = _close_open_list(html_lines, in_ul, in_ol)

        # Italic line (e.g. *Session saved: ...*)
        if stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
            text = stripped[1:-1]
            html_lines.append(
                f'<p style="color: #8B949E; font-style: italic; '
                f'margin: 4px 0; font-size: 11px;">{text}</p>'
            )
            continue

        # Regular paragraph
        text = _inline_fmt(stripped)
        html_lines.append(f'<p style="margin: 4px 0;">{text}</p>')

    # Close any tags left open
    if in_ul:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")
    if in_code_block:
        html_lines.append("</code></pre>")

    body = "\n".join(html_lines)
    return (
        '<div style="font-family: \'Segoe UI\', sans-serif; color: #E6EDF3; '
        f'line-height: 1.6;">{body}</div>'
    )
