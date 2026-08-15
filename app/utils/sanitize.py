"""HTML sanitization utilities for XSS prevention.

Streamlit's ``unsafe_allow_html=True`` renders raw HTML, which creates
an XSS vector when user-controlled text is included in the HTML string.
This module provides a single function — ``html_escape`` — that
neutralizes HTML metacharacters before they reach the browser.

Usage:
    >>> from app.utils.sanitize import html_escape
    >>> html_escape('<script>alert(1)</script>')
    '&lt;script&gt;alert(1)&lt;/script&gt;'
"""

from __future__ import annotations

import html as _html


def html_escape(text: object) -> str:
    """Escape HTML metacharacters in user-echoed text.

    Converts ``&``, ``<``, ``>``, ``"``, and ``'`` to their
    HTML entities so that any embedded HTML or JavaScript is
    rendered inert when passed to ``st.markdown(..., unsafe_allow_html=True)``.

    Args:
        text: The text to escape. Converted to str via ``str()``
            first so ``None`` and numeric types are handled gracefully.

    Returns:
        Escaped string safe for inclusion in ``unsafe_allow_html`` blocks.

    Examples:
        >>> html_escape(None)
        ''
        >>> html_escape(3.14)
        '3.14'
        >>> html_escape('<b>bold</b>')
        '&lt;b&gt;bold&lt;/b&gt;'
        >>> html_escape('AT&T')
        'AT&amp;T'
    """
    if text is None:
        return ""
    return _html.escape(str(text), quote=True)
