"""Shared utilities for CLI stats scripts."""

import re
import shutil

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def col(text: object, *styles: str, enabled: bool = True) -> str:
    """Colorize text with ANSI codes."""
    if not enabled:
        return str(text)
    return "".join(ANSI.get(s, "") for s in styles) + str(text) + ANSI["reset"]


def get_width() -> int:
    """Get terminal width."""
    return shutil.get_terminal_size((80, 24)).columns


def visible_len(s: str) -> int:
    """Get visible length of string (excluding ANSI codes)."""
    return len(ANSI_RE.sub("", s))


def shorten_plain(s: str, max_len: int) -> str:
    """Shorten string with ellipsis if needed."""
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return s[:max_len]
    return s[: max_len - 1] + "…"


def tok(n: int) -> str:
    """Format token count with K/M/B suffix."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 10_000:
        return f"{n / 1e3:.0f}K"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def bar(pct: float, width: int = 20) -> str:
    """Create a progress bar."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    return "█" * filled + "░" * (width - filled)


def pad_line(s: str, width: int) -> str:
    """Pad line to specified width based on visible length."""
    return s + " " * (width - visible_len(s))


def merge_columns(
    left: list[str], right: list[str], gap: int = 3, sep: str = "│"
) -> list[str]:
    """Merge two sections side-by-side with a separator."""
    lw = max((visible_len(l) for l in left), default=0)
    merged = []
    for i in range(max(len(left), len(right))):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        merged.append(pad_line(l, lw) + " " * gap + col(sep, "dim") + " " + r)
    return merged


def can_merge_columns(
    left: list[str], right: list[str], width: int, gap: int = 3, indent: int = 2
) -> bool:
    """Check if two columns can fit side-by-side."""
    if not left or not right:
        return False
    lw = max((visible_len(l) for l in left), default=0)
    rw = max((visible_len(r) for r in right), default=0)
    needed = indent + lw + gap + 2 + rw  # "│ " is 2 columns
    return needed <= width
