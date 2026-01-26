#!/usr/bin/env python3
"""Claude Code usage statistics."""

import json, os, subprocess, sys
from datetime import datetime
from pathlib import Path
import httpx
from utils import (
    col,
    get_width,
    shorten_plain,
    visible_len,
    tok,
    pad_line,
    merge_columns,
)

STATS = Path.home() / ".claude" / "stats-cache.json"
PROJECTS = Path.home() / ".claude.json"
HISTORY = Path.home() / ".claude" / "history.jsonl"
# Pricing per MTok: [input, output]. Cache read/write rates are derived from input price.
# Cache write multiplier defaults to 5-minute pricing (1.25x). Set CLAUDE_CACHE_WRITE_MULTIPLIER=2.0 for 1-hour writes.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = float(os.getenv("CLAUDE_CACHE_WRITE_MULTIPLIER", "1.25"))
PRICE = {
    "opus-4-5": [5, 25],
    "sonnet-4-5": [3, 15],
    "haiku-4-5": [1, 5],
}


def fdate(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%b %d")
    except:
        return s[-5:] if len(s) >= 5 else s


def freset(s):
    try:
        secs = (
            datetime.fromisoformat(s.replace("Z", "+00:00"))
            - datetime.now(datetime.fromisoformat(s.replace("Z", "+00:00")).tzinfo)
        ).total_seconds()
        if secs <= 0:
            return "now"
        h, m = divmod(int(secs) // 60, 60)
        d, h = divmod(h, 24)
        return f"{d}d {h}h" if d else f"{h}h {m}m" if h else f"{m}m"
    except:
        return "?"


def fdur(ms):
    h, m = divmod(ms // 60000, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def pkey(m):
    m = m.lower()
    for k in ["opus-4-5", "sonnet-4-5", "haiku-4-5"]:
        if (
            k in m
            or k.replace("-4-5", "-4.5")
            .replace("-4-1", "-4.1")
            .replace("-3-5", "-3.5")
            .replace("-3-7", "-3.7")
            in m
        ):
            return k
    return None


def cost(u, pk):
    if not pk or pk not in PRICE:
        return 0
    p = PRICE[pk]
    cr = p[0] * CACHE_READ_MULTIPLIER
    cw = p[0] * CACHE_WRITE_MULTIPLIER
    return (
        u.get("inputTokens", 0) * p[0]
        + u.get("outputTokens", 0) * p[1]
        + u.get("cacheReadInputTokens", 0) * cr
        + u.get("cacheCreationInputTokens", 0) * cw
    ) / 1e6


def creds():
    if sys.platform != "darwin":
        return None
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return None
        acct = next(
            (
                l.split("=")[1].strip().strip('"')
                for l in r.stdout.split("\n")
                if '"acct"<blob>=' in l
            ),
            None,
        )
        if not acct:
            return None
        r = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                "Claude Code-credentials",
                "-a",
                acct,
                "-w",
            ],
            capture_output=True,
            text=True,
        )
        return (
            json.loads(r.stdout.strip()).get("claudeAiOauth")
            if r.returncode == 0
            else None
        )
    except:
        return None


def build_header(st, cr, width):
    """Build header box that spans full terminal width."""
    tier = ""
    if cr:
        t = str(cr.get("rateLimitTier", "") or "")
        tier = (
            "Max 5x"
            if "max_5x" in t
            else "Max 20x"
            if "max_20x" in t
            else "Pro"
            if "pro" in t.lower()
            else ""
        )
    inner = width - 4  # Account for borders and padding

    parts = [
        tier,
        f"Since {fdate(st.get('firstSessionDate', ''))}"
        if st.get("firstSessionDate")
        else "",
    ]

    mu = st.get("modelUsage", {}) or {}
    ti = sum(u.get("inputTokens", 0) for u in mu.values())
    to = sum(u.get("outputTokens", 0) for u in mu.values())
    tcr = sum(u.get("cacheReadInputTokens", 0) for u in mu.values())
    tcw = sum(u.get("cacheCreationInputTokens", 0) for u in mu.values())
    tt = ti + to + tcr + tcw
    if tt:
        bits = [f"{tok(ti)} in", f"{tok(to)} out"]
        if tcr:
            bits.append(f"{tok(tcr)} cache read")
        if tcw:
            bits.append(f"{tok(tcw)} cache write")
        parts.append(f"{tok(tt)} tokens ({', '.join(bits)})")

    turns = None
    if HISTORY.exists():
        try:
            with HISTORY.open("r", encoding="utf-8") as f:
                turns = sum(1 for _ in f)
        except:
            turns = None
    if turns is not None:
        parts.append(f"{turns:,} turns")

    sub = shorten_plain(" • ".join(filter(None, parts)), inner)
    lines = [col("╭" + "─" * (width - 2) + "╮", "dim")]
    title = col("CLAUDE CODE USAGE", "bold", "cyan")
    title_len = 17  # "CLAUDE CODE USAGE"
    lines.append(
        f"{col('│', 'dim')}  {title}{' ' * (inner - title_len)}{col('│', 'dim')}"
    )
    if sub:
        lines.append(
            f"{col('│', 'dim')}  {col(sub, 'dim')}{' ' * (inner - len(sub))}{col('│', 'dim')}"
        )
    lines.append(col("╰" + "─" * (width - 2) + "╯", "dim"))
    return lines


def build_quick_stats(st, da):
    """Build quick stats line."""
    tool_calls = sum(d.get("toolCallCount", 0) for d in da)
    sessions = f"{st.get('totalSessions', 0):,}"
    messages = f"{st.get('totalMessages', 0):,}"
    tool_calls_s = f"{tool_calls:,}"
    return [
        f"  {col(sessions, 'bold')} sessions  {col('│', 'dim')}  "
        f"{col(messages, 'bold')} messages  {col('│', 'dim')}  "
        f"{col(tool_calls_s, 'bold')} tools  {col('│', 'dim')}  {col(len(da), 'bold')} days"
    ]


def build_usage_limits(cr):
    """Build usage limits section."""
    lines = []
    if not cr:
        return lines
    try:
        data = httpx.get(
            "https://api.anthropic.com/api/oauth/usage",
            headers={
                "Authorization": f"Bearer {cr['accessToken']}",
                "anthropic-beta": "oauth-2025-04-20",
            },
            timeout=30,
        ).json()
        for nm, ky in [("5-Hour", "five_hour"), ("7-Day", "seven_day")]:
            if (d := data.get(ky)) and (u := d.get("utilization")) is not None:
                f = int(25 * u / 100)
                bc = "green" if u < 50 else "yellow" if u < 80 else "magenta"
                lines.append(
                    f"  {col(f'{nm:<8}', 'bold')}{col('━' * f, bc)}{col('━' * (25 - f), 'dim')} {u:>3.0f}%  {col('resets', 'dim')} {col(freset(d.get('resets_at', '')), 'cyan')}"
                )
    except:
        pass
    return lines


def build_model_breakdown(mu, tot_out):
    """Build model breakdown section, returns (lines, totals_dict, cost_breakdown_dict)."""
    lines = [col("Model Breakdown", "bold"), col("─" * 45, "dim")]
    ct = {"in": 0, "out": 0, "cr": 0, "cw": 0, "cost": 0}
    cb = {"in": 0, "out": 0, "cr": 0, "cw": 0}
    for m, u in sorted(
        mu.items(), key=lambda x: x[1].get("outputTokens", 0), reverse=True
    ):
        out, pct = u.get("outputTokens", 0), u.get("outputTokens", 0) / tot_out * 100
        pts = m.split("-")
        nm = f"{pts[1]}-{pts[2]}-{pts[3]}" if len(pts) >= 4 else m[:10]
        pk, c = pkey(m), cost(u, pkey(m))
        ct["in"] += u.get("inputTokens", 0)
        ct["out"] += out
        ct["cr"] += u.get("cacheReadInputTokens", 0)
        ct["cw"] += u.get("cacheCreationInputTokens", 0)
        ct["cost"] += c
        if pk and pk in PRICE:
            p = PRICE[pk]
            cb["in"] += u.get("inputTokens", 0) * p[0] / 1e6
            cb["out"] += out * p[1] / 1e6
            cb["cr"] += (
                u.get("cacheReadInputTokens", 0) * (p[0] * CACHE_READ_MULTIPLIER) / 1e6
            )
            cb["cw"] += (
                u.get("cacheCreationInputTokens", 0)
                * (p[0] * CACHE_WRITE_MULTIPLIER)
                / 1e6
            )
        f = int(20 * pct / 100)
        lines.append(
            f"{col(f'{nm:<10}', 'cyan')} {col('█' * f + '░' * (20 - f), 'blue')} {pct:>3.0f}%  {col(f'${c:>6.0f}', 'green')}"
        )
    return lines, ct, cb


def build_cost_breakdown(ct, cb):
    """Build estimated cost section."""
    lines = [col("Estimated Cost", "bold"), col("─" * 42, "dim")]
    rows = [
        ("Input", ct["in"], cb["in"]),
        ("Output", ct["out"], cb["out"]),
        ("Cache Read", ct["cr"], cb["cr"]),
        ("Cache Write", ct["cw"], cb["cw"]),
    ]
    mx = max(r[2] for r in rows) or 1
    for lb, tk, cs in rows:
        bl = int(cs / mx * 16)
        lines.append(
            f"{lb:<12} {tok(tk):>6}  {col('█' * bl + '░' * (16 - bl), 'blue')}  {col(f'${cs:>6.2f}', 'green')}"
        )
    lines.append(col("─" * 42, "dim"))
    total_cost = f"${ct['cost']:>6.2f}"
    lines.append(
        f"{col('Total', 'bold'):<12} {'':>6}  {'':16}  {col(total_cost, 'bold', 'green')}"
    )
    return lines


def build_last7days(da):
    """Build last 7 days activity section."""
    last7 = da[-7:]
    if not last7:
        return []
    lines = [col("Last 7 Days", "bold"), col("─" * 45, "dim")]
    lines.append(f"{'':6}  {'':12} {'msgs':>5}  {'tools':>5}  {'sess':>4}")
    mx = max(d.get("messageCount", 0) for d in last7) or 1
    for d in last7:
        ms, tl, ss = (
            d.get("messageCount", 0),
            d.get("toolCallCount", 0),
            d.get("sessionCount", 0),
        )
        bars = "▇" * (int(ms / mx * 12) if mx else 0)
        lines.append(
            f"{col(fdate(d['date']), 'cyan')}  {col(f'{bars:<12}', 'blue')} {ms:>5}  {tl:>5}  {ss:>4}"
        )
    return lines


def build_daily_tokens(st):
    """Build daily output tokens section."""
    td = st.get("dailyModelTokens", [])[-7:]
    if not td:
        return []
    lines = [col("Daily Output Tokens", "bold"), col("─" * 36, "dim")]
    tots = [(d["date"], sum(d.get("tokensByModel", {}).values())) for d in td]
    mx = max(t[1] for t in tots) or 1
    for dt, tk in tots:
        bars = "▇" * int(tk / mx * 20)
        lines.append(
            f"{col(fdate(dt), 'cyan')}  {col(f'{bars:<20}', 'green')} {tok(tk):>6}"
        )
    return lines


def build_peak_hours(hc):
    """Build peak hours heatmap section."""
    if not hc:
        return []
    lines = [col("Peak Hours", "bold"), col("─" * 45, "dim")]
    ch, cn = "░▁▂▃▄▅▆▇█", [hc.get(str(h), 0) for h in range(24)]
    mx = max(cn) or 1
    # First row: hours 0-11
    lines.append(col("".join(f"{h:>3}" for h in range(12)), "dim"))
    lines.append(
        "".join(col(f"{ch[min(int(cn[h] / mx * 8), 8)]:>3}", "blue") for h in range(12))
    )
    # Second row: hours 12-23
    lines.append(col("".join(f"{h:>3}" for h in range(12, 24)), "dim"))
    lines.append(
        "".join(
            col(f"{ch[min(int(cn[h] / mx * 8), 8)]:>3}", "blue") for h in range(12, 24)
        )
    )
    return lines


def build_records(st, tools):
    """Build records section."""
    lines = [col("Records", "bold"), col("─" * 42, "dim")]
    if ls := st.get("longestSession"):
        lines.append(
            f"{'Longest Session':<18}{col(fdur(ls.get('duration', 0)), 'bold', 'green')}  •  {ls.get('messageCount', 0)} msgs"
        )
    if st.get("totalSessions"):
        lines.append(
            f"{'Avg Msgs/Sess':<18}{col(st['totalMessages'] // st['totalSessions'], 'bold', 'green')}"
        )
    lines.append(f"{'Total Tool Calls':<18}{col(f'{tools:,}', 'bold', 'green')}")
    return lines


def build_projects(width):
    """Build projects section."""
    if not PROJECTS.exists():
        return []
    try:
        pj = json.loads(PROJECTS.read_text()).get("projects", {})
    except (json.JSONDecodeError, OSError):
        return []
    sp = sorted(
        [(p, d) for p, d in pj.items() if d.get("lastCost", 0) > 0],
        key=lambda x: x[1]["lastCost"],
        reverse=True,
    )[:10]
    if not sp:
        return []
    line_width = max(50, width - 4)
    path_max = 38 if width < 100 else 45
    lines = [col("Projects (Last Session)", "bold"), col("─" * line_width, "dim")]
    tot = 0
    for pt, d in sp:
        c = d["lastCost"]
        tot += c
        sh = pt.replace(str(Path.home()), "~")
        sh = "..." + sh[-(path_max - 3) :] if len(sh) > path_max else sh
        la, lr = d.get("lastLinesAdded", 0), d.get("lastLinesRemoved", 0)
        ti, to = (
            tok(d.get("lastTotalInputTokens", 0)),
            tok(d.get("lastTotalOutputTokens", 0)),
        )
        tks = f"{ti:>5} in, {to:>5} out"
        # Fixed-width columns: lines (21 chars visible), duration (9 chars)
        if la or lr:
            ln = f"  {col(f'+{la:>5}', 'green')} {col(f'-{lr:>5}', 'red')} lines"
        else:
            ln = " " * 21  # Reserve space for lines column
        dr = f"  {fdur(d['lastDuration']):>7}" if d.get("lastDuration") else " " * 9
        lines.append(
            f"{col(f'${c:>6.2f}', 'green', 'bold')}  {col(f'{sh:<{path_max}}', 'cyan')}  {col(tks, 'dim')}{ln}{dr}"
        )
    lines.append(col("─" * line_width, "dim"))
    lines.append(f"{col(f'${tot:>6.2f}', 'green', 'bold')}  {col('Total', 'bold')}")
    return lines


def main():
    if not STATS.exists():
        print(col("✗ No local stats found", "magenta"))
        return 1
    try:
        st = json.loads(STATS.read_text())
    except (json.JSONDecodeError, OSError):
        print(col("✗ Failed to read local stats", "magenta"))
        return 1
    cr = creds()
    da = st.get("dailyActivity", [])
    mu = st.get("modelUsage", {})
    tot_out = sum(u.get("outputTokens", 0) for u in mu.values())
    tools = sum(d.get("toolCallCount", 0) for d in da)
    hc = st.get("hourCounts", {})

    width = get_width()
    wide = width >= 100  # Use side-by-side layout if terminal is wide enough

    # Header (full width)
    for line in build_header(st, cr, width):
        print(line)

    # Quick stats
    for line in build_quick_stats(st, da):
        print(line)

    # Usage limits
    limits = build_usage_limits(cr)
    if limits:
        print()
        for line in limits:
            print(line)

    # Model breakdown + Estimated cost
    if tot_out:
        mb_lines, ct, cb = build_model_breakdown(mu, tot_out)
        cost_lines = build_cost_breakdown(ct, cb)
        print()
        if wide:
            for line in merge_columns(mb_lines, cost_lines):
                print("  " + line)
        else:
            for line in mb_lines:
                print("  " + line)
            print()
            for line in cost_lines:
                print("  " + line)

    # Last 7 days + Daily output tokens
    last7_lines = build_last7days(da)
    daily_tokens_lines = build_daily_tokens(st)
    if last7_lines or daily_tokens_lines:
        print()
        if wide and last7_lines and daily_tokens_lines:
            for line in merge_columns(last7_lines, daily_tokens_lines):
                print("  " + line)
        else:
            if last7_lines:
                for line in last7_lines:
                    print("  " + line)
            if daily_tokens_lines:
                print()
                for line in daily_tokens_lines:
                    print("  " + line)

    # Peak hours + Records
    peak_lines = build_peak_hours(hc)
    records_lines = build_records(st, tools)
    if peak_lines or records_lines:
        print()
        if wide and peak_lines and records_lines:
            for line in merge_columns(peak_lines, records_lines):
                print("  " + line)
        else:
            if peak_lines:
                for line in peak_lines:
                    print("  " + line)
            if records_lines:
                print()
                for line in records_lines:
                    print("  " + line)

    # Projects (full width)
    proj_lines = build_projects(width)
    if proj_lines:
        print()
        for line in proj_lines:
            print("  " + line)

    print()
    return 0


if __name__ == "__main__":
    exit(main())
