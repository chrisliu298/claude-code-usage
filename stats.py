#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Claude Code usage statistics."""

import json, subprocess, sys
from datetime import datetime
from pathlib import Path
import httpx

STATS = Path.home() / ".claude" / "stats-cache.json"
PROJECTS = Path.home() / ".claude.json"
C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m", "red": "\033[31m",
     "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m"}
# Pricing per MTok: [input, output, cache_read, cache_write]
PRICE = {"opus-4-5": [5, 25, 0.5, 6.25], "opus-4": [15, 75, 1.5, 18.75], "sonnet-4-5": [3, 15, 0.3, 3.75],
         "sonnet-4": [3, 15, 0.3, 3.75], "haiku-4-5": [1, 5, 0.1, 1.25], "haiku-3-5": [0.8, 4, 0.08, 1.0]}

def col(t, *s): return "".join(C.get(x, "") for x in s) + str(t) + C["reset"]
def tok(n): return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.0f}K" if n >= 1e3 else str(n)
def fdate(s):
    try: return datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%b %d")
    except: return s[-5:] if len(s) >= 5 else s
def freset(s):
    try:
        secs = (datetime.fromisoformat(s.replace("Z", "+00:00")) - datetime.now(datetime.fromisoformat(s.replace("Z", "+00:00")).tzinfo)).total_seconds()
        if secs <= 0: return "now"
        h, m = divmod(int(secs) // 60, 60); d, h = divmod(h, 24)
        return f"{d}d {h}h" if d else f"{h}h {m}m" if h else f"{m}m"
    except: return "?"
def fdur(ms): h, m = divmod(ms // 60000, 60); return f"{h}h {m}m" if h else f"{m}m"
def sec(t): print(f"\n  {col(t, 'bold')}\n  {col('─' * 50, 'dim')}")
def pkey(m):
    m = m.lower()
    for k in ["opus-4-5", "opus-4", "sonnet-4-5", "sonnet-4", "haiku-4-5", "haiku-3-5"]:
        if k.replace("-", "-") in m or k.replace("-4-5", "-4.5").replace("-3-5", "-3.5") in m: return k
    return None
def cost(u, pk):
    if not pk or pk not in PRICE: return 0
    p = PRICE[pk]
    return (u.get("inputTokens", 0) * p[0] + u.get("outputTokens", 0) * p[1] +
            u.get("cacheReadInputTokens", 0) * p[2] + u.get("cacheCreationInputTokens", 0) * p[3]) / 1e6

def creds():
    if sys.platform != "darwin": return None
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials"], capture_output=True, text=True)
        if r.returncode != 0: return None
        acct = next((l.split("=")[1].strip().strip('"') for l in r.stdout.split("\n") if '"acct"<blob>=' in l), None)
        if not acct: return None
        r = subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials", "-a", acct, "-w"], capture_output=True, text=True)
        return json.loads(r.stdout.strip()).get("claudeAiOauth") if r.returncode == 0 else None
    except: return None

def main():
    if not STATS.exists(): print(col("✗ No local stats found", "magenta")); return 1
    st, cr, da = json.loads(STATS.read_text()), creds(), []
    da = st.get("dailyActivity", [])

    # Header
    tier = ""
    if cr:
        t = cr.get("rateLimitTier", "")
        tier = "Max 5x" if "max_5x" in t else "Max 20x" if "max_20x" in t else "Pro" if "pro" in t.lower() else ""
    sub = " • ".join(filter(None, [tier, f"Since {fdate(st.get('firstSessionDate', ''))}" if st.get("firstSessionDate") else ""]))
    print(f"{col('╭' + '─' * 50 + '╮', 'dim')}\n{col('│', 'dim')}  {col('CLAUDE CODE USAGE', 'bold', 'cyan')}{' ' * 31}{col('│', 'dim')}")
    if sub: print(f"{col('│', 'dim')}  {col(sub, 'dim')}{' ' * (48 - len(sub))}{col('│', 'dim')}")
    print(f"{col('╰' + '─' * 50 + '╯', 'dim')}")

    # Quick stats
    tools = sum(d.get("toolCallCount", 0) for d in da)
    print(f"\n  {col(f'{st.get('totalSessions', 0):,}', 'bold')} sessions  {col('│', 'dim')}  "
          f"{col(f'{st.get('totalMessages', 0):,}', 'bold')} messages  {col('│', 'dim')}  "
          f"{col(f'{tools:,}', 'bold')} tools  {col('│', 'dim')}  {col(len(da), 'bold')} days")

    # Usage limits
    if cr:
        try:
            data = httpx.get("https://api.anthropic.com/api/oauth/usage", headers={"Authorization": f"Bearer {cr['accessToken']}", "anthropic-beta": "oauth-2025-04-20"}, timeout=30).json()
            print()
            for nm, ky in [("5-Hour", "five_hour"), ("7-Day", "seven_day")]:
                if (d := data.get(ky)) and (u := d.get("utilization")) is not None:
                    f = int(25 * u / 100); bc = "green" if u < 50 else "yellow" if u < 80 else "magenta"
                    print(f"  {col(f'{nm:<8}', 'bold')}{col('━' * f, bc)}{col('━' * (25 - f), 'dim')} {u:>3.0f}%  {col('resets', 'dim')} {col(freset(d.get('resets_at', '')), 'cyan')}")
        except: pass

    # Model breakdown + cost
    mu = st.get("modelUsage", {})
    tot_out = sum(u.get("outputTokens", 0) for u in mu.values())
    if tot_out:
        sec("Model Breakdown")
        ct = {"in": 0, "out": 0, "cr": 0, "cw": 0, "cost": 0}
        cb = {"in": 0, "out": 0, "cr": 0, "cw": 0}
        for m, u in sorted(mu.items(), key=lambda x: x[1].get("outputTokens", 0), reverse=True):
            out, pct = u.get("outputTokens", 0), u.get("outputTokens", 0) / tot_out * 100
            pts = m.split("-"); nm = f"{pts[1]}-{pts[2]}" if len(pts) >= 3 else m[:10]
            pk, c = pkey(m), cost(u, pkey(m))
            ct["in"] += u.get("inputTokens", 0); ct["out"] += out
            ct["cr"] += u.get("cacheReadInputTokens", 0); ct["cw"] += u.get("cacheCreationInputTokens", 0); ct["cost"] += c
            if pk and pk in PRICE:
                p = PRICE[pk]
                cb["in"] += u.get("inputTokens", 0) * p[0] / 1e6; cb["out"] += out * p[1] / 1e6
                cb["cr"] += u.get("cacheReadInputTokens", 0) * p[2] / 1e6; cb["cw"] += u.get("cacheCreationInputTokens", 0) * p[3] / 1e6
            f = int(20 * pct / 100)
            print(f"  {col(f'{nm:<10}', 'cyan')}{col('█' * f + '░' * (20 - f), 'blue')} {pct:>3.0f}%  {tok(out):>5} out  {col(f'${c:>7.2f}', 'green')}")

        # Cost breakdown
        sec("Estimated Cost (Claude Code only)")
        rows = [("Input", ct["in"], cb["in"]), ("Output", ct["out"], cb["out"]), ("Cache Read", ct["cr"], cb["cr"]), ("Cache Write", ct["cw"], cb["cw"])]
        mx = max(r[2] for r in rows) or 1
        print(f"  {'Type':<14} {'Tokens':>10}  {'':18}  {'Cost':>8}")
        print(f"  {col('─' * 54, 'dim')}")
        for lb, tk, cs in rows:
            bl = int(cs / mx * 18); print(f"  {lb:<14} {tok(tk):>10}  {col('█' * bl + '░' * (18 - bl), 'blue')}  {col(f'${cs:>7.2f}', 'green')}")
        print(f"  {col('─' * 54, 'dim')}\n  {col('Total', 'bold'):<14} {' ':>10}  {' ':18}  {col(f'${ct['cost']:>7.2f}', 'bold', 'green')}")

    # Daily activity
    last7 = da[-7:]
    if last7:
        print(f"\n  {col('Last 7 Days', 'bold')}\n  {col('─' * 50, 'dim')}\n  {'':6}  {'':12} {'msgs':>5}  {'tools':>5}  {'sess':>4}")
        mx = max(d.get("messageCount", 0) for d in last7)
        for d in last7:
            ms, tl, ss = d.get("messageCount", 0), d.get("toolCallCount", 0), d.get("sessionCount", 0)
            print(f"  {col(fdate(d['date']), 'cyan')}  {col(f'{'▇' * (int(ms / mx * 12) if mx else 0):<12}', 'blue')} {ms:>5}  {tl:>5}  {ss:>4}")

    # Hour heatmap
    hc = st.get("hourCounts", {})
    if hc:
        sec("Peak Hours (sessions started)")
        print(f"  {col(''.join(f'{h:>3}' for h in range(24)), 'dim')}")
        ch, cn = "░▁▂▃▄▅▆▇█", [hc.get(str(h), 0) for h in range(24)]; mx = max(cn) or 1
        print(f"  {''.join(col(f'{ch[min(int(c/mx*8), 8)]:>3}', 'blue') for c in cn)}")

    # Records
    sec("Records")
    if ls := st.get("longestSession"):
        print(f"  {'Longest Session':<20}{col(fdur(ls.get('duration', 0)), 'bold', 'green')}  •  {ls.get('messageCount', 0)} msgs  •  {fdate(ls.get('timestamp', ''))}")
    if st.get("totalSessions"): print(f"  {'Avg Messages/Sess':<20}{col(st['totalMessages'] // st['totalSessions'], 'bold', 'green')}")
    print(f"  {'Total Tool Calls':<20}{col(f'{tools:,}', 'bold', 'green')}")

    # Token trends
    td = st.get("dailyModelTokens", [])[-7:]
    if td:
        sec("Daily Output Tokens (last 7 days)")
        tots = [(d["date"], sum(d.get("tokensByModel", {}).values())) for d in td]; mx = max(t[1] for t in tots) or 1
        for dt, tk in tots: print(f"  {col(fdate(dt), 'cyan')}  {col(f'{'▇' * int(tk / mx * 20):<20}', 'green')} {tok(tk):>6}")

    # Projects
    if PROJECTS.exists():
        pj = json.loads(PROJECTS.read_text()).get("projects", {})
        sp = sorted([(p, d) for p, d in pj.items() if d.get("lastCost", 0) > 0], key=lambda x: x[1]["lastCost"], reverse=True)[:10]
        if sp:
            sec("Projects (Last Session)")
            tot = 0
            for pt, d in sp:
                c = d["lastCost"]; tot += c
                sh = pt.replace(str(Path.home()), "~"); sh = "..." + sh[-39:] if len(sh) > 42 else sh
                la, lr = d.get("lastLinesAdded", 0), d.get("lastLinesRemoved", 0)
                ln = f"  •  {col(f'+{la}', 'green')} {col(f'-{lr}', 'red')} lines" if la or lr else ""
                dr = f"  •  {fdur(d['lastDuration'])}" if d.get("lastDuration") else ""
                cs = f"${c:.2f}"; tks = f"{tok(d.get('lastTotalInputTokens', 0))} in, {tok(d.get('lastTotalOutputTokens', 0))} out"
                print(f"  {col(f'{cs:>7}', 'green', 'bold')}  {col(sh, 'cyan')}\n  {'':7}  {col(tks, 'dim')}{ln}{dr}")
            ts = f"${tot:.2f}"; print(f"  {col('─' * 50, 'dim')}\n  {col(f'{ts:>7}', 'green', 'bold')}  {col('Total', 'bold')}")
    print(); return 0

if __name__ == "__main__": exit(main())
