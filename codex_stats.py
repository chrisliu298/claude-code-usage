#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# ///
"""Codex CLI usage statistics."""

import json, os, re, shutil, sys
from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


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
DATE_SUFFIX_RE = re.compile(r"^(?P<base>.+)-\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_usd_per_mtok: float
    cached_input_usd_per_mtok: float | None
    output_usd_per_mtok: float


# Prices are USD per 1M text tokens. Cached input tokens are a subset of input tokens and can be
# billed at a different rate (if supported by the model).
#
# Sources (as of 2026-01-26):
# - OpenAI API Pricing: https://openai.com/api/pricing
# - GPT-5.2 model page: https://platform.openai.com/docs/models/gpt-5.2
MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-5.2": ModelPricing(input_usd_per_mtok=1.75, cached_input_usd_per_mtok=0.175, output_usd_per_mtok=14.0),
    "gpt-5.2-pro": ModelPricing(input_usd_per_mtok=21.0, cached_input_usd_per_mtok=None, output_usd_per_mtok=168.0),
    "gpt-5.1": ModelPricing(input_usd_per_mtok=1.25, cached_input_usd_per_mtok=0.125, output_usd_per_mtok=10.0),
    "gpt-5.1-mini": ModelPricing(input_usd_per_mtok=0.25, cached_input_usd_per_mtok=0.025, output_usd_per_mtok=2.0),
    "gpt-5.1-nano": ModelPricing(input_usd_per_mtok=0.05, cached_input_usd_per_mtok=0.005, output_usd_per_mtok=0.4),
    "gpt-5": ModelPricing(input_usd_per_mtok=1.25, cached_input_usd_per_mtok=0.125, output_usd_per_mtok=10.0),
    "gpt-5-pro": ModelPricing(input_usd_per_mtok=15.0, cached_input_usd_per_mtok=None, output_usd_per_mtok=120.0),
    "gpt-5-mini": ModelPricing(input_usd_per_mtok=0.25, cached_input_usd_per_mtok=0.025, output_usd_per_mtok=2.0),
    "gpt-5-nano": ModelPricing(input_usd_per_mtok=0.05, cached_input_usd_per_mtok=0.005, output_usd_per_mtok=0.4),
    "gpt-4.1": ModelPricing(input_usd_per_mtok=2.0, cached_input_usd_per_mtok=0.5, output_usd_per_mtok=8.0),
    "gpt-4.1-mini": ModelPricing(input_usd_per_mtok=0.4, cached_input_usd_per_mtok=0.1, output_usd_per_mtok=1.6),
    "gpt-4.1-nano": ModelPricing(input_usd_per_mtok=0.1, cached_input_usd_per_mtok=0.025, output_usd_per_mtok=0.4),
    "gpt-4o": ModelPricing(input_usd_per_mtok=2.5, cached_input_usd_per_mtok=1.25, output_usd_per_mtok=10.0),
    "gpt-4o-mini": ModelPricing(input_usd_per_mtok=0.15, cached_input_usd_per_mtok=0.075, output_usd_per_mtok=0.6),
    "o1": ModelPricing(input_usd_per_mtok=15.0, cached_input_usd_per_mtok=7.5, output_usd_per_mtok=60.0),
    "o1-mini": ModelPricing(input_usd_per_mtok=1.1, cached_input_usd_per_mtok=0.55, output_usd_per_mtok=4.4),
    "o3": ModelPricing(input_usd_per_mtok=2.0, cached_input_usd_per_mtok=0.5, output_usd_per_mtok=8.0),
    "o3-mini": ModelPricing(input_usd_per_mtok=1.1, cached_input_usd_per_mtok=0.55, output_usd_per_mtok=4.4),
    "o3-pro": ModelPricing(input_usd_per_mtok=20.0, cached_input_usd_per_mtok=None, output_usd_per_mtok=80.0),
}


def col(text: object, *styles: str, enabled: bool = True) -> str:
    if not enabled:
        return str(text)
    return "".join(ANSI.get(s, "") for s in styles) + str(text) + ANSI["reset"]


def get_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def visible_len(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def shorten_plain(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return s[:max_len]
    return s[: max_len - 1] + "…"


def tok(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 10_000:
        return f"{n/1e3:.0f}K"
    if n >= 1_000:
        return f"{n/1e3:.1f}K"
    return str(n)


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def to_local(dt: datetime) -> datetime:
    # Convert an aware datetime into local time.
    return dt.astimezone()


def local_day(dt: datetime) -> date:
    return to_local(dt).date()


def fmt_day(d: date) -> str:
    return d.strftime("%b %d")


def fmt_reset(resets_at: int | float | None) -> str:
    if resets_at is None:
        return "?"
    try:
        delta = datetime.fromtimestamp(float(resets_at), tz=UTC) - datetime.now(tz=UTC)
    except (ValueError, OSError):
        return "?"
    secs = int(delta.total_seconds())
    if secs <= 0:
        return "now"
    mins, _ = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    if days:
        return f"{days}d {hrs}h"
    if hrs:
        return f"{hrs}h {mins}m"
    return f"{mins}m"


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_total_usage(cls, d: dict[str, Any] | None) -> "TokenUsage":
        if not d:
            return cls()
        return cls(
            input_tokens=int(d.get("input_tokens", 0) or 0),
            cached_input_tokens=int(d.get("cached_input_tokens", 0) or 0),
            output_tokens=int(d.get("output_tokens", 0) or 0),
            reasoning_output_tokens=int(d.get("reasoning_output_tokens", 0) or 0),
            total_tokens=int(d.get("total_tokens", 0) or 0),
        )

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self.total_tokens += other.total_tokens

    def to_json(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class CostSummary:
    non_cached_input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0

    input_usd: float = 0.0
    cached_input_usd: float = 0.0
    output_usd: float = 0.0

    cost_by_model: dict[str, float] = None  # type: ignore[assignment]
    unpriced_models: set[str] = None  # type: ignore[assignment]
    unpriced_tokens: int = 0

    def __post_init__(self) -> None:
        if self.cost_by_model is None:
            self.cost_by_model = {}
        if self.unpriced_models is None:
            self.unpriced_models = set()

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.cached_input_usd + self.output_usd

    def to_json(self) -> dict[str, Any]:
        return {
            "total_usd": self.total_usd,
            "breakdown_usd": {
                "input_usd": self.input_usd,
                "cached_input_usd": self.cached_input_usd,
                "output_usd": self.output_usd,
            },
            "tokens_priced": {
                "non_cached_input_tokens": self.non_cached_input_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "output_tokens": self.output_tokens,
            },
            "cost_by_model_usd": {m: float(c) for m, c in sorted(self.cost_by_model.items())},
            "unpriced_models": sorted(self.unpriced_models),
            "unpriced_tokens": int(self.unpriced_tokens),
        }


def normalize_model_name(model: str) -> str:
    m = str(model or "").strip().lower()
    if ":" in m:
        m = m.split(":", 1)[0]

    # Common suffixes.
    for suffix in ("-latest", "-preview"):
        if m.endswith(suffix):
            m = m[: -len(suffix)]

    # Versioned models like gpt-4.1-2025-04-14.
    if match := DATE_SUFFIX_RE.match(m):
        m = match.group("base")

    # Codex-tuned aliases often follow "<base>-codex" (or "-codex-max").
    for suffix in ("-codex-mini", "-codex-nano"):
        if m.endswith(suffix):
            m = m[: -len(suffix)] + suffix.replace("-codex", "")
            break
    for suffix in ("-codex-max", "-codex"):
        if m.endswith(suffix):
            m = m[: -len(suffix)]
            break

    return m


def pricing_for_model(model: str) -> ModelPricing | None:
    base = normalize_model_name(model)
    p = MODEL_PRICING.get(base)
    if p is not None:
        return p
    # Fallback: try prefix match for versioned variants we didn't normalize (e.g. "-2026-01-01").
    for key, pricing in MODEL_PRICING.items():
        if base.startswith(key + "-"):
            return pricing
    return None


def estimate_costs(tokens_by_model: dict[str, TokenUsage]) -> CostSummary:
    costs = CostSummary()
    for model, usage in tokens_by_model.items():
        pricing = pricing_for_model(model)
        if pricing is None:
            costs.unpriced_models.add(str(model))
            costs.unpriced_tokens += int(usage.total_tokens)
            continue

        cached = max(0, min(int(usage.cached_input_tokens), int(usage.input_tokens)))
        non_cached_in = max(0, int(usage.input_tokens) - cached)
        out = max(0, int(usage.output_tokens))

        input_cost = non_cached_in * pricing.input_usd_per_mtok / 1_000_000
        cached_rate = (
            pricing.input_usd_per_mtok
            if pricing.cached_input_usd_per_mtok is None
            else pricing.cached_input_usd_per_mtok
        )
        cached_cost = cached * cached_rate / 1_000_000
        output_cost = out * pricing.output_usd_per_mtok / 1_000_000

        costs.non_cached_input_tokens += non_cached_in
        costs.cached_input_tokens += cached
        costs.output_tokens += out

        costs.input_usd += input_cost
        costs.cached_input_usd += cached_cost
        costs.output_usd += output_cost

        costs.cost_by_model[str(model)] = costs.cost_by_model.get(str(model), 0.0) + (
            input_cost + cached_cost + output_cost
        )
    return costs


@dataclass(slots=True)
class SessionSummary:
    session_id: str
    started_at: datetime | None
    ended_at: datetime | None
    cwd: str | None
    repo_url: str | None
    branch: str | None
    cli_version: str | None
    turns: int
    tool_calls: int
    user_messages: int
    assistant_messages: int
    tokens: TokenUsage
    tokens_by_model: dict[str, TokenUsage]


@dataclass(slots=True)
class Aggregates:
    earliest: datetime | None = None
    latest: datetime | None = None
    sessions: list[SessionSummary] = None  # type: ignore[assignment]
    totals: TokenUsage = None  # type: ignore[assignment]
    tokens_by_day: dict[date, TokenUsage] = None  # type: ignore[assignment]
    tokens_by_model: dict[str, TokenUsage] = None  # type: ignore[assignment]
    tokens_by_hour: dict[int, int] = None  # type: ignore[assignment]
    messages_by_hour: dict[int, int] = None  # type: ignore[assignment]
    tool_calls_by_day: dict[date, int] = None  # type: ignore[assignment]
    turns_by_day: dict[date, int] = None  # type: ignore[assignment]
    user_messages_by_day: dict[date, int] = None  # type: ignore[assignment]
    assistant_messages_by_day: dict[date, int] = None  # type: ignore[assignment]
    sessions_by_day: dict[date, int] = None  # type: ignore[assignment]
    tool_calls_by_name: dict[str, int] = None  # type: ignore[assignment]
    latest_rate_limits_at: datetime | None = None
    latest_rate_limits: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.sessions is None:
            self.sessions = []
        if self.totals is None:
            self.totals = TokenUsage()
        if self.tokens_by_day is None:
            self.tokens_by_day = {}
        if self.tokens_by_model is None:
            self.tokens_by_model = {}
        if self.tokens_by_hour is None:
            self.tokens_by_hour = {h: 0 for h in range(24)}
        if self.messages_by_hour is None:
            self.messages_by_hour = {h: 0 for h in range(24)}
        if self.tool_calls_by_day is None:
            self.tool_calls_by_day = {}
        if self.turns_by_day is None:
            self.turns_by_day = {}
        if self.user_messages_by_day is None:
            self.user_messages_by_day = {}
        if self.assistant_messages_by_day is None:
            self.assistant_messages_by_day = {}
        if self.sessions_by_day is None:
            self.sessions_by_day = {}
        if self.tool_calls_by_name is None:
            self.tool_calls_by_name = {}


def ensure_usage_bucket(mapping: dict[str, TokenUsage], key: str) -> TokenUsage:
    bucket = mapping.get(key)
    if bucket is None:
        bucket = TokenUsage()
        mapping[key] = bucket
    return bucket


def ensure_day_bucket(mapping: dict[date, TokenUsage], key: date) -> TokenUsage:
    bucket = mapping.get(key)
    if bucket is None:
        bucket = TokenUsage()
        mapping[key] = bucket
    return bucket


def iter_session_files(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.exists():
        return []
    files = [p for p in sessions_dir.rglob("*.jsonl") if p.is_file()]
    files.sort()
    return files


def session_project_key(meta: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    cwd = meta.get("cwd")
    git = meta.get("git") or {}
    repo_url = git.get("repository_url")
    branch = git.get("branch")
    return cwd, repo_url, branch


def parse_session_file(path: Path, agg: Aggregates, *, since: datetime | None) -> None:
    meta: dict[str, Any] = {}
    started_at: datetime | None = None
    ended_at: datetime | None = None

    contexts: list[tuple[datetime, str]] = []
    token_snapshots: list[tuple[datetime, TokenUsage]] = []

    turns = 0
    tool_calls = 0
    user_messages = 0
    assistant_messages = 0
    active_days: set[date] = set()

    def track_session_time(dt: datetime) -> None:
        nonlocal started_at, ended_at
        if started_at is None or dt < started_at:
            started_at = dt
        if ended_at is None or dt > ended_at:
            ended_at = dt

    def track_agg_time(dt: datetime) -> None:
        if agg.earliest is None or dt < agg.earliest:
            agg.earliest = dt
        if agg.latest is None or dt > agg.latest:
            agg.latest = dt

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dt = parse_ts(obj.get("timestamp"))
                if dt is None:
                    continue
                in_range = since is None or dt >= since
                track_session_time(dt)
                if in_range:
                    track_agg_time(dt)

                typ = obj.get("type")
                payload = obj.get("payload") or {}

                if typ == "session_meta":
                    meta = payload
                    continue

                if typ == "turn_context":
                    model = payload.get("model") or "unknown"
                    contexts.append((dt, str(model)))
                    if in_range:
                        turns += 1
                        d = local_day(dt)
                        active_days.add(d)
                        agg.turns_by_day[d] = agg.turns_by_day.get(d, 0) + 1
                    continue

                if typ == "event_msg" and payload.get("type") == "token_count":
                    rl = payload.get("rate_limits")
                    if rl and (
                        agg.latest_rate_limits_at is None or dt > agg.latest_rate_limits_at
                    ):
                        agg.latest_rate_limits_at = dt
                        agg.latest_rate_limits = rl
                    if in_range:
                        active_days.add(local_day(dt))

                    info = payload.get("info")
                    if not info:
                        continue
                    total = info.get("total_token_usage")
                    if not isinstance(total, dict):
                        continue
                    totals = TokenUsage.from_total_usage(total)
                    token_snapshots.append((dt, totals))
                    continue

                if typ == "response_item":
                    item_type = payload.get("type")
                    if item_type == "function_call":
                        if in_range:
                            tool_calls += 1
                            d = local_day(dt)
                            active_days.add(d)
                            agg.tool_calls_by_day[d] = agg.tool_calls_by_day.get(d, 0) + 1
                            name = payload.get("name")
                            if isinstance(name, str) and name:
                                agg.tool_calls_by_name[name] = agg.tool_calls_by_name.get(name, 0) + 1
                        continue
                    if item_type == "message":
                        if in_range:
                            role = payload.get("role")
                            d = local_day(dt)
                            active_days.add(d)
                            if role == "user":
                                user_messages += 1
                                agg.user_messages_by_day[d] = agg.user_messages_by_day.get(d, 0) + 1
                                agg.messages_by_hour[to_local(dt).hour] += 1
                            elif role == "assistant":
                                assistant_messages += 1
                                agg.assistant_messages_by_day[d] = agg.assistant_messages_by_day.get(d, 0) + 1
                                agg.messages_by_hour[to_local(dt).hour] += 1
                        continue
    except OSError:
        return

    if since is not None and not active_days:
        return

    for d in active_days:
        agg.sessions_by_day[d] = agg.sessions_by_day.get(d, 0) + 1

    if not token_snapshots:
        # Still record session metadata/tool usage, but skip token aggregates.
        cwd, repo_url, branch = session_project_key(meta)
        agg.sessions.append(
            SessionSummary(
                session_id=str(meta.get("id") or path.stem),
                started_at=started_at,
                ended_at=ended_at,
                cwd=cwd,
                repo_url=repo_url,
                branch=branch,
                cli_version=meta.get("cli_version"),
                turns=turns,
                tool_calls=tool_calls,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                tokens=TokenUsage(),
                tokens_by_model={},
            )
        )
        return

    contexts.sort(key=lambda x: x[0])
    ctx_times = [c[0] for c in contexts]
    ctx_models = [c[1] for c in contexts]

    token_snapshots.sort(key=lambda x: x[0])
    prev = TokenUsage()

    session_tokens = TokenUsage()
    session_tokens_by_model: dict[str, TokenUsage] = {}

    def model_for(dt: datetime) -> str:
        if not ctx_times:
            return "unknown"
        i = bisect_right(ctx_times, dt) - 1
        if i < 0:
            return "unknown"
        return ctx_models[i] or "unknown"

    for dt, totals in token_snapshots:
        delta = TokenUsage(
            input_tokens=max(0, totals.input_tokens - prev.input_tokens),
            cached_input_tokens=max(0, totals.cached_input_tokens - prev.cached_input_tokens),
            output_tokens=max(0, totals.output_tokens - prev.output_tokens),
            reasoning_output_tokens=max(0, totals.reasoning_output_tokens - prev.reasoning_output_tokens),
            total_tokens=max(0, totals.total_tokens - prev.total_tokens),
        )

        # If the cumulative counters reset within a session, treat the new totals as the delta.
        if totals.input_tokens < prev.input_tokens:
            delta.input_tokens = totals.input_tokens
        if totals.cached_input_tokens < prev.cached_input_tokens:
            delta.cached_input_tokens = totals.cached_input_tokens
        if totals.output_tokens < prev.output_tokens:
            delta.output_tokens = totals.output_tokens
        if totals.reasoning_output_tokens < prev.reasoning_output_tokens:
            delta.reasoning_output_tokens = totals.reasoning_output_tokens
        if totals.total_tokens < prev.total_tokens:
            delta.total_tokens = totals.total_tokens

        prev = totals

        if since is not None and dt < since:
            continue

        m = model_for(dt)
        ensure_usage_bucket(session_tokens_by_model, m).add(delta)
        session_tokens.add(delta)

        ensure_usage_bucket(agg.tokens_by_model, m).add(delta)
        ensure_day_bucket(agg.tokens_by_day, local_day(dt)).add(delta)
        agg.totals.add(delta)
        agg.tokens_by_hour[to_local(dt).hour] += delta.total_tokens

    cwd, repo_url, branch = session_project_key(meta)
    agg.sessions.append(
        SessionSummary(
            session_id=str(meta.get("id") or path.stem),
            started_at=started_at,
            ended_at=ended_at,
            cwd=cwd,
            repo_url=repo_url,
            branch=branch,
            cli_version=meta.get("cli_version"),
            turns=turns,
            tool_calls=tool_calls,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            tokens=session_tokens,
            tokens_by_model=session_tokens_by_model,
        )
    )


def build_header(agg: Aggregates, *, width: int, color: bool) -> list[str]:
    inner = width - 4
    title = col("CODEX USAGE", "bold", "cyan", enabled=color)
    title_len = len("CODEX USAGE")

    tier = ""
    if isinstance(agg.latest_rate_limits, dict):
        plan_type = agg.latest_rate_limits.get("plan_type")
        if isinstance(plan_type, str) and plan_type.strip():
            tier = plan_type.strip()

    since = ""
    if agg.earliest is not None:
        since = f"Since {to_local(agg.earliest).strftime('%b %d')}"

    turns = sum(s.turns for s in agg.sessions)
    token_detail = f"{tok(agg.totals.total_tokens)} tokens"
    token_bits = [
        f"{tok(agg.totals.input_tokens)} in",
        f"{tok(agg.totals.output_tokens)} out",
    ]
    if agg.totals.cached_input_tokens:
        token_bits.append(f"{tok(agg.totals.cached_input_tokens)} cached")
    if agg.totals.reasoning_output_tokens:
        token_bits.append(f"{tok(agg.totals.reasoning_output_tokens)} reasoning")
    token_detail += " (" + ", ".join(token_bits) + ")"
    turns_s = f"{turns:,} turns"

    sub = " • ".join(x for x in (tier, since, token_detail, turns_s) if x)
    sub = shorten_plain(sub, inner)

    lines = [col("╭" + "─" * (width - 2) + "╮", "dim", enabled=color)]
    lines.append(
        f"{col('│', 'dim', enabled=color)}  {title}{' ' * (inner - title_len)}{col('│', 'dim', enabled=color)}"
    )
    if sub:
        lines.append(
            f"{col('│', 'dim', enabled=color)}  {col(sub, 'dim', enabled=color)}{' ' * (inner - len(sub))}{col('│', 'dim', enabled=color)}"
        )
    lines.append(col("╰" + "─" * (width - 2) + "╯", "dim", enabled=color))
    return lines


def build_quick_stats(agg: Aggregates, *, color: bool) -> list[str]:
    sessions = len(agg.sessions)
    messages = sum(s.user_messages + s.assistant_messages for s in agg.sessions)
    tool_calls = sum(s.tool_calls for s in agg.sessions)
    days = len(
        set(agg.tool_calls_by_day)
        | set(agg.user_messages_by_day)
        | set(agg.assistant_messages_by_day)
        | set(agg.sessions_by_day)
    )
    day_label = "day" if days == 1 else "days"
    return [
        f"  {col(f'{sessions:,}', 'bold', enabled=color)} sessions  {col('│', 'dim', enabled=color)}  "
        f"{col(f'{messages:,}', 'bold', enabled=color)} messages  {col('│', 'dim', enabled=color)}  "
        f"{col(f'{tool_calls:,}', 'bold', enabled=color)} tools  {col('│', 'dim', enabled=color)}  "
        f"{col(f'{days:,}', 'bold', enabled=color)} {day_label}"
    ]


def bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    return "█" * filled + "░" * (width - filled)


def build_usage_limits(agg: Aggregates, *, color: bool) -> list[str]:
    rl = agg.latest_rate_limits or {}
    primary = rl.get("primary") or {}
    secondary = rl.get("secondary") or {}

    def limit_bar(pct: float, width: int = 25) -> str:
        pct = max(0.0, min(100.0, pct))
        filled = int(width * pct / 100.0)
        if not color:
            return "━" * width
        return col("━" * filled, "green", enabled=True) + col(
            "━" * (width - filled), "dim", enabled=True
        )

    def line(label: str, window: dict[str, Any]) -> str | None:
        try:
            pct = float(window.get("used_percent", 0.0) or 0.0)
        except (TypeError, ValueError):
            pct = 0.0
        wm = window.get("window_minutes")
        resets_at = window.get("resets_at")
        if wm is None and resets_at is None and pct == 0.0:
            return None
        reset_s = fmt_reset(resets_at)
        return (
            f"{col(f'{label:<6}', 'cyan', enabled=color)}  "
            f"{limit_bar(pct, width=25)}  "
            f"{pct:>3.0f}%  {col('resets', 'dim', enabled=color)} {col(reset_s, 'cyan', enabled=color)}"
        )

    lines: list[str] = []
    l1 = line("5-Hour", primary)
    l2 = line("7-Day", secondary)
    if l1:
        lines.append(l1)
    if l2:
        lines.append(l2)
    return lines


def build_model_breakdown(agg: Aggregates, *, width: int, color: bool) -> list[str]:
    items = list(agg.tokens_by_model.items())
    if not items:
        return []
    costs = estimate_costs(agg.tokens_by_model)
    panel_width = min(45, max(24, width - 4))
    lines = [col("Model Breakdown", "bold", enabled=color), col("─" * panel_width, "dim", enabled=color)]
    tot_out = sum(u.output_tokens for _, u in items)
    denom = tot_out or (agg.totals.total_tokens or 1)
    key_fn = (lambda kv: kv[1].output_tokens) if tot_out else (lambda kv: kv[1].total_tokens)
    items.sort(key=key_fn, reverse=True)

    for model, usage in items[:12]:
        used = usage.output_tokens if tot_out else usage.total_tokens
        pct = used / denom * 100.0
        name = shorten_plain(str(model), 10)
        cost = costs.cost_by_model.get(str(model))
        cost_str = f"${cost:>6.0f}" if cost is not None else f"${'?':>6}"
        lines.append(
            f"{col(f'{name:<10}', 'cyan', enabled=color)}"
            f"{col(bar(pct, width=20), 'blue', enabled=color)} "
            f"{pct:>3.0f}%  {col(cost_str, 'green', enabled=color) if cost is not None else col(cost_str, 'dim', enabled=color)}"
        )
    return lines


def build_token_breakdown(agg: Aggregates, *, width: int, color: bool) -> list[str]:
    panel_width = min(45, max(24, width - 4))
    total = agg.totals
    denom = total.total_tokens or 1

    lines = [
        col("Token Breakdown", "bold", enabled=color),
        col("─" * panel_width, "dim", enabled=color),
    ]

    rows: list[tuple[str, int]] = [
        ("Input", total.input_tokens),
        ("Cached Input", total.cached_input_tokens),
        ("Output", total.output_tokens),
    ]
    if total.reasoning_output_tokens:
        rows.append(("Reasoning Out", total.reasoning_output_tokens))

    label_w = 13
    for label, value in rows:
        pct = value / denom * 100.0
        label_s = f"{shorten_plain(label, label_w):<{label_w}}"
        value_s = f"{tok(value):>6}"
        lines.append(
            f"{col(label_s, 'cyan', enabled=color)} "
            f"{col(value_s, 'bold', 'green', enabled=color)}  "
            f"{col(bar(pct, width=16), 'blue', enabled=color)} {pct:>3.0f}%"
        )

    lines.append(col("─" * panel_width, "dim", enabled=color))
    total_label = f"{'Total':<{label_w}}"
    total_value = f"{tok(total.total_tokens):>6}"
    lines.append(
        f"{col(total_label, 'bold', enabled=color)} "
        f"{col(total_value, 'bold', 'green', enabled=color)}"
    )
    return lines


def money(usd: float) -> str:
    return f"${usd:,.2f}"


def build_cost_breakdown(agg: Aggregates, *, width: int, color: bool) -> list[str]:
    costs = estimate_costs(agg.tokens_by_model)
    if costs.total_usd <= 0 and not costs.unpriced_tokens:
        return []

    panel_width = min(42, max(24, width - 4))
    lines = [
        col("Estimated Cost", "bold", enabled=color),
        col("─" * panel_width, "dim", enabled=color),
    ]

    rows: list[tuple[str, int, float]] = [
        ("Input", costs.non_cached_input_tokens, costs.input_usd),
        ("Output", costs.output_tokens, costs.output_usd),
        ("Cache Read", costs.cached_input_tokens, costs.cached_input_usd),
    ]
    mx_cost = max((c for _, _, c in rows), default=0.0) or 1.0
    bar_w = 16
    for label, tokens, usd in rows:
        bl = int(usd / mx_cost * bar_w) if mx_cost else 0
        bar_s = "█" * bl + "░" * (bar_w - bl)
        lines.append(
            f"{label:<12} {tok(tokens):>6}  "
            f"{col(bar_s, 'blue', enabled=color)}  "
            f"{col(f'${usd:>6.2f}', 'green', enabled=color)}"
        )

    lines.append(col("─" * panel_width, "dim", enabled=color))
    total_cost = f"${costs.total_usd:>6.2f}"
    if costs.unpriced_tokens:
        total_cost += "+"
    total_label = f"{'Total':<12}"
    lines.append(
        f"{col(total_label, 'bold', enabled=color)} "
        f"{'':>6}  {'':{bar_w}}  "
        f"{col(total_cost, 'bold', 'green', enabled=color)}"
    )

    return lines


def build_last_days(agg: Aggregates, *, days: int, width: int, color: bool) -> list[str]:
    panel_width = min(45, max(24, width - 4))
    active = sorted(
        set(agg.user_messages_by_day)
        | set(agg.assistant_messages_by_day)
        | set(agg.tool_calls_by_day)
        | set(agg.sessions_by_day)
    )
    if not active:
        return []
    series = active[-days:]

    points: list[tuple[date, int, int, int]] = []
    for d in series:
        msgs = agg.user_messages_by_day.get(d, 0) + agg.assistant_messages_by_day.get(d, 0)
        tools = agg.tool_calls_by_day.get(d, 0)
        sess = agg.sessions_by_day.get(d, 0)
        points.append((d, msgs, tools, sess))

    mx = max((m for _, m, _, _ in points), default=0) or 1
    bar_w = 12
    hdr = (
        f"{'':6}  {'':{bar_w}} "
        + col(f"{'msgs':>5}", "dim", enabled=color)
        + "  "
        + col(f"{'tools':>5}", "dim", enabled=color)
        + "  "
        + col(f"{'sess':>4}", "dim", enabled=color)
    )
    lines = [
        col(f"Last {days} Days", "bold", enabled=color),
        col("─" * panel_width, "dim", enabled=color),
        hdr,
    ]
    for d, m, tl, s in points:
        bars = "▇" * int((m / mx) * bar_w) if m else ""
        lines.append(
            f"{col(fmt_day(d), 'cyan', enabled=color)}  "
            f"{col(f'{bars:<{bar_w}}', 'blue', enabled=color)} "
            f"{m:>5}  {tl:>5}  {s:>4}"
        )
    return lines


def build_daily_output_tokens(agg: Aggregates, *, days: int, width: int, color: bool) -> list[str]:
    panel_width = min(36, max(24, width - 4))
    active = sorted(d for d, u in agg.tokens_by_day.items() if u.output_tokens)
    if not active:
        return []
    series = active[-days:]
    points = [(d, agg.tokens_by_day.get(d, TokenUsage()).output_tokens) for d in series]

    mx = max((t for _, t in points), default=0) or 1
    bar_w = min(20, max(10, panel_width - 6 - 2 - 6 - 1))

    lines = [
        col("Daily Output Tokens", "bold", enabled=color),
        col("─" * panel_width, "dim", enabled=color),
    ]
    for d, t in points:
        bars = "▇" * int((t / mx) * bar_w) if t else ""
        toks_s = f"{tok(t):>6}"
        lines.append(
            f"{col(fmt_day(d), 'cyan', enabled=color)}  "
            f"{col(f'{bars:<{bar_w}}', 'green', enabled=color)} "
            f"{toks_s}"
        )
    return lines


def build_peak_hours(agg: Aggregates, *, width: int, color: bool) -> list[str]:
    counts = [agg.messages_by_hour.get(h, 0) for h in range(24)]
    mx = max(counts) or 1
    chars = "░▁▂▃▄▅▆▇█"

    def cell(v: int) -> str:
        idx = min(int(v / mx * 8), 8) if mx else 0
        return chars[idx]

    panel_width = min(45, max(24, width - 4))
    lines = [col("Peak Hours", "bold", enabled=color), col("─" * panel_width, "dim", enabled=color)]
    lines.append(col("".join(f"{h:>3}" for h in range(12)), "dim", enabled=color))
    lines.append("".join(col(f"{cell(counts[h]):>3}", "blue", enabled=color) for h in range(12)))
    lines.append(col("".join(f"{h:>3}" for h in range(12, 24)), "dim", enabled=color))
    lines.append("".join(col(f"{cell(counts[h]):>3}", "blue", enabled=color) for h in range(12, 24)))
    return lines


def build_projects(agg: Aggregates, *, width: int, color: bool) -> list[str]:
    if not agg.sessions:
        return []

    last_by_project: dict[str, SessionSummary] = {}

    def stamp(s: SessionSummary) -> datetime:
        if s.ended_at is not None:
            return s.ended_at
        if s.started_at is not None:
            return s.started_at
        return datetime.min.replace(tzinfo=UTC)

    for s in agg.sessions:
        key = s.repo_url or s.cwd or "unknown"
        prev = last_by_project.get(key)
        if prev is None or stamp(s) > stamp(prev):
            last_by_project[key] = s

    rows: list[tuple[float, str, SessionSummary]] = []
    for key, sess in last_by_project.items():
        cost = estimate_costs(sess.tokens_by_model).total_usd
        if cost <= 0:
            continue
        rows.append((cost, key, sess))

    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:10]
    if not rows:
        return []

    line_width = max(50, width - 4)
    path_max = 38 if width < 100 else 45

    lines = [
        col("Projects (Last Session)", "bold", enabled=color),
        col("─" * line_width, "dim", enabled=color),
    ]

    tot = 0.0
    for cost, key, sess in rows:
        tot += cost
        shown = key.replace(str(Path.home()), "~")
        if len(shown) > path_max:
            shown = "..." + shown[-(path_max - 3) :]

        ti, to = tok(sess.tokens.input_tokens), tok(sess.tokens.output_tokens)
        tks = f"{ti:>5} in, {to:>5} out"

        ln = " " * 21  # Lines changed column (not available in Codex logs).

        dr = " " * 9
        if sess.started_at and sess.ended_at:
            dr = f"  {fmt_dur(sess.ended_at - sess.started_at):>7}"

        lines.append(
            f"{col(f'${cost:>6.2f}', 'green', 'bold', enabled=color)}  "
            f"{col(f'{shown:<{path_max}}', 'cyan', enabled=color)}  "
            f"{col(tks, 'dim', enabled=color)}{ln}{dr}"
        )

    lines.append(col("─" * line_width, "dim", enabled=color))
    lines.append(
        f"{col(f'${tot:>6.2f}', 'green', 'bold', enabled=color)}  "
        f"{col('Total', 'bold', enabled=color)}"
    )
    return lines


def merge_columns(left: list[str], right: list[str], *, color: bool, gap: int = 3) -> list[str]:
    lw = max((visible_len(l) for l in left), default=0)
    merged: list[str] = []
    sep = col("│", "dim", enabled=color)
    for i in range(max(len(left), len(right))):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        pad = " " * max(0, lw - visible_len(l))
        merged.append(f"{l}{pad}{' ' * gap}{sep} {r}")
    return merged


def can_merge_columns(
    left: list[str],
    right: list[str],
    *,
    width: int,
    gap: int = 3,
    indent: int = 2,
) -> bool:
    if not left or not right:
        return False
    lw = max((visible_len(l) for l in left), default=0)
    rw = max((visible_len(r) for r in right), default=0)
    needed = indent + lw + gap + 2 + rw  # "│ " is 2 columns
    return needed <= width


def fmt_dur(delta: timedelta) -> str:
    secs = int(delta.total_seconds())
    mins, _ = divmod(max(secs, 0), 60)
    hrs, mins = divmod(mins, 60)
    days, hrs = divmod(hrs, 24)
    if days:
        return f"{days}d {hrs}h"
    if hrs:
        return f"{hrs}h {mins}m"
    return f"{mins}m"


def build_records(agg: Aggregates, *, width: int, color: bool) -> list[str]:
    panel_width = min(42, max(24, width - 4))
    lines = [col("Records", "bold", enabled=color), col("─" * panel_width, "dim", enabled=color)]

    sessions = len(agg.sessions) or 1
    total_msgs = sum(s.user_messages + s.assistant_messages for s in agg.sessions)
    total_tools = sum(s.tool_calls for s in agg.sessions)

    longest: tuple[timedelta, SessionSummary] | None = None
    for s in agg.sessions:
        if not s.started_at or not s.ended_at:
            continue
        dur = s.ended_at - s.started_at
        if longest is None or dur > longest[0]:
            longest = (dur, s)

    if longest is not None:
        dur, s = longest
        msg_count = s.user_messages + s.assistant_messages
        lines.append(
            f"{'Longest Session':<18}{col(fmt_dur(dur), 'bold', 'green', enabled=color)}  "
            f"{col('•', 'dim', enabled=color)}  {msg_count} msgs"
        )

    avg_msgs = total_msgs // sessions
    lines.append(f"{'Avg Msgs/Sess':<18}{col(str(avg_msgs), 'bold', 'green', enabled=color)}")
    lines.append(f"{'Total Tool Calls':<18}{col(f'{total_tools:,}', 'bold', 'green', enabled=color)}")
    return lines


SESSIONS = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"


def main():
    files = iter_session_files(SESSIONS)
    if not files:
        print(col("✗ No Codex session logs found", "magenta") + f" in {SESSIONS}")
        return 1

    agg = Aggregates()
    for f in files:
        parse_session_file(f, agg, since=None)

    width = get_width()
    wide = width >= 100

    for line in build_header(agg, width=width, color=True):
        print(line)
    for line in build_quick_stats(agg, color=True):
        print(line)

    limits = build_usage_limits(agg, color=True)
    if limits:
        print()
        for line in limits:
            print("  " + line)

    models = build_model_breakdown(agg, width=width, color=True)
    cost_panel = build_cost_breakdown(agg, width=width, color=True)

    if models or cost_panel:
        print()
        if wide and models and cost_panel and can_merge_columns(models, cost_panel, width=width):
            for line in merge_columns(models, cost_panel, color=True):
                print("  " + line)
        else:
            if models:
                for line in models:
                    print("  " + line)
            if cost_panel:
                if models:
                    print()
                for line in cost_panel:
                    print("  " + line)

    last7 = build_last_days(agg, days=7, width=width, color=True)
    daily_out = build_daily_output_tokens(agg, days=7, width=width, color=True)
    if last7 or daily_out:
        print()
        if wide and last7 and daily_out and can_merge_columns(last7, daily_out, width=width):
            for line in merge_columns(last7, daily_out, color=True):
                print("  " + line)
        else:
            if last7:
                for line in last7:
                    print("  " + line)
            if daily_out:
                print()
                for line in daily_out:
                    print("  " + line)

    peak = build_peak_hours(agg, width=width, color=True)
    records = build_records(agg, width=width, color=True)
    if peak or records:
        print()
        if wide and peak and records and can_merge_columns(peak, records, width=width):
            for line in merge_columns(peak, records, color=True):
                print("  " + line)
        else:
            if peak:
                for line in peak:
                    print("  " + line)
            if records:
                print()
                for line in records:
                    print("  " + line)

    projects = build_projects(agg, width=width, color=True)
    if projects:
        print()
        for line in projects:
            print("  " + line)

    print()
    return 0


if __name__ == "__main__":
    exit(main())
