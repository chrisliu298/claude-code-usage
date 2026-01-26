# Agent CLI Usage Statistics

CLI tools to display usage statistics for **Claude Code** and **Codex CLI**.

<p align="center">
  <img src="claude_code.png" alt="Claude Code Usage Statistics" width="49%">
  <img src="codex.png" alt="Codex Usage Statistics" width="49%">
</p>

## Claude Code Stats

```bash
uv run claude_code_stats.py
# or
./claude_code_stats.py
```

Or add an alias:

```bash
alias cstats="/path/to/claude_code_stats.py"
```

### Features

- **Usage limits** - 5-hour and 7-day utilization with reset timers
- **Model breakdown** - Token usage and cost estimates by model
- **Activity tracking** - Daily messages, tool calls, and sessions
- **Peak hours** - Heatmap showing when you're most active
- **Token trends** - Daily output token usage over time
- **Project costs** - Per-project cost tracking with lines changed

### Data Sources

- `~/.claude/stats-cache.json` - Local usage statistics
- `~/.claude.json` - Project history
- Anthropic OAuth API - Rate limits (requires Claude Code login)

## Codex Stats

```bash
uv run codex_stats.py
# or
./codex_stats.py
```

Options:

```bash
./codex_stats.py --days 7
./codex_stats.py --json > codex_usage.json
./codex_stats.py --codex-home ~/.codex
```

Or add an alias:

```bash
alias xstats="/path/to/codex_stats.py"
```

### Cost Estimation

The CLI output includes an **Estimated Cost** panel based on OpenAI's pricing table (USD / 1M tokens). Cached input tokens are billed at the model's cached-input rate when available. Models not in the table show as "Unpriced".

### Data Source

Codex CLI writes per-session JSONL logs under `$CODEX_HOME/sessions/` (default: `~/.codex/sessions/`).

## Requirements

- macOS (uses Keychain for OAuth in Claude Code stats)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

## License

MIT
