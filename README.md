# Claude Code Usage

A CLI tool to display beautiful Claude Code usage statistics.

## Features

- Usage limits with reset timers
- Model breakdown with cost estimates
- Daily activity and peak hours heatmap
- Session records and token trends
- Project-level cost tracking

## Usage

```bash
uv run stats.py
```

Or add an alias:

```bash
alias cstats="/path/to/stats.py"
```

## Requirements

- macOS (uses Keychain for OAuth)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

## Data Sources

- `~/.claude/stats-cache.json` - Local usage statistics
- `~/.claude.json` - Project history
- Anthropic OAuth API - Rate limits (requires Claude Code login)
