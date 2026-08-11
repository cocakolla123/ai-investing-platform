"""Historical replay — a composition-root package (plan Phase 4).

Like ``orchestration``, this package may import everything (persistence,
data, research, risk, portfolio, backtest, learning); import-linter forbids
the core packages from importing it back. Replay is manual-only (CLI), never
scheduled, and writes ONLY replay_runs / replay_decisions /
decision_experience(source='replay').
"""
