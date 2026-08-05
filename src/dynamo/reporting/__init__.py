"""Reporting — delivery-layer daily activity reports (plain-language markdown).

Like ``dynamo.api``, this package sits at the edge: it may read from
persistence/config but nothing in the core ever imports it.
"""

from dynamo.reporting.daily import DailyReport, ReportSource, build_daily_report, render_markdown

__all__ = ["DailyReport", "ReportSource", "build_daily_report", "render_markdown"]
