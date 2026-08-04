"""Shared partition definitions (doc 02 §4 cadences)."""

from __future__ import annotations

from dagster import DailyPartitionsDefinition

TZ = "America/New_York"

# end_offset=1: the in-progress ET day is a valid partition — the 21:30 EOD
# schedule targets *today*, whose window has not closed yet.
daily_partitions = DailyPartitionsDefinition(start_date="2026-06-01", timezone=TZ, end_offset=1)
