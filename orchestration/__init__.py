"""Dagster orchestration — thin wiring over core functions (doc 01 §2).

Every asset body is: fetch via adapters -> call core function -> persist via
repository -> return metadata. Business logic stays in ``src/dynamo``.
"""
