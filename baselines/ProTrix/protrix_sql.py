#!/usr/bin/env python3
"""Restricted in-memory SQL execution adapted for ProTrix table reasoning."""

from __future__ import annotations

import math
import re
import sqlite3
from typing import Any

import pandas as pd


SQL_BLOCK = re.compile(r"```\s*sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
MAX_RESULT_ROWS = 50
MAX_CELL_CHARS = 120


def extract_sql_queries(text: str) -> list[str]:
    """Extract fenced SQL blocks, retaining their generated order."""
    return [query.strip() for query in SQL_BLOCK.findall(text) if query.strip()]


def _unique_headers(header: list[object]) -> list[str]:
    result: list[str] = []
    for index, value in enumerate(header):
        base = str(value).strip() or f"col_{index}"
        candidate = base
        suffix = 2
        while candidate in result:
            candidate = f"{base}_{suffix}"
            suffix += 1
        result.append(candidate)
    return result


def _parse_number(value: str) -> int | float | None:
    text = value.strip().replace(",", "")
    if not text or text in {"-", "/"}:
        return None
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    text = re.sub(r"^[\$£€]", "", text).strip()
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if abs(number - round(number)) < 1e-9:
        return int(round(number))
    return number


def table_to_dataframe(table_text: list[list[object]], transpose: bool = False) -> pd.DataFrame:
    table = [["" if cell is None else str(cell).strip() for cell in row] for row in table_text]
    if transpose:
        table = [list(row) for row in zip(*table)]
    if len(table) < 2:
        raise ValueError("A SQL table requires a header and at least one data row.")
    header = _unique_headers(table[0])
    frame = pd.DataFrame(table[1:], columns=header)
    for column in frame.columns:
        parsed = [_parse_number(value) for value in frame[column].tolist()]
        nonempty = [value for value in frame[column].tolist() if value not in {"", "-", "/"}]
        if nonempty and sum(value is not None for value in parsed) == len(nonempty):
            frame[column] = [value if value is not None else None for value in parsed]
    return frame


def _rewrite_sql(query: str, columns: list[str]) -> str:
    sql = query.strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    if ";" in sql:
        raise ValueError("Multiple SQL statements are not allowed.")
    if not re.match(r"^(select|with)\b", sql, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed.")
    # Never rewrite text inside SQL string literals: a cell value can equal a
    # column name, and changing that literal silently changes query semantics.
    segments = re.split(r"('(?:''|[^'])*')", sql)

    def replace_table_reference(match: re.Match[str]) -> str:
        keyword, raw_name = match.group(1), match.group(2)
        unquoted = raw_name.strip('`"[]').casefold()
        if unquoted in {"w", "table", "data", "df", "my_table"}:
            return f"{keyword.upper()} w"
        return match.group(0)

    for index in range(0, len(segments), 2):
        segment = segments[index]
        segment = re.sub(
            r"\b(from)\s+(`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w.]*)",
            replace_table_reference,
            segment,
            flags=re.IGNORECASE,
        )
        segment = re.sub(
            r"\b(join)\s+(`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][\w.]*)",
            replace_table_reference,
            segment,
            flags=re.IGNORECASE,
        )
        segment = re.sub(r"\bCHARINDEX\s*\(", "instr(", segment, flags=re.IGNORECASE)
        for column in sorted(columns, key=len, reverse=True):
            for variant in {column, column.replace(" ", "_"), column.replace(" ", "")}:
                if not variant:
                    continue
                pattern = rf"(?<![\w`\"\[])({re.escape(variant)})(?![\w`\"\]])"
                segment = re.sub(
                    pattern, lambda match, name=column: f"`{name}`", segment, flags=re.IGNORECASE
                )
        segments[index] = segment
    return "".join(segments)


def _authorizer(action: int, _arg1: str, _arg2: str, _db: str, _source: str) -> int:
    denied = {
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_PRAGMA,
    }
    return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK


def _execute_on_frame(query: str, frame: pd.DataFrame) -> tuple[str, list[dict[str, Any]]]:
    sql = _rewrite_sql(query, list(frame.columns))
    connection = sqlite3.connect(":memory:")
    try:
        connection.enable_load_extension(False)
        frame.to_sql("w", connection, index=False)
        connection.set_authorizer(_authorizer)
        steps = 0

        def progress() -> int:
            nonlocal steps
            steps += 1
            return 1 if steps > 10_000 else 0

        connection.set_progress_handler(progress, 100)
        cursor = connection.execute(sql)
        columns = [description[0] for description in cursor.description or []]
        values = cursor.fetchmany(MAX_RESULT_ROWS + 1)
        if len(values) > MAX_RESULT_ROWS:
            values = values[:MAX_RESULT_ROWS]
        rows = [
            {
                column: ("" if value is None else value)
                for column, value in zip(columns, row)
            }
            for row in values
        ]
        return sql, rows
    finally:
        connection.close()


def execute_queries(
    table_text: list[list[object]], queries: list[str]
) -> list[dict[str, Any]]:
    """Execute model SQL safely, with the official transpose fallback."""
    normal = table_to_dataframe(table_text)
    transposed: pd.DataFrame | None = None
    executions = []
    for query in queries:
        try:
            normalized_sql, rows = _execute_on_frame(query, normal)
            executions.append(
                {
                    "query": query,
                    "normalized_sql": normalized_sql,
                    "orientation": "normal",
                    "rows": rows,
                    "status": "ok",
                    "error": None,
                }
            )
            continue
        except Exception as normal_error:
            try:
                if transposed is None:
                    transposed = table_to_dataframe(table_text, transpose=True)
                normalized_sql, rows = _execute_on_frame(query, transposed)
                executions.append(
                    {
                        "query": query,
                        "normalized_sql": normalized_sql,
                        "orientation": "transposed",
                        "rows": rows,
                        "status": "ok",
                        "error": None,
                    }
                )
                continue
            except Exception as transpose_error:
                executions.append(
                    {
                        "query": query,
                        "normalized_sql": None,
                        "orientation": None,
                        "rows": [],
                        "status": "error",
                        "error": (
                            f"normal={type(normal_error).__name__}: {normal_error}; "
                            f"transpose={type(transpose_error).__name__}: {transpose_error}"
                        ),
                    }
                )
    return executions


def format_execution_blocks(
    executions: list[dict[str, Any]], max_rows_per_query: int = 10
) -> str:
    blocks = []
    for execution in executions:
        query = execution["normalized_sql"] or execution["query"]
        if execution["status"] != "ok":
            blocks.append(
                f"```sql\n{query}\n```\nExecution Error:\n{execution['error']}\n"
            )
            continue
        rows = execution["rows"]
        if not rows:
            rendered = "(empty result)"
        else:
            header = list(rows[0])
            lines = [" | ".join(header)]
            for row in rows[:max_rows_per_query]:
                lines.append(
                    " | ".join(str(row[column])[:MAX_CELL_CHARS] for column in header)
                )
            if len(rows) > max_rows_per_query:
                lines.append(f"... ({len(rows) - max_rows_per_query} more rows omitted)")
            rendered = "\n".join(lines)
        blocks.append(f"```sql\n{query}\n```\nExecution Result:\n```\n{rendered}\n```\n")
    return "".join(blocks)
