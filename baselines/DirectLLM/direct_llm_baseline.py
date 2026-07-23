#!/usr/bin/env python3
"""Direct LLM baseline for MACT table-QA JSONL files."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "output" / "crt_answerable.jsonl"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "output" / "direct_llm_results.jsonl"
DEFAULT_MODEL_CONFIG = SCRIPT_DIR / "gpt_5.yaml"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct LLM MACT baseline.")
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--model_config",
        default=None,
        help=f"Model config YAML. Defaults to {DEFAULT_MODEL_CONFIG} unless --model_name is used alone.",
    )
    parser.add_argument("--model_name", default=os.getenv("MODEL_NAME") or None)
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "4")))
    parser.add_argument("--env_file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument(
        "--max_table_chars",
        type=int,
        default=0,
        help="Maximum CSV characters to send. 0 means send the full table.",
    )
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--request_interval_s",
        type=float,
        default=float(os.getenv("REQUEST_INTERVAL_S", "0")),
        help="Minimum seconds between starting LLM HTTP attempts across all workers. 0 disables throttling.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


class RequestStartLimiter:
    def __init__(self, interval_s: float):
        self.interval_s = max(0.0, float(interval_s))
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        if self.interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_s = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self.interval_s
        if sleep_s > 0:
            time.sleep(sleep_s)


def load_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def resolve_path(path: str) -> Path:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path.resolve()


def resolve_config_value(llm_config: dict[str, Any], key: str, *, env_file: Path) -> str:
    value = llm_config.get(key)
    if value is None:
        env_key = llm_config.get(f"{key}_env")
        if env_key:
            value = os.getenv(str(env_key))

    if value is None:
        value = ""
    value = str(value).strip()

    if not value:
        env_key = llm_config.get(f"{key}_env")
        if env_key:
            raise ValueError(f"{env_key} is required in environment or {env_file}")
        raise ValueError(f"llm.{key} is required in the model config")

    return value


def load_model_config(config_path: Path, env_file: Path) -> dict[str, str]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llm_config = raw_config.get("llm")
    if not isinstance(llm_config, dict):
        raise ValueError(f"Model config must contain an llm mapping: {config_path}")

    return {
        "model_name": resolve_config_value(llm_config, "model", env_file=env_file),
        "base_url": resolve_config_value(llm_config, "base_url", env_file=env_file),
        "api_key": resolve_config_value(llm_config, "api_key", env_file=env_file),
    }


def load_legacy_openai_config(model_name: str, env_file: Path) -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY is required in {env_file} or the environment")

    return {
        "model_name": model_name,
        "base_url": base_url,
        "api_key": api_key,
    }


def ensure_local_base_url_bypasses_proxy(base_url: str) -> None:
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        return

    local_hosts = ["127.0.0.1", "localhost", "::1"]
    for env_key in ("NO_PROXY", "no_proxy"):
        existing = os.getenv(env_key, "")
        parts = [part.strip() for part in existing.split(",") if part.strip()]
        for host in local_hosts:
            if host not in parts:
                parts.append(host)
        os.environ[env_key] = ",".join(parts)


def read_jsonl(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be non-negative, got {limit}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
    if limit == 0:
        return []

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def truncate_table(table_text: str, max_table_chars: int) -> tuple[str, bool]:
    if max_table_chars > 0 and len(table_text) > max_table_chars:
        return table_text[:max_table_chars], True
    return table_text, False


def table_rows_to_csv(table_rows: list[object]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for row_index, row in enumerate(table_rows):
        if not isinstance(row, (list, tuple)):
            raise TypeError(
                f"table_text row {row_index} must be a list or tuple, "
                f"got {type(row).__name__}"
            )
        writer.writerow("" if value is None else value for value in row)
    return output.getvalue().rstrip("\n")


def read_table(item: dict[str, Any], max_table_chars: int) -> tuple[str, bool]:
    if "table_text" in item:
        raw_table = item["table_text"]
        if isinstance(raw_table, str):
            table_text = raw_table
        elif isinstance(raw_table, list):
            table_text = table_rows_to_csv(raw_table)
        else:
            raise TypeError(
                "table_text must be a CSV string or a list of rows, "
                f"got {type(raw_table).__name__}"
            )
    elif item.get("table_file"):
        table_path = Path(str(item["table_file"])).expanduser()
        if not table_path.is_absolute():
            table_path = PROJECT_ROOT / table_path
        table_text = table_path.read_text(encoding="utf-8")
    else:
        raise KeyError("Expected 'table_text' (MACT) or 'table_file' (TableZoomer)")

    return truncate_table(table_text, max_table_chars)


def read_question(item: dict[str, Any]) -> str:
    question = item.get("statement", item.get("question"))
    if question is None or not str(question).strip():
        raise KeyError("Expected 'statement' (MACT) or 'question' (TableZoomer)")
    return str(question).strip()


def build_prompt(question: str, table_text: str) -> str:
    return (
        "Table (CSV):\n"
        f"{table_text}\n\n"
        f"Question: {question}\n\n"
        "Answer the question based on the table. Output only the final answer."
    )


def normalize_answer(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff.]+", " ", text)
    return text.strip()


def is_correct_answer(prediction: str, gold_answer: object) -> bool:
    if gold_answer is None:
        return False

    if isinstance(gold_answer, list):
        gold_values = gold_answer
    else:
        gold_values = [gold_answer]

    gold_norms = [normalize_answer(value) for value in gold_values if normalize_answer(value)]
    pred_norm = normalize_answer(prediction)
    if not gold_norms or not pred_norm:
        return False

    if pred_norm in gold_norms:
        return True
    return all(gold in pred_norm for gold in gold_norms)


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def parse_chat_response(response_body: bytes) -> str:
    data = json.loads(response_body.decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"LLM response has no choices: {data}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise ValueError(f"LLM response has no message content: {data}")
    return str(content).strip()


def call_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: str,
    request_timeout: float,
    max_retries: int,
    request_limiter: RequestStartLimiter | None = None,
) -> str:
    url = f"{normalize_base_url(base_url)}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if request_limiter is not None:
            request_limiter.wait()
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                return parse_chat_response(response.read())
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body}")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"LLM request failed after {max_retries + 1} attempts: {last_error}")


def run_one(
    idx: int,
    item: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    max_table_chars: int,
    request_timeout: float,
    max_retries: int,
    request_limiter: RequestStartLimiter | None,
) -> tuple[int, dict[str, Any]]:
    result = dict(item)
    result["model_name"] = model_name

    try:
        table_text, truncated = read_table(item, max_table_chars)
        prompt = build_prompt(read_question(item), table_text)
        answer = call_chat_completion(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            prompt=prompt,
            request_timeout=request_timeout,
            max_retries=max_retries,
            request_limiter=request_limiter,
        )
        result["response"] = answer
        result["pred_answer"] = answer
        result["execute_status"] = "success"
        result["table_truncated"] = truncated
    except Exception as exc:
        result["response"] = ""
        result["pred_answer"] = ""
        result["execute_status"] = "fail"
        result["error"] = str(exc)

    return idx, result


def sync_output_file(output_file: TextIO) -> None:
    output_file.flush()
    os.fsync(output_file.fileno())


def write_jsonl_row(output_file: TextIO, row: dict[str, Any]) -> None:
    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    sync_output_file(output_file)


def stream_completed_results(
    completed_futures: Iterable[concurrent.futures.Future[tuple[int, dict[str, Any]]]],
    all_futures: Sequence[concurrent.futures.Future[tuple[int, dict[str, Any]]]],
    output_file: TextIO,
    total: int,
) -> tuple[int, int]:
    completed = 0
    failed = 0
    correct = 0
    incorrect = 0

    try:
        for future in completed_futures:
            idx, result = future.result()
            write_jsonl_row(output_file, result)
            completed += 1

            if result.get("execute_status") != "success":
                failed += 1
                progress_result = "error"
            elif "answer" in result:
                progress_result = (
                    "correct"
                    if is_correct_answer(str(result.get("pred_answer", "")), result["answer"])
                    else "incorrect"
                )
            else:
                progress_result = "unknown"
            if progress_result == "correct":
                correct += 1
            elif progress_result == "incorrect":
                incorrect += 1

            example_id = result.get("id") or f"row_{idx}"
            acc = correct / completed if completed else 0.0
            print(
                f"[PROGRESS] {completed}/{total} "
                f"id={example_id} result={progress_result} "
                f"status={result.get('execute_status')} "
                f"correct={correct} incorrect={incorrect} failed={failed} "
                f"acc={acc:.4f}",
                flush=True,
            )
    except BaseException:
        cancelled = sum(future.cancel() for future in all_futures)
        sync_output_file(output_file)
        print(
            f"Interrupted after saving {completed}/{total} results; "
            f"cancelled {cancelled} pending requests.",
            file=sys.stderr,
            flush=True,
        )
        raise

    return completed, failed


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if args.max_retries < 0:
        raise ValueError(f"--max_retries must be non-negative, got {args.max_retries}")
    if args.max_table_chars < 0:
        raise ValueError(f"--max_table_chars must be non-negative, got {args.max_table_chars}")
    if args.request_interval_s < 0:
        raise ValueError(f"--request_interval_s must be non-negative, got {args.request_interval_s}")

    env_file = Path(args.env_file).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    load_env_file(env_file)
    if args.model_config:
        model_config = load_model_config(resolve_path(args.model_config), env_file)
        model_name = args.model_name or model_config["model_name"]
    elif args.model_name:
        model_config = load_legacy_openai_config(args.model_name, env_file)
        model_name = model_config["model_name"]
    else:
        model_config = load_model_config(DEFAULT_MODEL_CONFIG, env_file)
        model_name = model_config["model_name"]

    api_key = model_config["api_key"]
    base_url = model_config["base_url"]
    ensure_local_base_url_bypasses_proxy(base_url)

    rows = read_jsonl(dataset_path, args.limit)

    print(
        f"Running direct LLM baseline: examples={len(rows)} workers={args.workers} "
        f"model={model_name} request_interval_s={args.request_interval_s}",
        flush=True,
    )

    request_limiter = RequestStartLimiter(args.request_interval_s)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_one,
                    idx,
                    item,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    max_table_chars=args.max_table_chars,
                    request_timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    request_limiter=request_limiter,
                )
                for idx, item in enumerate(rows)
            ]
            completed, failed = stream_completed_results(
                concurrent.futures.as_completed(futures),
                futures,
                output_file,
                len(rows),
            )

    print(f"Saved results to {output_path}", flush=True)

    if failed:
        print(f"Completed with {failed} failed examples.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
