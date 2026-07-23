#!/usr/bin/env python3
"""Send one tiny n=3 request to check endpoint compatibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DIRECT_LLM_DIR = SCRIPT_DIR.parent / "DirectLLM"
if str(DIRECT_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(DIRECT_LLM_DIR))

from direct_llm_baseline import (  # noqa: E402
    ensure_local_base_url_bypasses_proxy,
    load_env_file,
    load_model_config,
    resolve_path,
)
from reasoning_llm_baseline import send_chat_completion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send exactly one small Chat Completions request with n=3."
    )
    parser.add_argument(
        "--model_config",
        default=str(DIRECT_LLM_DIR / "gpt_5.yaml"),
    )
    parser.add_argument("--env_file", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Small completion ceiling for this compatibility request only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        raise ValueError(f"--max_tokens must be positive, got {args.max_tokens}")

    env_file = Path(args.env_file).expanduser().resolve()
    load_env_file(env_file)
    model_config = load_model_config(resolve_path(args.model_config), env_file)
    ensure_local_base_url_bypasses_proxy(model_config["base_url"])

    result = send_chat_completion(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
        model_name=model_config["model_name"],
        messages=[{"role": "user", "content": "Reply with A."}],
        n=3,
        temperature=0.6,
        top_p=0.95,
        max_tokens=args.max_tokens,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        request_timeout=args.request_timeout,
        max_retries=0,
        request_limiter=None,
    )

    choices = result["choices"]
    print(f"HTTP status: {result['http_status']}")
    print(f"Returned model: {result.get('model', '')}")
    print(f"Choice count: {len(choices)}")
    for choice in choices:
        compact_content = " ".join(str(choice["content"]).split())
        print(
            f"choice[{choice['index']}]: "
            f"finish_reason={choice.get('finish_reason')} "
            f"content={compact_content!r}"
        )
    print(
        "Usage: "
        + json.dumps(result.get("usage", {}), ensure_ascii=False, sort_keys=True)
    )

    if len(choices) != 3:
        raise SystemExit(
            f"n=3 is not compatible: endpoint returned {len(choices)} choices"
        )
    print("n=3 compatibility check passed.")


if __name__ == "__main__":
    main()
