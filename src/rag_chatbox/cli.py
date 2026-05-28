import argparse
from dataclasses import replace
import json
import logging
from typing import Sequence

from .config import load_config
from .rag_pipeline import build_chatbot


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Chatbox CLI")
    parser.add_argument("--llm-model", dest="llm_model", type=str, help="Override LLM model name")
    parser.add_argument("--embedding-model", dest="embedding_model", type=str, help="Override embedding model name")
    parser.add_argument(
        "--reader-artifact-dir",
        "--reader-model-dir",
        dest="reader_artifact_dir",
        type=str,
        help="Override reader artifact directory",
    )
    parser.add_argument("--top-k", dest="top_k", type=int, help="Override retrieval top-k")
    parser.add_argument("--fetch-k", dest="fetch_k", type=int, help="Override retrieval fetch-k")
    parser.add_argument("--score-threshold", dest="score_threshold", type=float, help="Override retrieval score threshold")
    parser.add_argument(
        "--query-rewrite-enabled",
        dest="query_rewrite_enabled",
        action="store_true",
        default=None,
        help="Enable AI query rewriting before retrieval",
    )
    parser.add_argument(
        "--no-query-rewrite",
        dest="query_rewrite_enabled",
        action="store_false",
        default=None,
        help="Disable AI query rewriting before retrieval",
    )
    parser.add_argument(
        "--query-rewrite-model",
        dest="query_rewrite_model",
        type=str,
        help="Override query rewrite model name",
    )
    parser.add_argument(
        "--query-rewrite-temperature",
        dest="query_rewrite_temperature",
        type=float,
        help="Override query rewrite temperature",
    )
    parser.add_argument(
        "--query-rewrite-max-variants",
        dest="query_rewrite_max_variants",
        type=int,
        help="Override number of rewritten query variants",
    )
    parser.add_argument(
        "--token-aware-chunking",
        action="store_true",
        default=None,
        help="Enable token-aware chunk splitting",
    )
    parser.add_argument(
        "--debug",
        dest="debug_trace",
        action="store_true",
        default=None,
        help="Print retrieval/reader/generation trace for each question",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug_trace",
        action="store_false",
        default=None,
        help="Disable retrieval/reader/generation trace output",
    )
    return parser.parse_args(argv)


def _print_debug_trace(chatbot, question: str) -> None:
    trace = chatbot.get_last_debug_trace()
    if not trace:
        return

    print("\n[DEBUG] Chat Trace")
    print(json.dumps({"question_input": question}, ensure_ascii=False, indent=2))
    print(json.dumps(trace, ensure_ascii=False, indent=2))


def run_cli(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    config = load_config()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    if overrides:
        config = replace(config, **overrides)
    chatbot = build_chatbot(config)

    print("\n" + "=" * 50)
    print("Gõ câu hỏi của bạn. Gõ 'exit' hoặc 'quit' để thoát.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChatbot: Tạm biệt! Hẹn gặp lại.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Tạm biệt! Hẹn gặp lại.")
            break

        if not user_input:
            continue

        print("Chatbot đang suy nghĩ...")
        answer = chatbot.ask(user_input)
        print(f"\nChatbot:\n{answer}")
        if config.debug_trace:
            _print_debug_trace(chatbot, question=user_input)
