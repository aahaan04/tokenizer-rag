"""CLI entry point: python -m tokrag <command> [options]."""

import argparse
import sys


def cmd_collect(args: argparse.Namespace) -> None:
    from tokrag.collect.pipeline import collect
    from tokrag.config import MANIFEST_PATH

    candidates = collect(sample=args.sample)
    included = sum(1 for c in candidates if c.included)
    print(f"Collected {len(candidates)} unique candidates: {included} included, {len(candidates) - included} rejected.")
    print(f"Manifest written to {MANIFEST_PATH}")


def cmd_parse(args: argparse.Namespace) -> None:
    from tokrag.parse.pipeline import run
    run(limit=args.limit)


def cmd_index(args: argparse.Namespace) -> None:
    raise NotImplementedError("index: lands in Phase 3")


def cmd_eval(args: argparse.Namespace) -> None:
    raise NotImplementedError("eval: lands in Phase 3/5")


def cmd_chat(args: argparse.Namespace) -> None:
    raise NotImplementedError("chat: lands in Phase 6")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="run the corpus collection pipeline")
    p_collect.add_argument("--sample", action="store_true", help="small ~20-candidate dry run instead of the full pass")
    p_collect.set_defaults(func=cmd_collect)

    p_parse = sub.add_parser("parse", help="parse and chunk collected papers")
    p_parse.add_argument("--limit", type=int, default=None, help="only process the first N included rows (for a quick test)")
    p_parse.set_defaults(func=cmd_parse)

    p_index = sub.add_parser("index", help="build BM25 / dense / hybrid indexes")
    p_index.add_argument("--model", default="all-MiniLM-L6-v2", help="embedding model name")
    p_index.set_defaults(func=cmd_index)

    p_eval = sub.add_parser("eval", help="run the evaluation set against an index config")
    p_eval.add_argument("--config", default="hybrid", help="retrieval config to evaluate")
    p_eval.set_defaults(func=cmd_eval)

    p_chat = sub.add_parser("chat", help="start the multi-turn chat loop")
    p_chat.set_defaults(func=cmd_chat)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print(f"Not implemented yet — {e}", file=sys.stderr)
        sys.exit(1)
