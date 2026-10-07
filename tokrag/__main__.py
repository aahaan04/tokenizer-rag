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
    from tokrag.index.bm25 import build_bm25_index
    from tokrag.index.dense import build_dense_index

    if args.model == "bm25":
        build_bm25_index()
    else:
        out_name = args.out or args.model.split("/")[-1].replace("-", "_")
        build_dense_index(args.model, out_name, batch_size=args.batch_size)


def cmd_dedup(args: argparse.Namespace) -> None:
    from tokrag.dedup.pipeline import run

    run()


def cmd_eval(args: argparse.Namespace) -> None:
    from tokrag.eval.runner import run_dense_eval

    result = run_dense_eval(args.index, args.model, k=args.k, use_diversify=args.diversify, use_gold_rewrite=args.gold_rewrite)
    import json

    print(json.dumps(result, indent=2))


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

    p_index = sub.add_parser("index", help="build a BM25 or dense index")
    p_index.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="embedding model name, or 'bm25'")
    p_index.add_argument("--out", default=None, help="output index name (default: derived from model name)")
    p_index.add_argument("--batch-size", type=int, default=64)
    p_index.set_defaults(func=cmd_index)

    p_dedup = sub.add_parser("dedup", help="assign canonical ids, flag surveys, write the false-merge audit")
    p_dedup.set_defaults(func=cmd_dedup)

    p_eval = sub.add_parser("eval", help="run the evaluation set against a dense index")
    p_eval.add_argument("--index", default="bge_small", help="dense index name (see data/index/)")
    p_eval.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="embedding model matching the index")
    p_eval.add_argument("--k", type=int, default=5)
    p_eval.add_argument("--diversify", action="store_true", help="apply Phase 4 dedup diversification (cap 1 chunk/paper + survey demotion)")
    p_eval.add_argument("--gold-rewrite", action="store_true", help="use each follow-up's gold standalone rewrite instead of the raw question")
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
