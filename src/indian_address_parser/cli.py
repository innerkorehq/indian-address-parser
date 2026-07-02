"""CLI for indian-address-parser.

Usage:
    indian-address-parser "FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029"
    cat addresses.txt | indian-address-parser --stdin
    indian-address-parser --file addresses.txt --out results.jsonl
    indian-address-parser --backend qwen "..."       # larger, more accurate model
    indian-address-parser --backend tinybert "..."   # smallest, fastest model
"""
from __future__ import annotations

import argparse
import json
import sys

from .parser import BACKENDS, DEFAULT_ADAPTER_REPO, DEFAULT_BASE_MODEL, AddressParser


def main():
    p = argparse.ArgumentParser(description="Parse Indian addresses into structured fields")
    p.add_argument("address", nargs="?", help="Single address to parse")
    p.add_argument("--stdin", action="store_true", help="Read addresses from stdin, one per line")
    p.add_argument("--file", help="Read addresses from a text file, one per line")
    p.add_argument("--out", help="Write JSONL output to file (default: stdout)")
    p.add_argument(
        "--backend", choices=BACKENDS, default="t5",
        help="Model backend: 't5' (default, flan-t5-small, lighter/faster), "
             "'qwen' (Qwen3-0.6B LoRA, most accurate), or "
             "'tinybert' (TinyBERT 4L/312D, smallest/fastest)",
    )
    p.add_argument("--model-repo", default=None,
                    help="HF Hub repo for the t5 or tinybert backend (defaults to each backend's own model)")
    p.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO, help="HF Hub repo for the qwen backend's LoRA adapter")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HF Hub repo for the qwen backend's base model")
    args = p.parse_args()

    if args.address:
        addresses = [args.address]
    elif args.stdin:
        addresses = [line.rstrip("\n") for line in sys.stdin if line.strip()]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            addresses = [line.rstrip("\n") for line in f if line.strip()]
    else:
        p.print_help()
        return

    parser = AddressParser(
        backend=args.backend,
        model_repo=args.model_repo,
        adapter_repo=args.adapter_repo,
        base_model=args.base_model,
    )

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    for addr in addresses:
        result = parser.parse(addr)
        result["_raw_address"] = addr
        line = json.dumps(result, ensure_ascii=False)
        if out_f:
            out_f.write(line + "\n")
        else:
            print(line)

    if out_f:
        out_f.close()
        print(f"Wrote {len(addresses):,} results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
