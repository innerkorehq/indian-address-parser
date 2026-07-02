"""Compare gagan1985/qwen3-0.6b-indian-address-parser against
shiprocket-ai/open-tinybert-indian-address-ner on a held-out gold test set.

The two models use DIFFERENT field taxonomies (ours: 13 fields including
houseNumber/houseName/street/subLocality/subsubLocality/village/subDistrict/
district/city/state/pincode/poi; theirs: an 11-entity BIO-NER schema with
building_name/house_details/road/sub_locality/locality/city/state/pincode/
country/landmarks/floor). This script only scores fields with a clear
conceptual overlap (see FIELD_MAP below) — fields unique to one schema
(our subsubLocality/village/subDistrict/district, their country/floor) are
reported separately as "not comparable," not silently dropped or forced
into a mapping that would misrepresent either model.

Usage:
    pip install indian-address-parser transformers torch
    python compare_models.py                      # full 237-example benchmark
    python compare_models.py --n 50                # quick subset
    python compare_models.py --out results.json    # save detailed results
"""
from __future__ import annotations

import argparse
import json
import time

# our_field -> shiprocket_entity_group. Only fields with a real conceptual
# match on both sides are included; this is the intersection, not a forced
# 1:1 mapping of every field.
FIELD_MAP = {
    "houseNumber": "house_details",
    "houseName": "building_name",
    "street": "road",
    "locality": "locality",
    "subLocality": "sub_locality",
    "city": "city",
    "state": "state",
    "pincode": "pincode",
    "poi": "landmarks",
}

# Fields each model has that the other doesn't — reported as coverage info,
# never scored against a field they don't actually attempt to predict.
OUR_ONLY_FIELDS = ("subsubLocality", "village", "subDistrict", "district")
THEIRS_ONLY_ENTITIES = ("country", "floor")


def _normalize(text) -> str:
    if not text:
        return ""
    return str(text).strip().strip(",.;:").lower()


def load_benchmark(path: str, n: int | None) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items[:n] if n else items


def run_ours(addresses: list[str]) -> tuple[list[dict], float]:
    from indian_address_parser import AddressParser

    # Pinned to "qwen" explicitly: these are the published benchmark numbers
    # (README, benchmarks/README.md) and must stay stable even though the
    # package's own default backend is "t5".
    parser = AddressParser(backend="qwen")
    results = []
    t0 = time.perf_counter()
    for addr in addresses:
        results.append(parser.parse(addr))
    elapsed = time.perf_counter() - t0
    return results, elapsed


def run_shiprocket(addresses: list[str]) -> tuple[list[dict], float]:
    from transformers import pipeline

    nlp = pipeline(
        "ner", model="shiprocket-ai/open-tinybert-indian-address-ner", aggregation_strategy="simple"
    )
    results = []
    t0 = time.perf_counter()
    for addr in addresses:
        entities = nlp(addr)
        fields: dict[str, str] = {}
        for ent in entities:
            group = ent["entity_group"]
            word = ent["word"]
            fields[group] = (fields[group] + " " + word) if group in fields else word
        results.append(fields)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def score(benchmark: list[dict], ours: list[dict], theirs: list[dict]) -> dict:
    n = len(benchmark)
    stats = {
        field: {"ours_correct": 0, "theirs_correct": 0, "gold_present": 0}
        for field in FIELD_MAP
    }
    our_exact = 0
    our_json_ok = 0

    for item, our_pred, their_pred in zip(benchmark, ours, theirs):
        gold = item["gold"]
        if "_parse_error" not in our_pred:
            our_json_ok += 1

        all_match = True
        for our_field, their_field in FIELD_MAP.items():
            gold_val = _normalize(gold.get(our_field))
            our_val = _normalize(our_pred.get(our_field))
            their_val = _normalize(their_pred.get(their_field))

            if gold_val:
                stats[our_field]["gold_present"] += 1
                if our_val == gold_val:
                    stats[our_field]["ours_correct"] += 1
                else:
                    all_match = False
                if their_val == gold_val:
                    stats[our_field]["theirs_correct"] += 1
        if all_match:
            our_exact += 1

    return {
        "n": n,
        "our_json_parse_rate": our_json_ok / n,
        "our_overall_exact_match": our_exact / our_json_ok if our_json_ok else 0,
        "per_field": stats,
    }


def print_report(results: dict, our_time: float, their_time: float, n: int):
    print(f"\nBenchmark: {n} held-out gold-labeled addresses")
    print(f"(neither model was trained on this exact held-out split)\n")

    print(f"{'Field':16s} {'Ours (acc)':>12s} {'Shiprocket (acc)':>18s} {'Gold presence':>14s}")
    print("-" * 64)
    for field, s in results["per_field"].items():
        n_gold = s["gold_present"]
        our_acc = s["ours_correct"] / n_gold if n_gold else float("nan")
        their_acc = s["theirs_correct"] / n_gold if n_gold else float("nan")
        pres = n_gold / results["n"]
        our_str = f"{100*our_acc:.1f}%" if n_gold else "n/a"
        their_str = f"{100*their_acc:.1f}%" if n_gold else "n/a"
        print(f"{field:16s} {our_str:>12s} {their_str:>18s} {100*pres:>13.1f}%")

    print()
    print(f"Our JSON parse rate:        {100*results['our_json_parse_rate']:.1f}%")
    print(f"Our overall exact match:    {100*results['our_overall_exact_match']:.1f}% (all {len(FIELD_MAP)} shared fields correct)")
    print()
    print(f"Fields we predict that shiprocket's schema has no equivalent for: {', '.join(OUR_ONLY_FIELDS)}")
    print(f"Entities shiprocket predicts that we don't have an equivalent field for: {', '.join(THEIRS_ONLY_ENTITIES)}")
    print()
    print(f"Inference time — ours:       {our_time:.1f}s total ({1000*our_time/n:.0f}ms/address)")
    print(f"Inference time — shiprocket: {their_time:.1f}s total ({1000*their_time/n:.0f}ms/address)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="gold_test_set.jsonl")
    p.add_argument("--n", type=int, default=None, help="Limit to N examples for a quick run")
    p.add_argument("--out", help="Write detailed per-example results to this JSON file")
    args = p.parse_args()

    benchmark = load_benchmark(args.benchmark, args.n)
    addresses = [item["raw_address"] for item in benchmark]
    print(f"Running our model on {len(addresses)} addresses...")
    ours, our_time = run_ours(addresses)
    print(f"Running shiprocket's model on {len(addresses)} addresses...")
    theirs, their_time = run_shiprocket(addresses)

    results = score(benchmark, ours, theirs)
    print_report(results, our_time, their_time, len(benchmark))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": results,
                    "timing": {"ours_seconds": our_time, "shiprocket_seconds": their_time},
                    "per_example": [
                        {"raw_address": b["raw_address"], "gold": b["gold"], "ours": o, "shiprocket": t}
                        for b, o, t in zip(benchmark, ours, theirs)
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nDetailed results written to {args.out}")


if __name__ == "__main__":
    main()
