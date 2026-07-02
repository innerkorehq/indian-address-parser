"""Compare our tinybert backend (huawei-noah/TinyBERT_General_4L_312D,
fine-tuned) against shiprocket-ai/open-tinybert-indian-address-ner on the
same held-out gold test set used by compare_models.py.

This is a separate, narrower comparison from compare_models.py's 4-way
benchmark: both models here share the "TinyBERT" name, but NOT the same
underlying architecture size — ours is the original TinyBERT_General_4L_312D
(4 layers, 312 hidden, ~14M params); shiprocket's is a larger BERT
(6 layers, 768 hidden, ~66.4M params, verified via `sum(p.numel() ...)`) that
merely reuses the "tinybert" name. That size difference (~4.7x more params)
is reported explicitly below rather than left implicit, since a same-name
comparison could otherwise look more like-for-like than it is.

Our model uses the SAME 13-field taxonomy as compare_models.py's other two
backends. Shiprocket's model uses the same 11-entity BIO-NER schema as their
modernbert model (verified: identical id2label). See compare_models.py's
docstring for why only the 9 conceptually-overlapping fields are scored.

Usage:
    pip install indian-address-parser transformers torch
    python compare_tinybert.py                      # full 237-example benchmark
    python compare_tinybert.py --n 50                # quick subset
    python compare_tinybert.py --out results.json    # save per-example detail
"""
from __future__ import annotations

import argparse
import json
import time

OUR_LABEL = "tinybert-4l-312d (ours)"
THEIRS_LABEL = "shiprocket open-tinybert (6L/768D)"
SHIPROCKET_MODEL_ID = "shiprocket-ai/open-tinybert-indian-address-ner"

# canonical_field -> shiprocket_entity_group. Same intersection as
# compare_models.py's FIELD_MAP (both shiprocket models share this schema).
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

    parser = AddressParser(backend="tinybert")
    results = []
    t0 = time.perf_counter()
    for addr in addresses:
        results.append(parser.parse(addr))
    elapsed = time.perf_counter() - t0
    return results, elapsed


def run_shiprocket(addresses: list[str]) -> tuple[list[dict], float]:
    from transformers import pipeline

    nlp = pipeline("ner", model=SHIPROCKET_MODEL_ID, aggregation_strategy="simple")
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
    stats = {field: {"ours": 0, "theirs": 0, "gold_present": 0} for field in FIELD_MAP}
    our_exact = 0

    for item, our_pred, their_pred in zip(benchmark, ours, theirs):
        gold = item["gold"]
        all_match = True
        for field, entity in FIELD_MAP.items():
            gold_val = _normalize(gold.get(field))
            our_val = _normalize(our_pred.get(field))
            their_val = _normalize(their_pred.get(entity))

            if gold_val:
                stats[field]["gold_present"] += 1
                if our_val == gold_val:
                    stats[field]["ours"] += 1
                else:
                    all_match = False
                if their_val == gold_val:
                    stats[field]["theirs"] += 1
        if all_match:
            our_exact += 1

    return {
        "n": n,
        "our_overall_exact_match": our_exact / n if n else 0,
        "per_field": stats,
    }


def print_report(results: dict, our_time: float, their_time: float, n: int):
    print(f"\nBenchmark: {n} held-out gold-labeled addresses")
    print("(neither model was trained on this exact held-out split)\n")
    print(f"{OUR_LABEL}: ~14M params, 4 layers, 312 hidden")
    print(f"{THEIRS_LABEL}: ~66.4M params, 6 layers, 768 hidden (~4.7x more params than ours)\n")

    print(f"{'Field':16s} {'ours (acc)':>14s} {'theirs (acc)':>14s} {'Gold presence':>14s}")
    print("-" * 60)
    for field, s in results["per_field"].items():
        n_gold = s["gold_present"]
        our_acc = s["ours"] / n_gold if n_gold else float("nan")
        their_acc = s["theirs"] / n_gold if n_gold else float("nan")
        pres = n_gold / results["n"]
        our_str = f"{100*our_acc:.1f}%" if n_gold else "n/a"
        their_str = f"{100*their_acc:.1f}%" if n_gold else "n/a"
        print(f"{field:16s} {our_str:>14s} {their_str:>14s} {100*pres:>13.1f}%")

    print()
    print(f"Our overall exact match (all {len(FIELD_MAP)} shared fields correct): {100*results['our_overall_exact_match']:.1f}%")
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

    print(f"Running {OUR_LABEL} on {len(addresses)} addresses...")
    ours, our_time = run_ours(addresses)
    print(f"Running {THEIRS_LABEL} on {len(addresses)} addresses...")
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
