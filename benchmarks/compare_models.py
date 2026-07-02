"""Compare this package's three models (TinyBERT 4L/312D default, flan-t5-small,
Qwen3-0.6B LoRA) against shiprocket-ai/open-modernbert-indian-address-ner on
a held-out gold test set.

Our three models use the SAME 13-field taxonomy (houseNumber/houseName/poi/
street/subsubLocality/subLocality/locality/village/subDistrict/district/city/
state/pincode). Shiprocket's model uses a different, 11-entity BIO-NER schema
(building_name/house_details/road/sub_locality/locality/city/state/pincode/
country/landmarks/floor). This script only scores fields with a clear
conceptual overlap between our schema and theirs (see FIELD_MAP below) —
fields unique to one schema (our subsubLocality/village/subDistrict/district,
their country/floor) are reported separately as "not comparable," not
silently dropped or forced into a mapping that would misrepresent either
model.

Usage:
    pip install indian-address-parser transformers torch
    python compare_models.py                          # full 237-example benchmark, all 4 models
    python compare_models.py --n 50                    # quick subset
    python compare_models.py --models t5 modernbert     # skip the slower qwen backend
    python compare_models.py --out results.json        # save detailed results
"""
from __future__ import annotations

import argparse
import json
import time

MODEL_LABELS = {
    "t5": "flan-t5-small (ours, previous default)",
    "qwen": "qwen3-0.6b (ours, most accurate)",
    "tinybert": "tinybert-4l-312d (ours, default)",
    "modernbert": "shiprocket modernbert",
}
# Short labels for the fixed-width table columns; MODEL_LABELS (above) is used
# everywhere else (prose lines, JSON output keys' human-readable form, etc.)
MODEL_SHORT_LABELS = {
    "t5": "t5 (ours)",
    "qwen": "qwen (ours)",
    "tinybert": "tinybert (ours)",
    "modernbert": "modernbert",
}
ALL_MODELS = tuple(MODEL_LABELS)
OUR_MODELS = ("t5", "qwen", "tinybert")

# canonical_field -> shiprocket_entity_group. Only fields with a real
# conceptual match on both sides are included; this is the intersection, not
# a forced 1:1 mapping of every field. Our two models (t5, qwen) share this
# exact field name for each canonical field, so no mapping is needed for them.
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

# Fields our models have that shiprocket's schema doesn't, and vice versa —
# reported as coverage info, never scored against a field a model doesn't
# actually attempt to predict.
OUR_ONLY_FIELDS = ("subsubLocality", "village", "subDistrict", "district")
THEIRS_ONLY_ENTITIES = ("country", "floor")

SHIPROCKET_MODEL_ID = "shiprocket-ai/open-modernbert-indian-address-ner"


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


def run_ours(addresses: list[str], backend: str) -> tuple[list[dict], float]:
    from indian_address_parser import AddressParser

    parser = AddressParser(backend=backend)
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


def run_model(name: str, addresses: list[str]) -> tuple[list[dict], float]:
    if name in OUR_MODELS:
        return run_ours(addresses, backend=name)
    if name == "modernbert":
        return run_shiprocket(addresses)
    raise ValueError(f"unknown model {name!r}")


def score(benchmark: list[dict], preds: dict[str, list[dict]]) -> dict:
    n = len(benchmark)
    stats = {field: {m: 0 for m in preds} | {"gold_present": 0} for field in FIELD_MAP}
    # "json_ok" is trivially 100% for tinybert — token classification always
    # produces a well-formed field dict, there's no JSON-parse failure mode.
    # Kept in the same overall-stats loop as t5/qwen for a uniform report
    # rather than a special-cased branch.
    overall = {m: {"exact": 0, "json_ok": 0} for m in preds if m in OUR_MODELS}

    for i, item in enumerate(benchmark):
        gold = item["gold"]
        all_match = {m: True for m in overall}

        for m in overall:
            pred = preds[m][i]
            if "_parse_error" not in pred:
                overall[m]["json_ok"] += 1

        for field, entity in FIELD_MAP.items():
            gold_val = _normalize(gold.get(field))
            if gold_val:
                stats[field]["gold_present"] += 1

            for m, model_preds in preds.items():
                pred = model_preds[i]
                pred_val = _normalize(pred.get(field if m != "modernbert" else entity))
                if gold_val:
                    if pred_val == gold_val:
                        stats[field][m] += 1
                    elif m in all_match:
                        all_match[m] = False

        for m in overall:
            if all_match[m]:
                overall[m]["exact"] += 1

    return {
        "n": n,
        "models": list(preds),
        "overall": {
            m: {
                "json_parse_rate": overall[m]["json_ok"] / n,
                "exact_match": overall[m]["exact"] / overall[m]["json_ok"] if overall[m]["json_ok"] else 0,
            }
            for m in overall
        },
        "per_field": stats,
    }


def print_report(results: dict, timings: dict[str, float], n: int):
    models = results["models"]
    print(f"\nBenchmark: {n} held-out gold-labeled addresses")
    print("(none of these models were trained on this exact held-out split)\n")

    col_w = 14
    header = f"{'Field':16s} " + " ".join(f"{MODEL_SHORT_LABELS[m]:>{col_w}s}" for m in models) + f" {'Gold presence':>14s}"
    print(header)
    print("-" * len(header))
    for field, s in results["per_field"].items():
        n_gold = s["gold_present"]
        pres = n_gold / results["n"]
        row = f"{field:16s} "
        for m in models:
            acc = s[m] / n_gold if n_gold else float("nan")
            row += f"{(f'{100*acc:.1f}%' if n_gold else 'n/a'):>{col_w}s} "
        row += f"{100*pres:>13.1f}%"
        print(row)

    print()
    for m in models:
        if m in results["overall"]:
            o = results["overall"][m]
            # tinybert (token classification) always produces a well-formed
            # dict — there's no JSON-parse failure mode, so that line is
            # skipped for it rather than printing a trivially-always-100% stat.
            if m in ("t5", "qwen"):
                print(f"{MODEL_LABELS[m]} JSON parse rate:     {100*o['json_parse_rate']:.1f}%")
            print(f"{MODEL_LABELS[m]} overall exact match: {100*o['exact_match']:.1f}% (all {len(FIELD_MAP)} shared fields correct)")
    print()
    if any(m in models for m in OUR_MODELS):
        print(f"Fields our models predict that shiprocket's schema has no equivalent for: {', '.join(OUR_ONLY_FIELDS)}")
    if "modernbert" in models:
        print(f"Entities shiprocket predicts that we don't have an equivalent field for: {', '.join(THEIRS_ONLY_ENTITIES)}")
    print()
    for m in models:
        t = timings[m]
        print(f"Inference time — {MODEL_LABELS[m]:32s} {t:.1f}s total ({1000*t/n:.0f}ms/address)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="gold_test_set.jsonl")
    p.add_argument("--n", type=int, default=None, help="Limit to N examples for a quick run")
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS),
                    help="Which models to run (default: all three)")
    p.add_argument("--out", help="Write detailed per-example results to this JSON file")
    args = p.parse_args()

    benchmark = load_benchmark(args.benchmark, args.n)
    addresses = [item["raw_address"] for item in benchmark]

    preds: dict[str, list[dict]] = {}
    timings: dict[str, float] = {}
    for m in args.models:
        print(f"Running {MODEL_LABELS[m]} on {len(addresses)} addresses...")
        preds[m], timings[m] = run_model(m, addresses)

    results = score(benchmark, preds)
    print_report(results, timings, len(benchmark))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": results,
                    "timing": {m: timings[m] for m in args.models},
                    "per_example": [
                        {"raw_address": b["raw_address"], "gold": b["gold"],
                         **{m: preds[m][i] for m in args.models}}
                        for i, b in enumerate(benchmark)
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nDetailed results written to {args.out}")


if __name__ == "__main__":
    main()
