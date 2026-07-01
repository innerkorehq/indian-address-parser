# Model comparison: gagan1985/qwen3-0.6b-indian-address-parser vs shiprocket-ai/open-tinybert-indian-address-ner

`compare_models.py` benchmarks this package's model against Shiprocket's
[open-tinybert-indian-address-ner](https://huggingface.co/shiprocket-ai/open-tinybert-indian-address-ner)
on a shared, held-out gold-labeled test set.

## Why this isn't a simple apples-to-apples score

The two models use **different field taxonomies** and different architectures:

| | This model | Shiprocket's model |
|---|---|---|
| Architecture | Qwen3-0.6B + LoRA, causal LM generating JSON | 6-layer TinyBERT, BIO token classification |
| Fields / entities | 13: houseNumber, houseName, poi, street, subsubLocality, subLocality, locality, village, subDistrict, district, city, state, pincode | 11: building_name, house_details, road, sub_locality, locality, city, state, pincode, country, landmarks, floor |

`compare_models.py` only scores the **9 fields with a clear conceptual overlap**
(see `FIELD_MAP` in the script). Fields unique to one schema — our
`subsubLocality`/`village`/`subDistrict`/`district`, their `country`/`floor` —
are reported separately as "not comparable," never forced into a mapping that
would misrepresent either model or silently dropped from the report.

## Benchmark set

`gold_test_set.jsonl` — 237 held-out, human/LLM-gold-labeled addresses from the
training pipeline's test split (never used to train either model). Each line:
```json
{"raw_address": "...", "gold": {"houseNumber": "...", ..., "pincode": "..."}}
```

## Running it

```bash
pip install indian-address-parser transformers torch
python compare_models.py                    # full 237-example benchmark
python compare_models.py --n 50             # quick subset
python compare_models.py --out results.json # save per-example detail
```

## What's reported

- Per-field accuracy for both models against gold, restricted to the 9 shared fields
- Our JSON parse rate and overall exact-match rate
- Wall-clock inference time per model (a 6-layer TinyBERT classifier is
  architecturally much faster than a 0.6B causal LM doing autoregressive
  generation — this is an expected, not surprising, result and is reported
  for completeness, not as a quality judgment)

## Results (full 237-example benchmark)

Run with `python compare_models.py --out full_results.json` (per-example detail
in [`full_results.json`](full_results.json)).

| Field | Ours (acc) | Shiprocket (acc) | Gold presence |
|---|---|---|---|
| houseNumber | 84.5% | 27.1% | 54.4% |
| houseName | 88.5% | 72.1% | 43.9% |
| street | 54.0% | 27.0% | 53.2% |
| locality | 34.6% | 6.7% | 43.9% |
| subLocality | 23.5% | 0.0% | 21.5% |
| city | 91.3% | 17.4% | 62.9% |
| state | 96.2% | 41.5% | 98.7% |
| pincode | 100.0% | 69.2% | 100.0% |
| poi | 30.8% | 10.3% | 16.5% |

- **Our JSON parse rate**: 100.0%
- **Our overall exact match** (all 9 shared fields correct): 30.8%
- **Inference time — ours**: 1087.4s total, 4588ms/address
- **Inference time — shiprocket**: 4.6s total, 19ms/address (**~240x faster**)

Ours scores higher on every one of the 9 shared fields, in some cases by a wide
margin (city, state, pincode, houseNumber). Shiprocket's model is dramatically
faster and far smaller (6-layer TinyBERT vs 0.6B causal LM) — a real,
architecture-driven tradeoff, not a quality artifact. If your use case needs
high-throughput/low-latency parsing and can tolerate lower field accuracy on
the harder fields (locality/subLocality/poi/street), Shiprocket's model may
still be the better fit; if accuracy matters more than raw speed, this
package's model is the stronger choice on this benchmark.

One important caveat: the gold labels themselves were generated for *this
project's* 13-field taxonomy, so field-boundary conventions (e.g. exactly
where "street" starts/ends, whether a landmark word is included) reflect this
project's labeling choices, not Shiprocket's. This is an inherent limitation
of scoring two different taxonomies against one schema's gold labels, not
something this script can fully correct for.
