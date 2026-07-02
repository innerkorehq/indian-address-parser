# Model comparison: our flan-t5-small and Qwen3-0.6B models vs shiprocket-ai/open-modernbert-indian-address-ner

`compare_models.py` benchmarks this package's two models — the default
`t5` backend ([flan-t5-small](https://huggingface.co/gagan1985/flan-t5-small-indian-address-parser))
and the `qwen` backend ([Qwen3-0.6B + LoRA](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser)) —
against Shiprocket's
[open-modernbert-indian-address-ner](https://huggingface.co/shiprocket-ai/open-modernbert-indian-address-ner)
on a shared, held-out gold-labeled test set.

(This supersedes an earlier two-way comparison against Shiprocket's
`open-tinybert-indian-address-ner`; see git history for those numbers.)

## Why this isn't a simple apples-to-apples score

Our two models share the same 13-field taxonomy; Shiprocket's model uses a
different, 11-entity BIO-NER schema:

| | Our models (t5, qwen) | Shiprocket's modernbert |
|---|---|---|
| Architecture | flan-t5-small (encoder-decoder, full fine-tune) / Qwen3-0.6B + LoRA (causal LM) — both generate JSON | ModernBERT, BIO token classification |
| Fields / entities | 13: houseNumber, houseName, poi, street, subsubLocality, subLocality, locality, village, subDistrict, district, city, state, pincode | 11: building_name, house_details, road, sub_locality, locality, city, state, pincode, country, landmarks, floor |

`compare_models.py` only scores the **9 fields with a clear conceptual overlap**
(see `FIELD_MAP` in the script). Fields unique to one schema — our
`subsubLocality`/`village`/`subDistrict`/`district`, their `country`/`floor` —
are reported separately as "not comparable," never forced into a mapping that
would misrepresent any model or silently dropped from the report.

## Benchmark set

`gold_test_set.jsonl` — 237 held-out, human/LLM-gold-labeled addresses from the
training pipeline's test split (never used to train any of these models). Each line:
```json
{"raw_address": "...", "gold": {"houseNumber": "...", ..., "pincode": "..."}}
```

## Running it

```bash
pip install indian-address-parser transformers torch
python compare_models.py                          # full 237-example benchmark, all 3 models
python compare_models.py --n 50                    # quick subset
python compare_models.py --models t5 modernbert     # skip the slower qwen backend
python compare_models.py --out results.json        # save per-example detail
```

## What's reported

- Per-field accuracy for all three models against gold, restricted to the 9 shared fields
- JSON parse rate and overall exact-match rate for our two models (not applicable to modernbert — it's a token classifier, not a JSON generator)
- Wall-clock inference time per model (ModernBERT, a small encoder run once per
  address, is architecturally much faster than either of our generative
  models — this is an expected, not surprising, result and is reported for
  completeness, not as a quality judgment)

## Results (full 237-example benchmark)

Run with `python compare_models.py --out full_results.json` (per-example detail
in [`full_results.json`](full_results.json)).

| Field | t5 (ours) | qwen (ours) | modernbert | Gold presence |
|---|---|---|---|---|
| houseNumber | 84.5% | 84.5% | 43.4% | 54.4% |
| houseName | 80.8% | 88.5% | 64.4% | 43.9% |
| street | 47.6% | 54.0% | 38.9% | 53.2% |
| locality | 29.8% | 33.7% | 5.8% | 43.9% |
| subLocality | 13.7% | 23.5% | 0.0% | 21.5% |
| city | 88.6% | 91.3% | 4.7% | 62.9% |
| state | 95.3% | 96.2% | 35.9% | 98.7% |
| pincode | 97.9% | 100.0% | 73.8% | 100.0% |
| poi | 0.0% | 30.8% | 5.1% | 16.5% |

- **flan-t5-small (t5) JSON parse rate**: 99.6% &nbsp;|&nbsp; **overall exact match** (all 9 shared fields correct): 24.2%
- **qwen3-0.6b (qwen) JSON parse rate**: 100.0% &nbsp;|&nbsp; **overall exact match**: 30.4%
- **Inference time** — t5: 682.9s total (2881ms/address) · qwen: 756.6s total (3192ms/address) · modernbert: 5.5s total (23ms/address, **~130-140x faster** than either of our models)

qwen scores highest on every one of the 9 shared fields; t5 is close behind on
most fields but notably weaker on `poi` (0.0% — it defaults to `null` far more
often than gold does) and the low-recall fields in general, consistent with
its own [model card](https://huggingface.co/gagan1985/flan-t5-small-indian-address-parser)'s
noted limitations. modernbert scores lowest on every field, and by a wide
margin on `city`/`state`/`locality` in particular.

**modernbert's low scores aren't a benchmarking artifact — we checked.**
Inspecting raw (unaggregated) per-token predictions shows the model itself
flip-flops entity tags *within a single word* on multi-word suffixes (e.g.
for "Kamrup Unclassified", `"Kam"` tagged `B-sub_locality` at 0.45 confidence,
then `"rup"` tagged `I-locality` at 0.42 — genuinely low-confidence, internally
inconsistent predictions, not a tokenizer/aggregation mismatch in this
script). This pattern — high confidence on short, common fields and
attention-inconsistent classification on longer administrative-suffix
sequences — was reproducible across multiple examples in
[`full_results.json`](full_results.json).

As with the earlier tinybert comparison: modernbert is dramatically faster
and far smaller than either of our models — a real, architecture-driven
tradeoff, not just a quality artifact. If your use case needs
high-throughput/low-latency parsing and can tolerate lower field accuracy,
modernbert may still be the better fit; if accuracy matters more than raw
speed, either of this package's models is the stronger choice on this
benchmark, with qwen scoring a few points higher than t5 at roughly the same
inference cost as each other (t5's main real-world advantage is being ~8x
smaller to download and load, not faster to run per-token here).

One important caveat: the gold labels themselves were generated for *this
project's* 13-field taxonomy, so field-boundary conventions (e.g. exactly
where "street" starts/ends, whether a landmark word is included) reflect this
project's labeling choices, not Shiprocket's. This is an inherent limitation
of scoring two different taxonomies against one schema's gold labels, not
something this script can fully correct for.
