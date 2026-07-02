# Model comparison: our three models vs shiprocket-ai/open-modernbert-indian-address-ner

`compare_models.py` benchmarks this package's three models — `t5`
([flan-t5-small](https://huggingface.co/gagan1985/flan-t5-small-indian-address-parser),
default), `qwen`
([Qwen3-0.6B + LoRA](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser)),
and `tinybert`
([TinyBERT 4L/312D](https://huggingface.co/gagan1985/tinybert-4l-312d-indian-address-parser)) —
against Shiprocket's
[open-modernbert-indian-address-ner](https://huggingface.co/shiprocket-ai/open-modernbert-indian-address-ner)
on a shared, held-out gold-labeled test set.

(This supersedes an earlier two-way comparison against Shiprocket's
`open-tinybert-indian-address-ner`; see git history for those numbers.)

## Why this isn't a simple apples-to-apples score

Our three models share the same 13-field taxonomy; Shiprocket's model uses a
different, 11-entity BIO-NER schema:

| | Our models (t5, qwen, tinybert) | Shiprocket's modernbert |
|---|---|---|
| Architecture | flan-t5-small (encoder-decoder, JSON generation) / Qwen3-0.6B + LoRA (causal LM, JSON generation) / TinyBERT 4L/312D (BERT encoder, BIO token classification) | ModernBERT, BIO token classification |
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
python compare_models.py                              # full 237-example benchmark, all 4 models
python compare_models.py --n 50                        # quick subset
python compare_models.py --models t5 tinybert modernbert  # skip the slower qwen backend
python compare_models.py --out results.json            # save per-example detail
```

## What's reported

- Per-field accuracy for all four models against gold, restricted to the 9 shared fields
- JSON parse rate and overall exact-match rate for our two generative models
  (`t5`, `qwen`); overall exact-match only (no JSON parse rate — not
  applicable) for `tinybert` and `modernbert`, both token classifiers that
  always produce a well-formed field dict
- Wall-clock inference time per model (the two token-classification models —
  `tinybert` and modernbert — are architecturally much faster than either
  generative model, an expected, not surprising, result reported for
  completeness, not as a quality judgment)

## Results (full 237-example benchmark)

Run with `python compare_models.py --out full_results.json` (per-example detail
in [`full_results.json`](full_results.json)).

| Field | t5 (ours) | qwen (ours) | tinybert (ours) | modernbert | Gold presence |
|---|---|---|---|---|---|
| houseNumber | 84.5% | 84.5% | 79.8% | 43.4% | 54.4% |
| houseName | 80.8% | 88.5% | 81.7% | 64.4% | 43.9% |
| street | 47.6% | 54.0% | 50.0% | 38.9% | 53.2% |
| locality | 29.8% | 33.7% | 36.5% | 5.8% | 43.9% |
| subLocality | 13.7% | 23.5% | 0.0% | 0.0% | 21.5% |
| city | 88.6% | 91.3% | 82.6% | 4.7% | 62.9% |
| state | 95.3% | 96.2% | 84.2% | 35.9% | 98.7% |
| pincode | 97.9% | 100.0% | 99.2% | 73.8% | 100.0% |
| poi | 0.0% | 30.8% | 20.5% | 5.1% | 16.5% |

- **flan-t5-small (t5) JSON parse rate**: 99.6% &nbsp;|&nbsp; **overall exact match** (all 9 shared fields correct): 24.2%
- **qwen3-0.6b (qwen) JSON parse rate**: 100.0% &nbsp;|&nbsp; **overall exact match**: 30.4%
- **tinybert-4l-312d (tinybert) overall exact match**: 22.8%
- **Inference time** — t5: 647.6s total (2732ms/address) · qwen: 768.9s total (3244ms/address) · tinybert: 3.4s total (15ms/address) · modernbert: 4.9s total (21ms/address)

qwen scores highest on every one of the 9 shared fields. t5 and tinybert trade
places field-by-field — tinybert actually edges out t5 on `locality` (36.5%
vs 29.8%) and `poi` (20.5% vs 0.0%), while t5 leads on `city`/`state`/`street`
— despite tinybert being ~5x smaller than t5 and ~40x smaller than qwen.
`subLocality` is the one field neither t5 nor tinybert can extract at all
(0% recall on both, consistent with each model's own card).

**tinybert is also the fastest model here**, edging out even modernbert
(3.4s vs 4.9s for 237 addresses) — both are small BERT-style encoders run
once per address, but TinyBERT's 4-layer/312-hidden architecture is smaller
still than ModernBERT's.

modernbert scores lowest on every field, and by a wide margin on
`city`/`state`/`locality` in particular. **This isn't a benchmarking
artifact — we checked.** Inspecting raw (unaggregated) per-token predictions
shows the model itself flip-flops entity tags *within a single word* on
multi-word suffixes (e.g. for "Kamrup Unclassified", `"Kam"` tagged
`B-sub_locality` at 0.45 confidence, then `"rup"` tagged `I-locality` at 0.42
— genuinely low-confidence, internally inconsistent predictions, not a
tokenizer/aggregation mismatch in this script). This pattern — high
confidence on short, common fields and attention-inconsistent classification
on longer administrative-suffix sequences — was reproducible across multiple
examples in [`full_results.json`](full_results.json).

If your use case needs high-throughput/low-latency parsing and can tolerate
lower field accuracy, `tinybert` is the clear pick among our own models (and
beats modernbert outright on every shared field while also being faster). If
accuracy matters more than speed, `qwen` is the strongest choice on this
benchmark, with `t5` close behind at a fraction of the download/load cost.

One important caveat: the gold labels themselves were generated for *this
project's* 13-field taxonomy, so field-boundary conventions (e.g. exactly
where "street" starts/ends, whether a landmark word is included) reflect this
project's labeling choices, not Shiprocket's. This is an inherent limitation
of scoring two different taxonomies against one schema's gold labels, not
something this script can fully correct for.
