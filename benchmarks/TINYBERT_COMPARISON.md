# TinyBERT vs TinyBERT: our tinybert-4l-312d vs shiprocket-ai/open-tinybert-indian-address-ner

`compare_tinybert.py` is a separate, narrower comparison from
[`compare_models.py`](compare_models.py)'s 4-way benchmark — a same-name
matchup between our `tinybert` backend
([gagan1985/tinybert-4l-312d-indian-address-parser](https://huggingface.co/gagan1985/tinybert-4l-312d-indian-address-parser))
and Shiprocket's
[open-tinybert-indian-address-ner](https://huggingface.co/shiprocket-ai/open-tinybert-indian-address-ner),
on the same 237-example held-out gold test set
([`gold_test_set.jsonl`](gold_test_set.jsonl)) used throughout this repo.

## Not actually the same size, despite the shared name

Both models are called "TinyBERT," but they are **not** the same architecture:

| | Ours | Shiprocket's |
|---|---|---|
| Base | huawei-noah/TinyBERT_General_4L_312D | a larger BERT variant, also branded "tinybert" |
| Layers | 4 | 6 |
| Hidden size | 312 | 768 |
| Params | ~14M | ~66.4M (verified via `sum(p.numel() for p in model.parameters())`) |

Shiprocket's "tinybert" has **~4.7x more parameters** than ours. This is
reported explicitly up front rather than left implicit, since a same-name
comparison could otherwise look more like-for-like than it actually is.

Both models share the same task (BIO token classification) and the same
11-entity schema Shiprocket uses across their model family (verified:
identical `id2label` to their modernbert model) — see
[`compare_models.py`](compare_models.py)'s docstring for why only the 9
fields with a clear conceptual match to our 13-field taxonomy are scored.

## Results (full 237-example benchmark)

Run with `python compare_tinybert.py --out tinybert_vs_shiprocket_results.json`
(per-example detail in [`tinybert_vs_shiprocket_results.json`](tinybert_vs_shiprocket_results.json)).

| Field | ours (4L/312D, ~14M) | shiprocket (6L/768D, ~66.4M) | Gold presence |
|---|---|---|---|
| houseNumber | 79.8% | 27.1% | 54.4% |
| houseName | 81.7% | 72.1% | 43.9% |
| street | 50.0% | 27.0% | 53.2% |
| locality | 36.5% | 6.7% | 43.9% |
| subLocality | 0.0% | 0.0% | 21.5% |
| city | 82.6% | 17.4% | 62.9% |
| state | 84.2% | 41.5% | 98.7% |
| pincode | 99.2% | 69.2% | 100.0% |
| poi | 20.5% | 10.3% | 16.5% |

- **Our overall exact match** (all 9 shared fields correct): 22.8%
- **Inference time** — ours: 2.7s total (11ms/address) · shiprocket: 3.8s total (16ms/address)

**Our model wins on every one of the 9 shared fields** — some by a wide
margin (city 82.6% vs 17.4%, houseNumber 79.8% vs 27.1%, state 84.2% vs
41.5%) — despite being ~4.7x smaller, and it's also faster per address
(11ms vs 16ms). `subLocality` is the one field neither model extracts at
all (0% on both).

This is a meaningfully different result from the
[modernbert comparison](README.md#results-full-237-example-benchmark), where
our tinybert only *edged out* modernbert on some fields. Here it dominates
outright — plausibly because fine-tuning on this project's own gold-labeled
data (verbatim-extraction, 13-field taxonomy, ~4,110 training examples) is
a bigger factor than the ~4.7x parameter gap, at least on this benchmark's
distribution of addresses.

## Known limitations of this comparison

Same caveat as the rest of `benchmarks/`: gold labels were generated for
*this project's* 13-field taxonomy, so field-boundary conventions reflect
this project's labeling choices, not Shiprocket's. Both models were
evaluated cold — Shiprocket's model was never fine-tuned on this project's
data, so this is a comparison of "the model as published" against "our model
fine-tuned for this exact task," not a controlled ablation isolating
architecture size alone.
