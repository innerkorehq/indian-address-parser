# indian-address-parser

Parse raw, unstructured Indian address strings into 13 structured fields. Two model
backends, both downloaded automatically from Hugging Face — this package ships only
inference code, no weights:

- **`t5` (default)** — a full fine-tune of
  [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
  ([model card](https://huggingface.co/gagan1985/flan-t5-small-indian-address-parser)).
  ~77M params, single download, faster and lighter.
- **`qwen`** — a LoRA adapter on
  [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
  ([model card](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser)).
  ~596M params, needs both the adapter and the base model, ~2 points higher mean
  field accuracy.

```
Input:  "FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029"
Output: {"houseNumber": "FLAT NO.32", "houseName": "UTTARA TOWERS", "poi": null,
         "street": "MG ROAD", "subsubLocality": null, "subLocality": null, "locality": null,
         "village": null, "subDistrict": null, "district": "Kamrup", "city": "GUWAHATI",
         "state": "AS", "pincode": "781029"}
```

## Install

```bash
pip install indian-address-parser
```

## Usage

### Python

```python
from indian_address_parser import AddressParser

parser = AddressParser()  # default backend "t5" — downloads model weights from HF on first use
result = parser.parse("FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029")
print(result)

# Batch
results = parser.parse_batch([addr1, addr2, addr3])

# Larger, more accurate model
parser = AddressParser(backend="qwen")
```

### CLI

```bash
# Single address (defaults to the t5 backend)
indian-address-parser "FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029"

# Batch from stdin
cat addresses.txt | indian-address-parser --stdin

# Batch from a file, JSONL output
indian-address-parser --file addresses.txt --out results.jsonl

# Larger, more accurate model
indian-address-parser --backend qwen "..."
```

## Fields

```
houseNumber, houseName, poi, street, subsubLocality, subLocality,
locality, village, subDistrict, district, city, state, pincode
```

Any field not present in the address is `null`. If the model output can't be parsed as
JSON, all fields are `null` and a `_parse_error` key holds the raw model output.

## Choosing a backend

| | `t5` (default) | `qwen` |
|---|---|---|
| Base model | google/flan-t5-small | Qwen/Qwen3-0.6B + LoRA |
| Params | ~77M | ~596M |
| Download | single model | adapter + base model |
| Mean field accuracy (held-out test) | 80.6% | 82.4% |
| JSON parse rate | 100% | 100% |

`t5` is the better default for most uses — it's smaller, faster, and CPU-friendly with a
small accuracy trade-off. Reach for `qwen` when the extra couple of points matter more
than latency or footprint (its edge is largest on the lower-recall fields like `poi`,
`subDistrict`, `subLocality`).

## Model details, evaluation metrics, and known limitations

See the model cards for training data, config, full per-field evaluation results, and
known limitations (locality/subLocality/subsubLocality/village field-boundary ambiguity,
etc.):
- [gagan1985/flan-t5-small-indian-address-parser](https://huggingface.co/gagan1985/flan-t5-small-indian-address-parser) (t5, default)
- [gagan1985/qwen3-0.6b-indian-address-parser](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser) (qwen)

## Datasets

- [gagan1985/indian-addresses-gold](https://huggingface.co/datasets/gagan1985/indian-addresses-gold) — the gold-labeled training data behind both models
- [gagan1985/indian-addresses-raw](https://huggingface.co/datasets/gagan1985/indian-addresses-raw) — the 4.37M-record raw, unlabeled corpus this gold set was drawn from (PII-redacted; see the dataset card for methodology)

## Comparison to other models

[`benchmarks/`](benchmarks/) has a head-to-head comparison of both our backends
(`t5`, `qwen`) against Shiprocket's
[open-modernbert-indian-address-ner](https://huggingface.co/shiprocket-ai/open-modernbert-indian-address-ner)
on a 237-example held-out gold test set. Summary: both of our models score higher
than modernbert on every one of the 9 conceptually-shared fields, often by a wide
margin (e.g. city 91.3%/88.6% vs 4.7%, state 96.2%/95.3% vs 35.9%) — modernbert's
raw per-token predictions show genuinely low-confidence, internally inconsistent
tagging on longer administrative-suffix text, not just a benchmarking artifact.
modernbert is ~130-140x faster per address, a real architecture-driven tradeoff
(small BERT-style encoder vs. our generative models). See
[`benchmarks/README.md`](benchmarks/README.md) for the full field-by-field
results and methodology (the models use different field taxonomies, so only
overlapping fields are scored).

## Apple Silicon (MLX) users

This package uses `transformers` (+`peft` for the `qwen` backend), which works on CUDA,
MPS, and CPU but is not the fastest path on Apple Silicon. For MLX-native inference of
the `qwen` backend, see the `mlx/` subfolder of its
[Hugging Face repo](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser/tree/main/mlx)
instead.

## License

Apache 2.0 (matching the base models, google/flan-t5-small and Qwen/Qwen3-0.6B).
