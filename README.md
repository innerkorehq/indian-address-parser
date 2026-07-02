# indian-address-parser

Parse raw, unstructured Indian address strings into 13 structured fields. Three model
backends, all downloaded automatically from Hugging Face — this package ships only
inference code, no weights:

- **`tinybert` (default)** — a full fine-tune of
  [huawei-noah/TinyBERT_General_4L_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
  ([model card](https://huggingface.co/gagan1985/tinybert-4l-312d-indian-address-parser)).
  ~14M params, token classification (BIO tagging) rather than JSON generation —
  by far the smallest and fastest option.
- **`t5`** — a full fine-tune of
  [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
  ([model card](https://huggingface.co/gagan1985/flan-t5-small-indian-address-parser)).
  ~77M params, single download, a couple points more accurate than `tinybert`, at
  noticeably higher inference cost.
- **`qwen`** — a LoRA adapter on
  [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
  ([model card](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser)).
  ~596M params, needs both the adapter and the base model, the most accurate option.

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

parser = AddressParser()  # default backend "tinybert" — downloads model weights from HF on first use
result = parser.parse("FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029")
print(result)

# Batch
results = parser.parse_batch([addr1, addr2, addr3])

# A couple points more accurate, slower
parser = AddressParser(backend="t5")

# Most accurate model
parser = AddressParser(backend="qwen")
```

### CLI

```bash
# Single address (defaults to the tinybert backend)
indian-address-parser "FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029"

# Batch from stdin
cat addresses.txt | indian-address-parser --stdin

# Batch from a file, JSONL output
indian-address-parser --file addresses.txt --out results.jsonl

# A couple points more accurate, slower
indian-address-parser --backend t5 "..."

# Most accurate model
indian-address-parser --backend qwen "..."
```

## Fields

```
houseNumber, houseName, poi, street, subsubLocality, subLocality,
locality, village, subDistrict, district, city, state, pincode
```

Any field not present in the address is `null`. For the generative backends (`t5`,
`qwen`), output that can't be parsed as JSON leaves all fields `null` and a
`_parse_error` key holds the raw model output; `tinybert` is token classification, not
generation, so it always produces a well-formed dict and `_parse_error` never applies.

## Choosing a backend

| | `tinybert` (default) | `t5` | `qwen` |
|---|---|---|---|
| Base model | TinyBERT_General_4L_312D | google/flan-t5-small | Qwen/Qwen3-0.6B + LoRA |
| Task | token classification (BIO) | JSON generation | JSON generation |
| Params | ~14M | ~77M | ~596M |
| Download | single model | single model | adapter + base model |
| Mean field accuracy (held-out test) | 78.8% | 80.6% | 82.4% |

`tinybert` is the default — at ~14M params (5x smaller than `t5`, 40x smaller than
`qwen`) it's by far the cheapest to download and run, a single forward pass instead of
autoregressive generation, and still lands within 2-4 points of both larger models on
mean field accuracy (though its `village`/`subLocality` recall is essentially 0% — see
its model card for the full breakdown). Reach for `t5` when those couple of points
matter and the extra inference cost is acceptable, or `qwen` when accuracy matters most
(its edge is largest on the lower-recall fields like `poi`, `subDistrict`, `subLocality`).

## Model details, evaluation metrics, and known limitations

See the model cards for training data, config, full per-field evaluation results, and
known limitations (locality/subLocality/subsubLocality/village field-boundary ambiguity,
etc.):
- [gagan1985/tinybert-4l-312d-indian-address-parser](https://huggingface.co/gagan1985/tinybert-4l-312d-indian-address-parser) (tinybert, default)
- [gagan1985/flan-t5-small-indian-address-parser](https://huggingface.co/gagan1985/flan-t5-small-indian-address-parser) (t5)
- [gagan1985/qwen3-0.6b-indian-address-parser](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser) (qwen)

## Datasets

- [gagan1985/indian-addresses-gold](https://huggingface.co/datasets/gagan1985/indian-addresses-gold) — the gold-labeled training data behind all three models
- [gagan1985/indian-addresses-raw](https://huggingface.co/datasets/gagan1985/indian-addresses-raw) — the 4.37M-record raw, unlabeled corpus this gold set was drawn from (PII-redacted; see the dataset card for methodology)

## Comparison to other models

[`benchmarks/`](benchmarks/) has a head-to-head comparison of all three of our backends
against Shiprocket's
[open-modernbert-indian-address-ner](https://huggingface.co/shiprocket-ai/open-modernbert-indian-address-ner)
on a 237-example held-out gold test set. See
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

Apache 2.0 (matching the base models: google/flan-t5-small, Qwen/Qwen3-0.6B, huawei-noah/TinyBERT_General_4L_312D).
