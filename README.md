# indian-address-parser

Parse raw, unstructured Indian address strings into 13 structured fields using a
Qwen3-0.6B model fine-tuned with LoRA. Model weights are downloaded automatically from
[Hugging Face](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser) — this
package ships only inference code, no weights.

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

parser = AddressParser()  # downloads model weights from HF on first use
result = parser.parse("FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029")
print(result)

# Batch
results = parser.parse_batch([addr1, addr2, addr3])
```

### CLI

```bash
# Single address
indian-address-parser "FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029"

# Batch from stdin
cat addresses.txt | indian-address-parser --stdin

# Batch from a file, JSONL output
indian-address-parser --file addresses.txt --out results.jsonl
```

## Fields

```
houseNumber, houseName, poi, street, subsubLocality, subLocality,
locality, village, subDistrict, district, city, state, pincode
```

Any field not present in the address is `null`. If the model output can't be parsed as
JSON, all fields are `null` and a `_parse_error` key holds the raw model output.

## Model details, evaluation metrics, and known limitations

See the [model card](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser)
for training data, LoRA config, per-field evaluation results (100% JSON parse rate, 82.4%
mean field accuracy on held-out test data), and known limitations (locality/subLocality/
subsubLocality/village field-boundary ambiguity, etc.).

## Datasets

- [gagan1985/indian-addresses-gold](https://huggingface.co/datasets/gagan1985/indian-addresses-gold) — the 4,834-example span-labeled training data behind this model
- [gagan1985/indian-addresses-raw](https://huggingface.co/datasets/gagan1985/indian-addresses-raw) — the 4.37M-record raw, unlabeled corpus this gold set was drawn from (PII-redacted; see the dataset card for methodology)

## Comparison to other models

[`benchmarks/`](benchmarks/) has a head-to-head comparison against Shiprocket's
[open-tinybert-indian-address-ner](https://huggingface.co/shiprocket-ai/open-tinybert-indian-address-ner)
on a 237-example held-out gold test set. Summary: this model scores higher on
every one of the 9 conceptually-shared fields (sometimes by a wide margin —
e.g. city 91.3% vs 17.4%, pincode 100% vs 69.2%), while Shiprocket's 6-layer
TinyBERT is ~240x faster per address. See [`benchmarks/README.md`](benchmarks/README.md)
for the full field-by-field results and methodology (the two models use
different field taxonomies, so only overlapping fields are scored).

## Apple Silicon (MLX) users

This package uses `transformers`+`peft`, which works on CUDA, MPS, and CPU but is not the
fastest path on Apple Silicon. For MLX-native inference, see the `mlx/` subfolder of the
[Hugging Face repo](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser/tree/main/mlx)
instead.

## License

Apache 2.0 (matching the base model, Qwen/Qwen3-0.6B).
