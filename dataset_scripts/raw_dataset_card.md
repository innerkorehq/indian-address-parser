---
license: apache-2.0
task_categories:
  - token-classification
  - text-generation
language:
  - en
tags:
  - address-parsing
  - india
  - ner
size_categories:
  - 1M<n<10M
---

# Indian Addresses (Raw)

Raw, unstructured Indian address strings — the unlabeled corpus behind
[gagan1985/qwen3-0.6b-indian-address-parser](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser).
For the labeled/parsed version, see
[gagan1985/indian-addresses-gold](https://huggingface.co/datasets/gagan1985/indian-addresses-gold).

## Sources

Two raw source types, distinguished by `source_type`:
- **`mca`** — Indian Ministry of Corporate Affairs company-registration addresses
  (public disclosure data; CIN numbers are part of India's public company registry)
- **`bank`** — bank/business-correspondent branch address records

## Fields

| Field | Description |
|---|---|
| `record_id` | Source-specific identifier (CIN for MCA, an internal id for bank) |
| `source_type` | `mca` or `bank` |
| `raw_address` | The raw, unparsed address string |
| `lifecycle_state` | Where this record ended up in the labeling pipeline: `AUTO_ACCEPTED` (rule-based high confidence), `PENDING_REVIEW` (not yet gold-labeled), `QUARANTINED` (unparseable — no pincode found), `REVIEWED_CORRECTED` (has a gold label — see the gold dataset) |
| `aux_state` | Declared state, when available from the source record itself |
| `aux_pincode_declared` | Declared pincode, when available separately from `raw_address` |
| `aux_lat` / `aux_lon` | Latitude/longitude of the branch/BC center (bank source only) |
| `source_metadata` | Free-text provenance string (e.g. `"MCA \| CIN: ... \| RoC: ..."` or `"Bank \| CANARA BANK \| type: BC \| center: ..."`) |

## PII redaction

Bank/BC address records are KYC-style customer data and can embed real
customer phone numbers and relational name markers (`S/O`/`D/O`/`W/O`/`C/O` +
name — "son of"/"daughter of"/"wife of"/"care of", standard on Indian
address forms). These are redacted (`[PHONE]`, `[REDACTED]`) in the `bank`
source before publication.

MCA source is left untouched: its own `C/O <name>` convention names a
**company director**, already public information via MCA's CIN disclosure —
categorically different from a bank customer's personal address.

This redaction targets the concrete patterns found in this specific corpus
(validated against real examples, including a false-positive check for
"Door No." colliding with the relational-marker pattern) — it is not a
general-purpose PII scrubber and should not be assumed to catch every
possible embedded personal detail.

## Related

- [Gold-labeled dataset](https://huggingface.co/datasets/gagan1985/indian-addresses-gold) — 4,834 span-labeled examples derived from this corpus
- [Model](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser) — Qwen3-0.6B LoRA fine-tuned on the gold dataset
- [Python package](https://pypi.org/project/indian-address-parser/) — `pip install indian-address-parser`
- [GitHub](https://github.com/innerkorehq/indian-address-parser)
