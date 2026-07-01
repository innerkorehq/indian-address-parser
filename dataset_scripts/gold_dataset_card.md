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
  - 1K<n<10K
---

# Indian Addresses (Gold Standard)

Span-labeled Indian addresses — the training data behind
[gagan1985/qwen3-0.6b-indian-address-parser](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser).
For the raw, unlabeled corpus, see
[gagan1985/indian-addresses-raw](https://huggingface.co/datasets/gagan1985/indian-addresses-raw).

**4,834 records**, split by provenance:

| `reviewer` | Count | Description |
|---|---|---|
| `llm:deepseek/deepseek-v4-pro` | 4,825 | LLM-reviewed via OpenRouter, span-verified against the raw address text (a field value not found verbatim in the source string is dropped, not guessed) |
| `gagan` | 9 | Human-reviewed via Label Studio |

| `source_type` | Count |
|---|---|
| `mca` | 2,508 |
| `bank` | 2,326 |

## Fields

13 structured fields, one column each: `houseNumber`, `houseName`, `poi`,
`street`, `subsubLocality`, `subLocality`, `locality`, `village`,
`subDistrict`, `district`, `city`, `state`, `pincode` (`null`/`None` when not
present in the address).

Plus:

| Field | Description |
|---|---|
| `record_id` | Source-specific identifier |
| `source_type` | `mca` or `bank` |
| `raw_address` | The raw address string these fields were extracted from |
| `review_state` | `REVIEWED_CONFIRMED` (reviewer's spans matched the rule-based prediction exactly) or `REVIEWED_CORRECTED` (reviewer changed at least one span) |
| `reviewer` | `gagan` (human) or `llm:<model>` (LLM-reviewed) — always check this if you care about provenance |
| `reviewed_at` | ISO timestamp |
| `rule_confidence_at_review_time` | The rule-based pipeline's own confidence score for this record before review |
| `spans` | The canonical span-offset format: a list of `{"start": int, "end": int, "label": str}` dicts, character offsets into `raw_address`. The per-field columns above are derived from this — `spans` is the source of truth if the two ever appear to disagree (e.g. a field with multiple spans is joined with a space in the flat column). |

## A note on label quality

99.8% of these labels (4,825 / 4,834) come from an LLM review pass, not
human review — see the reviewer field to filter to human-only if that
matters for your use case. The LLM's own accuracy was spot-checked against
a small (9-example) human-reviewed sample before scaling up, and known
systematic gaps are documented in the
[model card](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser)'s
"known limitations" section (`locality`/`subLocality`/`subsubLocality`/`village`
field-boundary ambiguity, `street` sometimes over-absorbing multi-part
location clusters).

## PII handling

174 bank-source records that contained embedded customer PII (phone
numbers, `S/O`/`D/O`/`W/O`/`C/O` relational name markers referring to a
private individual) were **dropped** rather than redacted in place — the
`spans` offsets are character positions into `raw_address`, and redacting
text changes string length, which would silently shift every downstream
offset (some of which, e.g. `poi`, can legitimately overlap the affected
region). Dropping was the safer choice for a curated gold set; see the
[raw dataset](https://huggingface.co/datasets/gagan1985/indian-addresses-raw)
for the redaction-based approach used there instead.

MCA records containing `C/O <name>` were **not** dropped — there it names a
company director, already public via MCA's own CIN disclosure, and is often
itself the correct `poi` label.

## Related

- [Raw corpus](https://huggingface.co/datasets/gagan1985/indian-addresses-raw) — 4.37M unlabeled addresses this gold set was drawn from
- [Model](https://huggingface.co/gagan1985/qwen3-0.6b-indian-address-parser) — Qwen3-0.6B LoRA fine-tuned on this data
- [Python package](https://pypi.org/project/indian-address-parser/) — `pip install indian-address-parser`
- [GitHub](https://github.com/innerkorehq/indian-address-parser) (includes a benchmark comparison against another Indian address NER model)
