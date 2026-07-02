"""Core AddressParser class.

Three backends, all fetched from Hugging Face Hub — this package ships only
inference code, no weights:

- "t5" (default): a full fine-tune of google/flan-t5-small
  (gagan1985/flan-t5-small-indian-address-parser). ~77M params, encoder-decoder,
  single model download, no adapter/base-model split. Faster and lighter; a couple
  points lower accuracy on rare/ambiguous fields.
- "qwen": a LoRA adapter on Qwen/Qwen3-0.6B
  (gagan1985/qwen3-0.6b-indian-address-parser). ~596M params, causal LM, needs
  both the adapter and the base model. The most accurate option.
- "tinybert": a full fine-tune of huawei-noah/TinyBERT_General_4L_312D
  (gagan1985/tinybert-4l-312d-indian-address-parser). ~14M params, BERT-style
  encoder doing token classification (BIO tagging) rather than JSON
  generation — by far the smallest and fastest option, with lower field
  accuracy than either generative model, particularly on longer/rarer fields.
"""
from __future__ import annotations

import json
import os

# This package only needs the PyTorch backend. Without this, `transformers`/`peft`
# can pull in a TensorFlow import chain (transformers.models.bloom -> ... ->
# image_transforms.py -> `import tensorflow`) purely as a side effect of module
# loading — and in environments where TF's own numpy/h5py binary versions don't
# match (a common conda/Anaconda footgun, unrelated to this package), that import
# fails with things like "numpy.dtype size changed, may indicate binary
# incompatibility" and crashes AddressParser() before it ever touches TensorFlow
# functionality. Setting these before transformers/peft are imported anywhere
# short-circuits transformers' TF detection so that chain is never triggered.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

DEFAULT_T5_MODEL = "gagan1985/flan-t5-small-indian-address-parser"
DEFAULT_ADAPTER_REPO = "gagan1985/qwen3-0.6b-indian-address-parser"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_TINYBERT_MODEL = "gagan1985/tinybert-4l-312d-indian-address-parser"

BACKENDS = ("t5", "qwen", "tinybert")

ALL_FIELDS = (
    "houseNumber",
    "houseName",
    "poi",
    "street",
    "subsubLocality",
    "subLocality",
    "locality",
    "village",
    "subDistrict",
    "district",
    "city",
    "state",
    "pincode",
)

SYSTEM_PROMPT = (
    "You are an Indian address parser. Given a raw address string, extract address "
    "fields and return them as a JSON object. Use null for fields not present in the "
    "address. Output only the JSON object, no explanation.\n\n"
    "Fields: " + ", ".join(ALL_FIELDS)
)

USER_TEMPLATE = "Parse this address:\n{raw_address}"

# T5 was pretrained on task-prefixed inputs; this is the exact prefix the
# gagan1985/flan-t5-small-indian-address-parser model was fine-tuned with.
T5_TASK_PREFIX = "parse indian address: "
T5_MAX_INPUT_LENGTH = 160

TINYBERT_MAX_LENGTH = 160
TINYBERT_LABELS = ["O"] + [f"{prefix}-{field}" for field in ALL_FIELDS for prefix in ("B", "I")]
TINYBERT_ID2LABEL = {i: label for i, label in enumerate(TINYBERT_LABELS)}


def _extract_fields_from_bio(raw_address: str, offsets: list[tuple[int, int]], pred_label_ids: list[int]) -> dict:
    """Reconstruct the ALL_FIELDS dict from per-token BIO predictions by
    slicing the raw address text at each contiguous B-/I- run's char span —
    never `tokenizer.decode`, which would lose casing and introduce
    WordPiece-merge artifacts (e.g. "##" continuation glue)."""
    result = {f: None for f in ALL_FIELDS}
    current_field = None
    current_start = None
    current_end = None

    def flush():
        if current_field is not None and result[current_field] is None:
            result[current_field] = raw_address[current_start:current_end]

    for (start, end), label_id in zip(offsets, pred_label_ids):
        if start == end:
            continue
        label = TINYBERT_ID2LABEL[label_id]
        if label == "O":
            flush()
            current_field = None
            continue
        prefix, field = label.split("-", 1)
        if prefix == "B" or field != current_field:
            flush()
            current_field, current_start, current_end = field, start, end
        else:
            current_end = end
    flush()
    return result


def _pick_device_and_dtype(device: str | None):
    import torch

    dtype = torch.float32
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
    return device, dtype


class AddressParser:
    """Parses raw Indian address strings into 13 structured fields.

    Example:
        >>> parser = AddressParser()  # defaults to the flan-t5-small backend
        >>> parser.parse("FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029")
        {'houseNumber': 'FLAT NO.32', 'houseName': 'UTTARA TOWERS', 'poi': None, ...}

        >>> parser = AddressParser(backend="qwen")  # most accurate, LoRA model
        >>> parser = AddressParser(backend="tinybert")  # smallest, fastest model
        >>> parser.parse("...")
    """

    def __init__(
        self,
        backend: str = "t5",
        model_repo: str | None = None,
        adapter_repo: str | None = None,
        base_model: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        self.max_new_tokens = max_new_tokens

        if backend == "t5":
            self._init_t5(model_repo or DEFAULT_T5_MODEL, device)
        elif backend == "qwen":
            self._init_qwen(adapter_repo or DEFAULT_ADAPTER_REPO, base_model or DEFAULT_BASE_MODEL, device)
        else:
            self._init_tinybert(model_repo or DEFAULT_TINYBERT_MODEL, device)

    def _init_t5(self, model_repo: str, device: str | None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device, dtype = _pick_device_and_dtype(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_repo, dtype=dtype)
        except TypeError:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_repo, torch_dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def _init_qwen(self, adapter_repo: str, base_model: str, device: str | None):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, dtype = _pick_device_and_dtype(device)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_repo)

        # `torch_dtype` is deprecated in favor of `dtype` in newer transformers, but
        # `dtype` doesn't exist yet on older ones (verified: fails on 4.51.0) — try the
        # new name first, fall back to the old one, so this doesn't break either way
        # as transformers eventually drops the deprecated kwarg entirely.
        try:
            base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        self.model = PeftModel.from_pretrained(base, adapter_repo)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def _init_tinybert(self, model_repo: str, device: str | None):
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        device, dtype = _pick_device_and_dtype(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self.model = AutoModelForTokenClassification.from_pretrained(model_repo)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def parse(self, raw_address: str) -> dict:
        """Parse a single raw address string. Returns a dict with all 13 ALL_FIELDS
        keys (null for fields not present). For the generative backends (t5, qwen),
        output that isn't valid JSON leaves all fields null and sets `_parse_error`
        to the raw model output; the tinybert backend (token classification) always
        produces a well-formed dict, so `_parse_error` never applies to it."""
        if self.backend == "tinybert":
            return self._predict_tinybert(raw_address)

        generated = self._generate_t5(raw_address) if self.backend == "t5" else self._generate_qwen(raw_address)
        try:
            result = json.loads(generated.strip())
            result = {f: result.get(f) for f in ALL_FIELDS}
        except json.JSONDecodeError:
            result = {f: None for f in ALL_FIELDS}
            result["_parse_error"] = generated.strip()
        return result

    def _generate_t5(self, raw_address: str) -> str:
        import torch

        input_text = T5_TASK_PREFIX + raw_address
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=T5_MAX_INPUT_LENGTH
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, num_beams=1, do_sample=False
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _generate_qwen(self, raw_address: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(raw_address=raw_address)},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _predict_tinybert(self, raw_address: str) -> dict:
        import torch

        enc = self.tokenizer(
            raw_address, return_tensors="pt", return_offsets_mapping=True,
            truncation=True, max_length=TINYBERT_MAX_LENGTH,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = enc.to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        pred_ids = out.logits[0].argmax(-1).tolist()
        return _extract_fields_from_bio(raw_address, offsets, pred_ids)

    def parse_batch(self, addresses: list[str]) -> list[dict]:
        """Parse multiple addresses sequentially. See `parse` for the per-item shape."""
        return [self.parse(addr) for addr in addresses]
