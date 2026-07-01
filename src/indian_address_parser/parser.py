"""Core AddressParser class.

Loads a LoRA adapter from Hugging Face Hub on top of a base Qwen3 model and parses
raw Indian address strings into structured fields. Model tensors are always fetched
from the Hugging Face repo (default: gagan1985/qwen3-0.6b-indian-address-parser) —
this package ships only inference code, no weights.
"""
from __future__ import annotations

import json

DEFAULT_ADAPTER_REPO = "gagan1985/qwen3-0.6b-indian-address-parser"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"

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


class AddressParser:
    """Parses raw Indian address strings into 13 structured fields.

    Example:
        >>> parser = AddressParser()
        >>> parser.parse("FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029")
        {'houseNumber': 'FLAT NO.32', 'houseName': 'UTTARA TOWERS', 'poi': None, ...}
    """

    def __init__(
        self,
        adapter_repo: str = DEFAULT_ADAPTER_REPO,
        base_model: str = DEFAULT_BASE_MODEL,
        device: str | None = None,
        max_new_tokens: int = 256,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_repo)

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

        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        self.model = PeftModel.from_pretrained(base, adapter_repo)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def parse(self, raw_address: str) -> dict:
        """Parse a single raw address string. Returns a dict with all 13 ALL_FIELDS
        keys (null for fields not present). On generation output that isn't valid
        JSON, all fields are null and `_parse_error` holds the raw model output."""
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
        generated = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        try:
            result = json.loads(generated.strip())
            result = {f: result.get(f) for f in ALL_FIELDS}
        except json.JSONDecodeError:
            result = {f: None for f in ALL_FIELDS}
            result["_parse_error"] = generated.strip()
        return result

    def parse_batch(self, addresses: list[str]) -> list[dict]:
        """Parse multiple addresses sequentially. See `parse` for the per-item shape."""
        return [self.parse(addr) for addr in addresses]
