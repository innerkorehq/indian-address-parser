"""PII redaction for bank/BC source raw addresses.

Bank/BC address records are KYC-style customer data and can embed real
customer phone numbers and relational name markers (S/O, D/O, W/O, C/O +
name — "son of"/"daughter of"/"wife of"/"care of", standard on Indian
address forms). MCA company-registration addresses also use "C/O <name>"
but there it names a company director (already public via MCA's own CIN
disclosure) — so redaction is applied to bank source only.

Patterns were derived and validated against the actual corpus, not
assumed: phone-number regex checked for false positives against pincode-
length collisions (6 digits, no overlap with 10-digit mobile pattern);
the digit-glued DO/SO/WO/CO variant (tail digits running directly into
the marker with no separator, e.g. "744302DO GURU") was found to false-
positive on "DO NO" (Door Number, not a relation marker) in 12/481 cases
and is excluded explicitly.
"""
import re

PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")
RELATION_SLASH_RE = re.compile(r"(D/O|S/O|W/O|C/O)\s*[A-Za-z][A-Za-z .]*", re.IGNORECASE)
RELATION_GLUED_RE = re.compile(r"(?<=\d)(DO|SO|WO|CO)\s+(?!NO\b)[A-Z][A-Za-z]*")


def has_pii(text: str) -> bool:
    return bool(PHONE_RE.search(text) or RELATION_SLASH_RE.search(text) or RELATION_GLUED_RE.search(text))


def redact(text: str) -> str:
    text = PHONE_RE.sub("[PHONE]", text)
    text = RELATION_SLASH_RE.sub("[REDACTED]", text)
    text = RELATION_GLUED_RE.sub("[REDACTED]", text)
    return text
