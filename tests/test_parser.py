"""Tests for indian_address_parser.

test_import_and_constants runs offline. The @pytest.mark.slow tests actually
download the model(s) from HF Hub and run real inference — slow and requires
network, but the only way to verify the package genuinely works end to end.
"""
import json

import pytest


def test_import_and_constants():
    from indian_address_parser import (
        ALL_FIELDS,
        DEFAULT_ADAPTER_REPO,
        DEFAULT_BASE_MODEL,
        DEFAULT_T5_MODEL,
        SYSTEM_PROMPT,
    )

    assert len(ALL_FIELDS) == 13
    assert "pincode" in ALL_FIELDS
    assert DEFAULT_T5_MODEL == "gagan1985/flan-t5-small-indian-address-parser"
    assert DEFAULT_ADAPTER_REPO == "gagan1985/qwen3-0.6b-indian-address-parser"
    assert DEFAULT_BASE_MODEL == "Qwen/Qwen3-0.6B"
    assert "Fields:" in SYSTEM_PROMPT


def test_invalid_backend_raises():
    from indian_address_parser import AddressParser

    with pytest.raises(ValueError):
        AddressParser(backend="not-a-real-backend")


@pytest.mark.slow
def test_parse_real_address_t5():
    from indian_address_parser import AddressParser

    parser = AddressParser()  # default backend is "t5"
    result = parser.parse("FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029")

    assert result["houseNumber"] == "FLAT NO.32"
    assert result["houseName"] == "UTTARA TOWERS"
    assert result["pincode"] == "781029"
    assert "_parse_error" not in result


@pytest.mark.slow
def test_parse_real_address_qwen():
    from indian_address_parser import AddressParser

    parser = AddressParser(backend="qwen")
    result = parser.parse("FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029")

    assert result["houseNumber"] == "FLAT NO.32"
    assert result["houseName"] == "UTTARA TOWERS"
    assert result["pincode"] == "781029"
    assert "_parse_error" not in result


@pytest.mark.slow
def test_parse_batch():
    from indian_address_parser import AddressParser

    parser = AddressParser()
    results = parser.parse_batch([
        "FLAT NO.32, UTTARA TOWERS, MG ROAD GUWAHATI , Kamrup Unclassified AS 781029",
        "SHOP NO 5, GANDHI MARKET, NEAR CIVIL HOSPITAL, SILCHAR , Cachar Unclassified AS 788001",
    ])
    assert len(results) == 2
    assert all(json.dumps(r) for r in results)  # each is JSON-serializable
