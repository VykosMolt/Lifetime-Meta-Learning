"""Hashing, seed derivation, sealing and the leak invariant (contract §3)."""
from __future__ import annotations

import hashlib
import os

import pytest

from foundation_learner.ecology import base


def test_canonical_json_is_sorted_and_minimal():
    assert base.canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'
    with pytest.raises(ValueError):
        base.canonical_json({"x": float("nan")})


def test_domain_sha256_domains_cannot_collide():
    obj = {"x": 1}
    assert base.domain_sha256("A", obj) != base.domain_sha256("B", obj)
    expected = hashlib.sha256(
        b"A\0" + base.canonical_json(obj).encode("utf-8")).hexdigest()
    assert base.domain_sha256("A", obj) == expected


def test_derive_seed_matches_the_frozen_formula():
    seed = base.derive_seed(20260809, "tag", 3)
    manual = int(base.domain_sha256("FL_V0_SEED", [20260809, "tag", 3])[:16],
                 16) % (2 ** 63)
    assert seed == manual
    assert 0 <= seed < 2 ** 63
    assert base.derive_seed(20260809, "tag", 3) == seed
    assert base.derive_seed(20260810, "tag", 3) != seed


def test_make_rng_is_reproducible_and_independent():
    a = base.make_rng(7).integers(0, 1 << 30, size=5).tolist()
    b = base.make_rng(7).integers(0, 1 << 30, size=5).tolist()
    c = base.make_rng(8).integers(0, 1 << 30, size=5).tolist()
    assert a == b and a != c


def test_sha256_file_and_tree(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "sub" / "b.txt").write_bytes(b"world")
    assert base.sha256_file(str(tmp_path / "a.txt")) == \
        hashlib.sha256(b"hello").hexdigest()
    digest = base.sha256_tree(str(tmp_path))
    manual = hashlib.sha256()
    for rel, payload in (("a.txt", b"hello"), ("sub/b.txt", b"world")):
        manual.update(rel.encode("utf-8"))
        manual.update(b"\0")
        manual.update(hashlib.sha256(payload).digest())
        manual.update(b"\n")
    assert digest == manual.hexdigest()
    (tmp_path / "sub" / "b.txt").write_bytes(b"WORLD")
    assert base.sha256_tree(str(tmp_path)) != digest


def test_sealed_key_matches_amendment_1():
    split_hex = "a" * 64
    assert base.sealed_key(split_hex) == hashlib.sha256(
        b"FL_V0_SEALED_KEY\0" + split_hex.encode("utf-8")).digest()
    with pytest.raises(ValueError):
        base.sealed_key("not-a-digest")


def test_seal_roundtrip_and_keystream_shape():
    key = base.sealed_key("b" * 64)
    payload = os.urandom(0) + b'{"episode":"x"}\n' * 40
    sealed = base.seal_bytes(payload, key)
    assert sealed != payload
    assert len(sealed) == len(payload)
    assert base.unseal_bytes(sealed, key) == payload
    # block i of the keystream is sha256(key + i as 8 big-endian bytes)
    block0 = hashlib.sha256(key + (0).to_bytes(8, "big")).digest()
    assert bytes(a ^ b for a, b in zip(payload[:32], block0)) == sealed[:32]
    block1 = hashlib.sha256(key + (1).to_bytes(8, "big")).digest()
    assert bytes(a ^ b for a, b in zip(payload[32:64], block1)) == sealed[32:64]
    other = base.sealed_key("c" * 64)
    assert base.unseal_bytes(sealed, other) != payload


def test_kind_encoding_roundtrip():
    for kind in base.KIND_NAMES:
        seed = base.encode_seed(123456789, kind)
        assert base.decode_seed(seed) == (123456789, kind)
    with pytest.raises(ValueError):
        base.encode_seed(1, 9)


def test_letters_is_digit_free_and_injective():
    codes = [base.letters(i) for i in range(200)]
    assert len(set(codes)) == 200
    assert all(c.isupper() and c.isalpha() for c in codes)
    assert base.letters(0) == "A" and base.letters(25) == "Z"
    assert base.letters(26) == "AA"


def test_answer_leak_detects_token_and_substring_exposure():
    assert base.answer_leak("HINT: value=1", "1")
    assert base.answer_leak("the target is a b c", "a b c")
    assert base.answer_leak("HINT-X: pick=YES", "YES")
    assert not base.answer_leak("HINT-PIVOT: pivot=VAR_C", "1")
    assert not base.answer_leak("HINT-DEGREE: at=PT_A", "NO")
    assert not base.answer_leak("HINT-PRED: fails=PRD_A", "OUT")
    assert not base.answer_leak("HINT-MATCH: aligned=CNT_B", "abcab")


def test_hint_field_and_feedback_reject_malformed_records():
    with pytest.raises(ValueError):
        base.HintField("bad name", "TAG", 0, 2)
    with pytest.raises(ValueError):
        base.HintField("name", "tag", 0, 2)
    with pytest.raises(ValueError):
        base.HintField("name", "TAG", 5, 2)
    with pytest.raises(ValueError):
        base.Feedback("hint-x", (base.HintField("a", "T", 0, 2),))
    with pytest.raises(ValueError):
        base.Feedback("HINT-X", ())
    with pytest.raises(ValueError):
        # every record must carry at least one corruptible field
        base.Feedback("HINT-X", (base.HintField("a", "T", 0, 1),))
    good = base.Feedback("HINT-X", (base.HintField("a", "TAG", 1, 3),))
    assert good.render() == "HINT-X: a=TAG_B"
    assert not any(ch.isdigit() for ch in good.render())
