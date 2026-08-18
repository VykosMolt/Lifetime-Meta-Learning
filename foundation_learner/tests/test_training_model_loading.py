"""Frozen-backbone binding and hash conventions (contract §1, §3, §22).

These tests NEVER load the 2.6 B checkpoint weights.  They touch the small
files of the checkpoint directory (config/tokenizer/code) and the tokenizer
only, which contract §22 explicitly allows.
"""

from __future__ import annotations

import hashlib
import os
import shutil

import pytest

from foundation_learner.training import model_loading as ml


def test_frozen_constants_match_contract_section_1():
    assert ml.FROZEN_CHECKPOINT_TREE_SHA256 == (
        "a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889"
    )
    assert ml.REQUIRED_TRANSFORMERS_VERSION == "4.54.1"
    assert ml.FROZEN_NUM_HIDDEN_LAYERS == 48
    assert ml.FROZEN_TOTAL_UT_STEPS == 4
    assert ml.FROZEN_HIDDEN_SIZE == 2048
    assert ml.FROZEN_VOCAB_SIZE == 49152
    assert ml.FROZEN_EARLY_EXIT_THRESHOLD == 1.0


def test_transformers_version_assert_is_hard():
    assert ml.assert_transformers_version() == "4.54.1"
    import transformers

    original = transformers.__version__
    try:
        transformers.__version__ = "4.55.0"
        with pytest.raises(ml.FrozenBindingError):
            ml.assert_transformers_version()
    finally:
        transformers.__version__ = original


def test_canonical_json_and_domain_hash_conventions():
    assert ml.canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    expected = hashlib.sha256(b"D\x00" + b'{"a":1}').hexdigest()
    assert ml.domain_sha256("D", {"a": 1}) == expected
    with pytest.raises(ValueError):
        ml.canonical_json({"a": float("nan")})


def test_derive_seed_is_deterministic_and_in_range():
    a = ml.derive_seed(20260809, "DATA_ORDER", "FL3", 0)
    b = ml.derive_seed(20260809, "DATA_ORDER", "FL3", 0)
    c = ml.derive_seed(20260809, "DATA_ORDER", "FL3", 1)
    assert a == b and a != c
    assert 0 <= a < 2**63


def test_sha256_tree_matches_the_o1_convention(tmp_path):
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"alpha")
    (root / "sub" / "b.txt").write_bytes(b"beta")

    h = hashlib.sha256()
    for rel, data in (("a.txt", b"alpha"), ("sub/b.txt", b"beta")):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(data).digest())
        h.update(b"\n")
    assert ml.sha256_tree(str(root)) == h.hexdigest()
    assert ml.sha256_tree(str(root / "a.txt")) == hashlib.sha256(b"alpha").hexdigest()


def test_local_hash_helpers_agree_with_ecology_base(tmp_path):
    """Contract §3 asks for one convention; §22 asks FL not to import O1 code.

    The training package carries its own copy, so it must agree exactly with
    ``ecology/base.py`` wherever that module exists.
    """
    base = pytest.importorskip("foundation_learner.ecology.base")
    root = tmp_path / "tree"
    root.mkdir()
    (root / "x.bin").write_bytes(os.urandom(4096))
    assert base.sha256_tree(str(root)) == ml.sha256_tree(str(root))
    assert base.sha256_file(str(root / "x.bin")) == ml.sha256_file(str(root / "x.bin"))
    assert base.canonical_json({"z": 1, "a": 2}) == ml.canonical_json({"z": 1, "a": 2})
    assert base.domain_sha256("FL_V0_X", [1, "a"]) == ml.domain_sha256("FL_V0_X", [1, "a"])
    assert base.derive_seed(20260809, "t", 3) == ml.derive_seed(20260809, "t", 3)


@pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory not present",
)
def test_frozen_small_file_hashes_verify():
    seen = ml.verify_frozen_files(ml.FROZEN_CHECKPOINT_DIR)
    assert set(seen) == set(ml.FROZEN_FILE_SHA256)


@pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory not present",
)
def test_tampered_checkpoint_directory_is_refused(tmp_path):
    fake = tmp_path / "ckpt"
    fake.mkdir()
    for name in ml.FROZEN_FILE_SHA256:
        src = os.path.join(ml.FROZEN_CHECKPOINT_DIR, name)
        shutil.copy2(src, fake / name)
    # Untampered copy passes the per-file table but fails the tree hash
    # (the weight shards are absent).
    ml.verify_frozen_files(str(fake))
    with pytest.raises(ml.FrozenBindingError):
        ml.verify_checkpoint_tree(str(fake))
    # A single flipped byte in config.json is caught by the file table.
    data = bytearray((fake / "config.json").read_bytes())
    data[0] ^= 0x01
    (fake / "config.json").write_bytes(bytes(data))
    with pytest.raises(ml.FrozenBindingError):
        ml.verify_frozen_files(str(fake))


@pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory not present",
)
def test_load_frozen_backbone_refuses_a_missing_directory():
    with pytest.raises(ml.FrozenBindingError):
        ml.load_frozen_backbone("/nonexistent/checkpoint", device="cpu")


@pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory not present",
)
def test_loading_recipe_matches_contract_section_22_exactly(monkeypatch):
    """Pin the §22 recipe WITHOUT materialising the 2.6 B weights.

    Loading the real checkpoint weights is forbidden in this test suite, so the
    weight-loading call itself is intercepted; everything around it (version
    assert, file verification, tokenizer load, UT/early-exit pinning,
    architecture assertion, identity construction) runs for real, and the exact
    keyword arguments of ``from_pretrained`` are asserted.
    """
    import torch
    import transformers
    from transformers import AutoConfig

    real_config = AutoConfig.from_pretrained(
        ml.FROZEN_CHECKPOINT_DIR, trust_remote_code=True, local_files_only=True
    )
    calls = {}

    class _WeightlessModel:
        def __init__(self, config):
            self.config = config
            self.model = type("Inner", (), {"total_ut_steps": None})()
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    def fake_from_pretrained(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        config = AutoConfig.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        config.total_ut_steps = 1  # deliberately unpinned; the loader must fix it
        return _WeightlessModel(config)

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained
    )
    bundle = ml.load_frozen_backbone(ml.FROZEN_CHECKPOINT_DIR, device="cpu")

    assert calls["path"] == os.path.realpath(ml.FROZEN_CHECKPOINT_DIR)
    assert calls["kwargs"] == {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    assert bundle.model.moved_to == "cpu"
    assert bundle.model.config.total_ut_steps == 4
    assert bundle.model.config.early_exit_threshold == 1.0
    assert bundle.model.model.total_ut_steps == 4
    assert bundle.tokenizer.is_fast
    assert bundle.scientific is True
    assert bundle.identity["kind"] == "FROZEN_BACKBONE"
    assert bundle.identity["tree_hash_verified"] is False
    assert bundle.identity["checkpoint_tree_sha256"] == (
        ml.FROZEN_CHECKPOINT_TREE_SHA256
    )
    assert bundle.identity["dtype"] == "bfloat16"
    assert bundle.identity["transformers_version"] == "4.54.1"
    assert bundle.identity["torch_version"] == torch.__version__
    assert len(bundle.identity["config_digest"]) == 64
    assert bundle.identity["environment"]["transformers"] == "4.54.1"
    bundle.assert_scientific()

    # an unpinned or foreign architecture is refused
    def bad_from_pretrained(path, **kwargs):
        config = AutoConfig.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        config.num_hidden_layers = 24
        model = _WeightlessModel(config)
        return model

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", bad_from_pretrained
    )
    with pytest.raises(ml.FrozenBindingError):
        ml.load_frozen_backbone(ml.FROZEN_CHECKPOINT_DIR, device="cpu")
    del real_config


def test_environment_digest_is_stable_and_records_versions():
    a = ml.environment_digest()
    b = ml.environment_digest()
    assert a == b
    assert a["transformers"] == "4.54.1"
    assert a["torch"].startswith("2.")
    assert len(a["digest"]) == 64


def test_identity_hash_ignores_volatile_fields():
    class Cfg:
        model_type = "ouro"
        architectures = ["OuroForCausalLM"]
        hidden_size = 2048
        num_hidden_layers = 48
        num_attention_heads = 16
        total_ut_steps = 4
        early_exit_threshold = 1.0
        vocab_size = 49152

        def to_dict(self):
            return {
                "model_type": "ouro",
                "hidden_size": 2048,
                "_name_or_path": "/somewhere",
                "transformers_version": "4.54.1",
            }

    cfg = Cfg()
    a = ml.build_identity(
        kind="FROZEN_BACKBONE",
        scientific=True,
        checkpoint_dir="/tmp",
        tree_sha256=ml.FROZEN_CHECKPOINT_TREE_SHA256,
        tree_hash_verified=True,
        config=cfg,
        dtype="bfloat16",
        device="cuda",
    )
    b = ml.build_identity(
        kind="FROZEN_BACKBONE",
        scientific=True,
        checkpoint_dir="/other",
        tree_sha256=ml.FROZEN_CHECKPOINT_TREE_SHA256,
        tree_hash_verified=False,
        config=cfg,
        dtype="bfloat16",
        device="cpu",
    )
    assert a["identity_hash"] == b["identity_hash"]

    c = ml.build_identity(
        kind="FROZEN_BACKBONE",
        scientific=True,
        checkpoint_dir="/tmp",
        tree_sha256="0" * 64,
        tree_hash_verified=True,
        config=cfg,
        dtype="bfloat16",
        device="cuda",
    )
    assert c["identity_hash"] != a["identity_hash"]
