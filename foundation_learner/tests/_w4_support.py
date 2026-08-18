"""Shared support for the W4 (evaluation / analysis) test modules.

OWNERSHIP: this file belongs to the evaluation+analysis worker.  It is not a
test module (its name does not match ``test_*``) and it defines no test.

It provides:

1. Resolution of the producer modules with a REAL-FIRST policy.  Where the
   real ``ecology`` / ``episodes`` / ``training.tiny_model`` modules exist they
   are used, so the evaluation tests exercise the true integration; a labelled
   stand-in is installed only for a module that is genuinely absent, and
   :func:`backend_report` states which was used so a pass can never be
   mistaken for evidence about a module that was not there.
2. Real episode construction (``real_episode``) through
   ``episodes.assemble.assemble_episode`` with a real DEVELOPMENT family, plus
   the matching exact-verifier environment.
3. Deliberately broken doubles used BY the hostile fixtures (a renderer whose
   reset query keeps the learning history), and marker planting helpers.
4. Synthetic episode records for the pure metric / statistics tests.

Nothing here weakens a check: the stand-ins implement the frozen conventions
exactly, and the broken doubles exist only to prove that a guard fires.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import traceback
import types
from dataclasses import dataclass, field, replace
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CHECKPOINT_DIR = "/home/moloch/ouro_project/models/ouro_rltt_local"

#: a DEVELOPMENT family (contract Amendment 2) with short prompts
DEFAULT_TEST_FAMILY = "boolean_rule"

BACKENDS: dict[str, str] = {}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


# --------------------------------------------------------------------------
# flhash (contract section 3): real ecology.base, else an exact stand-in
# --------------------------------------------------------------------------
def _standin_flhash_module() -> types.ModuleType:
    mod = types.ModuleType("foundation_learner.ecology.base")
    mod.__FL_STANDIN__ = True  # type: ignore[attr-defined]

    def canonical_json(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)

    def domain_sha256(domain: str, obj: Any) -> str:
        return hashlib.sha256(domain.encode() + b"\0"
                              + canonical_json(obj).encode()).hexdigest()

    def derive_seed(root_seed: int, *tags: Any) -> int:
        return int(domain_sha256("FL_V0_SEED", [root_seed, *tags])[:16], 16) % 2 ** 63

    mod.canonical_json = canonical_json          # type: ignore[attr-defined]
    mod.domain_sha256 = domain_sha256            # type: ignore[attr-defined]
    mod.derive_seed = derive_seed                # type: ignore[attr-defined]
    return mod


def flhash_backend() -> str:
    """Ensure ``foundation_learner.ecology.base`` resolves; return the backend."""
    if "foundation_learner.ecology.base" in sys.modules:
        return BACKENDS.setdefault("flhash", "real")
    if _module_available("foundation_learner.ecology.base"):
        importlib.import_module("foundation_learner.ecology.base")
        BACKENDS["flhash"] = "real"
        return "real"
    pkg_name = "foundation_learner.ecology"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules[pkg_name] = pkg
    module = _standin_flhash_module()
    sys.modules["foundation_learner.ecology.base"] = module
    setattr(sys.modules[pkg_name], "base", module)
    BACKENDS["flhash"] = "standin"
    return "standin"


def episodes_available() -> bool:
    ok = all(_module_available(f"foundation_learner.episodes.{m}")
             for m in ("schema", "render", "parse", "assemble"))
    BACKENDS["episodes"] = "real" if ok else "absent"
    return ok


def ecology_available() -> bool:
    ok = all(_module_available(f"foundation_learner.ecology.{m}")
             for m in ("base", "poison", "families"))
    BACKENDS["ecology"] = "real" if ok else "absent"
    return ok


def integration_available() -> bool:
    """True when real episodes can be assembled (all producer modules present)."""
    return ecology_available() and episodes_available()


# --------------------------------------------------------------------------
# tiny model bundle
# --------------------------------------------------------------------------
@dataclass
class StandinBundle:
    model: Any
    tokenizer: Any
    device: Any
    identity: dict = field(default_factory=dict)


_TINY_BUNDLE: Any = None


def build_standin_tiny_model(seed: int = 20260809, device: str = "cpu") -> StandinBundle:
    """Tiny OuroForCausalLM from the checkpoint's OWN code; random init, NO weights.

    hidden 64 / 2 layers / 2 ut steps (contract section 18).  The real
    ``model.safetensors`` shards are never opened.  Used only when
    ``training/tiny_model.py`` is absent.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    cfg.hidden_size = 64
    cfg.num_hidden_layers = 2
    cfg.total_ut_steps = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    cfg.intermediate_size = 128
    cfg.torch_dtype = "float32"
    cfg.use_cache = True
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    model = model.to(torch.float32).eval()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.empty(p.shape, dtype=torch.float32).normal_(
                0.0, 0.02, generator=gen))
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    return StandinBundle(model=model, tokenizer=tokenizer, device=torch.device(device),
                         identity={"kind": "TINY_STANDIN", "scientific": False,
                                   "seed": int(seed), "source": "tests/_w4_support"})


def tiny_bundle(seed: int = 20260809) -> Any:
    """Real ``training.tiny_model.build_tiny_model`` when present, else stand-in."""
    global _TINY_BUNDLE
    if _TINY_BUNDLE is not None:
        return _TINY_BUNDLE
    if _module_available("foundation_learner.training.tiny_model"):
        mod = importlib.import_module("foundation_learner.training.tiny_model")
        build = getattr(mod, "build_tiny_model", None)
        if callable(build):
            bundle = build(seed=seed, device="cpu")
            bundle.model.eval()
            BACKENDS["tiny_model"] = "real"
            _TINY_BUNDLE = bundle
            return _TINY_BUNDLE
    _TINY_BUNDLE = build_standin_tiny_model(seed=seed)
    BACKENDS["tiny_model"] = "standin"
    return _TINY_BUNDLE


def checkpoint_present() -> bool:
    return os.path.isdir(CHECKPOINT_DIR) and os.path.isfile(
        os.path.join(CHECKPOINT_DIR, "modeling_ouro.py"))


# --------------------------------------------------------------------------
# real episodes
# --------------------------------------------------------------------------
def get_family(family_id: str = DEFAULT_TEST_FAMILY):
    from foundation_learner.ecology.families import FAMILY_REGISTRY

    return FAMILY_REGISTRY[family_id]


def real_episode(*, family_id: str = DEFAULT_TEST_FAMILY, index: int = 0,
                 root_seed: int = 20260809, split: str = "DEVELOPMENT",
                 difficulty: int = 1, condition: str = "clean",
                 structured: bool = True, reveal: bool = False,
                 mode: str = "scripted"):
    """Assemble one REAL pre-generated episode plus its family and rule."""
    from foundation_learner.ecology.base import derive_seed, make_rng
    from foundation_learner.episodes.assemble import assemble_episode

    family = get_family(family_id)
    rule_seed = derive_seed(root_seed, "FL_V0_TEST_RULE", family_id, index)
    rule = family.sample_rule(make_rng(rule_seed))
    episode_seed = derive_seed(root_seed, "FL_V0_TEST_EPISODE", family_id, index)
    ep = assemble_episode(family, rule, split=split, difficulty=difficulty,
                          root_seed=root_seed, episode_seed=episode_seed,
                          mode=mode, condition=condition, structured=structured,
                          reveal=reveal)
    return ep, family, rule


def env_for(ep: Any, family: Any, rule: Any = None):
    from foundation_learner.evaluation.learning_curve import environment_from_episode

    return environment_from_episode(ep, family, rule=rule)


def episode_batch(n: int = 2, *, family_id: str = DEFAULT_TEST_FAMILY, **kwargs):
    """``(episodes, family, env_factory)`` for ``n`` real episodes."""
    built = [real_episode(family_id=family_id, index=i, **kwargs) for i in range(n)]
    episodes = [b[0] for b in built]
    family = built[0][1]
    rules = {id(b[0]): b[2] for b in built}
    return episodes, family, (lambda ep: env_for(ep, family, rules.get(id(ep))))


# --------------------------------------------------------------------------
# marker planting (hostile fixtures)
# --------------------------------------------------------------------------
SCRIPTED_ATTEMPT_MARKER = "SCRIPTED_PLACEHOLDER_ATTEMPT_ZZQ"
SCRIPTED_FEEDBACK_MARKER = "SCRIPTED_PLACEHOLDER_FEEDBACK_ZZQ"
SUPPORT_MARKER = "SUPPORT_SECRET_MARKER_ZZQ"


def plant_scripted_markers(ep: Any) -> Any:
    """Replace every scripted attempt / feedback body with a unique marker.

    The ONLINE walker must never show any of these to the model: it fills
    attempt slots from real generations and feedback slots from the real
    verifier.  Any marker appearing in a prompt is a regression.
    """
    from foundation_learner.episodes.schema import Role

    events = []
    for ev in ep.events:
        if ev.role is Role.MODEL_ATTEMPT:
            events.append(replace(ev, text=f"ANSWER: {SCRIPTED_ATTEMPT_MARKER}"))
        elif ev.role in (Role.FEEDBACK, Role.HINT):
            events.append(replace(ev, text=SCRIPTED_FEEDBACK_MARKER))
        else:
            events.append(ev)
    return replace(ep, events=events)


def plant_support_marker(ep: Any, marker: str = SUPPORT_MARKER) -> Any:
    """Append a unique marker to every SUPPORT task statement."""
    from foundation_learner.episodes.schema import Role

    events = []
    for ev in ep.events:
        if ev.role is Role.TASK:
            events.append(replace(ev, text=f"{ev.text} {marker}"))
        else:
            events.append(ev)
    return replace(ep, events=events)


def leaky_render_module() -> types.ModuleType:
    """A BROKEN renderer double whose reset query keeps the learning history.

    Used only by the context-reset hostile fixture, to prove that the leak
    check fires instead of passing vacuously.
    """
    from foundation_learner.episodes import render as real_render

    mod = types.ModuleType("tests.leaky_render")
    mod.__FL_BROKEN_DOUBLE__ = True  # type: ignore[attr-defined]
    mod.render_episode = real_render.render_episode      # type: ignore[attr-defined]
    mod.render_events = real_render.render_events        # type: ignore[attr-defined]
    mod.render_subset = real_render.render_subset        # type: ignore[attr-defined]

    def render_reset_query(ep, query_event_index):
        return real_render.render_subset(ep, list(range(query_event_index + 1)))

    mod.render_reset_query = render_reset_query          # type: ignore[attr-defined]
    return mod


# --------------------------------------------------------------------------
# synthetic episode records (pure metric / statistics tests)
# --------------------------------------------------------------------------
def make_record(episode_id: str, family_id: str, R: dict[int, float],
                **extra: Any) -> dict[str, Any]:
    record = {
        "schema": "flb200.episode_record.v1",
        "episode_id": episode_id,
        "family_id": family_id,
        "split": "DEVELOPMENT",
        "condition": "clean",
        "R": {str(k): float(v) for k, v in R.items()},
        "interaction_indices": sorted(int(k) for k in R),
        "aulc": (sum(R.values()) / len(R)) if R else None,
        "attempts": [],
    }
    record.update(extra)
    return record


def constant_records(family_id: str, n: int, value: float, *,
                     prefix: str = "", indices=(0, 1, 2, 3, 4, 5, 6)
                     ) -> list[dict[str, Any]]:
    return [make_record(f"{prefix}{family_id}-{i}", family_id,
                        {k: value for k in indices}) for i in range(n)]


# --------------------------------------------------------------------------
# plain runner glue (usable without pytest)
# --------------------------------------------------------------------------
def run_module_tests(namespace: dict[str, Any], name: str = "") -> int:
    """Execute every ``test_*`` callable in ``namespace``; return exit status."""
    tests = [(k, v) for k, v in sorted(namespace.items())
             if k.startswith("test_") and callable(v)]
    passed, failures = 0, []
    for label, fn in tests:
        try:
            fn()
            passed += 1
        except BaseException:  # noqa: BLE001 - report, never swallow
            failures.append((label, traceback.format_exc(limit=12)))
    print(f"[{name or namespace.get('__name__', '?')}] {passed} passed, "
          f"{len(failures)} failed")
    for label, tb in failures:
        print(f"  FAIL {label}\n{tb}")
    return 1 if failures else 0


def backend_report() -> dict[str, str]:
    """Which modules were real and which were stand-ins for this run."""
    return dict(BACKENDS)


# Resolve the hash helpers at import time so record hashing never depends on
# which test module happened to touch them first.
flhash_backend()


__all__ = [
    "CHECKPOINT_DIR",
    "DEFAULT_TEST_FAMILY",
    "BACKENDS",
    "backend_report",
    "flhash_backend",
    "episodes_available",
    "ecology_available",
    "integration_available",
    "checkpoint_present",
    "StandinBundle",
    "build_standin_tiny_model",
    "tiny_bundle",
    "get_family",
    "real_episode",
    "env_for",
    "episode_batch",
    "SCRIPTED_ATTEMPT_MARKER",
    "SCRIPTED_FEEDBACK_MARKER",
    "SUPPORT_MARKER",
    "plant_scripted_markers",
    "plant_support_marker",
    "leaky_render_module",
    "make_record",
    "constant_records",
    "run_module_tests",
]
