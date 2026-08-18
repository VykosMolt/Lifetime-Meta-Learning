"""FL4 — future-competence value head (contract §7, §9, §23).

What this module is
===================
A small MLP on a STOP-GRADIENT hidden state that predicts the REALIZED
learning value of one feedback / hint / reveal item:

    y_j = [mean per-token log-likelihood of the correct QUERY answers,
           item j PRESENT in the context]
        - [same, item j ABLATED from the context]

It is NOT a correctness predictor of the current answer, and it never sees a
query answer, a verifier verdict, a poison flag, or any realized future
outcome.

Frozen architecture (contract §7)
---------------------------------
``Linear(hidden -> 256) + GELU + Linear(256 -> 1)``, fp32 head on the bf16
backbone's hidden state.  The input is the FINAL-LAYER, FINAL-LOOP hidden
state at the LAST token of feedback/hint item j (``reduce="last"``, exactly the
summary ``evaluation/learning_curve.hidden_summary_fn`` exposes online).

Structural no-leak guarantee (hostile-tested)
---------------------------------------------
The public head-input path is :func:`head_input_hidden`, which renders the
episode with ``episodes.render.render_subset`` over events ``0..j`` ONLY and
REFUSES if any kept event is a ``QUERY_TASK`` / ``TRANSFER_TASK`` block or a
query/transfer ``MODEL_ATTEMPT``.  There is therefore no public way to hand the
head a hidden state computed from a context that contains a query answer.  The
target pipeline likewise refuses any ablation plan that would remove a
non-feedback event (a plan that ablated a QUERY event would silently redefine
the estimand).  Target computation is TRAIN-families-only by default
(``ecology.split.split_of`` must return ``TRAIN``); the §7 DEVELOPMENT
evaluation of the head opts in to DEVELOPMENT explicitly with
``allowed_splits=TARGET_SPLITS_DEV_EVAL``, and SEALED_TEST is refused under
every setting (``assert_target_split`` filters it out of any caller-supplied
list).

Constants introduced HERE (documented, recorded in CONTRACT_AMENDMENTS.md)
--------------------------------------------------------------------------
* ``VALUE_HEAD_INNER = 256`` and the fp32 dtype are contract §7 (not new).
* ``MSE_WEIGHT = 0.5`` is contract §7 (not new).
* ``RANKING_MARGIN = 1.0`` — the margin of the pairwise ranking hinge.  §7
  fixes the loss FORM ("pairwise within-episode ranking hinge") but not the
  margin; 1.0 is the standard ``torch.nn.MarginRankingLoss`` default and is
  frozen here before any run.  Predictions are unbounded, so the margin is a
  scale convention, not a strength knob: the auxiliary MSE term on Z-SCORED
  targets is what fixes the scale.
* ``TARGET_Z_SCORE_SCOPE = "dataset"`` — targets are z-scored with the mean and
  standard deviation of the TRAINING set, stored in the head
  (``target_mean`` / ``target_std`` buffers) so that DEV predictions live on
  the same scale.  Ranking metrics are scale invariant; calibration slope /
  intercept are reported against the z-scored target, which is the quantity the
  head is trained on.
* Head optimizer: AdamW with the §8 campaign constants
  (betas (0.9, 0.95), eps 1e-8, weight decay 0.01, grad-norm clip 1.0) and
  ``VALUE_HEAD_LR = 1e-3``.  §8's LR grid binds the BACKBONE arms only; the
  head is a new 0.5 M-parameter module, so its LR is declared here.
* ``VALUE_HEAD_EPOCHS = 40`` and ``VALUE_HEAD_GROUPS_PER_BATCH = 8`` are the
  defaults of :class:`ValueHeadTrainConfig`; every caller may set them, and the
  campaign records the values it used.

Determinism: initialisation and shuffling use explicit ``torch.Generator``
objects seeded through ``ecology.base.derive_seed``.  No global RNG, no wall
clock.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from foundation_learner.ecology.base import derive_seed, domain_sha256
from foundation_learner.ecology.split import split_of
from foundation_learner.episodes.parse import format_answer
from foundation_learner.episodes.render import render_subset
from foundation_learner.episodes.schema import (
    EPISODE_STRUCTURE_V0,
    Episode,
    Event,
    Role,
)
from foundation_learner.evaluation.scoring import ScoreRequest, answer_logprob_batch
from foundation_learner.mechanisms.hidden_states import item_hidden_summary
from foundation_learner.training.tokenization import (
    MASK_STRADDLE_POLICY,
    answer_spans_for_event,
    char_span_to_token_span,
    encode_with_offsets,
)

__all__ = [
    "VALUE_HEAD_INNER",
    "VALUE_HEAD_DTYPE",
    "RANKING_MARGIN",
    "MSE_WEIGHT",
    "VALUE_HEAD_LR",
    "TARGET_Z_SCORE_SCOPE",
    "SURFACE_HEURISTICS",
    "ValueHeadError",
    "ValueHeadLeakError",
    "SplitRefusedError",
    "AblationPlanError",
    "ValueHead",
    "ValueItem",
    "ValueHeadTrainConfig",
    "TARGET_SPLITS_TRAIN",
    "TARGET_SPLITS_DEV_EVAL",
    "assert_target_split",
    "assert_train_family",
    "head_context_indices",
    "head_input_hidden",
    "ablation_plan_for_episode",
    "validate_ablation_plan",
    "token_aligned_answer_span",
    "query_answer_score",
    "episode_targets",
    "build_value_dataset",
    "train_value_head",
    "value_head_predictions",
    "surface_heuristic_predictions",
    "evaluate_value_head",
    "FEEDBACK_ROLE_NAMES",
    "HIDDEN_LABEL_ROLE_NAMES",
    "run_fl4_stage",
]

VALUE_HEAD_INNER = 256
VALUE_HEAD_DTYPE = torch.float32
RANKING_MARGIN = 1.0
MSE_WEIGHT = 0.5
VALUE_HEAD_LR = 1e-3
TARGET_Z_SCORE_SCOPE = "dataset"
#: minimum target spread below which z-scoring would amplify numerical noise
MIN_TARGET_STD = 1e-8

FEEDBACK_ROLE_NAMES = (Role.FEEDBACK.value, Role.HINT.value, Role.REVEAL.value)
HIDDEN_LABEL_ROLE_NAMES = (Role.QUERY_TASK.value, Role.TRANSFER_TASK.value)

#: contract §7 DEV comparison baselines ("vs random, vs surface heuristics")
SURFACE_HEURISTICS = ("item_length", "item_type", "lexical_overlap")

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class ValueHeadError(RuntimeError):
    """Base class for FL4 failures."""


class ValueHeadLeakError(ValueHeadError):
    """A head input or target computation would expose a hidden query label."""


class SplitRefusedError(ValueHeadError):
    """Target computation was requested for a non-TRAIN family."""


class AblationPlanError(ValueHeadError):
    """A pre-generated ablation plan does not match its episode."""


# ---------------------------------------------------------------------------
# the head
# ---------------------------------------------------------------------------
def _init_linear_(layer: nn.Linear, gen: torch.Generator) -> None:
    """``nn.Linear.reset_parameters`` driven by an EXPLICIT generator."""
    fan_in = int(layer.weight.shape[1])
    # kaiming_uniform_(a=sqrt(5)) -> bound = sqrt(6 / ((1 + 5) * fan_in))
    bound = math.sqrt(6.0 / ((1.0 + 5.0) * fan_in))
    with torch.no_grad():
        layer.weight.copy_(
            torch.empty(layer.weight.shape, dtype=torch.float32).uniform_(
                -bound, bound, generator=gen))
        if layer.bias is not None:
            b = 1.0 / math.sqrt(fan_in)
            layer.bias.copy_(
                torch.empty(layer.bias.shape, dtype=torch.float32).uniform_(
                    -b, b, generator=gen))


class ValueHead(nn.Module):
    """Contract §7 FL4 head: ``Linear(H,256) + GELU + Linear(256,1)``, fp32.

    ``forward`` DETACHES its input: the head can never backpropagate into the
    backbone, whatever a caller does.  ``target_mean`` / ``target_std`` are the
    z-scoring statistics of the training targets, stored so that predictions
    are interpretable on the same scale at evaluation time.
    """

    def __init__(self, hidden_size: int, *, inner: int = VALUE_HEAD_INNER,
                 seed: int = 0) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.inner = int(inner)
        self.seed = int(seed)
        self.fc1 = nn.Linear(self.hidden_size, self.inner).to(VALUE_HEAD_DTYPE)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.inner, 1).to(VALUE_HEAD_DTYPE)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(derive_seed(self.seed, "FL4_VALUE_HEAD_INIT",
                                    self.hidden_size, self.inner))
        _init_linear_(self.fc1, gen)
        _init_linear_(self.fc2, gen)
        self.register_buffer("target_mean", torch.zeros((), dtype=VALUE_HEAD_DTYPE))
        self.register_buffer("target_std", torch.ones((), dtype=VALUE_HEAD_DTYPE))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[..., hidden] -> [...]`` prediction on the z-scored target scale."""
        if hidden.shape[-1] != self.hidden_size:
            raise ValueHeadError(
                f"value head expects hidden size {self.hidden_size}, got "
                f"{tuple(hidden.shape)}")
        # STOP GRADIENT (contract §7): structural, not a caller convention.
        x = hidden.detach().to(dtype=VALUE_HEAD_DTYPE,
                               device=self.fc1.weight.device)
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)

    @torch.no_grad()
    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.forward(hidden)

    def config(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "inner": self.inner,
            "seed": self.seed,
            "dtype": "float32",
            "arch": "Linear+GELU+Linear",
            "ranking_margin": RANKING_MARGIN,
            "mse_weight": MSE_WEIGHT,
            "target_z_score_scope": TARGET_Z_SCORE_SCOPE,
        }

    def config_hash(self) -> str:
        return domain_sha256("FL_V0_FL4_HEAD", self.config())

    def set_target_stats(self, mean: float, std: float) -> None:
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
            raise ValueHeadError(f"invalid target statistics mean={mean} std={std}")
        with torch.no_grad():
            self.target_mean.fill_(float(mean))
            self.target_std.fill_(float(std))

    def freeze_(self) -> "ValueHead":
        """Freeze every parameter (contract §7 FL6 uses a FROZEN head)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self


# ---------------------------------------------------------------------------
# episode access helpers
# ---------------------------------------------------------------------------
def _events(episode: Episode) -> list[Event]:
    return list(episode.events)


def _role_name(event: Event) -> str:
    return event.role.value if isinstance(event.role, Role) else str(event.role)


#: frozen interaction indices of the hidden-label attempts (§6 structure)
QUERY_INDEX = EPISODE_STRUCTURE_V0.query_index
TRANSFER_INDEX = EPISODE_STRUCTURE_V0.transfer_index


def _query_attempt_indices(episode: Episode, *,
                           structure_query_index: int = QUERY_INDEX
                           ) -> list[int]:
    return [i for i, e in enumerate(_events(episode))
            if _role_name(e) == Role.MODEL_ATTEMPT.value
            and e.interaction_index == structure_query_index]


def _spans_by_event(rendered: Any) -> dict[int, tuple[int, int]]:
    return {int(s.event_index): (int(s.start), int(s.end))
            for s in rendered.spans}


def _item_answer(episode: Episode, item_id: str) -> str:
    """Canonical answer of an item, from ``Episode.items`` (Amendment 5)."""
    record = (episode.items or {}).get(item_id)
    if isinstance(record, dict):
        instance = record.get("instance", record)
        if isinstance(instance, dict) and instance.get("answer_canonical"):
            return str(instance["answer_canonical"])
    raise ValueHeadError(
        f"episode {episode.episode_id!r} has no canonical answer for item "
        f"{item_id!r}; FL4 targets require the episode's own item table")


# ---------------------------------------------------------------------------
# head input path (structurally query-free)
# ---------------------------------------------------------------------------
def head_context_indices(episode: Episode, event_index: int) -> list[int]:
    """Event indices of the head-input context: ``0..event_index`` inclusive.

    REFUSES if the resulting context would contain a query/transfer task block
    or a query/transfer attempt, and refuses any ``event_index`` that is not a
    FEEDBACK / HINT / REVEAL event.
    """
    events = _events(episode)
    index = int(event_index)
    if not 0 <= index < len(events):
        raise ValueHeadError(
            f"event index {index} outside episode {episode.episode_id!r}")
    target = events[index]
    if _role_name(target) not in FEEDBACK_ROLE_NAMES:
        raise ValueHeadError(
            f"FL4 scores FEEDBACK / HINT / REVEAL items; event {index} is "
            f"{_role_name(target)}")
    keep = list(range(index + 1))
    for i in keep:
        role = _role_name(events[i])
        if role in HIDDEN_LABEL_ROLE_NAMES:
            raise ValueHeadLeakError(
                f"head-input context for event {index} would include a {role} "
                f"block at event {i}; the FL4 head must never read a query or "
                "transfer item")
        if (role == Role.MODEL_ATTEMPT.value
                and events[i].interaction_index in (QUERY_INDEX, TRANSFER_INDEX)):
            raise ValueHeadLeakError(
                f"head-input context for event {index} would include the "
                f"query/transfer attempt at event {i}")
    return keep


def head_input_hidden(bundle: Any, episode: Episode, event_index: int,
                      *, prefix_embeds: torch.Tensor | None = None
                      ) -> torch.Tensor:
    """Final-loop hidden state at the LAST token of feedback item ``j``.

    The context is rendered by :func:`head_context_indices`, i.e. it ENDS at
    item ``j`` and can contain no query material at all.
    """
    keep = head_context_indices(episode, event_index)
    rendered = render_subset(episode, keep)
    span = _spans_by_event(rendered)[int(event_index)]
    return item_hidden_summary(bundle, rendered.text, span, reduce="last",
                               prefix_embeds=prefix_embeds)


# ---------------------------------------------------------------------------
# ablation plans (pre-generated by W1)
# ---------------------------------------------------------------------------
def ablation_plan_for_episode(episode: Episode) -> dict:
    """The SAME plan the shards carry, produced by W1's own generator.

    ``data/generate_shards._ablation_plans`` is the single source of truth for
    the plan shape; importing it (rather than re-deriving the plan) is what
    keeps an in-memory plan byte-identical to the pre-generated one.
    """
    from foundation_learner.data.generate_shards import _ablation_plans

    return _ablation_plans(episode)


def validate_ablation_plan(episode: Episode, plan: dict) -> dict:
    """Hard-check a plan against its episode; return it unchanged.

    A plan may only remove FEEDBACK / HINT / REVEAL events.  A plan that
    removes a QUERY / TRANSFER block, a task statement or a model attempt would
    change the estimand (the target is defined as "item j present" vs "item j
    ablated"), so it is REFUSED rather than executed.
    """
    events = _events(episode)
    if str(plan.get("episode_id", episode.episode_id)) != episode.episode_id:
        raise AblationPlanError(
            f"plan episode_id {plan.get('episode_id')!r} != "
            f"{episode.episode_id!r}")
    if int(plan.get("n_events", len(events))) != len(events):
        raise AblationPlanError(
            f"plan was generated for {plan.get('n_events')} events; the "
            f"episode has {len(events)}")
    entries = plan.get("plans")
    if not entries:
        raise AblationPlanError("plan carries no entries")
    if entries[0].get("ablation_id") != "FULL" or entries[0].get(
            "ablated_event_indices"):
        raise AblationPlanError("the first plan entry must be the FULL context")
    for entry in entries:
        for index in entry.get("ablated_event_indices", []):
            index = int(index)
            if not 0 <= index < len(events):
                raise AblationPlanError(
                    f"plan {entry.get('ablation_id')!r} ablates event {index}, "
                    f"outside the episode")
            role = _role_name(events[index])
            if role not in FEEDBACK_ROLE_NAMES:
                raise AblationPlanError(
                    f"REFUSED: plan {entry.get('ablation_id')!r} would ablate a "
                    f"{role} event (index {index}). FL4 targets are defined "
                    "over FEEDBACK / HINT / REVEAL items only; ablating a "
                    "query or task event silently redefines the estimand")
    return plan


#: splits a target computation may ever touch.  SEALED_TEST is absent by
#: construction, so no argument can admit it (contract §12).
TARGET_SPLITS_TRAIN: tuple[str, ...] = ("TRAIN",)
TARGET_SPLITS_DEV_EVAL: tuple[str, ...] = ("TRAIN", "DEVELOPMENT")
SEALED_SPLIT = "SEALED_TEST"


def assert_target_split(episode: Episode,
                        allowed_splits: Sequence[str] = TARGET_SPLITS_TRAIN
                        ) -> str:
    """Refuse a target computation outside ``allowed_splits`` (never sealed).

    Contract §7 computes the FL4 TRAINING targets on TRAIN families only; the
    §7 DEV evaluation of the head (Spearman, pairwise ranking accuracy,
    calibration, regret) needs the same realized quantity on DEVELOPMENT
    families, which is what ``TARGET_SPLITS_DEV_EVAL`` admits.  SEALED_TEST is
    never admissible: it is filtered out of any caller-supplied list, so even
    an explicit request for it is refused.
    """
    allowed = tuple(s for s in allowed_splits if s != SEALED_SPLIT)
    declared = str(episode.split)
    computed = split_of(str(episode.family_id))
    if declared and declared != computed:
        raise SplitRefusedError(
            f"episode {episode.episode_id!r} declares split {declared!r} but "
            f"family {episode.family_id!r} is in {computed!r}")
    if computed not in allowed:
        raise SplitRefusedError(
            f"REFUSED: FL4 targets may be computed on {list(allowed)} only "
            f"(contract §7/§12); family {episode.family_id!r} is {computed}")
    return computed


def assert_train_family(episode: Episode) -> None:
    """Target computation is TRAIN-families-only (contract §7)."""
    assert_target_split(episode, TARGET_SPLITS_TRAIN)


# ---------------------------------------------------------------------------
# target computation
# ---------------------------------------------------------------------------
def _scoring_episode(episode: Episode) -> Episode:
    """Episode whose QUERY attempts carry the CORRECT canonical answers.

    Pre-generated scripted query attempts are already the true answers in the
    frozen policy, but the target is defined over the CORRECT answers, so this
    substitutes them explicitly rather than trusting the policy.
    """
    events = _events(episode)
    out: list[Event] = []
    for event in events:
        if (_role_name(event) == Role.MODEL_ATTEMPT.value
                and event.interaction_index == QUERY_INDEX):
            answer = _item_answer(episode, event.item_id)
            out.append(dataclasses.replace(event, text=format_answer(answer)))
        else:
            out.append(event)
    return dataclasses.replace(episode, events=out)


def token_aligned_answer_span(offsets: Sequence[tuple[int, int]], text: str,
                               line_span: tuple[int, int]) -> tuple[int, int]:
    """Snap an ``ANSWER:`` line span to EXACT token boundaries.

    W2's loss masks select the tokens of the ``ANSWER: <payload>`` line under
    the frozen ``expand_whitespace`` straddle policy — a byte-level BPE token
    such as ``ĠANSWER`` carries the preceding separator space.  W4's
    ``answer_logprob_batch`` deliberately requires a STRICTLY token-aligned
    character span (a silently snapped span would change the estimand).

    This selects exactly the tokens W2 would put loss on, then returns THEIR
    character boundaries, so the FL4 target measures the log-likelihood of
    exactly the tokens FL3 trains on, with no re-alignment inside the scorer.
    Any straddle that is not pure whitespace is a hard error from W2's policy.
    """
    t0, t1 = char_span_to_token_span(
        offsets, int(line_span[0]), int(line_span[1]),
        on_straddle=MASK_STRADDLE_POLICY, what="FL4 query answer span",
        text=text)
    return int(offsets[t0][0]), int(offsets[t1 - 1][1])


def query_answer_score(bundle: Any, scoring_episode: Episode,
                       keep_indices: Sequence[int]) -> dict[str, Any]:
    """Mean per-token logprob of the correct QUERY answers under one context.

    ``keep_indices`` is the rendered event subset (the presence / ablation
    variant).  Every query attempt in the subset is scored teacher-forced with
    W4's :func:`evaluation.scoring.answer_logprob_batch`, and the reported
    score is the MEAN OVER QUERY ITEMS of their per-item mean per-token
    log-probabilities (both halves of the FL4 difference use this same
    definition, so the difference is well defined).
    """
    keep = sorted(int(i) for i in keep_indices)
    events = _events(scoring_episode)
    query_events = [i for i in keep
                    if _role_name(events[i]) == Role.MODEL_ATTEMPT.value
                    and events[i].interaction_index == QUERY_INDEX]
    if not query_events:
        raise ValueHeadError(
            "the scored context contains no QUERY attempt; the FL4 target is "
            "undefined without query answers")
    rendered = render_subset(scoring_episode, keep)
    spans = _spans_by_event(rendered)
    _ids, offsets = encode_with_offsets(getattr(bundle, "tokenizer"),
                                        rendered.text)
    requests: list[ScoreRequest] = []
    for index in query_events:
        char_start, char_end = spans[index]
        answer_spans = answer_spans_for_event(
            rendered.text, char_start, char_end, events[index].text)
        if len(answer_spans) != 1:
            raise ValueHeadError(
                f"query attempt event {index} has {len(answer_spans)} ANSWER "
                "lines; exactly one is required for scoring")
        requests.append(ScoreRequest(
            rendered.text,
            token_aligned_answer_span(offsets, rendered.text, answer_spans[0])))
    values = answer_logprob_batch(bundle, requests)
    per_item = [float(v) for v in values]
    return {
        "score": float(sum(per_item) / len(per_item)),
        "per_item": per_item,
        "query_event_indices": query_events,
        "n_kept_events": len(keep),
    }


def episode_targets(bundle: Any, episode: Episode, plan: dict | None = None,
                    *, require_train_split: bool = True,
                    allowed_splits: Sequence[str] | None = None) -> dict[str, Any]:
    """Realized learning value ``y_j`` for every ablatable item of an episode.

    Teacher-forced, computed with whatever model ``bundle`` carries (the
    campaign passes the final FL3 checkpoint, contract §7).  By default TRAIN
    families only.  ``allowed_splits=TARGET_SPLITS_DEV_EVAL`` additionally
    admits DEVELOPMENT, which the §7 DEV evaluation of the head requires;
    SEALED_TEST is refused under every setting.
    """
    if allowed_splits is None:
        allowed_splits = (TARGET_SPLITS_TRAIN if require_train_split
                          else TARGET_SPLITS_DEV_EVAL)
    assert_target_split(episode, allowed_splits)
    plan = validate_ablation_plan(
        episode, plan if plan is not None else ablation_plan_for_episode(episode))
    scoring = _scoring_episode(episode)
    events = _events(scoring)
    query_events = _query_attempt_indices(scoring)
    if not query_events:
        raise ValueHeadError(
            f"episode {episode.episode_id!r} has no QUERY attempts")
    # Events after the last query attempt cannot influence the log-probability
    # of the query answers under a causal LM, so they are dropped: the score is
    # numerically identical and the forward is shorter.
    horizon = max(query_events)
    universe = [i for i in range(len(events)) if i <= horizon]

    full = query_answer_score(bundle, scoring, universe)
    targets: list[dict[str, Any]] = []
    for entry in plan["plans"]:
        if entry.get("ablation_id") == "FULL":
            continue
        ablated = [int(i) for i in entry.get("ablated_event_indices", [])]
        if any(i > horizon for i in ablated):
            # a feedback item after the last query cannot change the query
            # log-probability; it carries no realized learning value signal
            continue
        keep = [i for i in universe if i not in set(ablated)]
        part = query_answer_score(bundle, scoring, keep)
        index = ablated[0]
        targets.append({
            "ablation_id": entry["ablation_id"],
            "event_index": index,
            "item_id": entry.get("ablated_item_id") or events[index].item_id,
            "role": _role_name(events[index]),
            "target": float(full["score"] - part["score"]),
            "score_present": float(full["score"]),
            "score_ablated": float(part["score"]),
        })
    return {
        "episode_id": episode.episode_id,
        "family_id": episode.family_id,
        "split": episode.split,
        "horizon_event_index": horizon,
        "full": full,
        "targets": targets,
    }


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
@dataclass
class ValueItem:
    """One scoreable feedback item: head input, realized target, surfaces."""

    episode_id: str
    family_id: str
    split: str
    item_id: str
    event_index: int
    role: str
    ablation_id: str
    target: float
    hidden: torch.Tensor
    surface: dict[str, float] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "family_id": self.family_id,
            "split": self.split,
            "item_id": self.item_id,
            "event_index": self.event_index,
            "role": self.role,
            "ablation_id": self.ablation_id,
            "target": self.target,
            "surface": dict(self.surface),
        }


def _surface_features(episode: Episode, event: Event) -> dict[str, float]:
    """Contract §7 surface baselines (never head inputs)."""
    text = event.text or ""
    words = set(w.lower() for w in _WORD_RE.findall(text))
    query_words: set[str] = set()
    for record in (episode.items or {}).values():
        if isinstance(record, dict) and record.get("role") == "query":
            instance = record.get("instance", {})
            query_words |= set(
                w.lower() for w in _WORD_RE.findall(str(instance.get("prompt_text", ""))))
    overlap = (len(words & query_words) / len(words)) if words else 0.0
    role_code = {Role.FEEDBACK.value: 0.0, Role.HINT.value: 1.0,
                 Role.REVEAL.value: 2.0}.get(_role_name(event), -1.0)
    return {
        "item_length": float(len(text)),
        "item_type": role_code,
        "lexical_overlap": float(overlap),
    }


def build_value_dataset(bundle: Any, episodes: Iterable[Episode],
                        plans: dict[str, dict] | None = None,
                        *, require_train_split: bool = True,
                        allowed_splits: Sequence[str] | None = None,
                        with_hidden: bool = True) -> list[ValueItem]:
    """Head inputs + realized targets for a set of episodes.

    ``plans`` maps ``episode_id -> pre-generated plan``; a missing plan is
    produced by W1's own generator.  ``with_hidden=False`` skips the head-input
    forward (target-only use, e.g. dataset statistics).  ``allowed_splits``
    is passed through to :func:`episode_targets`; SEALED_TEST is never
    admissible.
    """
    plans = dict(plans or {})
    items: list[ValueItem] = []
    for episode in episodes:
        result = episode_targets(bundle, episode, plans.get(episode.episode_id),
                                 require_train_split=require_train_split,
                                 allowed_splits=allowed_splits)
        events = _events(episode)
        for record in result["targets"]:
            index = int(record["event_index"])
            hidden = (head_input_hidden(bundle, episode, index)
                      if with_hidden
                      else torch.zeros(0, dtype=VALUE_HEAD_DTYPE))
            items.append(ValueItem(
                episode_id=episode.episode_id,
                family_id=episode.family_id,
                split=str(episode.split),
                item_id=str(record["item_id"]),
                event_index=index,
                role=str(record["role"]),
                ablation_id=str(record["ablation_id"]),
                target=float(record["target"]),
                hidden=hidden,
                surface=_surface_features(episode, events[index]),
            ))
    return items


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ValueHeadTrainConfig:
    """Frozen-by-default head training loop (see the module docstring)."""

    learning_rate: float = VALUE_HEAD_LR
    epochs: int = 40
    groups_per_batch: int = 8
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    ranking_margin: float = RANKING_MARGIN
    mse_weight: float = MSE_WEIGHT
    seed: int = 20260809

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["betas"] = list(self.betas)
        return data


def _group_items(items: Sequence[ValueItem]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(item.episode_id, []).append(i)
    return [groups[k] for k in sorted(groups)]


def pairwise_ranking_hinge(preds: torch.Tensor, targets: torch.Tensor,
                           margin: float = RANKING_MARGIN) -> torch.Tensor:
    """Mean ``relu(margin - (v_i - v_k))`` over ordered pairs with ``y_i > y_k``.

    Returns a zero scalar (with graph) when the group has no comparable pair.
    """
    diff_pred = preds.unsqueeze(0) - preds.unsqueeze(1)      # [k, k] = v_i - v_j
    diff_targ = targets.unsqueeze(0) - targets.unsqueeze(1)
    mask = (diff_targ > 0).to(preds.dtype)
    n_pairs = float(mask.sum().item())
    if n_pairs == 0.0:
        return preds.sum() * 0.0
    losses = torch.relu(margin - diff_pred) * mask
    return losses.sum() / n_pairs


def train_value_head(head: ValueHead, items: Sequence[ValueItem],
                     cfg: ValueHeadTrainConfig | None = None
                     ) -> dict[str, Any]:
    """Own small training loop: pairwise hinge (primary) + 0.5 * MSE(z-target).

    Deterministic: the epoch permutation comes from a ``torch.Generator``
    seeded through ``derive_seed``; nothing touches the global RNG.
    """
    cfg = cfg or ValueHeadTrainConfig()
    if not items:
        raise ValueHeadError("no value items supplied")
    groups = _group_items(items)
    targets = torch.tensor([it.target for it in items], dtype=VALUE_HEAD_DTYPE)
    mean = float(targets.mean().item())
    std = float(targets.std(unbiased=False).item())
    if std < MIN_TARGET_STD:
        raise ValueHeadError(
            f"targets have (near) zero spread (std={std}); z-scoring them "
            "would amplify numerical noise into a training signal")
    head.set_target_stats(mean, std)
    z = (targets - mean) / std

    hidden = torch.stack([it.hidden.to(VALUE_HEAD_DTYPE) for it in items], dim=0)
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=cfg.learning_rate, betas=tuple(cfg.betas), eps=cfg.eps,
        weight_decay=cfg.weight_decay)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(derive_seed(cfg.seed, "FL4_HEAD_TRAIN", len(items),
                                len(groups), cfg.epochs))
    history: list[dict[str, float]] = []
    head.train()
    for epoch in range(int(cfg.epochs)):
        order = torch.randperm(len(groups), generator=gen).tolist()
        epoch_loss = 0.0
        epoch_rank = 0.0
        epoch_mse = 0.0
        n_batches = 0
        for start in range(0, len(order), int(cfg.groups_per_batch)):
            batch = [groups[i] for i in order[start:start + int(cfg.groups_per_batch)]]
            flat = [i for g in batch for i in g]
            preds = head(hidden[flat])
            zb = z[flat]
            offset = 0
            rank_terms: list[torch.Tensor] = []
            for g in batch:
                sl = slice(offset, offset + len(g))
                rank_terms.append(pairwise_ranking_hinge(
                    preds[sl], zb[sl], cfg.ranking_margin))
                offset += len(g)
            rank_loss = torch.stack(rank_terms).mean()
            mse_loss = torch.mean((preds - zb) ** 2)
            loss = rank_loss + float(cfg.mse_weight) * mse_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_rank += float(rank_loss.detach())
            epoch_mse += float(mse_loss.detach())
            n_batches += 1
        history.append({
            "epoch": epoch,
            "loss": epoch_loss / max(1, n_batches),
            "ranking_hinge": epoch_rank / max(1, n_batches),
            "mse": epoch_mse / max(1, n_batches),
        })
    head.eval()
    return {
        "schema": "flb200.value_head_training.v1",
        "config": cfg.to_dict(),
        "head_config": head.config(),
        "head_config_hash": head.config_hash(),
        "n_items": len(items),
        "n_groups": len(groups),
        "target_mean": mean,
        "target_std": std,
        "history": history,
        "final_loss": history[-1]["loss"] if history else None,
    }


# ---------------------------------------------------------------------------
# evaluation helpers (feed W4's analysis.stats / evaluation.metrics)
# ---------------------------------------------------------------------------
def value_head_predictions(head: ValueHead, items: Sequence[ValueItem]
                           ) -> dict[str, list]:
    """Per-item predictions on the head's z-scored target scale."""
    if not items:
        return {"predictions": [], "targets": [], "groups": [], "item_ids": []}
    hidden = torch.stack([it.hidden.to(VALUE_HEAD_DTYPE) for it in items], dim=0)
    preds = head.predict(hidden)
    mean = float(head.target_mean.item())
    std = float(head.target_std.item())
    return {
        "predictions": [float(v) for v in preds],
        "targets": [(it.target - mean) / std for it in items],
        "raw_targets": [it.target for it in items],
        "groups": [it.episode_id for it in items],
        "item_ids": [it.item_id for it in items],
        "families": [it.family_id for it in items],
    }


def surface_heuristic_predictions(items: Sequence[ValueItem], name: str
                                  ) -> list[float]:
    """Contract §7 DEV baseline: item length / item type / lexical overlap."""
    if name not in SURFACE_HEURISTICS:
        raise ValueHeadError(
            f"unknown surface heuristic {name!r}; frozen: {list(SURFACE_HEURISTICS)}")
    return [float(it.surface.get(name, 0.0)) for it in items]


def evaluate_value_head(head: ValueHead, items: Sequence[ValueItem]
                        ) -> dict[str, Any]:
    """METRIC 14 for the head plus the frozen comparison baselines.

    Delegates every statistic to W4's ``evaluation.metrics.value_ranking_metrics``
    (which delegates to ``analysis.stats.value_head_metrics``); nothing is
    re-implemented here.
    """
    from foundation_learner.evaluation.metrics import value_ranking_metrics

    preds = value_head_predictions(head, items)
    out: dict[str, Any] = {
        "schema": "flb200.value_head_eval.v1",
        "n_items": len(items),
        "head": value_ranking_metrics(preds["predictions"], preds["targets"],
                                      preds["groups"]),
        "baselines": {},
    }
    for name in SURFACE_HEURISTICS:
        out["baselines"][name] = value_ranking_metrics(
            surface_heuristic_predictions(items, name), preds["targets"],
            preds["groups"])
    return out


# ---------------------------------------------------------------------------
# campaign stage adapter (contract §7 FL4, §10 promotion, §11 stage table)
# ---------------------------------------------------------------------------
def run_fl4_stage(ctx: Any, stage: Any) -> dict[str, Any]:
    """FL4 rung entry point for ``campaign/stage_definitions.py``.

    Work, at the scale the context allows:

    1. restore the FINAL FL3 checkpoint into a FRESH bundle (contract §7's
       target definition); when no FL3 checkpoint exists in this session the
       base checkpoint is used and the payload SAYS SO;
    2. compute realized learning values on TRAIN episodes with their
       PRE-GENERATED ablation plans, and train the head on them;
    3. compute the same realized values on DEVELOPMENT episodes and evaluate
       the head there (Spearman, pairwise ranking accuracy, calibration,
       top-1 regret, versus the frozen surface heuristics).

    Promotion interface (``campaign/promotion.py``):
    ``pairwise_ranking_accuracy`` is surfaced at the TOP LEVEL of the payload
    because ``stage_definitions.fl6_entry`` reads exactly
    ``ctx.results["FL4"]["pairwise_ranking_accuracy"]`` and passes it to
    ``promotion.promote_fl6`` (threshold 0.55).  ``status == "COMPLETE"`` is
    what ``fl7_entry`` reads as ``fl4_succeeded``.  The realized per-episode
    scoreable-item counts are published to
    ``ctx.extra["fl4_scoreable_items_per_episode"]`` for
    ``promotion.promote_fl4``'s data-sufficiency evidence.
    """
    from foundation_learner.mechanisms import stage_support as _s

    n_train = _s.stage_episode_count(ctx, stage, "fl4_train_episodes")
    n_dev = _s.stage_episode_count(ctx, stage, "fl4_dev_episodes")
    train_eps = _s.train_episodes(ctx, n_train)
    dev_eps = _s.dev_episodes(ctx, stage, count=n_dev)
    if not train_eps or not dev_eps:
        return _s.finish_stage(ctx, stage, _s.mechanism_payload(
            stage, _s.STATUS_SKIPPED_INPUT,
            reason="no TRAIN or DEVELOPMENT episodes are available"))
    train_plans = _s.load_ablation_plans(ctx, "TRAIN")
    dev_plans = _s.load_ablation_plans(ctx, "DEVELOPMENT")

    bundle = ctx.bundle_factory()                       # FRESH load (§7)
    target_model = _s.arm_checkpoint_state(ctx, bundle, "FL3")

    train_items = build_value_dataset(bundle, train_eps, train_plans)
    dev_items = build_value_dataset(bundle, dev_eps, dev_plans,
                                    allowed_splits=TARGET_SPLITS_DEV_EVAL)
    if not train_items or not dev_items:
        return _s.finish_stage(ctx, stage, _s.mechanism_payload(
            stage, _s.STATUS_SKIPPED_INPUT,
            reason="the episodes carry no scoreable feedback items"))

    cfg = ValueHeadTrainConfig(
        seed=int(ctx.root_seed),
        epochs=int(ctx.extra.get("fl4_epochs",
                                 5 if ctx.rehearsal else ValueHeadTrainConfig.epochs)))
    head = ValueHead(int(bundle.model.config.hidden_size), seed=int(ctx.root_seed))
    training = train_value_head(head, train_items, cfg)
    head.freeze_()

    dev_report = evaluate_value_head(head, dev_items)
    train_report = evaluate_value_head(head, train_items)
    accuracy = dev_report["head"]["pairwise"]["accuracy"]

    # cross-stage handoff: FL6 and FL7 need the FROZEN head object, and the
    # tensors are also persisted next to the report for provenance.
    ctx.extra[_s.HEAD_CTX_KEY] = head
    head_path = _s.save_head_state(ctx, stage.stage_id, head)
    counts = _s.count_scoreable_items(train_eps)
    ctx.extra["fl4_scoreable_items_per_episode"] = counts

    payload = _s.mechanism_payload(
        stage, _s.STATUS_COMPLETE,
        # --- promotion interface (campaign/promotion.py) -------------------
        pairwise_ranking_accuracy=accuracy,
        # --- the rung's real results ---------------------------------------
        target_model=target_model,
        n_train_episodes=len(train_eps), n_dev_episodes=len(dev_eps),
        n_train_items=len(train_items), n_dev_items=len(dev_items),
        scoreable_items_per_episode=counts,
        training=training,
        dev=dev_report, train=train_report,
        head_config=head.config(), head_config_hash=head.config_hash(),
        head_state_path=head_path,
        target_definition=("mean per-token logprob of the correct QUERY "
                           "answers, item present minus item ablated"),
        dev_split_note=("head TRAINING targets are TRAIN-families-only "
                        "(contract §7); the ranking / calibration / regret "
                        "numbers above are the §7 DEVELOPMENT evaluation"),
    )
    return _s.finish_stage(ctx, stage, payload)
