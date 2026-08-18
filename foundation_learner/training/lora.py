"""Custom LoRA (Amendment 3: no ``peft`` at B200 runtime).

The B200 container lock contains no ``peft``, so every low-rank-adapter need of
this campaign (PEFT_MODE for the core arms, the FL7 fast adapter, the FL8 slow
bank) is served by this module.  ``peft==0.19.1`` exists in the local venv only
and is used exclusively as a numerical oracle in a unit test.

Mathematics (exactly what is implemented)
-----------------------------------------
For a wrapped ``nn.Linear`` with weight ``W`` the forward is::

    y = W x + b  +  (alpha / r) * B @ (A @ dropout(x))  +  sum_i s_i * B_i @ (A_i @ x)

* ``A`` has shape ``(r, in_features)`` and is Kaiming-uniform initialised
  (``a = sqrt(5)``, the peft/torch convention), ``B`` has shape
  ``(out_features, r)`` and is ZERO initialised, so the adapter is an exact
  no-op at attachment time.
* ``(alpha / r)`` is the scaling; ``dropout`` is identity for the frozen specs
  (dropout 0.0 everywhere in this campaign).
* The ``s_i, A_i, B_i`` terms are FROZEN merged components appended by
  :func:`merge_scaled_` (see its docstring).  They are buffers, never trained.

The "effective delta" of a layer is
``D = (alpha/r) B A + sum_i s_i B_i A_i``; the "trainable delta" is only the
first term.  Base weights are never modified by any function in this module,
so the frozen checkpoint artifact stays byte-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import torch
from torch import nn

from foundation_learner.training.model_loading import derive_seed


class LoraError(RuntimeError):
    """Raised on malformed LoRA specs, targets, or state dicts."""


@dataclass(frozen=True)
class LoraSpec:
    """Frozen description of a LoRA attachment.

    ``target_suffixes`` are matched against the DOTTED MODULE NAME suffix, i.e.
    a module named ``model.layers.7.self_attn.q_proj`` matches the suffix
    ``"q_proj"`` and also ``"self_attn.q_proj"``.  Only ``nn.Linear`` modules
    are eligible; a suffix that matches nothing is a hard error.
    """

    target_suffixes: tuple[str, ...]
    rank: int
    alpha: float
    dropout: float = 0.0
    seed: int = 0
    freeze_base: bool = True
    name: str = "lora"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise LoraError(f"rank must be positive, got {self.rank}")
        if not self.target_suffixes:
            raise LoraError("target_suffixes must be non-empty")
        if not 0.0 <= self.dropout < 1.0:
            raise LoraError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target_suffixes": list(self.target_suffixes),
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "seed": self.seed,
            "freeze_base": self.freeze_base,
        }


class LoraLinear(nn.Module):
    """``nn.Linear`` wrapper implementing the equations in the module docstring."""

    def __init__(self, base: nn.Linear, spec: LoraSpec, layer_name: str) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise LoraError(f"{layer_name}: LoRA target must be nn.Linear, got {type(base)}")
        self.base_layer = base
        self.spec = spec
        self.layer_name = layer_name
        self.r = int(spec.rank)
        self.scaling = spec.scaling
        self.enabled = True
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(
            torch.zeros(self.r, base.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.r, device=device, dtype=dtype)
        )
        self.lora_dropout = (
            nn.Dropout(p=spec.dropout) if spec.dropout > 0.0 else nn.Identity()
        )
        self._merged_scales: list[float] = []
        self._n_merged = 0
        self.reset_lora_parameters()

    # -- initialisation ----------------------------------------------------
    def _init_generator(self) -> torch.Generator:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(derive_seed(int(self.spec.seed), "LORA_INIT", self.layer_name))
        return gen

    def reset_lora_parameters(self) -> None:
        """Kaiming-uniform ``A``, zero ``B`` -> delta is exactly zero."""
        gen = self._init_generator()
        fan_in = self.lora_A.shape[1]
        # torch.nn.init.kaiming_uniform_(a=sqrt(5)) on a (r, fan_in) tensor.
        gain = math.sqrt(2.0 / (1 + 5.0))
        bound = gain * math.sqrt(3.0 / fan_in)
        sample = torch.empty(
            self.lora_A.shape, dtype=torch.float32, device="cpu"
        ).uniform_(-bound, bound, generator=gen)
        with torch.no_grad():
            self.lora_A.copy_(sample.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
            self.lora_B.zero_()

    # -- merged components -------------------------------------------------
    def merged_components(self) -> list[tuple[float, torch.Tensor, torch.Tensor]]:
        out = []
        for i, scale in enumerate(self._merged_scales):
            out.append(
                (scale, getattr(self, f"merged_A_{i}"), getattr(self, f"merged_B_{i}"))
            )
        return out

    def add_merged_component(
        self, scale: float, a: torch.Tensor, b: torch.Tensor
    ) -> None:
        idx = self._n_merged
        self.register_buffer(
            f"merged_A_{idx}",
            a.detach().clone().to(device=self.lora_A.device, dtype=self.lora_A.dtype),
            persistent=True,
        )
        self.register_buffer(
            f"merged_B_{idx}",
            b.detach().clone().to(device=self.lora_B.device, dtype=self.lora_B.dtype),
            persistent=True,
        )
        self._merged_scales.append(float(scale))
        self._n_merged += 1

    def clear_merged_components(self) -> None:
        for i in range(self._n_merged):
            delattr(self, f"merged_A_{i}")
            delattr(self, f"merged_B_{i}")
        self._merged_scales = []
        self._n_merged = 0

    # -- deltas ------------------------------------------------------------
    def trainable_delta(self) -> torch.Tensor:
        """Dense ``(out, in)`` trainable delta ``(alpha/r) B A`` (test/debug use)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def effective_delta(self) -> torch.Tensor:
        delta = self.trainable_delta()
        for scale, a, b in self.merged_components():
            delta = delta + scale * (b @ a)
        return delta

    def trainable_delta_fro_sq(self) -> torch.Tensor:
        """``||(alpha/r) B A||_F^2`` without materialising the dense delta.

        Uses ``||BA||_F^2 = tr(A^T B^T B A) = tr((B^T B)(A A^T))`` — an ``r x r``
        computation.  Exactness is pinned by a unit test against the dense form.
        """
        a = self.lora_A.float()
        b = self.lora_B.float()
        btb = b.t() @ b
        aat = a @ a.t()
        return (self.scaling**2) * torch.sum(btb * aat.t())

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        out = self.base_layer(x)
        if not self.enabled:
            return out
        dropped = self.lora_dropout(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(dropped, self.lora_A), self.lora_B
        )
        out = out + self.scaling * delta
        for scale, a, b in self.merged_components():
            out = out + scale * torch.nn.functional.linear(
                torch.nn.functional.linear(x, a), b
            )
        return out

    def extra_repr(self) -> str:  # pragma: no cover - debug helper
        return (
            f"name={self.layer_name}, r={self.r}, alpha={self.spec.alpha}, "
            f"scaling={self.scaling}, merged={self._n_merged}, enabled={self.enabled}"
        )


@dataclass
class LoraHandles:
    """Handle object returned by :func:`attach_lora`."""

    spec: LoraSpec
    model: nn.Module
    layers: dict[str, LoraLinear] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.layers)

    def names(self) -> list[str]:
        return sorted(self.layers)

    def parameters(self) -> Iterator[nn.Parameter]:
        for name in self.names():
            layer = self.layers[name]
            yield layer.lora_A
            yield layer.lora_B

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _matches(name: str, suffixes: Sequence[str]) -> bool:
    parts = name.split(".")
    for suffix in suffixes:
        want = suffix.split(".")
        if len(want) <= len(parts) and parts[-len(want):] == want:
            return True
    return False


def attach_lora(model: nn.Module, spec: LoraSpec) -> LoraHandles:
    """Wrap every matching ``nn.Linear`` with a :class:`LoraLinear` (§23)."""
    targets: list[tuple[str, nn.Module, str, nn.Linear]] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoraLinear):
            raise LoraError(f"{name}: LoRA already attached to this module")
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and _matches(full, spec.target_suffixes):
                targets.append((full, module, child_name, child))
    if not targets:
        raise LoraError(
            f"LoRA spec {spec.name!r} matched no nn.Linear module; "
            f"target_suffixes={list(spec.target_suffixes)}"
        )
    matched_suffixes = {
        suffix
        for suffix in spec.target_suffixes
        if any(_matches(full, [suffix]) for full, _, _, _ in targets)
    }
    missing = sorted(set(spec.target_suffixes) - matched_suffixes)
    if missing:
        raise LoraError(f"LoRA spec {spec.name!r}: no module matched suffixes {missing}")

    if spec.freeze_base:
        for param in model.parameters():
            param.requires_grad_(False)

    handles = LoraHandles(spec=spec, model=model)
    for full, parent, child_name, child in targets:
        wrapper = LoraLinear(child, spec, full)
        for param in wrapper.base_layer.parameters():
            if spec.freeze_base:
                param.requires_grad_(False)
        wrapper.lora_A.requires_grad_(True)
        wrapper.lora_B.requires_grad_(True)
        setattr(parent, child_name, wrapper)
        handles.layers[full] = wrapper
    return handles


def detach_lora_(handles: LoraHandles) -> None:
    """Restore the original ``nn.Linear`` modules (adapters discarded)."""
    for full, wrapper in list(handles.layers.items()):
        parent_name, _, child_name = full.rpartition(".")
        parent = handles.model.get_submodule(parent_name) if parent_name else handles.model
        setattr(parent, child_name, wrapper.base_layer)
    handles.layers.clear()


def set_lora_enabled(handles: LoraHandles, enabled: bool) -> None:
    """Enable/disable the adapter contribution (base forward is unaffected)."""
    for layer in handles.layers.values():
        layer.enabled = bool(enabled)


def lora_enabled(handles: LoraHandles) -> bool:
    values = {layer.enabled for layer in handles.layers.values()}
    if len(values) > 1:
        raise LoraError("inconsistent per-layer enabled flags")
    return values.pop() if values else False


def zero_lora_(handles: LoraHandles, *, reinit_a: bool = True) -> None:
    """Reset the trainable delta to EXACTLY zero (FL7 per-episode reset).

    ``reinit_a=True`` (default) additionally re-draws ``A`` from the layer's
    deterministic init generator, so a reset adapter is bitwise identical to a
    freshly attached one.  ``reinit_a=False`` only zeroes ``B`` (delta zero,
    ``A`` preserved).  Merged components (FL8 bank) are NOT touched.
    """
    with torch.no_grad():
        for layer in handles.layers.values():
            if reinit_a:
                layer.reset_lora_parameters()
            else:
                layer.lora_B.zero_()


def lora_state_dict(handles: LoraHandles) -> dict[str, torch.Tensor]:
    """Detached CPU copy of every adapter tensor, keyed by module path."""
    state: dict[str, torch.Tensor] = {}
    for name in handles.names():
        layer = handles.layers[name]
        state[f"{name}.lora_A"] = layer.lora_A.detach().to("cpu").clone()
        state[f"{name}.lora_B"] = layer.lora_B.detach().to("cpu").clone()
        for i, (scale, a, b) in enumerate(layer.merged_components()):
            state[f"{name}.merged_{i}.scale"] = torch.tensor(scale, dtype=torch.float64)
            state[f"{name}.merged_{i}.A"] = a.detach().to("cpu").clone()
            state[f"{name}.merged_{i}.B"] = b.detach().to("cpu").clone()
    return state


def load_lora_state_(handles: LoraHandles, state: dict[str, torch.Tensor]) -> None:
    """Strict inverse of :func:`lora_state_dict` (shape and key checked)."""
    expected = set(lora_state_dict(handles))
    got = set(state)
    trainable_expected = {k for k in expected if k.endswith((".lora_A", ".lora_B"))}
    trainable_got = {k for k in got if k.endswith((".lora_A", ".lora_B"))}
    if trainable_expected != trainable_got:
        missing = sorted(trainable_expected - trainable_got)
        unknown = sorted(trainable_got - trainable_expected)
        raise LoraError(
            f"LoRA state dict mismatch; missing={missing[:6]} unknown={unknown[:6]}"
        )
    with torch.no_grad():
        for name in handles.names():
            layer = handles.layers[name]
            for attr in ("lora_A", "lora_B"):
                tensor = state[f"{name}.{attr}"]
                target = getattr(layer, attr)
                if tuple(tensor.shape) != tuple(target.shape):
                    raise LoraError(
                        f"{name}.{attr}: shape {tuple(tensor.shape)} != "
                        f"{tuple(target.shape)}"
                    )
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            layer.clear_merged_components()
            i = 0
            while f"{name}.merged_{i}.A" in state:
                layer.add_merged_component(
                    float(state[f"{name}.merged_{i}.scale"].item()),
                    state[f"{name}.merged_{i}.A"],
                    state[f"{name}.merged_{i}.B"],
                )
                i += 1


def lora_frobenius_norm(handles: LoraHandles, *, include_merged: bool = False) -> float:
    """Global Frobenius norm over all layers' deltas.

    ``sqrt(sum_layers ||delta_layer||_F^2)`` where ``delta_layer`` is the
    TRAINABLE delta ``(alpha/r) B A`` by default (this is the quantity FL7
    bounds with beta = 1.0).  With ``include_merged=True`` the frozen merged
    components are included, which requires materialising dense deltas.
    """
    total = torch.zeros((), dtype=torch.float64)
    for layer in handles.layers.values():
        if include_merged:
            delta = layer.effective_delta().detach().double()
            total = total + torch.sum(delta * delta).cpu()
        else:
            total = total + layer.trainable_delta_fro_sq().detach().double().cpu()
    return float(torch.sqrt(total).item())


def clip_lora_frobenius_(
    handles: LoraHandles, max_norm: float, *, include_merged: bool = False
) -> float:
    """Clip the global trainable-delta Frobenius norm; return the PRE-clip norm.

    The delta is linear in ``B``, so scaling every ``B`` by
    ``max_norm / norm`` scales the global norm by exactly that factor.  Merged
    components are frozen and are never rescaled.
    """
    if max_norm <= 0:
        raise LoraError(f"max_norm must be positive, got {max_norm}")
    norm = lora_frobenius_norm(handles, include_merged=include_merged)
    if norm > max_norm and norm > 0.0:
        factor = max_norm / norm
        with torch.no_grad():
            for layer in handles.layers.values():
                layer.lora_B.mul_(factor)
    return norm


def merge_scaled_(dst_h: LoraHandles, src_h: LoraHandles, scale: float) -> int:
    """Add ``scale * (alpha_src/r_src) * B_src A_src`` into ``dst``'s effective delta.

    EXACT SEMANTICS (FL8 consolidation, contract §7):

    * For every module name present in BOTH handles, a new FROZEN low-rank
      component ``(scale * alpha_src/r_src, A_src.clone(), B_src.clone())`` is
      appended to the destination layer's merged-component list.  The
      destination's effective delta afterwards is
      ``(alpha_d/r_d) B_d A_d + sum_i s_i B_i A_i`` and therefore contains the
      source delta EXACTLY.
    * The destination's trainable ``(A, B)`` pair is NOT modified: projecting a
      rank-``r_src`` delta into a rank-``r_dst`` pair would be lossy, and a
      lossy merge must never masquerade as consolidation.
    * Base weights are NOT modified (the frozen checkpoint stays intact).
    * Merged components are non-trainable buffers, are carried by
      :func:`lora_state_dict`, and are removable with
      ``LoraLinear.clear_merged_components``.

    Returns the number of layers merged.  Module names present in only one of
    the two handles are a hard error (a silent partial merge would corrupt the
    consolidation claim).
    """
    dst_names = set(dst_h.names())
    src_names = set(src_h.names())
    if dst_names != src_names:
        raise LoraError(
            "merge_scaled_ requires identical target modules; "
            f"only-in-dst={sorted(dst_names - src_names)[:6]} "
            f"only-in-src={sorted(src_names - dst_names)[:6]}"
        )
    merged = 0
    with torch.no_grad():
        for name in sorted(dst_names):
            src = src_h.layers[name]
            dst = dst_h.layers[name]
            dst.add_merged_component(
                float(scale) * src.scaling, src.lora_A, src.lora_B
            )
            merged += 1
    return merged


def iter_lora_parameters(handles: Iterable[LoraHandles]) -> Iterator[nn.Parameter]:
    for handle in handles:
        yield from handle.parameters()
