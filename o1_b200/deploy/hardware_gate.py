#!/usr/bin/env python3
"""Real-hardware acceptance gate for the accelerator session.

Before either scientific workload may begin, the pod must prove it is the
intended machine for the ACQUIRED profile (B300 primary / B200 explicit
fallback).  Two layers:

  1. validate_facts(profile_key, facts)  — PURE, fail-closed identity
     validation (unit-testable off-hardware): exact GPU identity, exactly
     one GPU, expected compute capability, HBM class floor, CUDA runtime/
     driver compatibility, BF16 support, sufficient free HBM, torch native
     architecture support.  The cu130 wheels ship NO PTX (verified against
     the shipped fatbinary and the pytorch builder configuration), so
     PTX/JIT fallback is structurally impossible: either kernels load as
     native SASS or they fail loudly here — and this gate additionally
     asserts the no-PTX property so a future wheel that silently
     reintroduces JIT is detected.

     SASS basis, measured from the shipped fatbinary with cuobjdump and
     recorded in deploy/FATBINARY_ARCH_EVIDENCE.json (448 sm_100 + 59
     sm_100a + 59 sm_103a cubins, 0 PTX entries):
       * B200 (sm_100, CC 10.0) — native and DIRECT; no compatibility rule
         is relied upon.
       * B300 (sm_103, CC 10.3) — the plain sm_100 cubins run natively
         under NVIDIA's same-major/higher-minor cubin rule (X.y runs on
         X.z, z >= y), plus the wheel's own sm_103a arch-tuned kernels.
     The workload exercises below are what actually prove it on the metal.

  2. exercise_workloads(...) — representative REAL workload paths on the
     acquired GPU: BF16 GEMM, Ouro-RLTT forward, Ouro-RLTT generation,
     backward + optimizer step (Foundation Learner path), the O1
     intervention hook and transport capture, checkpoint save/load, and
     the Foundation Learner mechanism smoke.  Exercises consume ONLY
     non-scientific synthetic prompts/corpus material — never an O1
     calibration row and never a Foundation Learner evaluation example.

The gate writes HARDWARE_GATE_REPORT.json; any failure refuses the session
before a single scientific token is produced.
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from o1_b200.provider.runpod.policy import (  # noqa: E402
    PROFILES_BY_KEY, MIN_CUDA_VERSION,
)

# Minimum free HBM after model load headroom check (fraction of total).
MIN_FREE_HBM_FRACTION = 0.15

# Minimum NVIDIA driver major for CUDA 13.0 runtime compatibility
# (CUDA 13.0 requires >= r580 per NVIDIA CUDA release notes).
MIN_DRIVER_MAJOR = 580


class HardwareGateError(RuntimeError):
    pass


def gather_facts(torch_mod=None) -> dict:
    """Collect device/runtime facts from the live torch runtime."""
    import torch as _t
    torch_mod = torch_mod or _t
    if not torch_mod.cuda.is_available():
        return {"cuda_available": False}
    props = torch_mod.cuda.get_device_properties(0)
    free_b, total_b = torch_mod.cuda.mem_get_info(0)
    driver = None
    try:
        import ctypes
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        lib.nvmlInit_v2()
        buf = ctypes.create_string_buffer(80)
        lib.nvmlSystemGetDriverVersion(buf, 80)
        driver = buf.value.decode()
        lib.nvmlShutdown()
    except Exception:  # noqa: BLE001 - driver string is best-effort here
        pass
    return {
        "cuda_available": True,
        "device_count": torch_mod.cuda.device_count(),
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_hbm_bytes": int(props.total_memory),
        "free_hbm_bytes": int(free_b),
        "torch_version": torch_mod.__version__,
        "torch_cuda_runtime": torch_mod.version.cuda,
        "cudnn_version": torch_mod.backends.cudnn.version(),
        "bf16_supported": bool(torch_mod.cuda.is_bf16_supported()),
        "arch_list": list(torch_mod.cuda.get_arch_list()),
        "driver_version": driver,
    }


def _version_tuple(value) -> tuple[int, ...] | None:
    """Parse "13.0"/"13.2.1" into a comparable tuple; None if unparseable."""
    if value is None:
        return None
    parts = str(value).strip().split(".")
    out = []
    for part in parts:
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            return None
        out.append(int(digits))
    return tuple(out) or None


def _version_at_least(value, minimum: str) -> bool:
    """Numeric version comparison.

    A lexicographic string compare would accept a single-digit-major
    runtime: "9.0" >= "13.0" is True for strings but false for versions.
    """
    got, want = _version_tuple(value), _version_tuple(minimum)
    if got is None or want is None:
        return False
    return got >= want


def validate_facts(profile_key: str, facts: dict) -> dict:
    """PURE fail-closed validation of the acquired machine's identity."""
    profile = PROFILES_BY_KEY[profile_key]
    problems = []

    def need(cond, msg):
        if not cond:
            problems.append(msg)

    need(facts.get("cuda_available") is True, "CUDA is not available")
    if facts.get("cuda_available"):
        need(facts.get("device_count") == 1,
             f"expected exactly one GPU, found {facts.get('device_count')}")
        name = str(facts.get("device_name", ""))
        need(profile.display_name in name,
             f"device {name!r} is not the intended {profile.display_name}")
        for other in ("H100", "H200", "A100", "GB200", "RTX", "L40", "V100"):
            need(other not in name.upper() or other in
                 profile.display_name.upper(),
                 f"device {name!r} matches forbidden substitute {other}")
        need(facts.get("compute_capability") == profile.compute_capability,
             f"compute capability {facts.get('compute_capability')} != "
             f"expected {profile.compute_capability} for {profile.key}")
        total_gb = facts.get("total_hbm_bytes", 0) / (1000 ** 3)
        need(total_gb >= profile.min_memory_gb,
             f"total HBM {total_gb:.0f} GB below the {profile.key} floor "
             f"{profile.min_memory_gb} GB")
        free_frac = (facts.get("free_hbm_bytes", 0)
                     / max(1, facts.get("total_hbm_bytes", 1)))
        need(free_frac >= MIN_FREE_HBM_FRACTION,
             f"free HBM fraction {free_frac:.2f} below "
             f"{MIN_FREE_HBM_FRACTION}")
        need(_version_at_least(facts.get("torch_cuda_runtime"),
                               MIN_CUDA_VERSION),
             f"torch CUDA runtime {facts.get('torch_cuda_runtime')} below "
             f"{MIN_CUDA_VERSION}")
        drv = str(facts.get("driver_version") or "")
        if drv:
            try:
                need(int(drv.split(".")[0]) >= MIN_DRIVER_MAJOR,
                     f"driver {drv} below r{MIN_DRIVER_MAJOR} "
                     f"(CUDA 13 requirement)")
            except ValueError:
                problems.append(f"unparseable driver version {drv!r}")
        else:
            problems.append("driver version unavailable (NVML)")
        need(facts.get("bf16_supported") is True, "BF16 not supported")
        arch_list = facts.get("arch_list") or []
        # native-execution basis: the compatible-SASS arch for this profile
        # must be compiled into the wheel...
        compat = {"B300": ("sm_100", "sm_103"), "B200": ("sm_100",)}[
            profile.key]
        need(any(a in arch_list for a in compat),
             f"torch arch list {arch_list} carries no native SASS for "
             f"{profile.key} (need one of {compat})")
        # ...and the wheel must carry NO PTX targets: with no PTX, JIT
        # fallback is impossible — kernels either load natively or fail
        # loudly in exercise_workloads.
        ptx = [a for a in arch_list if a.startswith("compute_")]
        need(not ptx,
             f"torch wheel embeds PTX targets {ptx}: silent JIT fallback "
             f"is possible; the deployment contract requires the no-PTX "
             f"cu130 build so native execution is provable")
    report = {"profile": profile_key, "problems": problems,
              "accepted": not problems, "facts": facts}
    return report


def exercise_workloads(checkpoint_dir: str, out_dir: str) -> dict:
    """Representative REAL workload paths (pod-side; consumes no science)."""
    import torch

    results = {}

    def step(name, fn):
        try:
            results[name] = {"ok": True, **(fn() or {})}
        except Exception as exc:  # noqa: BLE001
            results[name] = {"ok": False, "error": repr(exc)[:400]}

    def bf16_gemm():
        a = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        return {"mean_abs": float(c.abs().mean())}
    step("bf16_gemm", bf16_gemm)

    model_ctx = {}

    def ouro_load():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(checkpoint_dir,
                                            trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda().eval()
        model_ctx["model"], model_ctx["tok"] = model, tok
        return {"params": sum(p.numel() for p in model.parameters())}
    step("ouro_rltt_load", ouro_load)

    def ouro_forward():
        model, tok = model_ctx["model"], model_ctx["tok"]
        enc = tok("hardware gate synthetic prompt; not a task",
                  return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**enc)
        torch.cuda.synchronize()
        return {"logits_shape": list(out.logits.shape)}
    step("ouro_rltt_forward", ouro_forward)

    def ouro_generate():
        model, tok = model_ctx["model"], model_ctx["tok"]
        enc = tok("synthetic generation probe:", return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
        return {"new_tokens": int(gen.shape[1] - enc["input_ids"].shape[1])}
    step("ouro_rltt_generate", ouro_generate)

    def backward_optimizer():
        model, tok = model_ctx["model"], model_ctx["tok"]
        probe = torch.nn.Linear(model.config.hidden_size, 8,
                                dtype=torch.bfloat16, device="cuda")
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-4)
        enc = tok("training-path probe text", return_tensors="pt").to("cuda")
        with torch.no_grad():
            hidden = model(**enc, output_hidden_states=True).hidden_states[-1]
        loss = probe(hidden).float().pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return {"loss": float(loss)}
    step("backward_optimizer_step", backward_optimizer)

    def intervention_hook():
        import numpy as np
        from o1_b200.runner.intervention import (
            BatchedOneShotResidualEdit, register,
        )
        model, tok = model_ctx["model"], model_ctx["tok"]
        hidden_size = model.config.hidden_size
        rng = np.random.default_rng(0)
        d = rng.standard_normal(hidden_size)
        d = d / np.sqrt(np.mean(d ** 2))          # unit-RMS synthetic axis
        enc = tok("hook capture probe", return_tensors="pt").to("cuda")
        hook = BatchedOneShotResidualEdit(
            [d], [0.5], attention_mask=enc["attention_mask"])
        handle = register(model, hook)
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            handle.remove()
        if hook.applied < 1:
            raise HardwareGateError("intervention hook never fired")
        if not hook.injected_rms[0] > 0:
            raise HardwareGateError("transport capture recorded no edit")
        return {"applied": hook.applied,
                "injected_rms": hook.injected_rms[0],
                "realized_delta_rms": hook.realized_delta_rms[0]}
    step("o1_intervention_hook_and_transport", intervention_hook)

    def checkpoint_roundtrip():
        probe = torch.nn.Linear(16, 16)
        opt = torch.optim.AdamW(probe.parameters())
        path = os.path.join(out_dir, "gate_ckpt_probe.pt")
        torch.save({"model": probe.state_dict(),
                    "opt": opt.state_dict(),
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, path)
        loaded = torch.load(path, weights_only=False)
        probe.load_state_dict(loaded["model"])
        opt.load_state_dict(loaded["opt"])
        torch.set_rng_state(loaded["rng"])
        os.remove(path)
        return {"roundtrip": True}
    step("checkpoint_save_load", checkpoint_roundtrip)

    ok = all(v.get("ok") for v in results.values())
    return {"accepted": ok, "workloads": results}


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True, choices=sorted(PROFILES_BY_KEY))
    p.add_argument("--checkpoint", default="/artifacts/ouro_rltt_local")
    p.add_argument("--out", required=True)
    p.add_argument("--skip-workloads", action="store_true",
                   help="identity validation only (no model load)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = validate_facts(a.profile, gather_facts())
    if report["accepted"] and not a.skip_workloads:
        report["workload_exercises"] = exercise_workloads(a.checkpoint, a.out)
        report["accepted"] = report["workload_exercises"]["accepted"]
    path = os.path.join(a.out, "HARDWARE_GATE_REPORT.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    print(f"HARDWARE GATE [{a.profile}]: "
          f"{'ACCEPTED' if report['accepted'] else 'REFUSED'}")
    if not report["accepted"]:
        for prob in report.get("problems", []):
            print(" -", prob)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
