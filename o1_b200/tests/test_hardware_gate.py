"""Hardware gate: pure fact-validation matrix for both profiles."""
from __future__ import annotations

from _h import Runner, hermetic_mock_credentials

MOCK_KEY = hermetic_mock_credentials()

from o1_b200.deploy.hardware_gate import validate_facts


def _b300_facts(**over):
    facts = {
        "cuda_available": True, "device_count": 1,
        "device_name": "NVIDIA B300 SXM6 AC",
        "compute_capability": "10.3",
        "total_hbm_bytes": 288 * 1000 ** 3,
        "free_hbm_bytes": 280 * 1000 ** 3,
        "torch_version": "2.12.1+cu130", "torch_cuda_runtime": "13.0",
        "cudnn_version": 92000, "bf16_supported": True,
        "arch_list": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100",
                      "sm_120"],
        "driver_version": "580.65.06",
    }
    facts.update(over)
    return facts


def _b200_facts(**over):
    return _b300_facts(device_name="NVIDIA B200",
                       compute_capability="10.0",
                       total_hbm_bytes=180 * 1000 ** 3,
                       free_hbm_bytes=170 * 1000 ** 3, **over)


def run() -> Runner:
    r = Runner("hardware_gate")

    r.check("qualifying B300 accepted under the B300 profile",
            lambda: (_ for _ in ()).throw(AssertionError(
                validate_facts("B300", _b300_facts())["problems"]))
            if not validate_facts("B300", _b300_facts())["accepted"] else None)

    r.check("qualifying B200 accepted under the B200 fallback profile",
            lambda: None if validate_facts("B200", _b200_facts())["accepted"]
            else (_ for _ in ()).throw(AssertionError(
                validate_facts("B200", _b200_facts())["problems"])))

    def refused(profile, facts, needle):
        rep = validate_facts(profile, facts)
        assert not rep["accepted"], f"accepted: {facts}"
        assert any(needle in p for p in rep["problems"]), rep["problems"]

    r.check("B200 hardware refused under the B300 profile (no silent "
            "profile swap)",
            lambda: refused("B300", _b200_facts(), "not the intended B300"))

    r.check("B300 hardware refused under the B200 profile",
            lambda: refused("B200", _b300_facts(), "not the intended B200"))

    r.check("two GPUs refused",
            lambda: refused("B300", _b300_facts(device_count=2),
                            "exactly one GPU"))

    r.check("wrong compute capability refused",
            lambda: refused("B300", _b300_facts(compute_capability="10.0"),
                            "compute capability"))

    r.check("wrong HBM class refused (180 GB part sold as B300)",
            lambda: refused("B300", _b300_facts(
                total_hbm_bytes=180 * 1000 ** 3), "HBM"))

    r.check("insufficient free HBM refused",
            lambda: refused("B300", _b300_facts(
                free_hbm_bytes=20 * 1000 ** 3), "free HBM"))

    r.check("H100 identity refused outright",
            lambda: refused("B300", _b300_facts(
                device_name="NVIDIA H100 80GB HBM3",
                compute_capability="9.0"), "not the intended"))

    r.check("pre-CUDA-13 runtime refused",
            lambda: refused("B300", _b300_facts(torch_cuda_runtime="12.8"),
                            "CUDA runtime"))

    r.check("old driver refused",
            lambda: refused("B300", _b300_facts(driver_version="570.86.15"),
                            "driver"))

    r.check("missing BF16 refused",
            lambda: refused("B300", _b300_facts(bf16_supported=False),
                            "BF16"))

    r.check("wheel without compatible native SASS refused",
            lambda: refused("B300", _b300_facts(
                arch_list=["sm_75", "sm_80", "sm_90"]), "native SASS"))

    r.check("wheel embedding PTX (silent JIT possible) refused",
            lambda: refused("B300", _b300_facts(
                arch_list=["sm_100", "compute_120"]), "PTX"))

    r.check("no CUDA at all refused",
            lambda: refused("B300", {"cuda_available": False}, "CUDA"))

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
