"""
Verifier for the four-axis L3_24 artifact package. v1.5.3.

v1.3 had two holes that would have certified a fabricated package:
  * the random bank was checked only for Gram equality, which the STRUCTURED
    tensor trivially satisfies — a literal copy passed as SEALABLE
  * per-axis hashes, source hashes and the d4 cosine were checked for presence,
    never for truth; arbitrary strings and a fabricated 0.999 passed

v1.5 regenerates the random bank deterministically and recomputes every claim it
can. Two levels of assurance are reported separately and never conflated:

  STRUCTURALLY VERIFIED — recomputed here from the submitted bytes
  ATTESTED ONLY        — asserted about remote source runs; provable only if
                         the source artifacts or a hash-verified reconstruction
                         log are supplied

Package:
    axes_l3_24.npy              (4, 2048) float64, rows in the fixed order
    random_axes_l3_24.npy       (4, 2048) float64, deterministically derived
    d4_components_l3_24.npy     (2, 2048) float64, the two adapter PCs
    AXIS_MANIFEST.json          per-axis construction + provenance + row hashes
    GRAM_MATRIX.json            stored Gram, must equal the recomputed one
    RECONSTRUCTION_ATTESTATION.json
    SHA256SUMS                  must cover EVERY file in the package

Run:  python verify_axis_artifact.py <package_dir>
      python verify_axis_artifact.py --selftest
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

N_AXES, D_MODEL = 4, 2048
AXIS_ORDER = ("d1_readable_process_quality", "d2_empirical_outcome",
              "d3_writable_transport", "d4_learned_actuator")
LOCUS = "L3_24_raw_residual_2048"
DUP_COS, D4_MIN_COSINE = 0.98, 0.90
GRAM_TOL, RMS_TOL, RECON_TOL = 1e-6, 1e-6, 1e-8
RANDOM_DOMAIN = b"O1-v1.5-fixed-random-domain"
PROVENANCE_FIELDS = ("source_loop", "source_layer", "source_token_position",
                     "source_pooling_rule", "source_phase",
                     "target_injection_loop", "target_injection_layer",
                     "target_token_position", "coefficient_mapping")
PKG_FILES = ("axes_l3_24.npy", "random_axes_l3_24.npy", "d4_components_l3_24.npy",
             "AXIS_MANIFEST.json", "GRAM_MATRIX.json",
             "RECONSTRUCTION_ATTESTATION.json", "axis_reconstruction.py",
             "verify_axis_artifact.py")


class AxisArtifactError(RuntimeError):
    def __init__(self, code, msg, detail=None):
        super().__init__(f"[{code}] {msg}")
        self.code, self.detail = code, detail


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def canonical_row_bytes(v: np.ndarray) -> bytes:
    """Fixed dtype, byte order and layout, so a row hash is reproducible."""
    return np.ascontiguousarray(v, dtype="<f8").tobytes()


def row_hash(v: np.ndarray) -> str:
    return hashlib.sha256(canonical_row_bytes(v)).hexdigest()


def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def unit_rms(v):
    return v / rms(v)


def random_seed_int(D: np.ndarray) -> int:
    """
    Non-discretionary random-control seed. Once D is frozen there is no master
    seed to choose after inspecting the axes.
    """
    h = hashlib.sha256()
    h.update(RANDOM_DOMAIN)
    h.update(hashlib.sha256(canonical_row_bytes(D)).digest())
    return int(h.hexdigest()[:16], 16)


def regenerate_random(D: np.ndarray) -> np.ndarray:
    """
    R = L U^T where C = D D^T = L L^T and U is a deterministic orthonormal
    basis derived only from the fixed public domain and D. R R^T = C.
    """
    rng = np.random.default_rng(random_seed_int(D))
    Q, Rq = np.linalg.qr(rng.normal(size=(D.shape[1], D.shape[0])))
    signs = np.sign(np.where(np.diag(Rq) == 0, 1.0, np.diag(Rq)))
    Q = Q * signs
    C = D @ D.T
    L = np.linalg.cholesky(C + 1e-12 * np.eye(D.shape[0]))
    return L @ Q.T


def canonical_correlations(A: np.ndarray, B: np.ndarray) -> list[float]:
    """Between the two 4-d subspaces. Reported, not gating."""
    Qa = np.linalg.qr(A.T)[0]
    Qb = np.linalg.qr(B.T)[0]
    return sorted(np.linalg.svd(Qa.T @ Qb, compute_uv=False).round(6).tolist(),
                  reverse=True)


def verify(pkg: str) -> dict:
    for f in PKG_FILES + ("SHA256SUMS",):
        if not os.path.exists(os.path.join(pkg, f)):
            raise AxisArtifactError("A0_MISSING_FILE", f"package lacks {f}")

    # A15 — SHA256SUMS must cover every file, and every entry must be true
    sums = {}
    for ln in open(os.path.join(pkg, "SHA256SUMS")):
        if ln.strip():
            h, f = ln.split()
            sums[os.path.basename(f)] = h
    on_disk = {f for f in os.listdir(pkg) if f != "SHA256SUMS"}
    missing = sorted(on_disk - set(sums))
    if missing:
        raise AxisArtifactError(
            "A15_PACKAGE_HASH_COVERAGE",
            f"SHA256SUMS does not cover {missing}; a manifest, Gram file or "
            f"reconstruction script outside the hash set can be edited freely")
    for f, h in sums.items():
        p = os.path.join(pkg, f)
        if os.path.exists(p) and _sha(p) != h:
            raise AxisArtifactError("A11_HASH", f"{f}: SHA256SUMS disagrees with the file")

    man = json.load(open(os.path.join(pkg, "AXIS_MANIFEST.json")))
    att = json.load(open(os.path.join(pkg, "RECONSTRUCTION_ATTESTATION.json")))
    D = np.load(os.path.join(pkg, "axes_l3_24.npy"))
    R = np.load(os.path.join(pkg, "random_axes_l3_24.npy"))
    V = np.load(os.path.join(pkg, "d4_components_l3_24.npy"))

    for name, M, shape in (("structured", D, (N_AXES, D_MODEL)),
                           ("random", R, (N_AXES, D_MODEL)),
                           ("d4 components", V, (2, D_MODEL))):
        if M.shape != shape:
            raise AxisArtifactError(
                "A1_SHAPE", f"{name} has shape {M.shape}, expected {shape} — a "
                f"wrong hidden dimension is the signature of a cross-locus or "
                f"concatenated-tap reconstruction")
        if not np.all(np.isfinite(M)):
            raise AxisArtifactError("A2_NONFINITE", f"{name} contains non-finite values")
    for i in range(N_AXES):
        if abs(rms(D[i]) - 1.0) > RMS_TOL:
            raise AxisArtifactError(
                "A3_UNIT_RMS", f"structured axis {i} has RMS {rms(D[i]):.9f}, not 1")

    if [a.get("name") for a in man.get("axes", [])] != list(AXIS_ORDER):
        raise AxisArtifactError(
            "A4_AXIS_ORDER", f"axis order != sealed order {list(AXIS_ORDER)}")

    # A13 — per-axis row hashes must be TRUE, not merely present
    for i, a in enumerate(man["axes"]):
        for k in ("name", "sha256", "construction", "source_artifact_sha256", "locus"):
            if k not in a:
                raise AxisArtifactError("A5_AXIS_RECORD", f"{a.get('name')}: missing {k!r}")
        if a["locus"] != LOCUS:
            raise AxisArtifactError(
                "A6_CROSS_LOCUS",
                f"{a['name']}: locus {a['locus']!r} is not {LOCUS}; no L4 reuse, no "
                f"24/36/47 concatenation, no layer-36 projection, no pooled "
                f"four-loop direction")
        want = row_hash(D[i])
        if a["sha256"] != want:
            raise AxisArtifactError(
                "A13_AXIS_ROW_HASH",
                f"{a['name']}: manifest sha256 {a['sha256'][:16]}... but the "
                f"canonical row bytes hash to {want[:16]}...",
                detail={"axis": a["name"], "claimed": a["sha256"], "actual": want})
        # A16 — provenance, not just locus
        prov = a.get("provenance", {})
        miss = [f for f in PROVENANCE_FIELDS if f not in prov]
        if miss:
            raise AxisArtifactError(
                "A16_AXIS_PROVENANCE",
                f"{a['name']}: provenance missing {miss}. 'Same locus' does not "
                f"describe provenance: a direction learned from pooled completed-"
                f"candidate states and injected at the final prompt position is a "
                f"deliberate cross-phase transfer and must be recorded as one",
                detail={"axis": a["name"], "missing": miss})

    # cosines: rows are unit-RMS, not unit-L2, so the Gram is NOT the cosine matrix
    C = D @ D.T
    COS = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    for i in range(N_AXES):
        for j in range(i + 1, N_AXES):
            if abs(COS[i, j]) >= DUP_COS:
                raise AxisArtifactError(
                    "A7_DUPLICATE_AXIS",
                    f"|cos({AXIS_ORDER[i]}, {AXIS_ORDER[j]})| = {abs(COS[i, j]):.4f} "
                    f">= {DUP_COS}; replace the later with the next singular "
                    f"direction from the same provenance family, or do not seal",
                    detail={"cos": float(abs(COS[i, j]))})

    stored = np.array(json.load(open(os.path.join(pkg, "GRAM_MATRIX.json")))["gram"])
    if stored.shape != (N_AXES, N_AXES) or np.max(np.abs(stored - C)) > GRAM_TOL:
        raise AxisArtifactError("A8_GRAM_MISMATCH", "stored Gram != recomputed Gram")

    # A9 — Gram match is necessary
    dev = float(np.max(np.abs(R @ R.T - C)) / D_MODEL)
    if dev > 1e-6:
        raise AxisArtifactError(
            "A9_RANDOM_GRAM",
            f"random axes deviate from the structured Gram by {dev:.3e}; four "
            f"orthogonal random axes are not a matched control when the "
            f"structured axes are correlated", detail={"max_dev": dev})

    # A12 — Gram equality is necessary but not sufficient. The random bank is
    # regenerated from a fixed public domain and D alone; no self-attested seed.
    R_exp = regenerate_random(D)
    rdev = float(np.max(np.abs(R - R_exp)))
    if rdev > RECON_TOL:
        same = float(np.max(np.abs(R - D)))
        raise AxisArtifactError(
            "A12_RANDOM_ORIENTATION",
            f"random bank differs from its non-discretionary regeneration by "
            f"{rdev:.3e}" + (" — and is identical to the structured tensor"
                             if same == 0.0 else ""),
            detail={"max_dev": rdev, "identical_to_structured": same == 0.0})

    # A17 — the package must contain the exact verifier being executed and the
    # attested reconstruction-script hash must match the submitted script.
    pkg_verifier = os.path.join(pkg, "verify_axis_artifact.py")
    if _sha(pkg_verifier) != _sha(os.path.abspath(__file__)):
        raise AxisArtifactError(
            "A17_VERIFIER_IDENTITY",
            "packaged verify_axis_artifact.py is not the verifier currently executing")
    recon_actual = _sha(os.path.join(pkg, "axis_reconstruction.py"))
    if att.get("reconstruction_code_sha256") != recon_actual:
        raise AxisArtifactError(
            "A17_VERIFIER_IDENTITY",
            f"attested reconstruction_code_sha256 does not match axis_reconstruction.py")


    # A14 — recompute d4 from its components rather than believing the cosine
    v1, v2 = unit_rms(V[0]), unit_rms(V[1])
    c4 = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    claimed = man.get("d4_reconstruction_cosine")
    if claimed is None or abs(float(claimed) - c4) > 1e-6:
        raise AxisArtifactError(
            "A14_D4_RECONSTRUCTION",
            f"manifest d4_reconstruction_cosine {claimed} but the supplied "
            f"components give {c4:.6f}", detail={"claimed": claimed, "actual": c4})
    if c4 < D4_MIN_COSINE:
        raise AxisArtifactError(
            "A10_D4_COSINE",
            f"d4 reconstruction cosine {c4:.4f} < {D4_MIN_COSINE}. This is an "
            f"AXIS-RECONSTRUCTION failure, not evidence that the published adapter "
            f"proxies were inconsistent: the published 0.951 is between proxy "
            f"directions, not between leading PCs of induced L3_24 updates.",
            detail={"cosine": c4})
    d4 = unit_rms((v1 + np.sign(np.dot(v1, v2)) * v2) / 2.0)
    if float(np.max(np.abs(d4 - D[3]))) > RECON_TOL:
        raise AxisArtifactError(
            "A14_D4_RECONSTRUCTION",
            f"row 4 of axes_l3_24.npy does not equal unitRMS((v1 + sign*v2)/2) "
            f"recomputed from d4_components_l3_24.npy "
            f"(max abs dev {float(np.max(np.abs(d4 - D[3]))):.3e})")

    return {
        "verdict": "SEALABLE",
        "structurally_verified": [
            "shape, finiteness, unit RMS", "axis order and locus",
            "per-axis canonical row hashes", "duplicate rule |cos| < 0.98",
            "stored Gram == recomputed Gram",
            "random bank == non-discretionary regeneration from fixed domain + D",
            "d4 == unitRMS((v1 + sign*v2)/2) from its supplied components",
            "SHA256SUMS covers every package file"],
        "attested_only": [
            "source_artifact_sha256 for each axis refers to the stated remote run",
            "the construction text describes what was actually run",
            "training/reference task manifests are the ones named",
            "(provable only if those source artifacts or a hash-verified "
            "reconstruction log are supplied)"],
        "cosines": COS.round(6).tolist(),
        "max_abs_offdiag_cos": float(max(abs(COS[i, j]) for i in range(N_AXES)
                                         for j in range(N_AXES) if i != j)),
        "random_gram_max_dev": dev,
        "random_regeneration_max_dev": rdev,
        "subspace_canonical_correlations": canonical_correlations(D, R),
        "d4_cosine": c4,
        "random_seed_derivation": "SHA256(fixed_public_domain || SHA256(D))",
        "attestation_keys": sorted(att),
    }


# ---------------------------------------------------------------- self-test


def _mk(tmp, *, random_mode="deterministic", d4_cos_claim=None, dup=False,
        locus=LOCUS, shape=(N_AXES, D_MODEL), drop_sums=(), row_hash_lie=False,
        drop_provenance=False, d4_angle=0.25):
    rng = np.random.default_rng(7)
    A = unit_rms_rows = rng.normal(size=shape)
    A = A / np.sqrt(np.mean(A ** 2, axis=1, keepdims=True))
    if dup:
        A[1] = A[0]
    # build d4 from two components so row 4 is genuinely reconstructible
    v1 = A[0] if shape[0] < 4 else rng.normal(size=shape[1])
    v1 = v1 / np.sqrt(np.mean(v1 ** 2))
    w = rng.normal(size=shape[1]); w -= (w @ v1) / (v1 @ v1) * v1
    w = w / np.sqrt(np.mean(w ** 2))
    v2 = np.cos(d4_angle) * v1 + np.sin(d4_angle) * w
    v2 = v2 / np.sqrt(np.mean(v2 ** 2))
    d4 = (v1 + np.sign(v1 @ v2) * v2) / 2.0
    d4 = d4 / np.sqrt(np.mean(d4 ** 2))
    if shape == (N_AXES, D_MODEL):
        A[3] = d4
    C = A @ A.T
    if random_mode == "deterministic":
        R = regenerate_random(A)
    elif random_mode == "copy":
        R = A.copy()
    else:  # orthogonal, unit-RMS: the plausible-looking wrong control
        R = np.linalg.qr(rng.normal(size=(shape[1], shape[0])))[0].T * np.sqrt(shape[1])
    np.save(os.path.join(tmp, "axes_l3_24.npy"), A)
    np.save(os.path.join(tmp, "random_axes_l3_24.npy"), R)
    np.save(os.path.join(tmp, "d4_components_l3_24.npy"), np.stack([v1, v2]))
    json.dump({"gram": C.tolist()}, open(os.path.join(tmp, "GRAM_MATRIX.json"), "w"))
    c4 = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    prov = {f: "recorded" for f in PROVENANCE_FIELDS}
    json.dump({"axes": [{"name": n,
                         "sha256": ("deadbeef" if row_hash_lie else row_hash(A[i])),
                         "construction": "…", "source_artifact_sha256": f"s{i}",
                         "locus": locus,
                         **({} if (drop_provenance and i == 0)
                            else {"provenance": prov})}
                        for i, n in enumerate(AXIS_ORDER)],
               "d4_reconstruction_cosine": (c4 if d4_cos_claim is None
                                            else d4_cos_claim)},
              open(os.path.join(tmp, "AXIS_MANIFEST.json"), "w"))
    # The package must include the exact verifier and a reconstruction script.
    import shutil
    shutil.copy(os.path.abspath(__file__), os.path.join(tmp, "verify_axis_artifact.py"))
    with open(os.path.join(tmp, "axis_reconstruction.py"), "w") as fh:
        fh.write("# synthetic reconstruction fixture\n")
    json.dump({"reconstruction_code_sha256": _sha(os.path.join(tmp, "axis_reconstruction.py")),
               "source_runs": {}}, open(os.path.join(tmp,
                                                     "RECONSTRUCTION_ATTESTATION.json"), "w"))
    with open(os.path.join(tmp, "SHA256SUMS"), "w") as fh:
        for f in PKG_FILES:
            if f in drop_sums:
                continue
            fh.write(f"{_sha(os.path.join(tmp, f))}  {f}\n")
    return tmp


def selftest() -> int:
    import tempfile
    fails = 0

    def case(name, expect, mutate=None, **kw):
        nonlocal fails
        d = _mk(tempfile.mkdtemp(), **kw)
        if mutate:
            mutate(d)
        try:
            got, extra = None, verify(d)["verdict"]
        except AxisArtifactError as e:
            got, extra = e.code, str(e)[:78]
        ok = got == expect
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: {got or extra}")

    def copy_random(d):
        import shutil
        shutil.copy(os.path.join(d, "axes_l3_24.npy"),
                    os.path.join(d, "random_axes_l3_24.npy"))
        with open(os.path.join(d, "SHA256SUMS"), "w") as fh:
            for f in PKG_FILES:
                fh.write(f"{_sha(os.path.join(d, f))}  {f}\n")

    case("clean package", None)
    case("random bank is a COPY of the structured tensor", "A12_RANDOM_ORIENTATION",
         mutate=copy_random)
    case("random bank orthogonal, not Gram-matched", "A9_RANDOM_GRAM",
         random_mode="orthogonal")
    case("fabricated per-axis row hash", "A13_AXIS_ROW_HASH", row_hash_lie=True)
    case("fabricated d4 cosine (0.999)", "A14_D4_RECONSTRUCTION", d4_cos_claim=0.999)
    case("genuine d4 components disagree (cos < 0.90)", "A10_D4_COSINE", d4_angle=1.1)
    case("SHA256SUMS omits AXIS_MANIFEST.json", "A15_PACKAGE_HASH_COVERAGE",
         drop_sums=("AXIS_MANIFEST.json",))
    case("axis provenance absent", "A16_AXIS_PROVENANCE", drop_provenance=True)
    case("wrong hidden dim (cross-locus signature)", "A1_SHAPE", shape=(4, 1024))
    case("duplicate axes", "A7_DUPLICATE_AXIS", dup=True)
    case("axis reconstructed at the wrong locus", "A6_CROSS_LOCUS", locus="L4_47")

    def wrong_but_gram_matched_orientation(d):
        D = np.load(os.path.join(d, "axes_l3_24.npy"))
        rng = np.random.default_rng(999)
        Q, Rq = np.linalg.qr(rng.normal(size=(D.shape[1], D.shape[0])))
        Q = Q * np.sign(np.where(np.diag(Rq) == 0, 1.0, np.diag(Rq)))
        L = np.linalg.cholesky(D @ D.T + 1e-12 * np.eye(D.shape[0]))
        np.save(os.path.join(d, "random_axes_l3_24.npy"), L @ Q.T)
        with open(os.path.join(d, "RECONSTRUCTION_ATTESTATION.json")) as fh:
            att = json.load(fh)
        att["master_seed"] = "CHOSEN_AFTER_SEEING_AXES"
        json.dump(att, open(os.path.join(d, "RECONSTRUCTION_ATTESTATION.json"), "w"))
        with open(os.path.join(d, "SHA256SUMS"), "w") as fh:
            for f in PKG_FILES:
                fh.write(f"{_sha(os.path.join(d, f))}  {f}\n")

    def wrong_packaged_verifier(d):
        with open(os.path.join(d, "verify_axis_artifact.py"), "a") as fh:
            fh.write("\n# tampered verifier\n")
        with open(os.path.join(d, "SHA256SUMS"), "w") as fh:
            for f in PKG_FILES:
                fh.write(f"{_sha(os.path.join(d, f))}  {f}\n")

    def wrong_reconstruction_attestation(d):
        p = os.path.join(d, "RECONSTRUCTION_ATTESTATION.json")
        att = json.load(open(p)); att["reconstruction_code_sha256"] = "0" * 64
        json.dump(att, open(p, "w"))
        with open(os.path.join(d, "SHA256SUMS"), "w") as fh:
            for f in PKG_FILES:
                fh.write(f"{_sha(os.path.join(d, f))}  {f}\n")

    case("self-attested chosen seed cannot change orientation",
         "A12_RANDOM_ORIENTATION", mutate=wrong_but_gram_matched_orientation)
    case("packaged verifier differs from executing verifier",
         "A17_VERIFIER_IDENTITY", mutate=wrong_packaged_verifier)
    case("reconstruction script hash is fabricated",
         "A17_VERIFIER_IDENTITY", mutate=wrong_reconstruction_attestation)
    case("axis_reconstruction.py missing", "A0_MISSING_FILE",
         mutate=lambda d: os.remove(os.path.join(d, "axis_reconstruction.py")))
    n = 15
    print(f"\n  {n - fails}/{n} axis-artifact checks behaved as specified")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    print(json.dumps(verify(sys.argv[1]), indent=2))
