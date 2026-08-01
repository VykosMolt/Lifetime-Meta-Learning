"""
Deterministic assembler for the O1 L3_24 axis package, v1.5.3.

This script does not infer axes from remote runs. It consumes four frozen source
artifacts (d1, d2, d3, and two d4 components), a provenance JSON, and writes the
canonical package verified by verify_axis_artifact.py. Source-run truth remains
attested unless the source artifacts/reconstruction log are also archived.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from verify_axis_artifact import (
    AXIS_ORDER, D4_MIN_COSINE, D_MODEL, LOCUS, PKG_FILES,
    canonical_row_bytes, regenerate_random, row_hash, unit_rms,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_vector(path: str) -> np.ndarray:
    v = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if v.shape != (D_MODEL,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{path}: expected finite ({D_MODEL},) vector, got {v.shape}")
    return unit_rms(v)


def assemble(args: argparse.Namespace) -> None:
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    d1, d2, d3 = [load_vector(p) for p in (args.d1, args.d2, args.d3)]
    v1, v2 = load_vector(args.d4_component1), load_vector(args.d4_component2)
    c4 = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if c4 < D4_MIN_COSINE:
        raise ValueError(
            f"d4 component cosine {c4:.6f} < {D4_MIN_COSINE}; reconstruction halts. "
            "This diagnoses the extraction, not necessarily the published adapter proxies."
        )
    d4 = unit_rms((v1 + np.sign(np.dot(v1, v2)) * v2) / 2.0)
    D = np.stack([d1, d2, d3, d4]).astype("<f8", copy=False)
    R = regenerate_random(D).astype("<f8", copy=False)
    V = np.stack([v1, v2]).astype("<f8", copy=False)

    with open(args.provenance) as fh:
        provenance = json.load(fh)
    axes_meta = []
    for i, name in enumerate(AXIS_ORDER):
        if name not in provenance:
            raise ValueError(f"provenance JSON missing {name}")
        rec = dict(provenance[name])
        for key in ("construction", "source_artifact_sha256", "provenance"):
            if key not in rec:
                raise ValueError(f"{name}: provenance record missing {key}")
        axes_meta.append({
            "name": name,
            "sha256": row_hash(D[i]),
            "construction": rec["construction"],
            "source_artifact_sha256": rec["source_artifact_sha256"],
            "locus": LOCUS,
            "provenance": rec["provenance"],
        })

    np.save(out / "axes_l3_24.npy", D)
    np.save(out / "random_axes_l3_24.npy", R)
    np.save(out / "d4_components_l3_24.npy", V)
    with open(out / "GRAM_MATRIX.json", "w") as fh:
        json.dump({"gram": (D @ D.T).tolist()}, fh, indent=2)
        fh.write("\n")
    with open(out / "AXIS_MANIFEST.json", "w") as fh:
        json.dump({"axes": axes_meta, "d4_reconstruction_cosine": c4}, fh, indent=2)
        fh.write("\n")

    shutil.copy(os.path.abspath(__file__), out / "axis_reconstruction.py")
    verifier = Path(__file__).with_name("verify_axis_artifact.py")
    shutil.copy(verifier, out / "verify_axis_artifact.py")
    with open(out / "RECONSTRUCTION_ATTESTATION.json", "w") as fh:
        json.dump({
            "reconstruction_code_sha256": sha256_file(out / "axis_reconstruction.py"),
            "input_artifacts": {
                "d1": sha256_file(args.d1), "d2": sha256_file(args.d2),
                "d3": sha256_file(args.d3),
                "d4_component1": sha256_file(args.d4_component1),
                "d4_component2": sha256_file(args.d4_component2),
                "provenance": sha256_file(args.provenance),
            },
        }, fh, indent=2)
        fh.write("\n")

    with open(out / "SHA256SUMS", "w") as fh:
        for name in PKG_FILES:
            fh.write(f"{sha256_file(out / name)}  {name}\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d1", required=True)
    p.add_argument("--d2", required=True)
    p.add_argument("--d3", required=True)
    p.add_argument("--d4-component1", required=True)
    p.add_argument("--d4-component2", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--output", required=True)
    assemble(p.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
