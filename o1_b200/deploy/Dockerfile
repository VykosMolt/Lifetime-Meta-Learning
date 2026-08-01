# O1 B200 runner deployment image — BUILT LOCALLY, VALIDATED ON B200 LATER.
# Scientific execution is BF16 PyTorch/Transformers eager only. FP8/FP4,
# quantization, speculative decoding, vLLM, TensorRT-LLM, and SGLang are
# deliberately absent. Optional performance modes (torch.compile, CUDA
# graphs, alternate attention) exist in the runner behind flags that default
# OFF and stay scientifically ineligible until a future B200 equivalence pass.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    CUBLAS_WORKSPACE_CONFIG=":4096:8"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.14 python3.14-venv python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/o1_b200
COPY deploy/requirements.lock /opt/o1_b200/deploy/requirements.lock

RUN python3.14 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r deploy/requirements.lock \
    && /opt/venv/bin/python -c "import transformers; assert transformers.__version__ == '4.54.1', transformers.__version__"

# Runner source + sealed v2.1.0 package + policies + corpus (hash-verified at
# runtime by runner/sealed_import.py and deploy/verify_artifacts.py).
COPY o1_b200 /opt/o1_b200/o1_b200
COPY o1_packages/O1_oracle_reachability_v2.1.0_source \
     /opt/o1_b200/o1_packages/O1_oracle_reachability_v2.1.0_source
COPY o1_runs/O1_V2_AXIS_BANK_REDESIGN/AXIS_PACKAGE_V2 \
     /opt/o1_b200/o1_runs/O1_V2_AXIS_BANK_REDESIGN/AXIS_PACKAGE_V2
COPY o1_runs/O1_V2_AXIS_BANK_REDESIGN/COHORTS \
     /opt/o1_b200/o1_runs/O1_V2_AXIS_BANK_REDESIGN/COHORTS

ENV PYTHONPATH=/opt/o1_b200 \
    PATH=/opt/venv/bin:$PATH

# The model checkpoint is NEVER baked into the image; it is transferred
# out of band and mounted at /artifacts/ouro_rltt_local, then tree-hash
# verified against the transfer manifest before anything scientific runs.
ENTRYPOINT ["/opt/o1_b200/o1_b200/deploy/start_b200.sh"]
