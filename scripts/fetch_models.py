"""Download the two models into ./models.

Default source is the Hugging Face Hub. `--source dockerhub` pulls the same weights from
Docker Hub instead (Docker's `ai/qwen2.5:0.5B-F16` GGUF and Weaviate's MiniLM inference
image); this is how the published results were produced, on a machine where the Hub was
not reachable. GGUF F16 -> bf16 safetensors conversion goes through transformers.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

QWEN_HF = "Qwen/Qwen2.5-0.5B-Instruct"
MINILM_HF = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path("models")


def from_hf():
    from huggingface_hub import snapshot_download

    snapshot_download(QWEN_HF, local_dir=OUT / "qwen2.5-0.5b-instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    snapshot_download(MINILM_HF, local_dir=OUT / "all-MiniLM-L6-v2", allow_patterns=["*.json", "model.safetensors", "vocab.txt"])


def _token(repo: str) -> str:
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    return json.load(urllib.request.urlopen(url))["token"]


def _get(repo: str, path: str, accept: str | None = None) -> bytes:
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/{path}", headers={"Authorization": f"Bearer {_token(repo)}"})
    if accept:
        req.add_header("Accept", accept)
    return urllib.request.urlopen(req).read()


def from_dockerhub():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT.mkdir(exist_ok=True)
    repo = "ai/qwen2.5"
    man = json.loads(_get(repo, "manifests/0.5B-F16", "application/vnd.oci.image.manifest.v1+json"))
    gguf = next(lay for lay in man["layers"] if lay["mediaType"].endswith("gguf.v3"))
    (OUT / "qwen2.5-0.5b-f16.gguf").write_bytes(_get(repo, f"blobs/{gguf['digest']}"))
    tok = AutoTokenizer.from_pretrained(str(OUT), gguf_file="qwen2.5-0.5b-f16.gguf")
    m = AutoModelForCausalLM.from_pretrained(str(OUT), gguf_file="qwen2.5-0.5b-f16.gguf", dtype=torch.float32)
    cfg = m.config
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    clean = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)  # GGUF-loaded models refuse save_pretrained
    clean.load_state_dict({k: v.to(torch.bfloat16) for k, v in m.state_dict().items()})
    clean.generation_config = m.generation_config
    clean.save_pretrained(OUT / "qwen2.5-0.5b-instruct")
    tok.save_pretrained(OUT / "qwen2.5-0.5b-instruct")

    repo = "semitechnologies/transformers-inference"
    idx = json.loads(
        _get(
            repo,
            "manifests/sentence-transformers-all-MiniLM-L6-v2",
            "application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json",
        )
    )
    amd = next(x for x in idx["manifests"] if x.get("platform", {}).get("architecture") == "amd64")
    man = json.loads(
        _get(
            repo,
            f"manifests/{amd['digest']}",
            "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json",
        )
    )
    dest = OUT / "all-MiniLM-L6-v2"
    dest.mkdir(exist_ok=True)
    for lay in sorted(man["layers"], key=lambda x: x["size"]):
        if lay["size"] > 500_000_000:  # skip the multi-GB python/torch layer
            continue
        with tarfile.open(fileobj=io.BytesIO(_get(repo, f"blobs/{lay['digest']}")), mode="r:gz") as tf:
            members = [mm for mm in tf.getmembers() if mm.name.startswith("app/models/model/") and mm.isfile()]
            for mm in members:
                (dest / Path(mm.name).name).write_bytes(tf.extractfile(mm).read())
            if members:
                return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "dockerhub"], default="hf")
    a = ap.parse_args()
    from_hf() if a.source == "hf" else from_dockerhub()
