"""Experiment CLI. Every command writes predictions to results/preds/<name>.jsonl and a
metrics file to results/metrics/<name>.json; `report` turns those into results/RESULTS.md.

    python -m ftvr.run baselines
    python -m ftvr.run zeroshot
    python -m ftvr.run rag --k 10 --split val
    python -m ftvr.run train --name lora-r16 --rank 16
    python -m ftvr.run eval-ft --name lora-r16
    python -m ftvr.run newintent --stage all
    python -m ftvr.run report
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import llm, metrics, prompts, retrieval
from .finetune import FTConfig, plain_items, train

MODEL = "models/qwen2.5-0.5b-instruct"
EMB = "models/all-MiniLM-L6-v2"
RES = Path("results")
RUNS = Path("runs")
CACHE = Path(".cache")


def env() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "threads": torch.get_num_threads(),
    }


def save(name: str, examples: list[D.Example], preds: list, extra: dict) -> dict:
    (RES / "preds").mkdir(parents=True, exist_ok=True)
    (RES / "metrics").mkdir(parents=True, exist_ok=True)
    gold = [e.label for e in examples]
    labs = [p.label if hasattr(p, "label") else p for p in preds]
    conf = [p.confidence for p in preds] if preds and hasattr(preds[0], "confidence") else None
    with (RES / "preds" / f"{name}.jsonl").open("w") as f:
        for e, p in zip(examples, preds):
            rec = {"text": e.text, "gold": e.label, "pred": p.label if hasattr(p, "label") else p}
            if hasattr(p, "confidence"):
                rec.update(conf=round(p.confidence, 4), raw=p.raw)
            f.write(json.dumps(rec) + "\n")
    m = metrics.summarise(gold, labs, conf)
    if preds and hasattr(preds[0], "latency_s"):
        m["throughput_ms_per_example"] = 1000 * float(np.mean([p.latency_s for p in preds]))
        m["prompt_tokens_mean"] = float(np.mean([p.prompt_tokens for p in preds]))
    m.update(extra)
    m["name"] = name
    m["env"] = env()
    (RES / "metrics" / f"{name}.json").write_text(json.dumps(m, indent=2))
    print(
        name,
        {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in m.items()
            if k in ("accuracy", "macro_f1", "invalid_rate", "ece", "throughput_ms_per_example")
        },
        flush=True,
    )
    return m


def get_split(name: str, n: int | None = None) -> list[D.Example]:
    ex = D.splits()[name]
    return ex[:n] if n else ex


def embed_cached(emb: retrieval.Embedder, key: str, texts: list[str]) -> np.ndarray:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"emb-{key}-{len(texts)}.npy"
    if p.exists():
        return np.load(p)
    v = emb.encode(texts)
    np.save(p, v)
    return v


def online_latency(tok, model, prompts_: list[list[dict]], labels, candidates=None, n: int = 100) -> dict:
    """Batch-size-1 latency on the first n prompts (what a single user request would see)."""
    preds = llm.classify(tok, model, prompts_[:n], labels, candidates[:n] if candidates else None, batch_size=1, prefix_cache_min=64)
    return {"online_latency": metrics.latency_summary([p.latency_s for p in preds]), "online_latency_n": n}


# ---------------------------------------------------------------- baselines
def cmd_baselines(a):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    tr, te = get_split("train"), get_split(a.split, a.n)
    ytr = [e.label for e in tr]
    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    Xtr = vec.fit_transform([e.text for e in tr])
    clf = LogisticRegression(C=10, max_iter=3000).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = list(clf.predict(vec.transform([e.text for e in te])))
    save(
        f"tfidf-lr-{a.split}",
        te,
        pred,
        {
            "method": "TF-IDF + logistic regression",
            "train_seconds": fit_s,
            "throughput_ms_per_example": 1000 * (time.perf_counter() - t0) / len(te),
        },
    )

    emb = retrieval.Embedder(EMB)
    t0 = time.perf_counter()
    vtr = embed_cached(emb, "train", [e.text for e in tr])
    clf = LogisticRegression(C=10, max_iter=3000).fit(vtr, ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    vte = emb.encode([e.text for e in te])
    pred = list(clf.predict(vte))
    save(
        f"minilm-lr-{a.split}",
        te,
        pred,
        {
            "method": "MiniLM embeddings + logistic regression",
            "train_seconds": fit_s,
            "throughput_ms_per_example": 1000 * (time.perf_counter() - t0) / len(te),
        },
    )

    idx = retrieval.DenseIndex(tr, emb, vtr)
    shots, sims = idx.search([], a.k, query_vecs=vte)
    pred = [retrieval.knn_vote(s, w) for s, w in zip(shots, sims)]
    save(f"minilm-knn{a.k}-{a.split}", te, pred, {"method": f"MiniLM kNN vote (k={a.k})"})
    recall = float(np.mean([e.label in {s.label for s in ss} for e, ss in zip(te, shots)]))
    print(f"label recall@{a.k} of retrieval: {recall:.4f}")


# ---------------------------------------------------------------- prompting
def cmd_zeroshot(a):
    te = get_split(a.split, a.n)
    labels = D.labels_of(get_split("train"))
    tok, model = llm.load(MODEL)
    P = [prompts.zero_shot(e.text, labels) for e in te]
    preds = llm.classify(tok, model, P, labels, batch_size=a.batch_size, constrained=not a.free)
    extra = {"method": "zero-shot prompting, 77 intents listed in the prompt", "constrained": not a.free}
    if a.latency:
        extra |= online_latency(tok, model, P, labels)
    save(f"zeroshot{'-free' if a.free else ''}-{a.split}", te, preds, extra)


def rag_prompts(te, pool, k, cand_mode, emb=None, pool_key="train"):
    emb = emb or retrieval.Embedder(EMB)
    idx = retrieval.DenseIndex(pool, emb, embed_cached(emb, pool_key, [e.text for e in pool]))
    shots, _ = idx.search([e.text for e in te], k)
    shots = [s[::-1] for s in shots]  # most similar example closest to the query
    cands = [retrieval.candidates_from_shots(s) for s in shots]
    P = [prompts.retrieval_few_shot(e.text, s, c if cand_mode != "none" else None) for e, s, c in zip(te, shots, cands)]
    recall = float(np.mean([e.label in set(c) for e, c in zip(te, cands)]))
    return P, cands, recall


def cmd_rag(a):
    te = get_split(a.split, a.n)
    pool = get_split("train")
    labels = D.labels_of(pool)
    P, cands, recall = rag_prompts(te, pool, a.k, a.candidates)
    tok, model = llm.load(MODEL)
    restrict = cands if a.constrain_to == "retrieved" else None
    preds = llm.classify(tok, model, P, labels, restrict, batch_size=a.batch_size, constrained=not a.free)
    extra = {
        "method": f"retrieval-augmented few-shot (MiniLM top-{a.k} labelled examples)",
        "k": a.k,
        "constrain_to": a.constrain_to,
        "candidates_in_prompt": a.candidates != "none",
        "label_recall_at_k": recall,
    }
    if a.latency:
        extra |= online_latency(tok, model, P, labels, restrict)
    save(f"rag-k{a.k}-{a.constrain_to}{'-free' if a.free else ''}-{a.split}", te, preds, extra)


# ---------------------------------------------------------------- fine-tuning
def cmd_train(a):
    tr = get_split("train")  # val held out for selection
    if a.per_label:
        tr = D.per_label_subset(tr, a.per_label)
    if a.drop_intents:
        held = set(D.holdout_intents(D.labels_of(tr)))
        tr = [e for e in tr if e.label not in held]
    cfg = FTConfig(rank=a.rank, targets=a.targets, lr=a.lr, epochs=a.epochs, batch_size=a.batch_size, quant4=a.quant4)
    if a.raft:
        items = raft_items(tr, a.k)
    else:
        items = plain_items(tr)
    meta = train(MODEL, items, str(RUNS / a.name), cfg)
    print(json.dumps({k: v for k, v in meta.items() if k != "loss_log"}, indent=2))


def raft_items(pool: list[D.Example], k: int):
    """Retrieval-augmented fine-tuning data: each example sees its k nearest *other* pool examples."""
    emb = retrieval.Embedder(EMB)
    idx = retrieval.DenseIndex(pool, emb)
    shots, _ = idx.search([], k, query_vecs=idx.vecs, exclude_self=True)
    items = []
    for e, s in zip(pool, shots):
        s = s[::-1]
        items.append((prompts.retrieval_few_shot(e.text, s, retrieval.candidates_from_shots(s)), e.label))
    return items


def cmd_eval_ft(a):
    te = get_split(a.split, a.n)
    pool = get_split("train")
    labels = D.labels_of(pool)
    meta = json.loads((RUNS / a.name / "train_meta.json").read_text())
    quant4 = meta["config"]["quant4"]
    tok, model = llm.load(MODEL, adapter=str(RUNS / a.name / "adapter"), quant4=quant4)
    restrict = None
    if a.raft:
        P, cands, recall = rag_prompts(te, pool, a.k, "retrieved")
    else:
        P = [prompts.finetune(e.text) for e in te]
    preds = llm.classify(tok, model, P, labels, restrict, batch_size=a.batch_size, constrained=not a.free)
    extra = {
        "method": f"{'QLoRA' if quant4 else 'LoRA'} fine-tune ({a.name})",
        "train": {k: v for k, v in meta.items() if k != "loss_log"},
        "constrained": not a.free,
    }
    if a.latency:
        extra |= online_latency(tok, model, P, labels)
    save(f"{a.name}{'-free' if a.free else ''}-{a.split}", te, preds, extra)


# ---------------------------------------------------------------- new intents
def cmd_newintent(a):
    """Ten intents 'launch' after the model was trained. Which approach copes without retraining?"""
    pool_full = get_split("train")
    labels = D.labels_of(pool_full)
    held = D.holdout_intents(labels)
    te = get_split("test")
    gold = [e.label for e in te]
    is_new = np.array([g in held for g in gold])
    out = {"held_out_intents": held, "n_test_new": int(is_new.sum()), "n_test_seen": int((~is_new).sum()), "rows": []}

    def row(name, pred, note, extra=None):
        pred = list(pred)
        acc_new = metrics.accuracy([g for g, n in zip(gold, is_new) if n], [p for p, n in zip(pred, is_new) if n])
        acc_seen = metrics.accuracy([g for g, n in zip(gold, is_new) if not n], [p for p, n in zip(pred, is_new) if not n])
        r = {
            "approach": name,
            "acc_new_intents": acc_new,
            "acc_seen_intents": acc_seen,
            "acc_all": metrics.accuracy(gold, pred),
            "retraining": note,
        }
        r |= extra or {}
        out["rows"].append(r)
        print(json.dumps(r), flush=True)

    # Predictions of the full-data models are re-scored from their saved files.
    for name, label, note in a.reuse:
        name = name.format(k=a.k)
        p = RES / "preds" / f"{name}.jsonl"
        if p.exists():
            pred = [json.loads(line)["pred"] for line in p.open()]
            row(label, pred, note)

    for name in a.adapters:  # adapters trained without the held-out intents
        tok, model = llm.load(MODEL, adapter=str(RUNS / name / "adapter"))
        raft = name.startswith("raft")
        if raft:
            P, cands, _ = rag_prompts(te, pool_full, a.k, "retrieved")
        else:
            P = [prompts.finetune(e.text) for e in te]
        preds = llm.classify(tok, model, P, labels, batch_size=a.batch_size)
        save(f"newintent-{name}-test", te, preds, {"method": name, "held_out_intents": held})
        note = "none (index updated with new examples)" if raft else "required (labels live in the weights)"
        row(name, [p.label for p in preds], note)
        del model

    (RES / "metrics" / "newintent.json").write_text(json.dumps(out, indent=2))


# ---------------------------------------------------------------- report
def cmd_report(a):
    from .report import build

    build(RES)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--split", default="test", choices=["val", "test"])
        p.add_argument("--n", type=int, default=None, help="evaluate on the first n examples only")
        p.add_argument("--batch-size", type=int, default=16)
        p.add_argument("--free", action="store_true", help="unconstrained decoding")
        p.add_argument("--latency", action="store_true", help="also measure batch-1 online latency")

    p = sub.add_parser("baselines")
    common(p)
    p.add_argument("--k", type=int, default=10)
    p.set_defaults(fn=cmd_baselines)
    p = sub.add_parser("zeroshot")
    common(p)
    p.set_defaults(fn=cmd_zeroshot)
    p = sub.add_parser("rag")
    common(p)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", default="retrieved", choices=["retrieved", "none"])
    p.add_argument("--constrain-to", default="all", choices=["all", "retrieved"])
    p.set_defaults(fn=cmd_rag)
    p = sub.add_parser("train")
    p.add_argument("--name", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--targets", default="all-linear", choices=["all-linear", "attn"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--quant4", action="store_true", help="QLoRA: NF4 4-bit frozen base")
    p.add_argument("--per-label", type=int, default=None)
    p.add_argument("--drop-intents", action="store_true", help="exclude the 10 held-out intents")
    p.add_argument("--raft", action="store_true", help="train with retrieved examples in the prompt")
    p.add_argument("--k", type=int, default=10)
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("eval-ft")
    common(p)
    p.add_argument("--name", required=True)
    p.add_argument("--raft", action="store_true")
    p.add_argument("--k", type=int, default=10)
    p.set_defaults(fn=cmd_eval_ft)
    p = sub.add_parser("newintent")
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.set_defaults(
        fn=cmd_newintent,
        reuse=[
            ("zeroshot-test", "zero-shot prompt (77 intents listed)", "none (edit the label list)"),
            ("minilm-knn10-test", "MiniLM kNN vote, full index", "none (index updated)"),
            ("rag-k{k}-all-test", "retrieval-augmented few-shot, full index", "none (index updated)"),
            ("lora-r16-test", "LoRA retrained on all 77 intents (upper bound)", "required"),
            ("minilm-lr-test", "MiniLM + logistic regression retrained on all 77", "required (seconds)"),
        ],
    )
    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
