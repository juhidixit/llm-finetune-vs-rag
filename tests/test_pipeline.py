from __future__ import annotations

import numpy as np

from ftvr import data as D
from ftvr import finetune, llm, metrics, prompts, retrieval
from ftvr.data import Example

LABELS = ["card arrival", "lost or stolen card", "exchange rate", "change pin"]


def test_verbalise_and_splits():
    assert D.verbalise("Refund_not_showing_up") == "refund not showing up"
    ex = [Example(f"t{i}", LABELS[i % 4]) for i in range(80)]
    rest, held = D.stratified_split(ex, 0.1)
    assert len(held) == 8 and len(rest) == 72
    assert {e.label for e in held} == set(LABELS)
    sub = D.per_label_subset(ex, 3)
    assert len(sub) == 12
    h = D.holdout_intents([f"l{i}" for i in range(77)], 10)
    assert len(h) == 10 and h == D.holdout_intents([f"l{i}" for i in range(77)], 10)


def test_candidates_and_knn():
    shots = [Example("a", "x"), Example("b", "y"), Example("c", "y"), Example("d", "z")]
    assert retrieval.candidates_from_shots(shots) == ["y", "x", "z"]
    assert retrieval.knn_vote(shots, np.array([0.9, 0.5, 0.5, 0.1])) == "y"


def test_metrics():
    g = ["a", "b", "c", "d"]
    p = ["a", "b", "x", None]
    s = metrics.summarise(g, p, [0.9, 0.8, 0.2, 0.1])
    assert s["accuracy"] == 0.5 and s["invalid_rate"] == 0.25
    assert metrics.selective_accuracy(g, p, [0.9, 0.8, 0.2, 0.1], 0.5) == 1.0
    assert 0 <= s["ece"] <= 1
    assert metrics.paired_bootstrap_p(g, g, p, n=200) < 0.2


def test_normalise_free_text():
    assert llm.normalise_free_text(" Card_Arrival. ", LABELS) == "card arrival"
    assert llm.normalise_free_text("something else", LABELS) is None


def test_constrained_decoding_always_valid_and_prefix_cache_exact(tiny_model_dir):
    tok, model = llm.load(tiny_model_dir)
    texts = ["my card has not arrived yet", "i want to change pin", "exchange rate ?", "lost card"]
    P = [prompts.zero_shot(t, LABELS) for t in texts]
    cached = llm.classify(tok, model, P, LABELS, batch_size=3, prefix_cache_min=4, bf16=False)
    plain = llm.classify(tok, model, P, LABELS, batch_size=1, prefix_cache_min=10**9, bf16=False)
    assert all(p.label in LABELS for p in cached)
    assert [p.label for p in cached] == [p.label for p in plain]
    np.testing.assert_allclose([p.confidence for p in cached], [p.confidence for p in plain], rtol=1e-4)
    # per-example candidate restriction
    out = llm.classify(tok, model, P, LABELS, candidates=[["change pin"]] * 4)
    assert {p.label for p in out} == {"change pin"}


def test_answer_only_labels(tiny_model_dir):
    tok, _ = llm.load(tiny_model_dir)
    ids, lab = finetune.build_example(tok, prompts.finetune("lost card"), "lost or stolen card")
    answer = [t for t in lab if t != -100]
    assert len(ids) == len(lab)
    assert answer == tok.encode("lost or stolen card", add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]


def test_lora_training_overfits_tiny_set(tiny_model_dir, tmp_path):
    items = [
        (prompts.finetune("my card has not arrived yet"), "card arrival"),
        (prompts.finetune("i want to change pin"), "change pin"),
    ] * 8
    cfg = finetune.FTConfig(rank=4, lr=2e-2, epochs=40, batch_size=4, warmup_frac=0.0)
    meta = finetune.train(tiny_model_dir, items, str(tmp_path / "run"), cfg, log_every=5)
    assert meta["loss_log"][-1]["loss"] < meta["loss_log"][0]["loss"]
    tok, model = llm.load(tiny_model_dir, adapter=str(tmp_path / "run" / "adapter"))
    preds = llm.classify(tok, model, [m for m, _ in items[:2]], LABELS)
    assert [p.label for p in preds] == ["card arrival", "change pin"]
