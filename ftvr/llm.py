"""Small-LLM inference with label-constrained greedy decoding.

Constrained decoding walks a token trie built from the allowed label strings, so the
model can only emit a valid intent followed by end-of-turn. The probability of the
emitted path (renormalised over the allowed tokens at each step) is returned as a
confidence score, which we use for calibration and abstention analysis.

When every prompt in a run shares a long prefix (the zero-shot prompt carries all 77
intent names), the prefix is encoded once and its KV cache is reused for every batch.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclass
class Prediction:
    label: str | None
    confidence: float
    raw: str
    latency_s: float = 0.0  # amortised per example
    prompt_tokens: int = 0


@dataclass
class _Node:
    children: dict[int, _Node] = field(default_factory=dict)
    label: str | None = None


class LabelTrie:
    def __init__(self, tok, labels: list[str]):
        self.root = _Node()
        self.eos = tok.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tok.get_vocab() else tok.eos_token_id
        for lab in labels:
            node = self.root
            for t in tok.encode(lab, add_special_tokens=False) + [self.eos]:
                node = node.children.setdefault(t, _Node())
            node.label = lab


def load(model_dir: str, adapter: str | None = None, quant4: bool = False, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(model_dir)
    kw = {"dtype": dtype}
    if quant4:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True
        )
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if not quant4:
            model = model.merge_and_unload()
    model.eval()
    return tok, model


def encode_chat(tok, messages: list[dict]) -> list[int]:
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def _common_prefix(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs)
    first = seqs[0]
    for i in range(n):
        t = first[i]
        if any(s[i] != t for s in seqs):
            return i
    return n


def classify(*args, bf16: bool = True, **kw) -> list[Prediction]:
    """See `_classify`. Runs under bf16 autocast by default (AMX-accelerated on recent Xeons)."""
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        return _classify(*args, **kw)


@torch.inference_mode()
def _classify(
    tok,
    model,
    prompts: list[list[dict]],
    labels: list[str],
    candidates: list[list[str]] | None = None,
    batch_size: int = 8,
    constrained: bool = True,
    max_new_tokens: int = 16,
    prefix_cache_min: int = 64,
) -> list[Prediction]:
    """Classify each chat prompt. `candidates[i]` optionally narrows the label set for prompt i."""
    enc = [encode_chat(tok, p) for p in prompts]
    full_trie = LabelTrie(tok, labels)
    tries = [LabelTrie(tok, c) if c else full_trie for c in candidates] if candidates else [full_trie] * len(enc)

    # Shared-prefix KV cache (only when it pays off).
    plen = _common_prefix(enc) if len(enc) > 1 else 0
    plen = plen if plen >= prefix_cache_min else 0
    plen = min(plen, min(len(e) for e in enc) - 1) if plen else 0
    prefix_cache = None
    if plen:
        prefix_cache = DynamicCache()
        model(input_ids=torch.tensor([enc[0][:plen]]), past_key_values=prefix_cache, use_cache=True)

    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))  # length-bucketing
    out: list[Prediction | None] = [None] * len(enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else full_trie.eos
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        t0 = time.perf_counter()
        suf = [enc[i][plen:] for i in idx]
        L = max(len(s) for s in suf)
        B = len(idx)
        ids = torch.full((B, L), pad, dtype=torch.long)
        mask = torch.zeros((B, plen + L), dtype=torch.long)
        mask[:, :plen] = 1
        for r, s in enumerate(suf):
            ids[r, L - len(s) :] = torch.tensor(s)
            mask[r, plen + L - len(s) :] = 1
        cache = DynamicCache()
        if prefix_cache is not None:
            cache = copy.deepcopy(prefix_cache)
            cache.batch_repeat_interleave(B)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=pos[:, plen:], past_key_values=cache, use_cache=True).logits[
            :, -1, :
        ]
        nodes = [tries[i].root for i in idx]
        gen: list[list[int]] = [[] for _ in idx]
        logp = torch.zeros(B)
        done = [False] * B
        for _step in range(max_new_tokens):
            nxt = torch.full((B,), pad, dtype=torch.long)
            for r in range(B):
                if done[r]:
                    continue
                lp = torch.log_softmax(logits[r].float(), -1)
                if constrained:
                    allowed = list(nodes[r].children)
                    sub = lp[allowed]
                    j = int(sub.argmax())
                    t = allowed[j]
                    logp[r] += float(torch.log_softmax(sub, -1)[j])
                    nodes[r] = nodes[r].children[t]
                else:
                    t = int(lp.argmax())
                    logp[r] += float(lp[t])
                gen[r].append(t)
                nxt[r] = t
                if t == full_trie.eos or t == tok.eos_token_id:
                    done[r] = True
            if all(done):
                break
            mask = torch.cat([mask, torch.ones((B, 1), dtype=torch.long)], 1)
            pos = mask.sum(-1, keepdim=True) - 1
            logits = model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True).logits[
                :, -1, :
            ]
        dt = (time.perf_counter() - t0) / B
        for r, i in enumerate(idx):
            raw = tok.decode([t for t in gen[r] if t not in (full_trie.eos, tok.eos_token_id)]).strip()
            if constrained:
                lab = nodes[r].label
            else:
                lab = normalise_free_text(raw, labels)
            out[i] = Prediction(lab, float(torch.exp(logp[r])), raw, dt, len(enc[i]))
    return out  # type: ignore[return-value]


def normalise_free_text(raw: str, labels: list[str]) -> str | None:
    """Map unconstrained output to a label: exact match after normalisation, else None (invalid)."""
    s = raw.strip().strip(".'\"`").lower().replace("_", " ")
    s = " ".join(s.split())
    return s if s in set(labels) else None
