"""LoRA / QLoRA supervised fine-tuning of a causal LM to emit the intent name.

Loss is computed on the answer tokens only (label + end-of-turn). The training loop is
deliberately small and explicit (no Trainer) so every knob in the ablations is visible.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from . import llm, prompts
from .data import Example

TARGETS = {
    "all-linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


@dataclass
class FTConfig:
    rank: int = 16
    alpha: int | None = None  # defaults to 2*rank
    dropout: float = 0.05
    targets: str = "all-linear"
    lr: float = 2e-4
    epochs: float = 1.0
    batch_size: int = 16
    warmup_frac: float = 0.05
    weight_decay: float = 0.0
    quant4: bool = False
    seed: int = 13
    max_len: int = 512
    bf16: bool = True  # bf16 autocast (AMX on recent Xeons); LoRA master weights stay fp32


def answer_only_loss(model, ids, att, lab):
    """Cross-entropy on answer tokens, projecting only those positions through the 151k-way LM head.

    Equivalent to `model(labels=lab).loss` but skips the vocabulary projection for prompt
    tokens, which dominates the cost on short-answer classification data.
    """
    inner = model.base_model.model  # PeftModel -> LoraModel -> *ForCausalLM
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    sel = tgt != -100
    logits = inner.lm_head(h[:, :-1][sel])
    return torch.nn.functional.cross_entropy(logits.float(), tgt[sel])


def build_example(tok, messages: list[dict], label: str) -> tuple[list[int], list[int]]:
    p = llm.encode_chat(tok, messages)
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    a = tok.encode(label, add_special_tokens=False) + [eot]
    return p + a, [-100] * len(p) + a


def _batches(n_items: list[int], bs: int, rng: random.Random) -> list[list[int]]:
    """Length-grouped batches: sort within shuffled mega-chunks, then shuffle batches."""
    idx = list(range(len(n_items)))
    rng.shuffle(idx)
    out = []
    chunk = bs * 50
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c : c + chunk], key=lambda i: n_items[i])
        out += [part[i : i + bs] for i in range(0, len(part), bs)]
    rng.shuffle(out)
    return out


def train(
    model_dir: str,
    train_items: list[tuple[list[dict], str]],
    out_dir: str,
    cfg: FTConfig,
    log_every: int = 10,
    ckpt_every: int = 50,
) -> dict:
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    tok, model = llm.load(model_dir, quant4=cfg.quant4)
    if cfg.quant4:
        for p in model.parameters():
            p.requires_grad_(False)
    model.config.use_cache = False
    lcfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha or 2 * cfg.rank,
        lora_dropout=cfg.dropout,
        target_modules=TARGETS[cfg.targets],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    data = [build_example(tok, m, y) for m, y in train_items]
    data = [(i[: cfg.max_len], lab[: cfg.max_len]) for i, lab in data]
    steps_per_epoch = math.ceil(len(data) / cfg.batch_size)
    total_steps = max(1, int(round(steps_per_epoch * cfg.epochs)))
    warm = max(1, int(total_steps * cfg.warmup_frac))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))),
    )
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lengths = [len(d[0]) for d in data]
    model.train()
    log, step, t0 = [], 0, time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pt"
    resume_step, prior_s = 0, 0.0
    if ckpt.exists():  # resume after an interrupted run (preemptible / restarted machines)
        from peft import set_peft_model_state_dict

        st = torch.load(ckpt, weights_only=False)
        set_peft_model_state_dict(model, st["lora"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        resume_step, log, prior_s = st["step"], st["log"], st["elapsed_s"]
        torch.set_rng_state(st["torch_rng"])
        print(f"resuming from step {resume_step}", flush=True)
    t0 -= prior_s
    while step < total_steps:
        for b in _batches(lengths, cfg.batch_size, rng):  # same seed -> same batch order on resume
            if step >= total_steps:
                break
            if step < resume_step:
                step += 1
                continue
            L = -(-max(lengths[i] for i in b) // 32) * 32  # bucket shapes: bounds oneDNN kernel-cache growth on CPU
            ids = torch.full((len(b), L), pad, dtype=torch.long)
            lab = torch.full((len(b), L), -100, dtype=torch.long)
            att = torch.zeros((len(b), L), dtype=torch.long)
            for r, i in enumerate(b):  # right padding for training
                x, y = data[i]
                ids[r, : len(x)] = torch.tensor(x)
                lab[r, : len(y)] = torch.tensor(y)
                att[r, : len(x)] = 1
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16):
                loss = answer_only_loss(model, ids, att, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0 or step == total_steps:
                rec = {"step": step, "loss": float(loss.detach()), "lr": sched.get_last_lr()[0], "elapsed_s": time.perf_counter() - t0}
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step % ckpt_every == 0 and step < total_steps:
                from peft import get_peft_model_state_dict

                torch.save(
                    {
                        "lora": get_peft_model_state_dict(model),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "step": step,
                        "log": log,
                        "elapsed_s": time.perf_counter() - t0,
                        "torch_rng": torch.get_rng_state(),
                    },
                    ckpt.with_suffix(".tmp"),
                )
                ckpt.with_suffix(".tmp").replace(ckpt)
    train_s = time.perf_counter() - t0
    model.save_pretrained(str(out / "adapter"))
    size = sum(f.stat().st_size for f in (out / "adapter").glob("*"))
    meta = {
        "config": asdict(cfg),
        "n_train": len(data),
        "steps": total_steps,
        "train_seconds": train_s,
        "seconds_per_step": train_s / total_steps,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "adapter_bytes": size,
        "torch_threads": torch.get_num_threads(),
        "loss_log": log,
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    ckpt.unlink(missing_ok=True)
    return meta


def plain_items(examples: list[Example]) -> list[tuple[list[dict], str]]:
    return [(prompts.finetune(e.text), e.label) for e in examples]
