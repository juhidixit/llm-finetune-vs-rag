# Fine-tune, RAG, or just a classifier?

**A measured comparison of LoRA, QLoRA, retrieval-augmented prompting, zero-shot prompting and classic
baselines for a fintech support task — on a CPU, with a small open model, and with every number reproducible.**

Teams that put LLMs into production keep asking the same question: *should we fine-tune, add retrieval, or
neither?* This repo answers it for one realistic task, routing customer messages of a digital bank to one of
77 fine-grained intents ([Banking77](https://arxiv.org/abs/2003.04807)), and reports accuracy together with
the things that decide a real deployment: calibration, latency, serving cost, training cost, data needs,
and what happens when the business launches new intents after the model shipped.

Model: **Qwen2.5-0.5B-Instruct** (Apache-2.0). Hardware: **one 4-vCPU Intel Xeon, no GPU** (bf16 via AMX).
Everything, including fine-tuning, runs on that machine; the full suite of 10 training runs and all evaluations takes about 12 hours.

## Headline results (Banking77 official test split, 3,080 messages)

| Approach | Accuracy | Calibration (ECE) | Serving cost / 1M msgs* | Handles new intents without retraining? |
|---|---|---|---|---|
| Zero-shot prompt with all 77 intents listed | 26.6% | 36.6% | $5.14 | yes, but it is wrong 3 times out of 4 |
| RAG: 5 retrieved labelled examples in the prompt | 87.5% | 7.6% | $11.33 | yes (80.2% on new intents) |
| LoRA fine-tune (r=16, 9k examples, 27 min on CPU) | 91.2% | **1.1%** | $4.25 | **no (0.8% on new intents)** |
| QLoRA fine-tune (NF4 4-bit base) | 90.9% | 1.0% | $10.50 | no |
| MiniLM embeddings + logistic regression (37 s) | 92.9% | – | **$0.21** | no, but retrains in seconds |
| **RAFT-style: LoRA trained with retrieved examples in the prompt** | **93.5%** | 4.7% | $14.70 | **yes (88.0% on new intents)** |

\*Measured throughput converted at an assumed $0.18 per instance-hour; see [results/RESULTS.md](results/RESULTS.md)
for latency, confidence intervals, paired significance tests and every ablation.

### What the numbers say

1. **Start with the boring baseline.** With 9,000 labelled examples, sentence embeddings + logistic regression
   (92.9%, 37 s to train) beat the default LoRA run (91.2%; −1.7 pp, paired bootstrap p < 0.001) at about 1/20 of
   the serving cost. Tuning LoRA closes the gap but does not open a new one: rank 64 reaches 92.6% and two epochs
   reach 93.0%. The LLM has to earn its place with something other than accuracy.
2. **Zero-shot does not work at this granularity** for a small model: 26.6%. Intents such as
   `pending transfer` vs `transfer not received by recipient` need examples, not instructions.
3. **For a small model, more retrieved examples hurt.** On the validation split, accuracy fell from 86.7% (k=5)
   to 80.2% (k=10) to 71.4% (k=20) even though the right label was retrieved more often (96.5% → 98.5%).
   k was chosen on validation, never on test.
4. **Fine-tuning buys calibration.** The LoRA model's confidence is nearly exact (ECE 1.1%). Sending the 20% least
   confident messages to a human agent raises accuracy on the rest to 97.8%, which is what a support team needs
   for automated routing.
5. **QLoRA matches LoRA's accuracy (−0.3 pp, not significant) but is 2.5x slower to serve on CPU**, because
   there are no fast 4-bit kernels there. QLoRA saves memory; it does not save time unless the hardware has
   4-bit kernels.
6. **Plain fine-tuning breaks when the label set changes.** Ten intents were hidden from training and "launched"
   afterwards. The plain LoRA model got 0.8% of them right. A LoRA model trained to *read retrieved examples*
   (RAFT-style, only 30 examples for each of the 67 known intents) got **88.0%** of the new intents right with no
   retraining, just by adding their examples to the index, and was the most accurate system overall (93.5%,
   +2.3 pp over plain LoRA, p < 0.001; +0.6 pp over embeddings + LR, p = 0.057, not significant). With the same
   30 examples per intent, plain LoRA reached 88.7%, so most of the gain comes from training the model to use
   retrieved evidence.
7. **Rank and placement matter more than quantisation.** Attention-only LoRA lost 6.5 pp against all linear layers
   (84.7% vs 91.2%); rank 4 / 16 / 64 gave 88.9% / 91.2% / 92.6%. With 10 examples per intent LoRA got 77.3%.
8. **Constrained decoding is free accuracy for prompted models**: free-form generation produced invalid labels
   27.3% of the time zero-shot and 6.7% with RAG (accuracy −5.9 and −5.5 pp); the fine-tuned model barely needs it
   (0.3% invalid).

**Decision guide derived from these runs:** if labels are stable and you have thousands of examples, ship
the embedding classifier first. If you need calibrated confidences for human handoff, fine-tune. If the label
set will change, combine them: fine-tune the model *to use* retrieval rather than to memorise labels.

## How it works

```
                 ┌──────────── labelled pool (Banking77 train minus validation) ───────────┐
                 │                                                                          │
message ──► MiniLM embed ──► top-k similar labelled messages ──┐                            │
                                                                ├─► prompt ─► Qwen2.5-0.5B ─► label-constrained
                     zero-shot: all 77 intent names ────────────┤        (+ LoRA / QLoRA     decoding over a
                     fine-tuned: message only ──────────────────┘          adapter)          token trie of intents
```

- **Label-constrained decoding** ([ftvr/llm.py](ftvr/llm.py)): generation walks a token trie built from the allowed
  intents, so every output is a valid label and the probability of the chosen path is a usable confidence
  score. Section 5 of RESULTS.md compares it with free-form decoding.
- **Prefix KV caching:** the 467-token zero-shot prompt shares everything up to the user message, so it is
  encoded once and its KV cache reused for every batch (a unit test checks the outputs match the uncached path).
- **Answer-only loss** ([ftvr/finetune.py](ftvr/finetune.py)): only answer positions go through the 151k-way LM head,
  which is most of the compute for short-label classification.
- **Resumable training:** LoRA state, optimizer, scheduler and RNG are checkpointed every 50 steps, because the
  cloud machine used here was reset twice while these experiments ran.
- **RAFT-style training** (retrieval-augmented fine-tuning): each training message is shown with its 5 nearest
  *other* training messages and their labels, as at inference time.
- **Honest evaluation:** stratified 10% validation split for choices (k for RAG), official test split only for
  reporting, 95% bootstrap CIs, paired bootstrap tests, and the retrieval pool never contains the query.

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,fetch]"
bash scripts/fetch_data.sh                 # Banking77, pinned commit
python scripts/fetch_models.py             # Hugging Face Hub; or --source dockerhub (what was used here)
pytest -q                                  # hermetic tests with a tiny random Qwen2 model
bash scripts/run_all.sh                    # every experiment; resumable; ~12 h on 4 vCPUs
python -m ftvr.run report                  # regenerates results/RESULTS.md from results/metrics/*.json
```

Single experiments: `python -m ftvr.run {baselines,zeroshot,rag,train,eval-ft,newintent,report} --help`.
On a GPU, the same code runs unchanged apart from device placement and is much faster; the numbers in this
repo were all measured on CPU.

## Repository layout

| Path | What it is |
|---|---|
| `ftvr/data.py` | Banking77 loading, stratified validation split, per-intent subsets, held-out intents |
| `ftvr/prompts.py` | The three prompt formats (zero-shot, retrieval few-shot, fine-tuned) |
| `ftvr/llm.py` | Model loading (LoRA / 4-bit), batched label-constrained decoding, prefix KV cache |
| `ftvr/retrieval.py` | MiniLM embedder, dense index, candidate labels, kNN vote |
| `ftvr/finetune.py` | LoRA / QLoRA / RAFT training loop with answer-only loss and checkpoints |
| `ftvr/metrics.py` | Accuracy, macro-F1, bootstrap CI, paired bootstrap, ECE, selective accuracy, latency |
| `ftvr/run.py`, `ftvr/report.py` | Experiment CLI and the results-table generator |
| `results/` | Per-message predictions (`preds/`), metrics JSON (`metrics/`), `RESULTS.md` |
| `runs/*/train_meta.json` | Training config, loss curve, time, trainable parameters for every adapter |
| `MODEL_CARD.md` | Model card for the fine-tuned adapters |

## Limitations

- One task, one small model, one seed per configuration. Differences under about 1 pp are within noise
  (see the confidence intervals).
- CPU latencies are for a 4-vCPU Xeon with AMX; absolute numbers will differ on other hardware, ratios less so.
- The serving-cost column is measured throughput times an assumed hourly price, not a billed cost.
- No hyperparameter search was run for LoRA (lr 2e-4, 1 epoch, batch 16, except where an ablation changes one of them). The ablations in section 4 of RESULTS.md are reported on the test split, so treat the best ablation numbers as optimistic; the main rows used fixed defaults.
- Banking77 messages are short, English and single-intent; real support traffic is messier.

## Data and model licences

Banking77: CC-BY-4.0 (PolyAI). Qwen2.5-0.5B-Instruct: Apache-2.0. all-MiniLM-L6-v2: Apache-2.0. Code: MIT.
