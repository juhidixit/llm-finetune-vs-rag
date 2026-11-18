# Model card: Banking77 intent adapters for Qwen2.5-0.5B-Instruct

This card covers the LoRA adapters trained in this repository. The adapters are produced by
`scripts/run_all.sh` under `runs/<name>/adapter/` (not committed, about 35 MB each); training metadata,
including the loss curve, is committed as `runs/<name>/train_meta.json`.

| Adapter | What it is | Test accuracy | Macro-F1 | ECE |
|---|---|---|---|---|
| `lora-r16` | LoRA r=16, alpha 32, all linear projections, 9,000 examples, 1 epoch | 91.2% | 91.2 | 1.1% |
| `qlora-r16` | Same, frozen base quantised to NF4 (bitsandbytes, double quant) | 90.9% | 90.9 | 1.0% |
| `raft-r16-drop10` | LoRA r=16 trained with 5 retrieved labelled examples in the prompt; 30 examples for each of 67 intents (2,010), 10 intents never seen | 93.5% | 93.5 | 4.7% |

See [results/RESULTS.md](results/RESULTS.md) for confidence intervals, ablations and latency.

## Intended use

- Routing short English customer-support messages of a retail or neobank to one of the 77 Banking77 intents,
  with the predicted label's probability used to send low-confidence messages to a human.
- Research and teaching: a reproducible reference point for fine-tuning vs retrieval trade-offs on CPU.

## Out of scope

- Any decision with financial or legal effect on a customer (credit, fraud, account closure).
- Languages other than English, multi-intent messages, long conversations, or intents outside the 77
  (the plain adapters can only output one of the 77 labels; the RAFT adapter can only output labels present
  in its retrieval index).

## Training data

Banking77 (Casanueva et al., 2020; CC-BY-4.0), train split minus a stratified 10% validation split
(seed 13). The test split was used only for the reported numbers. Messages are synthetic-style but written
to resemble real customer queries; they contain no real customer data.

## Training procedure

Causal-LM loss on the answer tokens only (intent name + end-of-turn), AdamW, lr 2e-4, cosine schedule with
5% warmup, batch 16 (RAFT: 4, for memory), gradient clipping 1.0, LoRA dropout 0.05, bf16 autocast with fp32
LoRA weights, single seed (13). Hardware: 4-vCPU Intel Xeon with AMX, no GPU; 27–30 minutes per adapter.

## Evaluation

Official Banking77 test split (3,080 messages, 40 per intent). Decoding is constrained to valid intent
names with a token trie. Metrics: accuracy with 95% bootstrap CI, macro-F1, expected calibration error
(10 bins, on the probability of the emitted label path), accuracy at 80% coverage, paired bootstrap tests.

## Limitations and risks

- Single seed; differences below about 1 pp are within noise.
- A 0.5B model is weak at world knowledge and long instructions; it is only reliable here because the task is
  narrow and the label set is fixed (or, for RAFT, supplied by retrieval).
- RAFT accuracy depends on the retriever: if no example of the right intent is retrieved, the model is unlikely
  to produce it. Retriever label recall at k=5 was 96.5% on validation.
- Banking77 has known label noise and near-duplicate intents (for example card-arrival vs card-delivery
  estimate), so some "errors" are ambiguous.
- Calibration was measured in-distribution; confidences on off-topic or adversarial input are not validated.

## Licences

Base model Qwen2.5-0.5B-Instruct: Apache-2.0. Adapters and code: MIT. Data: CC-BY-4.0.
