#!/usr/bin/env bash
# Reproduces every number in results/RESULTS.md (about 8-10 CPU-hours on 4 vCPUs).
# Each step is skipped if its metrics file already exists, so the script can be resumed.
set -euo pipefail
R="python -m ftvr.run"
done_() { [ -f "results/metrics/$1.json" ]; }
trained() { [ -f "runs/$1/train_meta.json" ]; }

done_ minilm-lr-test || $R baselines
# RAG: choose k on validation, then test
for k in 5 10 20; do done_ rag-k$k-all-val || $R rag --k $k --split val; done
K=$(python -c "import json;print(max((5,10,20),key=lambda k:json.load(open(f'results/metrics/rag-k{k}-all-val.json'))['accuracy']))")
echo "best k on validation: $K"
done_ zeroshot-test || $R zeroshot --latency
done_ rag-k$K-all-test || $R rag --k $K --latency
# main fine-tunes
trained lora-r16 || $R train --name lora-r16 --rank 16
done_ lora-r16-test || $R eval-ft --name lora-r16 --latency
trained qlora-r16 || $R train --name qlora-r16 --rank 16 --quant4
done_ qlora-r16-test || $R eval-ft --name qlora-r16 --latency
# new-intent (drift) experiment
trained lora-r16-drop10 || $R train --name lora-r16-drop10 --rank 16 --drop-intents
trained raft-r16-drop10 || $R train --name raft-r16-drop10 --rank 16 --drop-intents --raft --k $K --per-label 30 --batch-size 4
[ -f results/metrics/newintent.json ] || $R newintent --k $K --adapters lora-r16-drop10 raft-r16-drop10
# ablations
for n in 10 30; do
  trained lora-r16-n$n || $R train --name lora-r16-n$n --rank 16 --per-label $n --epochs 3
  done_ lora-r16-n$n-test || $R eval-ft --name lora-r16-n$n
done
for r in 4 64; do
  trained lora-r$r || $R train --name lora-r$r --rank $r
  done_ lora-r$r-test || $R eval-ft --name lora-r$r
done
trained lora-r16-e2 || $R train --name lora-r16-e2 --rank 16 --epochs 2
done_ lora-r16-e2-test || $R eval-ft --name lora-r16-e2
trained lora-r16-attn || $R train --name lora-r16-attn --rank 16 --targets attn
done_ lora-r16-attn-test || $R eval-ft --name lora-r16-attn
done_ zeroshot-free-test || $R zeroshot --free
done_ rag-k$K-all-free-test || $R rag --k $K --free
done_ lora-r16-free-test || $R eval-ft --name lora-r16 --free
$R report || true
echo ALL DONE
