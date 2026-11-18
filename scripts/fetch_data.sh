#!/usr/bin/env bash
# Banking77 (PolyAI, CC-BY-4.0), pinned to the commit used for the published results.
set -euo pipefail
mkdir -p data && cd data
[ -d task-specific-datasets ] || git clone https://github.com/PolyAI-LDN/task-specific-datasets
cd task-specific-datasets && git checkout -q 57ec275d8078af65b7731c2a98be812d844a6d6b
