#!/usr/bin/env bash
# The ablations behind the design decisions in the README.
#
# Each one exists because a claim in the write-up would otherwise be an
# assertion. Run after the main evaluation; they reuse the same corpus and
# the same cases.
set -euo pipefail
cd "$(dirname "$0")/.."

CASES="everything,structure_rot,leak_and_dupes,duplicate_farm"

# 1. Does curating finding ids actually prevent evidence loss?
#    `agent_retype` gets the same tools and a prompt that differs only in the
#    submit step, but re-emits findings itself instead of referring to ids.
echo "=== ablation 1: curate-by-id vs retype ==="
python eval/run_eval.py --arms agent,agent_retype --cases "$CASES" \
  --out runs/ablation-retype

# 2. What does the verification pass over suppressions actually buy?
echo "=== ablation 2: verifier on/off ==="
python eval/run_eval.py --arms agent,agent_noverify --cases "$CASES" \
  --out runs/ablation-verifier

# 3. What happens if the retired detector is switched back on? This is the
#    measurement behind iteration 4, at the level of the whole pipeline.
echo "=== ablation 3: the retired detector, re-enabled ==="
python eval/run_eval.py --arms script,agent --cases pristine,everything \
  --experimental --out runs/ablation-experimental

echo
echo "wrote runs/ablation-retype, runs/ablation-verifier, runs/ablation-experimental"
