# Reproduction guide

Written for someone starting from an empty directory who has never seen this
project. Every number in `README.md` comes out of
the commands below.

## 1. What you need

| | |
|---|---|
| OS | Linux or macOS (developed on Ubuntu 22.04, kernel 5.15) |
| Python | 3.11 (3.10 also works) |
| Disk | ~350 MB (COCO labels 48 MB, 600 JPEGs ~95 MB, CPU PyTorch ~200 MB) |
| Network | needed once, to fetch the public corpus |
| GPU | **not required** for the solution or the evaluation |
| An LLM endpoint | anything OpenAI-compatible with tool calling |

The `script` arm needs no model at all. The `baseline` and `agent` arms need an
endpoint.

## 2. Set up the environment

Using `pyenv`, which is what this was developed with:

```bash
pyenv install -s 3.11.12
pyenv virtualenv 3.11.12 micro1
cd dataset-doctor
pyenv local micro1                 # the repo already contains .python-version
python -m pip install -e .
```

Plain `venv` works identically:

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -e .
```

The optional `vision` extra is only needed to reproduce the *retired*
experiment in iteration 4 of the changelog (README.md). Skip it unless
you want to:

```bash
pip install -e ".[vision]"         # adds ultralytics + CPU PyTorch, ~200 MB
```

## 3. Point at a model

The defaults target a local vLLM server. Override with environment variables or
CLI flags; nothing in the code is specific to a provider.

```bash
export DSDOCTOR_BASE_URL="http://localhost:8000/v1"   # any OpenAI-compatible URL
export DSDOCTOR_MODEL="qwen3.8-27b"
export DSDOCTOR_API_KEY="..."                          # ignored by local vLLM
```

The published results were produced against **Qwen3.8-27B (W4A16 AWQ)** served
by **vLLM 0.28.0** on a single RTX 3090, started like this:

```bash
vllm serve philbert440/Qwen3.8-27B-W4A16-AWQ \
  --served-model-name qwen3.8-27b \
  --max-model-len 65536 --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 --max-num-seqs 4 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching
```

Tool calling must be enabled on the server or the agent arm cannot run.

## 4. Build the corpus

```bash
python eval/build_corpus.py --images 600 --out data/corpus_clean
```

Downloads public COCO 2017 labels and 600 val2017 JPEGs, filters to the twelve
best-populated classes, repairs or discards anything a detector would flag, and
splits train/val. It ends by running the full detector suite over the result and
**fails if a single finding remains** — the evaluation is only valid because the
base corpus is provably clean.

Expected tail:

```
[attempt 1] 12 classes, 468 train / 132 val -> 0 finding(s)

--- clean corpus ---
  0 findings across 7 detectors
  classes (12): person, chair, car, dining table, cup, bottle, bowl, handbag, truck, bench, book, backpack
  splits: {'train': {'images': 468, ...}, 'val': {'images': 132, ...}}
  boxes: 6763
```

Runtime: ~3 minutes on a first run (dominated by the downloads), seconds after
that. The downloads are cached in `data/_cache/`.

> Image selection is greedy over a seeded shuffle, so the exact 600 files depend
> on `--seed` (default 7). Keep the default to match the published numbers.

## 5. Run the evaluation

```bash
python eval/run_eval.py --arms script,baseline,agent
```

This builds all twelve cases from the clean corpus, runs each arm on each one,
scores against the injected ground truth, and writes everything to
`runs/<UTC timestamp>/`:

```
runs/<stamp>/
  scores.json                       aggregate + per-case metrics
  reports/<case>.ground_truth.json  exactly what was injected
  reports/<case>.<arm>.json         what the arm reported
  trajectories/<case>.<arm>.json    every model call and tool result
```

Approximate runtime and cost on the reference setup:

| arm | per case | all 12 | model calls | tokens (12 cases) |
|---|---|---|---|---|
| `script` | 5 s | 64 s | 0 | none — no model at all |
| `baseline` | 63 s | 12.6 min | 12 | 434k in / 15k out |
| `agent` | 74 s | 14.9 min | 55 | 215k in / 34k out |

Measured on the reference setup, one client at a time. Add roughly 3 minutes
for the first corpus build, and about 30 s for the one-off `yolov8n.pt`
download if you install the `vision` extra.

Self-hosted, so the monetary cost here is electricity. On metered API pricing
the whole three-arm, twelve-case run is a few dollars at Sonnet-class rates —
the agent arm is the cheaper of the two model arms in prompt tokens, because it
reads targeted tool results instead of being handed a large slice of the
dataset up front.

> **Run it on a server nothing else is using.** Three sequential runs of this
> evaluation reproduced identically, down to the individual findings. Two runs
> sharing the vLLM server with another client did not — batch composition
> changes the order of floating-point reductions, and that is enough to move
> the output. If your numbers differ from the published ones, check that
> nothing else was talking to the model at the time.

Useful subsets while iterating:

```bash
python eval/run_eval.py --arms script                        # no model needed
python eval/run_eval.py --arms agent --cases subtle_leak     # one case
python eval/run_eval.py --arms script --experimental         # the retired detector
```

To reproduce the measurement behind the removed experiment (iteration 4 of the
changelog), and the ablations behind the design decisions:

```bash
pip install -e ".[vision]"
python eval/experiment_class_swap.py                         # ~5 min, CPU only
python eval/run_eval.py --arms agent,agent_retype --cases everything,structure_rot,leak_and_dupes,duplicate_farm
```

To regenerate the README's tables from any run directory, and to turn the
trajectories into readable Markdown:

```bash
python eval/summarise.py runs/main
python eval/render_trajectory.py runs/main/trajectories/ --all
```

## 6. Run the tool on a real dataset

The evaluation is not the product. This is.

`pip install -e .` also puts a `dsdoctor` script on your PATH, so
`dsdoctor scan ...` works too; the `python -m` form below is used throughout
because it needs no shim rehash and behaves the same under every install
method.

```bash
# deterministic checks only, no model, a few seconds
python -m dsdoctor.cli scan /path/to/your/dataset

# full audit: report, ordered fix plan, trajectory
python -m dsdoctor.cli audit /path/to/your/dataset --out audit_out
cat audit_out/audit_report.md

# apply the plan - prompts before touching anything, backs up every file
python -m dsdoctor.cli apply audit_out/fix_plan.json
```

Your dataset needs the standard Ultralytics layout:

```
root/
  data.yaml            names: [...]
  images/train/*.jpg   images/val/*.jpg
  labels/train/*.txt   labels/val/*.txt
```

To try it without a dataset of your own, point it at a generated case:

```bash
python -m dsdoctor.cli audit data/cases/everything --out audit_out
```

## 7. Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # 66 tests, ~2 s, no network, no model
```

They cover the parts the results depend on:

- every detector fires on its own defect and stays silent on a clean dataset;
- every injected defect is recoverable by the detector suite, so a miss is the
  arm's fault and not an impossible case;
- injected filenames never reveal the defect, which is a regression test for a
  real flaw that let one arm "detect" leakage by reading the file listing;
- the baseline's answer is parsed from truncated as well as complete JSON;
- the agent loop recovers from a turn that produces no tool call, and gives up
  rather than spinning;
- the scorer's hit / miss / false-positive arithmetic;
- the safety gate — declining the prompt changes nothing, approving writes a
  backup, and steps marked `requires_human_review` are never applied.

## 8. Verifying the claims

| Claim | Command |
|---|---|
| the base corpus is clean | `python eval/build_corpus.py` — asserts 0 findings, exits 1 otherwise |
| ground truth is what we say | `cat runs/<stamp>/reports/<case>.ground_truth.json` |
| the agent did what we say | `cat runs/<stamp>/trajectories/<case>.agent.json` |
| the detectors use no model | `grep -rn "llm\|openai" src/dsdoctor/detectors/` returns nothing |
| nothing is modified without consent | `python -m dsdoctor.cli apply ...` and answer anything but `apply` |
