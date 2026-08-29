# What is in here

Every number in the project's `README.md` comes out of these directories, and
every claim in the changelog can be traced to a file here. Nothing was
hand-copied: `eval/summarise.py` regenerates the tables from `scores.json`.

## Directories

| directory | what it is |
|---|---|
| `main/`, `main-2/`, `main-3/` | three independent runs of all twelve cases with all three arms. Three, not one, because `temperature=0` does not make a served model deterministic — see the variance note in the main README. |
| `ablation-retype/` | does curating finding **ids** actually prevent evidence loss, versus letting the model re-emit findings itself? |
| `ablation-verifier/` | what the verification pass over suppressions buys. |
| `ablation-experimental/` | the retired vision detector switched back on, at the level of the whole pipeline. |
| `experiment-class-swap.json` | the measurement that retired that detector: what it claims on a corpus with no swaps in it, and a sweep over operating points. |

## Inside a run directory

```
scores.json                       aggregate + per-case metrics for every arm
reports/<case>.ground_truth.json  exactly what the injector did, and why
reports/<case>.<arm>.json         what that arm reported
trajectories/<case>.<arm>.json    every model call, tool call and tool result
trajectories/<case>.<arm>.md      the same, rendered readable
```

## Reading a trajectory

Start with the `.md` — it opens with the agent's instructions verbatim, then
the task, then every turn with its tool calls and the results that came back.
`note` steps are the interesting ones: they record retries, recoveries, and any
point where the loop had to intervene. The rendered file ends with a statement
of where the human checkpoint sits.

To regenerate them:

```bash
python eval/render_trajectory.py runs/main/trajectories/ --all
```

## Verifying a single number

Pick any cell in the README's results table and chase it:

1. `scores.json` → `per_case` → the row for that case and arm.
2. `missed_facts` and `fp_facts` on that row name the exact
   `(defect_type, file)` pairs that were missed or invented.
3. `reports/<case>.ground_truth.json` says what was injected, with the seed.
4. `trajectories/<case>.<arm>.json` shows how the arm got there.
