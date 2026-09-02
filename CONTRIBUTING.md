# Contributing

## The one rule that shapes everything else

**No detector may call a language model.** Every defect claim in this system
starts as a pure function over the files on disk. The agent decides which
checks to run, how to rank them and what to say about them; it never invents a
fact about the data. `grep -rn "openai" src/dsdoctor/detectors/` returns
nothing, and a change that alters this is a change to the project's premise,
not a feature.

## Getting set up

```bash
git clone https://github.com/Furqan3/dataset-doctor
cd dataset-doctor
pip install -e ".[dev]"
pytest                      # 217 tests, offline, about 55 seconds
```

You do not need a GPU, a model endpoint, or the evaluation corpus to work on
most of this. `dsdoctor scan`, `card`, `verify-card`, `recheck`, `diff`,
`resplit` and `convert` never contact a model at all.

To work on the agent you need any OpenAI-compatible endpoint:

```bash
export DSDOCTOR_BASE_URL=http://localhost:8000/v1
export DSDOCTOR_MODEL=your-model
```

## Adding a check

A detector is a function `Dataset -> list[Finding]` registered with a
decorator. The shape to copy is `src/dsdoctor/detectors/split.py`.

```python
@register("my_scan", "one line the agent reads when choosing checks",
          covers=("my_defect_type",), group="split")
def my_scan(ds: Dataset) -> list[Finding]:
    ...
```

Four things are required of a new check.

1. **Declare a defect type in `findings.py`.** The scorer matches on
   `(type, file)` pairs, so a finding with an ad-hoc type string cannot be
   evaluated. If your check reports something the vocabulary cannot express,
   extend the vocabulary in the same change.

2. **Put it in a group, and default to an optional one.** `core` is the set
   the published results were measured with. Anything added afterwards goes in
   `split`, `metadata`, `privacy` or a new group until it has been through
   `eval/run_eval.py` on all twelve cases. This is enforced by
   `tests/test_groups.py::test_default_detector_set_is_unchanged`, which will
   fail if you widen the default set — that failure is the test doing its job,
   and the fix is a measurement, not an edit to the expected value.

3. **Carry the evidence.** Populate `items` with `split/stem` keys and
   `evidence` with the raw rows or numbers the claim rests on. A finding a
   reviewer cannot check is a finding they have to take on faith.

4. **Prove it is silent on clean data.** Add a test that the check returns
   nothing on the `clean_root` fixture. A check with a false-positive rate is
   worse than no check: it trains people to skim the report.

## Writing a detector plugin

Organisations have conventions this repository should not try to guess — a
naming scheme, a class taxonomy, a licence policy. Ship them as a normal
package instead of forking:

```toml
# in your package's pyproject.toml
[project.entry-points."dsdoctor.detectors"]
acme = "acme_checks:register_all"
```

`register_all` takes no arguments and calls `dsdoctor.detectors.register` once
per check. `dsdoctor detectors` lists what loaded. A plugin that raises on
import is reported and skipped rather than taking the audit down.

## Things that need more care than usual

**The converters must be faithful, not corrective.** `src/dsdoctor/formats/`
turns COCO and VOC into a YOLO view, and the temptation is to clamp
coordinates, drop degenerate boxes and skip unknown categories. Every one of
those is a defect this tool exists to report. The only correction permitted is
sub-pixel quantisation, and `snap_subpixel` documents exactly why that one is
not a judgement call. See `tests/test_formats.py` — two of those tests exist
because the converter got this wrong during development.

**Anything that writes needs a backup and a prompt.** `scan`, `audit`, `card`
and `resplit` never modify the dataset; `tests/test_apply.py` asserts it across
every check group. `apply` is the only path that writes, and it backs up each
file first.

**Do not change the default check set to make a number look better.** If a
detector is not earning its place, the honest move is to measure that and
retire it, as iteration 4 did in the README.

## Adding a check to an opt-in group

If your check belongs to `split`, `metadata`, `privacy`, `training` or
`annotations`, it can and should be *measured* rather than argued for:

1. Add an injector to `eval/injector.py` that creates the defect the way it
   actually arrives in a delivered dataset.
2. Add a case to `EXTENDED_CASES` in `eval/cases.py`, naming the groups it
   needs. Do not add it to `CASES` — that list is the published twelve.
3. Run `python eval/run_extended.py` and put the numbers in the PR.

Watch the base-rate column. If your check reports a *property* of the data
rather than an injected defect, the clean corpus will produce findings too,
and those are not false positives — `undetectable_at_imgsz` fires on 96 files
of a provably clean corpus and is right to. The scorer separates them; do not
"fix" a check to make that column smaller.

## Running the evaluation

```bash
python eval/build_corpus.py           # downloads COCO; fails if not provably clean
python eval/run_eval.py --out runs/my-run
python eval/summarise.py runs/my-run  # regenerates the README tables
python eval/run_extended.py           # scores the opt-in groups, no model needed
```

See [REPRODUCTION.md](REPRODUCTION.md) for the full path from a clean
environment.

## Pull requests

Say what you measured, not only what you changed. The README's changelog is
written as "what I believed, what the measurement said, what I did about it" —
a PR that follows that shape is easy to review and easy to trust.
