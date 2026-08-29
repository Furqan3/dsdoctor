"""The simple baseline: one direct prompt with basic instructions.

This is the "reasonable basic way to handle the task before using the
solution" that the comparison is measured against, and it is written to be as
strong as a single prompt can honestly be:

  * it gets the same defect vocabulary the agent's detectors use, so scoring
    never punishes it for naming a defect differently;
  * it gets the full dataset summary and per-class counts;
  * it gets the complete file listing for both splits;
  * it gets the raw label rows for as many files as fit in a generous slice of
    the context window, sampled deterministically.

What it cannot do is the thing that needs tools: hash 600 images, decode them,
or cross-check every row. That gap is the point of the comparison, not a
handicap imposed on it.
"""

from __future__ import annotations

import json
import random
import re

from .dataset import Dataset
from .findings import DEFECT_TYPES
from .llm import LLM, Trajectory

# Characters of raw label text to hand the baseline.
#
# Two measurements set this. First, label rows are digits and decimal points,
# and they tokenise at roughly 1 char/token - not the ~3.5 of prose - so 42k
# characters produced a 52k-token prompt. Second, a reasoning model given this
# task spends thousands of tokens walking the rows, and the first version of
# this baseline burned its entire 4k output budget thinking and emitted no
# answer at all. Scoring that as "the baseline found nothing" would have been
# a fabricated result.
#
# So the budget reserves room for the answer rather than filling the window
# with input. That the single prompt can then only see a fraction of the
# dataset is not a handicap imposed on it - it is the actual constraint of
# doing this job in one prompt, and it is the finding.
#
# The output cap is 6144 because the answers that do arrive are ~5k characters,
# well inside it. It is not larger because the observed failure was a
# non-terminating enumeration rather than a slightly-too-small budget, and a
# bigger cap only makes that failure more expensive.
LABEL_BUDGET_CHARS = 26_000
OUTPUT_TOKENS = 6_144
BUDGET_BACKOFF = 0.6
MIN_BUDGET_CHARS = 4_000

BASELINE_PROMPT = """\
You are reviewing an object-detection dataset in YOLO format to decide whether \
it is safe to train on. Below is a summary of the dataset, the list of files in \
each split, and the raw label rows for a large sample of those files.

Report every defect you can find. These are the defect types, use these exact \
type ids:

{vocabulary}

Reply with a single JSON object and nothing else:

{{"verdict": "blocked" | "fix_before_training" | "usable_with_caveats",
  "headline": "<one sentence>",
  "findings": [
    {{"type": "<one of the type ids above>",
      "severity": "critical" | "major" | "minor",
      "files": ["train/000000012345", "..."],
      "rationale": "<why this matters>"}}
  ]}}

For dataset-wide defects that are not about particular files, use the single \
file key "<dataset>". List every affected file you can identify.
"""


def run_baseline(ds: Dataset, llm: LLM, seed: int = 0,
                 budget_chars: int = LABEL_BUDGET_CHARS) -> tuple[dict, Trajectory]:
    """One prompt, with as much of the dataset as the context window allows.

    The budget adapts: if the backend rejects the prompt as too long we shrink
    the label sample and try again, so the baseline stays as strong as the
    model's context permits instead of being tuned by hand per model.
    """
    traj = Trajectory(agent="baseline-single-prompt", model=llm.model)
    vocabulary = "\n".join(f"  {t} - {desc}" for t, (_, desc) in DEFECT_TYPES.items())

    summary = ds.summary()
    listing = {sp: sorted(s.stem for s in ds.in_split(sp)) for sp in ds.splits}

    rng = random.Random(seed)
    keys = [s.key() for s in ds.samples]
    rng.shuffle(keys)

    blocks: list[str] = []
    shown_keys: list[str] = []
    used = 0
    shown = 0
    for key in keys:
        s = ds.get(key)
        if s is None or s.label is None:
            continue
        rows = "\n".join(b.raw for b in s.label.boxes)
        for pe in s.label.parse_errors:
            rows += f"\n{pe.raw}"
        block = f"### {key}\n{rows or '(empty file)'}\n"
        if used + len(block) > budget_chars:
            continue
        blocks.append(block)
        shown_keys.append(key)
        used += len(block)
        shown += 1

    content = (
        f"## Dataset summary\n```json\n{json.dumps(summary, indent=2)}\n```\n\n"
        f"## File listing\n```json\n{json.dumps(listing, indent=2)}\n```\n\n"
        f"## Raw label rows ({shown} of {len(ds.samples)} files)\n"
        + "\n".join(blocks)
    )

    traj.note(f"baseline context: {shown}/{len(ds.samples)} label files, "
              f"{used} chars of label rows")

    messages = [{"role": "system",
                 "content": BASELINE_PROMPT.format(vocabulary=vocabulary)},
                {"role": "user", "content": content}]

    # Attempt order is set by measurement, not preference. With thinking on,
    # this prompt makes the model enumerate label rows one at a time: at a
    # 4,096-token cap every case ended with finish_reason=length and an empty
    # answer, which the scorer would have recorded as "the baseline found
    # nothing". Raising the cap does not fix a non-terminating enumeration, it
    # just pays more for the same empty result. With thinking disabled the
    # same prompt answers in about 83s. So the path that works goes first, and
    # the plain call is the fallback for any backend that rejects the switch.
    attempts = [
        ("thinking disabled", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("backend default", None),
    ]

    parsed = None
    for label, extra in attempts:
        try:
            msg = llm.chat(messages, traj=traj, max_tokens=OUTPUT_TOKENS,
                           retries=1, extra_body=extra, label="baseline")
        except RuntimeError as exc:
            if _is_context_error(exc):
                smaller = int(budget_chars * BUDGET_BACKOFF)
                if smaller < MIN_BUDGET_CHARS:
                    traj.note("context too long even at the minimum budget")
                    break
                traj.note(f"prompt exceeded the context window at "
                          f"{budget_chars} chars of labels; retrying at {smaller}")
                return run_baseline(ds, llm, seed=seed, budget_chars=smaller)
            traj.note(f"attempt '{label}' failed: {exc}")
            continue

        parsed = _parse(msg["content"], traj, quiet=True)
        if parsed is not None:
            break
        # Truncated but non-empty output still contains real findings.
        parsed = _salvage(msg["content"] or "")
        if parsed is not None:
            traj.note(f"attempt '{label}' was cut off at "
                      f"{msg.get('finish_reason')}; salvaged "
                      f"{len(parsed['findings'])} complete finding(s) from the "
                      f"partial JSON")
            break
        traj.note(f"attempt '{label}' produced no parseable answer "
                  f"(finish_reason={msg.get('finish_reason')}, "
                  f"{len(msg.get('content') or '')} content chars)")

    if parsed is None:
        traj.note("baseline produced no usable JSON; scored as zero findings")
        parsed = {"verdict": "unknown", "headline": "", "findings": []}
    parsed["_files_shown"] = shown_keys
    return parsed, traj


def _is_context_error(exc: Exception) -> bool:
    t = str(exc).lower()
    return ("maximum context length" in t or "context_length_exceeded" in t
            or "too long" in t or "reduce the length" in t)


def _salvage(text: str) -> dict | None:
    """Recover the findings from a truncated answer.

    The model reliably starts emitting well-formed JSON and then runs out of
    output budget part way through the findings array. Scoring that as "found
    nothing" would be a lie about what the baseline actually produced, so we
    walk the array and keep every complete object, discarding only the one it
    was in the middle of writing.
    """
    i = text.find('"findings"')
    if i == -1:
        return None
    j = text.find("[", i)
    if j == -1:
        return None

    objs: list[dict] = []
    depth = 0
    start = None
    in_str = False
    esc = False
    for k in range(j + 1, len(text)):
        c = text[k]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = k
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start:k + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
        elif c == "]" and depth == 0:
            break

    # The truncation often lands inside the *first* finding, when the model
    # emits one defect with a very long file list. Recovering only complete
    # objects would then throw away the entire answer, so pull the type and
    # whatever whole filenames it managed to write out of the trailing object
    # too. This can only ever recover claims the baseline actually made -
    # including wrong ones, which is the point.
    if start is not None:
        tail = text[start:]
        m_type = re.search(r'"type"\s*:\s*"([^"]+)"', tail)
        if m_type:
            m_sev = re.search(r'"severity"\s*:\s*"([^"]+)"', tail)
            files = []
            m_files = re.search(r'"files"\s*:\s*\[', tail)
            if m_files:
                files = re.findall(r'"([^"\n]+)"', tail[m_files.end():])
            objs.append({"type": m_type.group(1),
                         "severity": m_sev.group(1) if m_sev else "major",
                         "files": files,
                         "rationale": "(recovered from a truncated answer)"})

    if not objs:
        return None
    verdict = re.search(r'"verdict"\s*:\s*"([^"]+)"', text)
    headline = re.search(r'"headline"\s*:\s*"([^"]*)"', text)
    return {"verdict": verdict.group(1) if verdict else "unknown",
            "headline": headline.group(1) if headline else "",
            "findings": objs,
            "_salvaged": True}


def _parse(text: str, traj: Trajectory, quiet: bool = False) -> dict | None:
    t = (text or "").strip()
    if "```" in t:
        parts = t.split("```")
        t = max(parts, key=len).removeprefix("json").strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(t[start:end + 1])
            if isinstance(obj.get("findings"), list):
                return obj
        except json.JSONDecodeError as exc:
            if not quiet:
                traj.note(f"baseline JSON did not parse: {exc}")
    return None


def baseline_scope(report: dict) -> set[str]:
    """The file keys the baseline was actually shown.

    Reported alongside recall so that a coverage limit (it never saw the file)
    is never confused with a capability limit (it saw the file and still could
    not tell).
    """
    return set(report.get("_files_shown") or [])


def baseline_facts(report: dict) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for f in report.get("findings") or []:
        if not isinstance(f, dict):
            continue
        dtype = str(f.get("type", "")).strip()
        files = f.get("files") or ["<dataset>"]
        if isinstance(files, str):
            files = [files]
        for k in files:
            out.add((dtype, str(k).strip()))
    return out
