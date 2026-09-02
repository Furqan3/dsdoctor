"""The auditing agent: a tool loop plus a verification pass over its own cuts.

Division of labour, which is the whole design:

  detectors  supply recall. They read the files literally and are never wrong
             about what is on disk.
  the agent  supplies precision and priority. It decides which checks are worth
             running, which findings are real, and what the engineer should do
             first.
  the verifier
             guards the one thing the agent can break. Letting a model suppress
             findings is what makes the report usable; it is also the only way
             this system can lose a real defect. So every suppression is
             re-argued from the evidence by a second pass that starts from the
             opposite prior, and anything it will not defend is reinstated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .dataset import Dataset
from .findings import (Finding, sort_findings, SEVERITY_ORDER,
                       DEFECT_TYPES, MAJOR as MAJOR_DEFAULT)
from .llm import LLM, Trajectory, Step
from .tools import ToolBox, schemas, compact_json

MAX_STEPS = 24

# Per-turn output budget. 2048 was too small twice over: the model's reasoning
# could run past it and emit no tool call at all, and a submit_report carrying
# a dozen decisions is itself ~1600 tokens, so a busy dataset could truncate
# the report the whole run exists to produce.
MAX_TURN_TOKENS = 4096

# How many consecutive turns may fail to produce a tool call before we stop
# paying for them and fall back to the unfiltered detector output.
MAX_STUCK_TURNS = 3

# Applied when a turn runs out of budget mid-reasoning. Disabling thinking
# ends the enumeration that caused it; forcing a tool choice guarantees the
# turn produces something the loop can act on.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}

# The verifier answers with one small JSON object, but it needs room to
# reach it. 400 was not enough even with thinking off.
VERIFIER_TOKENS = 512

SYSTEM_PROMPT = """\
You audit object-detection training datasets in YOLO format.

Your user is an ML engineer who has just inherited a labelled dataset and has \
to decide today whether to train on it. They are not asking for a list of \
everything technically imperfect. They need three answers: will this train at \
all, will the validation number mean anything, and what has to be fixed first.

How to work:

1. Call dataset_summary first, to see the scale, the splits and the classes.

2. Call list_detectors, then run the ones that matter. Each detector reports a \
`reliability`. Detectors marked exact read the files directly and use no \
language model, so what they say about the bytes on disk is fact, not opinion; \
run all of them, they are cheap. Detectors marked experimental are known to \
produce false positives - treat anything they return as a hypothesis you have \
to check, never as a finding on its own.

3. Detector output is not a report. What needs your judgement is:

   Ranking. A dataset with a train/val leak and a handful of oversized boxes \
has one urgent problem and one cosmetic one. Lead with what actually blocks \
the engineer, and say plainly what it costs them.

   Reading the evidence. Before you rank a finding highly, call \
inspect_finding and look at the rows it is built on. Whether a defect sits in \
train or in val often matters more than how many files it touches: anything \
wrong in val corrupts the number the team will use to make a decision.

   Precision, where a detector is not exact. If an experimental detector \
raises something, inspect it and ask whether the pattern is what a real bug \
looks like - one-directional, near-total, concentrated - or the scattered \
disagreement you would expect from ordinary ambiguity. Suppress the second \
kind and say which pattern you saw.

4. Never suppress a finding from an exact detector. Those findings are \
statements about bytes and they are always true. You may re-rank them. You may \
not delete them.

5. Finish by calling submit_report exactly once. Include one decision per \
finding_id you have seen, ordered so the most urgent comes first. Anything you \
do not mention is reported anyway, so the only reason to list a finding is to \
rank it or to argue it away.

6. Take the verdict from the most severe finding you are actually reporting, \
not from an overall impression of the dataset. If any reported finding is \
critical, the verdict is "blocked" - a single zero-area box that puts a NaN in \
the loss blocks the run no matter how healthy the other 6000 boxes are. If the \
worst reported finding is major, the verdict is "fix_before_training". Only \
when everything left is minor is it "usable_with_caveats".

Write the rationales for a competent engineer who is short on time. No \
preamble, no restating the obvious.
"""

# The ablation has to be given instructions that match the tool it is handed,
# or it would be losing to a prompt/schema mismatch rather than to the design
# difference the ablation exists to measure.
RETYPE_STEP_5 = """\
5. Finish by calling submit_report exactly once. List every defect you are \
reporting, and for each one list every affected file. Completeness matters: a \
file you leave out is a defect the engineer never hears about."""

CURATE_STEP_5 = """\
5. Finish by calling submit_report exactly once. Include one decision per \
finding_id you have seen, ordered so the most urgent comes first. Anything you \
do not mention is reported anyway, so the only reason to list a finding is to \
rank it or to argue it away."""


def system_prompt(retype: bool = False) -> str:
    if not retype:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT.replace(CURATE_STEP_5, RETYPE_STEP_5)


VERIFIER_PROMPT = """\
You are checking one decision made by a dataset auditor: it chose to suppress a \
finding, meaning the finding will not be shown to the engineer.

Suppression is only justified when the evidence shows the finding is not a real \
defect. Your default is to reinstate. Uphold the suppression only if the \
evidence positively supports it.

The one case where suppression is normally correct: a class_swap finding where \
the reference detector disagrees with the labels in a scattered, two-directional \
way between visually similar classes. That is annotation ambiguity, not a defect.

The cases where suppression is never correct: any finding about file structure, \
box geometry, coordinate normalisation, class ids, duplicates or leakage. Those \
are measurements of the files themselves.

Reply with a single JSON object and nothing else:
{"uphold": true or false, "reason": "<one sentence>"}
"""


@dataclass
class Decision:
    finding: Finding
    finding_id: str
    action: str            # "report" | "suppress"
    severity: str
    rationale: str
    source: str = "agent"  # "agent" | "default" | "reinstated"


@dataclass
class AuditResult:
    verdict: str
    headline: str
    training_impact: str
    decisions: list[Decision] = field(default_factory=list)
    trajectory: Trajectory | None = None
    detectors_run: list[str] = field(default_factory=list)
    suppressed_total: int = 0
    reinstated_total: int = 0
    incomplete: str = ""
    swept_fallback: bool = False

    @property
    def reported(self) -> list[Decision]:
        return [d for d in self.decisions if d.action == "report"]

    @property
    def suppressed(self) -> list[Decision]:
        return [d for d in self.decisions if d.action == "suppress"]

    def as_facts(self) -> set[tuple[str, str]]:
        """(type, file_key) pairs, which is what the scorer compares."""
        out = set()
        for d in self.reported:
            for k in (d.finding.items or ["<dataset>"]):
                out.add((d.finding.type, k))
        return out


def audit(ds: Dataset, llm: LLM, *, experimental: bool = False,
          verify: bool = True, retype: bool = False,
          groups: list[str] | None = None,
          max_steps: int = MAX_STEPS) -> AuditResult:
    """Audit a dataset.

    `retype=True` selects the ablation in which the model re-emits findings
    instead of curating detector ids. It exists to measure the shipped
    design, not to be used.
    """
    tb = ToolBox(ds, experimental=experimental, retype=retype,
                 groups=list(groups or []))
    tools = schemas(retype)
    traj = Trajectory(agent="auditor", model=llm.model)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt(retype)},
        {"role": "user", "content":
            f"Audit the dataset at {ds.root}. Decide whether it is safe to "
            f"train on, and tell me what to fix first."},
    ]

    incomplete = ""
    stuck = 0          # consecutive turns that produced no tool call
    force_call = False  # next turn uses tool_choice=required
    no_think = False    # next turn disables thinking
    for step in range(max_steps):
        msg = llm.chat(messages, tools=tools,
                       tool_choice="required" if force_call else "auto",
                       traj=traj, max_tokens=MAX_TURN_TOKENS,
                       extra_body=NO_THINKING if no_think else None,
                       label="auditor")
        force_call = no_think = False

        if msg["tool_calls"]:
            stuck = 0
        else:
            # Two different failures land here and they need different
            # remedies, which the first version of this loop conflated.
            #
            #   finish_reason == "stop"   the model decided on a tool and then
            #                             emitted nothing. Forcing a tool
            #                             choice makes the server decode one.
            #   finish_reason == "length" the model was still reasoning when
            #                             the budget ran out. Forcing a tool
            #                             choice does not help - it just buys
            #                             another truncated turn. Observed:
            #                             one case spent 17 turns and 1,017s
            #                             this way. Disabling thinking ends
            #                             the enumeration.
            stuck += 1
            truncated = msg.get("finish_reason") == "length"
            if stuck >= MAX_STUCK_TURNS:
                traj.note(f"step {step}: {stuck} consecutive turns produced no "
                          f"tool call; abandoning the loop")
                incomplete = (f"agent stalled after {stuck} turns without a "
                              f"tool call")
                break
            if truncated:
                traj.note(f"step {step}: turn hit the output token limit while "
                          f"reasoning; retrying with thinking disabled and a "
                          f"forced tool choice")
                force_call = no_think = True
                continue
            if not (msg["content"] or "").strip():
                traj.note(f"step {step}: empty turn with no tool call; "
                          f"retrying with tool_choice=required")
                force_call = True
                continue
            messages.append({"role": "assistant", "content": msg["content"]})
            messages.append({"role": "user", "content":
                             "Continue. Use a tool, or call submit_report if "
                             "you have everything you need."})
            continue

        messages.append({"role": "assistant", "content": msg["content"] or None,
                         "tool_calls": msg["tool_calls"]})

        done = False
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError as exc:
                result = {"error": f"arguments were not valid JSON: {exc}"}
            else:
                result = tb.call(name, args)
            traj.steps.append(Step(kind="tool", name=name,
                                   request=args if isinstance(args, dict) else {},
                                   response=_trim_result(result)))
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": compact_json(result)})
            if name == "submit_report" and tb.report is not None:
                done = True
        if done:
            break
    else:
        incomplete = (f"agent did not call submit_report within {max_steps} "
                      f"steps; falling back to the raw detector output")

    if tb.report is not None and not any(_emits_findings(n) for n in tb.ran):
        # The agent concluded without running a single finding-producing
        # detector. Shipping "looks fine" on the strength of that would be the
        # worst possible failure for this tool, so fall back to a full sweep.
        # This is recorded, and the evaluation counts how often it fires - if
        # it fired often the agent's recall would be the script arm's in
        # disguise.
        traj.note("agent submitted a report without running any detector; "
                  "falling back to a full sweep")
        incomplete = "agent reported without running any detector; swept anyway"
        for det in _default_sweep(tb, experimental, groups):
            tb.run_detector(det)

    if tb.report is None:
        # Never return nothing: an audit that fails still has to hand over the
        # facts the detectors established.
        if not tb.ran:
            for det in _default_sweep(tb, experimental, groups):
                tb.run_detector(det)
        tb.report = {"verdict": "fix_before_training",
                     "headline": "Automated triage did not complete; showing "
                                 "all detector findings unfiltered.",
                     "decisions": [], "training_impact": ""}
        incomplete = incomplete or "agent produced no report"

    decisions = (_resolve_retyped(tb, tb.report, traj) if retype
                 else _resolve(tb, tb.report, traj))

    if verify:
        decisions = _verify_suppressions(decisions, llm, traj)

    reinstated = sum(1 for d in decisions if d.source == "reinstated")
    result = AuditResult(
        verdict=tb.report.get("verdict", "fix_before_training"),
        headline=tb.report.get("headline", ""),
        training_impact=tb.report.get("training_impact", ""),
        decisions=_rank(decisions),
        trajectory=traj,
        detectors_run=list(tb.ran),
        suppressed_total=sum(1 for d in decisions if d.action == "suppress"),
        reinstated_total=reinstated,
        incomplete=incomplete,
        swept_fallback="without running any detector" in incomplete,
    )
    return result


def _emits_findings(name: str) -> bool:
    from . import detectors
    det = detectors.REGISTRY.get(name)
    return bool(det and det.covers)


def _default_sweep(tb: ToolBox, experimental: bool,
                   groups: list[str] | None = None) -> list[str]:
    from . import detectors
    return [d.name for d in
            detectors.available(include_experimental=experimental,
                                groups=groups) if d.covers]


def _resolve_retyped(tb: ToolBox, report: dict, traj: Trajectory) -> list[Decision]:
    """Ablation path: build Decisions from what the model typed.

    Nothing here consults the detector's own file lists, so whatever the model
    failed to transcribe is simply gone - which is exactly the quantity the
    ablation is measuring.
    """
    out: list[Decision] = []
    for i, raw in enumerate(report.get("retyped_findings") or []):
        if not isinstance(raw, dict):
            continue
        dtype = str(raw.get("type", "")).strip()
        files = raw.get("files") or []
        if isinstance(files, str):
            files = [files]
        sev = raw.get("severity")
        if sev not in SEVERITY_ORDER:
            sev = DEFECT_TYPES.get(dtype, (MAJOR_DEFAULT, ""))[0]
        f = Finding(type=dtype, title=str(raw.get("rationale", ""))[:120],
                    detail=str(raw.get("rationale", "")), detector="retyped",
                    items=[str(k).strip() for k in files], severity=sev)
        out.append(Decision(finding=f, finding_id=f"retyped:{i}",
                            action="report", severity=sev,
                            rationale=str(raw.get("rationale", "")),
                            source="retyped"))
    return out


def _resolve(tb: ToolBox, report: dict, traj: Trajectory) -> list[Decision]:
    """Turn the model's decision list into one Decision per real finding.

    Unknown ids are dropped with a note; findings the model never mentioned
    default to being reported, so forgetting one cannot lose a defect.
    """
    by_id: dict[str, Decision] = {}
    for raw in report.get("decisions") or []:
        if not isinstance(raw, dict):
            continue
        fid = str(raw.get("finding_id", ""))
        f = tb.findings.get(fid)
        if f is None:
            traj.note(f"report referenced unknown finding_id {fid!r}; ignored")
            continue
        action = raw.get("action", "report")
        if action not in ("report", "suppress"):
            action = "report"
        sev = raw.get("severity")
        if sev not in SEVERITY_ORDER:
            sev = f.severity
        by_id[fid] = Decision(finding=f, finding_id=fid, action=action,
                              severity=sev,
                              rationale=str(raw.get("rationale", "")).strip())

    # A prompt instruction is not an invariant. Findings from exact detectors
    # are statements about bytes on disk, so suppressing one is never correct
    # and the code refuses it outright rather than asking a model to police it.
    for fid, d in by_id.items():
        if d.action == "suppress" and not _is_experimental(d.finding.detector):
            traj.note(f"refused to suppress {fid}: {d.finding.detector} is an "
                      f"exact detector, its findings are not suppressible")
            d.action = "report"
            d.source = "forced"
            d.rationale = ("Kept: this comes from an exact detector, so it is a "
                           "measurement rather than a judgement. "
                           + d.rationale)

    for fid, f in tb.findings.items():
        if fid not in by_id:
            by_id[fid] = Decision(finding=f, finding_id=fid, action="report",
                                  severity=f.severity,
                                  rationale="Reported by default: the agent did "
                                            "not rank this finding.",
                                  source="default")
    return list(by_id.values())


def _is_experimental(detector_name: str) -> bool:
    from . import detectors
    det = detectors.REGISTRY.get(detector_name)
    return bool(det and det.experimental)


def _verify_suppressions(decisions: list[Decision], llm: LLM,
                         traj: Trajectory) -> list[Decision]:
    """Re-argue every surviving suppression from the opposite prior.

    Only experimental findings can reach here at all - `_resolve` refuses to
    suppress anything else - so a verifier that cannot answer is deciding
    about a claim that was unreliable to begin with.
    """
    for d in decisions:
        if d.action != "suppress":
            continue
        f = d.finding
        payload = {
            "type": f.type,
            "detector": f.detector,
            "title": f.title,
            "affected_file_count": f.n_items,
            "evidence": f.evidence[:12],
            "auditor_rationale": d.rationale,
        }
        # Thinking disabled, for the same reason as the baseline: with it on,
        # this call spent its whole budget reasoning and returned empty
        # content on every single suppression, so the unparseable-response
        # fallback fired seven times out of seven and reinstated everything
        # the auditor had correctly argued away.
        msgs = [{"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user",
                 "content": json.dumps(payload, indent=2, default=str)}]
        uphold, reason, parsed = None, "", False
        for extra in (NO_THINKING, None):
            try:
                msg = llm.chat(msgs, traj=traj, max_tokens=VERIFIER_TOKENS,
                               retries=1, extra_body=extra, label="verifier")
            except RuntimeError as exc:
                traj.note(f"verifier call failed for {d.finding_id}: {exc}")
                continue
            uphold, reason = _parse_verdict(msg["content"])
            if "could not be parsed" not in reason:
                parsed = True
                break

        if not parsed:
            # It could not give an answer. The claim under review came from an
            # experimental detector, so deferring to the auditor's reading of
            # the evidence beats reinstating a finding nobody will defend.
            traj.note(f"verifier gave no parseable answer for {d.finding_id}; "
                      f"keeping the auditor's decision")
            d.rationale += " [verifier could not be reached for a second opinion]"
            continue

        if uphold:
            d.rationale = (d.rationale + f" [verifier upheld: {reason}]").strip()
        else:
            d.action = "report"
            d.source = "reinstated"
            d.rationale = (f"Reinstated by the verifier: {reason} "
                           f"(auditor had argued: {d.rationale})")
            traj.note(f"verifier reinstated {d.finding_id}: {reason}")
    return decisions


def _parse_verdict(text: str) -> tuple[bool, str]:
    """Be forgiving about fences and stray prose around the JSON."""
    t = (text or "").strip()
    if "```" in t:
        parts = t.split("```")
        t = max(parts, key=len).removeprefix("json").strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(t[start:end + 1])
            return bool(obj.get("uphold", False)), str(obj.get("reason", ""))[:200]
        except json.JSONDecodeError:
            pass
    # Unparseable means we did not get a defence, so the suppression falls.
    return False, "verifier response could not be parsed"


def _rank(decisions: list[Decision]) -> list[Decision]:
    order = {d.finding_id: i for i, d in enumerate(decisions)}
    return sorted(decisions,
                  key=lambda d: (d.action == "suppress",
                                 SEVERITY_ORDER.get(d.severity, 9),
                                 order[d.finding_id]))


def _trim_result(result: dict, limit: int = 4000) -> dict:
    s = compact_json(result)
    if len(s) <= limit:
        return result
    return {"_truncated_in_trajectory": True,
            "preview": s[:limit] + f"... [{len(s) - limit} more chars]"}
