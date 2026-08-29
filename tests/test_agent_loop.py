"""The agent loop's recovery paths, driven by a scripted model.

These exist because the loop's failure modes are invisible in the results: a
run that spends seventeen turns going nowhere still finishes with the right
findings, so nothing but wall-clock time says anything is wrong. Each test
here corresponds to a failure that actually happened.
"""

from __future__ import annotations

import json

import pytest

from dsdoctor.agent import audit, MAX_STUCK_TURNS
from dsdoctor.dataset import Dataset


class ScriptedLLM:
    """Returns a fixed sequence of assistant messages and records the calls."""

    model = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []          # (tool_choice, thinking_disabled)
        self.labels = []         # which agent each turn belonged to

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=2048,
             traj=None, retries=3, extra_body=None, label=""):
        thinking_off = bool(extra_body and not extra_body
                            .get("chat_template_kwargs", {})
                            .get("enable_thinking", True))
        self.labels.append(label)
        self.calls.append((tool_choice, thinking_off))
        turn = self.turns.pop(0) if self.turns else _stop("")
        if traj is not None:
            from dsdoctor.llm import Step
            traj.steps.append(Step(kind="llm", name=self.model,
                                   response={"content": turn["content"],
                                             "tool_calls": turn["tool_calls"],
                                             "finish_reason": turn["finish_reason"]}))
        return turn


def _stop(content, tool_calls=None):
    return {"role": "assistant", "content": content,
            "tool_calls": tool_calls or [], "finish_reason": "stop",
            "reasoning": ""}


def _truncated():
    """A turn that ran out of budget mid-reasoning: no call, finish=length."""
    return {"role": "assistant", "content": "", "tool_calls": [],
            "finish_reason": "length", "reasoning": "thinking and thinking"}


def _call(name, args, cid="c1"):
    return {"role": "assistant", "content": "", "finish_reason": "tool_calls",
            "reasoning": "",
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name,
                                         "arguments": json.dumps(args)}}]}


def _report():
    return _call("submit_report", {
        "verdict": "usable_with_caveats", "headline": "ok",
        "training_impact": "", "decisions": []}, cid="c9")


def test_truncated_turn_retries_with_thinking_disabled(clean_root):
    """finish_reason=length must disable thinking, not just force a tool call.

    Forcing a tool choice alone was the original fix and it bought nothing:
    one case spent 17 turns and 1,017s that way.
    """
    llm = ScriptedLLM([_truncated(), _call("run_detector", {"name": "geometry_scan"}),
                       _report()])
    audit(Dataset(clean_root), llm, verify=False)

    assert llm.calls[0] == ("auto", False)
    assert llm.calls[1] == ("required", True), \
        "a truncated turn must retry with thinking disabled and a forced call"


def test_empty_stop_retries_with_forced_tool_choice_only(clean_root):
    """finish_reason=stop with no content is the other failure, and thinking
    is not the problem there."""
    llm = ScriptedLLM([_stop(""), _call("run_detector", {"name": "geometry_scan"}),
                       _report()])
    audit(Dataset(clean_root), llm, verify=False)

    assert llm.calls[1] == ("required", False), \
        "an empty stop should force a tool call without disabling thinking"


def test_loop_gives_up_instead_of_spinning(clean_root):
    """Consecutive dead turns must abandon the loop, not run to max_steps."""
    llm = ScriptedLLM([_truncated()] * 20)
    res = audit(Dataset(clean_root), llm, verify=False, max_steps=24)

    assert len(llm.calls) == MAX_STUCK_TURNS, \
        f"spun for {len(llm.calls)} turns instead of stopping at {MAX_STUCK_TURNS}"
    assert "stalled" in res.incomplete


def test_a_stalled_run_still_returns_the_detector_findings(clean_root):
    """Giving up must not mean handing back an empty audit."""
    lbl = clean_root / "labels" / "train" / "t000.txt"
    lbl.write_text("0 0.9 0.5 0.40 0.20\n")          # out_of_bounds

    llm = ScriptedLLM([_truncated()] * 20)
    res = audit(Dataset(clean_root), llm, verify=False)

    assert res.incomplete
    assert any(d.finding.type == "out_of_bounds" for d in res.reported), \
        "a stalled audit must still report what the detectors found"


def test_a_counter_resets_after_a_good_turn(clean_root):
    """Two dead turns, a good one, then two more must not trip the limit."""
    llm = ScriptedLLM([
        _truncated(), _truncated(),
        _call("run_detector", {"name": "geometry_scan"}),
        _truncated(), _truncated(),
        _report(),
    ])
    res = audit(Dataset(clean_root), llm, verify=False)
    assert not res.incomplete, f"gave up early: {res.incomplete}"


def test_report_without_any_detector_falls_back_to_a_full_sweep(clean_root):
    """The worst possible failure is concluding 'looks fine' having looked at
    nothing."""
    lbl = clean_root / "labels" / "train" / "t000.txt"
    lbl.write_text("0 0.5 0.5 0.0 0.2\n")            # degenerate_box

    llm = ScriptedLLM([_report()])
    res = audit(Dataset(clean_root), llm, verify=False)

    assert res.swept_fallback
    assert any(d.finding.type == "degenerate_box" for d in res.reported)


# --------------------------------------------------------------- suppression

from dsdoctor.agent import Decision, _verify_suppressions, _resolve   # noqa: E402
from dsdoctor.findings import Finding                                 # noqa: E402
from dsdoctor.llm import Trajectory                                   # noqa: E402
from dsdoctor.tools import ToolBox                                    # noqa: E402


def _finding(detector, dtype="class_swap"):
    return Finding(type=dtype, title="t", detail="d", detector=detector,
                   items=["train/a"], evidence=["e"])


def _decisions(*pairs):
    return [Decision(finding=f, finding_id=f"id{i}", action=act,
                     severity=f.severity, rationale="because")
            for i, (f, act) in enumerate(pairs)]


class VerifierLLM:
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.labels = []

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=2048,
             traj=None, retries=3, extra_body=None, label=""):
        thinking_off = bool(extra_body and not extra_body
                            .get("chat_template_kwargs", {})
                            .get("enable_thinking", True))
        self.labels.append(label)
        self.calls.append(thinking_off)
        content = self.replies.pop(0) if self.replies else ""
        return {"role": "assistant", "content": content, "tool_calls": [],
                "finish_reason": "stop", "reasoning": ""}


def test_exact_detector_findings_cannot_be_suppressed(clean_root):
    """A prompt instruction is not an invariant; this one is enforced in code."""
    tb = ToolBox(Dataset(clean_root))
    f = _finding("geometry_scan", "out_of_bounds")
    tb.findings["g0"] = f
    report = {"decisions": [{"finding_id": "g0", "action": "suppress",
                             "rationale": "looks fine to me"}]}
    out = _resolve(tb, report, Trajectory(agent="t", model="m"))
    d = next(x for x in out if x.finding_id == "g0")
    assert d.action == "report"
    assert d.source == "forced"


def test_experimental_findings_may_be_suppressed(clean_root):
    tb = ToolBox(Dataset(clean_root))
    tb.findings["m0"] = _finding("model_disagreement_scan")
    report = {"decisions": [{"finding_id": "m0", "action": "suppress",
                             "rationale": "two-directional, so ambiguity"}]}
    out = _resolve(tb, report, Trajectory(agent="t", model="m"))
    assert next(x for x in out if x.finding_id == "m0").action == "suppress"


def test_verifier_tries_thinking_disabled_first():
    """Regression: with thinking on it burned its budget and returned nothing,
    so the fallback reinstated seven correct suppressions out of seven."""
    llm = VerifierLLM(['{"uphold": true, "reason": "scattered both ways"}'])
    decs = _decisions((_finding("model_disagreement_scan"), "suppress"))
    _verify_suppressions(decs, llm, Trajectory(agent="t", model="m"))
    assert llm.calls[0] is True, "first verifier attempt must disable thinking"
    assert decs[0].action == "suppress"


def test_verifier_overturns_a_bad_suppression():
    llm = VerifierLLM(['{"uphold": false, "reason": "one-directional and total"}'])
    decs = _decisions((_finding("model_disagreement_scan"), "suppress"))
    _verify_suppressions(decs, llm, Trajectory(agent="t", model="m"))
    assert decs[0].action == "report"
    assert decs[0].source == "reinstated"


def test_unparseable_verifier_keeps_the_auditors_decision():
    """The old fallback reinstated on a parse failure, which turned a broken
    verifier into 55 false positives. An unreachable second opinion is not
    evidence against the first."""
    llm = VerifierLLM(["", ""])          # both attempts unusable
    decs = _decisions((_finding("model_disagreement_scan"), "suppress"))
    _verify_suppressions(decs, llm, Trajectory(agent="t", model="m"))
    assert decs[0].action == "suppress"
    assert len(llm.calls) == 2, "should fall back to the default call once"


def test_trajectory_labels_each_agent(clean_root):
    """Deliverable: a reader must be able to tell the auditor's turns from the
    verifier's, since both share one trajectory file."""
    llm = ScriptedLLM([_call("run_detector", {"name": "geometry_scan"}), _report()])
    audit(Dataset(clean_root), llm, verify=False)
    assert set(llm.labels) == {"auditor"}


def test_verifier_turns_are_labelled_verifier():
    llm = VerifierLLM(['{"uphold": true, "reason": "ok"}'])
    decs = _decisions((_finding("model_disagreement_scan"), "suppress"))
    _verify_suppressions(decs, llm, Trajectory(agent="t", model="m"))
    assert llm.labels == ["verifier"]
