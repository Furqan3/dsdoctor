"""Make the verifier actually run, on purpose, every time the suite runs.

This exists because of the failure recorded in the README's "hot take": the
verifier never fires in the shipped configuration, so for most of the
project's life it looked identical to a verifier that worked - and when it was
finally forced to run it turned seven correct suppressions into fifty-five
false positives, because a reasoning model with a 400-token budget spent all
of it thinking and returned an empty string.

The lesson written there was "make it fire on purpose and watch what it does".
That was done by hand, once. These tests do it on every run, with a scripted
model, offline, in milliseconds - which is the difference between a lesson and
a guarantee.

Each test drives the one path that the real configuration cannot reach:
suppression is only permitted for findings from an experimental detector, so
the fixture registers one.
"""

from __future__ import annotations

import json

import pytest

from dsdoctor import detectors
from dsdoctor.agent import audit
from dsdoctor.dataset import Dataset
from dsdoctor.findings import Finding, MAJOR


@pytest.fixture
def unreliable_detector(monkeypatch):
    """An experimental detector that always fires: the only suppressible kind."""
    def scan(ds):
        return [Finding(type="class_swap", severity=MAJOR,
                        title="3 boxes may carry a swapped class",
                        detail="a hypothesis, not a fact",
                        detector="flaky_scan",
                        items=["train/t000", "train/t001", "train/t002"],
                        evidence=["car/truck confusion on 3 boxes"])]

    registry = dict(detectors.REGISTRY)
    registry["flaky_scan"] = detectors.Detector(
        name="flaky_scan", description="known to produce false positives",
        fn=scan, experimental=True, covers=("class_swap",))
    monkeypatch.setattr(detectors, "REGISTRY", registry)
    return registry


class VerifierLLM:
    """Auditor suppresses everything; the verifier answers however we say."""

    model = "scripted"

    def __init__(self, verifier_reply, finish_reason="stop"):
        self.verifier_reply = verifier_reply
        self.finish_reason = finish_reason
        self.verifier_calls = 0
        self.verifier_budgets = []
        self.thinking_flags = []
        self._submitted = False

    def chat(self, messages, tools=None, tool_choice="auto", max_tokens=2048,
             traj=None, retries=3, extra_body=None, label=""):
        if label == "verifier":
            self.verifier_calls += 1
            self.verifier_budgets.append(max_tokens)
            self.thinking_flags.append(
                bool(extra_body and not extra_body
                     .get("chat_template_kwargs", {})
                     .get("enable_thinking", True)))
            return {"role": "assistant", "content": self.verifier_reply,
                    "tool_calls": [], "finish_reason": self.finish_reason,
                    "reasoning": ""}

        if not self._submitted:
            self._submitted = True
            args = {"verdict": "usable_with_caveats",
                    "headline": "looks fine to me",
                    "training_impact": "",
                    "decisions": [{"finding_id": "flaky_scan:class_swap:0",
                                   "action": "suppress",
                                   "rationale": "two-directional car/truck "
                                                "confusion reads as annotation "
                                                "ambiguity, not a mapping bug"}]}
            return {"role": "assistant", "content": "",
                    "finish_reason": "tool_calls", "reasoning": "",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "submit_report",
                                                 "arguments": json.dumps(args)}}]}
        return {"role": "assistant", "content": "", "tool_calls": [],
                "finish_reason": "stop", "reasoning": ""}


def _audit(llm, root, **kw):
    return audit(Dataset(root), llm, experimental=True, verify=True,
                 max_steps=6, **kw)


def test_the_verifier_actually_runs(clean_root, unreliable_detector):
    """The guard must fire. If this fails, nothing below means anything."""
    llm = VerifierLLM('{"uphold": true, "reason": "agreed, ambiguity"}')
    res = _audit(llm, clean_root)
    assert llm.verifier_calls == 1, "the verifier never ran"


def test_upheld_suppression_stays_suppressed(clean_root, unreliable_detector):
    llm = VerifierLLM('{"uphold": true, "reason": "agreed, annotation ambiguity"}')
    res = _audit(llm, clean_root)
    assert res.reinstated_total == 0
    assert len(res.suppressed) == 1
    assert "verifier upheld" in res.suppressed[0].rationale


def test_overruled_suppression_is_reinstated(clean_root, unreliable_detector):
    llm = VerifierLLM('{"uphold": false, "reason": "the evidence shows a swap"}')
    res = _audit(llm, clean_root)
    assert res.reinstated_total == 1
    assert not res.suppressed
    assert any(d.source == "reinstated" for d in res.reported)


def test_a_silent_verifier_does_not_reinstate(clean_root, unreliable_detector):
    """The exact regression from the README's hot take.

    An empty answer is not dissent. Treating it as dissent turned seven
    correct judgements into fifty-five false positives, and it is the reason
    the guard was worse than not having one.
    """
    llm = VerifierLLM("", finish_reason="length")
    res = _audit(llm, clean_root)
    assert llm.verifier_calls >= 1
    assert res.reinstated_total == 0, "silence was treated as dissent again"
    assert len(res.suppressed) == 1
    assert "could not be reached" in res.suppressed[0].rationale


def test_unparseable_verifier_answer_does_not_reinstate(clean_root,
                                                        unreliable_detector):
    llm = VerifierLLM("I think, on balance, that this is complicated.")
    res = _audit(llm, clean_root)
    assert res.reinstated_total == 0
    assert len(res.suppressed) == 1


def test_verifier_answer_in_a_code_fence_is_understood(clean_root,
                                                       unreliable_detector):
    llm = VerifierLLM('```json\n{"uphold": false, "reason": "real swap"}\n```')
    res = _audit(llm, clean_root)
    assert res.reinstated_total == 1


def test_verifier_gets_room_to_answer_and_thinking_is_disabled(
        clean_root, unreliable_detector):
    """Both halves of the original bug, pinned as configuration."""
    from dsdoctor.agent import VERIFIER_TOKENS
    llm = VerifierLLM('{"uphold": true, "reason": "fine"}')
    _audit(llm, clean_root)
    assert VERIFIER_TOKENS >= 512
    assert llm.verifier_budgets[0] >= 512
    assert llm.thinking_flags[0] is True, (
        "the first verifier attempt must disable thinking; with it on, the "
        "call spends its budget reasoning and returns nothing")


def test_exact_findings_are_never_suppressible(clean_root, unreliable_detector):
    """The gate in front of the verifier, which is what makes it unnecessary.

    An agent asking to suppress a finding from an exact detector is refused in
    code. This is why the verifier never fires in the shipped configuration -
    and it must keep being why.
    """
    class SuppressExactLLM(VerifierLLM):
        def chat(self, messages, tools=None, tool_choice="auto",
                 max_tokens=2048, traj=None, retries=3, extra_body=None,
                 label=""):
            if label != "verifier" and not self._submitted:
                self._submitted = True
                args = {"verdict": "usable_with_caveats", "headline": "",
                        "training_impact": "",
                        "decisions": [{"finding_id": "structure_scan:"
                                                     "empty_label_file:0",
                                       "action": "suppress",
                                       "rationale": "I do not like it"}]}
                return {"role": "assistant", "content": "",
                        "finish_reason": "tool_calls", "reasoning": "",
                        "tool_calls": [{"id": "c1", "type": "function",
                                        "function": {
                                            "name": "submit_report",
                                            "arguments": json.dumps(args)}}]}
            return super().chat(messages, tools, tool_choice, max_tokens,
                                traj, retries, extra_body, label)

    # give the dataset a real empty-label finding to try to suppress
    (clean_root / "labels" / "train" / "t000.txt").write_text("")
    llm = SuppressExactLLM('{"uphold": true, "reason": "x"}')
    res = _audit(llm, clean_root)
    assert not any(d.finding.detector == "structure_scan"
                   for d in res.suppressed), \
        "an exact finding was suppressed; the gate is open"
