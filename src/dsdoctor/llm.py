"""OpenAI-compatible chat client with trajectory recording.

The project is developed against a locally served Qwen3.8-27B (vLLM, W4A16 on
one RTX 3090), but nothing here is specific to it: any endpoint that speaks the
OpenAI chat-completions API with tool calling works, and the backend is a flag.

Every request and response is written to a trajectory so that each claim in a
report can be traced back to the exact exchange that produced it. That is a
submission requirement, but it is also the only practical way to debug an agent
that is wrong in a plausible-sounding way.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("DSDOCTOR_BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.environ.get("DSDOCTOR_MODEL", "qwen3.8-27b")
DEFAULT_API_KEY = os.environ.get("DSDOCTOR_API_KEY", "not-needed-for-local-vllm")


@dataclass
class Step:
    """One turn of the loop, in enough detail to replay it by hand."""

    kind: str                      # "llm" | "tool" | "note"
    name: str = ""
    request: dict | None = None
    response: dict | None = None
    reasoning: str = ""
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Trajectory:
    agent: str
    model: str
    started: float = field(default_factory=time.time)
    steps: list[Step] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "llm")

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool")

    @property
    def seconds(self) -> float:
        return time.time() - self.started

    def note(self, text: str) -> None:
        self.steps.append(Step(kind="note", name="note",
                               response={"text": text}))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "agent": self.agent,
            "model": self.model,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_seconds": round(self.seconds, 2),
            "steps": [asdict(s) for s in self.steps],
        }, indent=2, default=str))
        return path


class LLM:
    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL,
                 api_key: str = DEFAULT_API_KEY,
                 temperature: float = 0.0,
                 timeout: float = 600.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str = "auto", max_tokens: int = 2048,
             traj: Trajectory | None = None, retries: int = 3,
             extra_body: dict | None = None, label: str = "") -> dict:
        """One chat completion, returned as a plain message dict."""
        kwargs: dict = {"model": self.model, "messages": messages,
                        "temperature": self.temperature, "max_tokens": max_tokens}
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        last_exc = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as exc:                # transient server/timeout
                last_exc = exc
                if traj:
                    traj.note(f"LLM call failed ({type(exc).__name__}: {exc}); "
                              f"retry {attempt + 1}/{retries}")
                time.sleep(1.5 * (attempt + 1))
                continue

            msg = resp.choices[0].message
            extra = getattr(msg, "model_extra", None) or {}
            out = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in (msg.tool_calls or [])],
                "finish_reason": resp.choices[0].finish_reason,
                "reasoning": extra.get("reasoning") or "",
            }
            if traj:
                traj.steps.append(Step(
                    kind="llm",
                    name=f"{self.model} · {label}" if label else self.model,
                    request={"messages": _trim(messages),
                             "tool_choice": tool_choice if tools else None,
                             "n_tools": len(tools or [])},
                    response={"content": out["content"],
                              "tool_calls": out["tool_calls"],
                              "finish_reason": out["finish_reason"]},
                    reasoning=out["reasoning"],
                    seconds=round(time.time() - t0, 2),
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                ))
            return out

        raise RuntimeError(f"LLM call failed after {retries} attempts: {last_exc}")


def _trim(messages: list[dict], limit: int = 1200) -> list[dict]:
    """Keep trajectories readable: tool results can be tens of kilobytes."""
    out = []
    for m in messages:
        c = m.get("content") or ""
        if isinstance(c, str) and len(c) > limit:
            c = c[:limit] + f"\n... [{len(c) - limit} more chars]"
        trimmed = {k: v for k, v in m.items() if k != "content"}
        trimmed["content"] = c
        out.append(trimmed)
    return out
