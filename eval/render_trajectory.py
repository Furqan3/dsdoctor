"""Render a saved trajectory as a readable walkthrough.

The JSON trajectories are complete but not pleasant to read, and a reviewer
should be able to follow an agent from its instructions to its final answer
without a JSON viewer.

    python eval/render_trajectory.py runs/full-v1/trajectories/everything.agent.json
    python eval/render_trajectory.py runs/full-v1/trajectories/ --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MAX_RESULT = 900
MAX_REASONING = 700


def clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n… [{len(text) - limit} more chars]"


def render(path: Path) -> str:
    d = json.loads(path.read_text())
    L: list[str] = []
    L.append(f"# Trajectory — `{path.stem}`")
    L.append("")
    L.append(f"- agent: **{d['agent']}**")
    L.append(f"- model: `{d['model']}`")
    L.append(f"- {d['llm_calls']} model call(s), {d['tool_calls']} tool call(s)")
    L.append(f"- {d['prompt_tokens']:,} prompt tokens, "
             f"{d['completion_tokens']:,} completion tokens")
    L.append(f"- wall time: {d['wall_seconds']}s")
    agents = sorted({st["name"].split("·")[-1].strip()
                     for st in d["steps"]
                     if st["kind"] == "llm" and "·" in (st.get("name") or "")})
    if agents:
        L.append(f"- agents appearing in this trajectory: "
                 + ", ".join(f"**{a}**" for a in agents))
    L.append("")

    # The instructions the agent was given, taken from the first request.
    for step in d["steps"]:
        if step["kind"] == "llm" and step.get("request"):
            msgs = step["request"].get("messages") or []
            sysm = next((m for m in msgs if m.get("role") == "system"), None)
            usr = next((m for m in msgs if m.get("role") == "user"), None)
            if sysm:
                L.append("## Agent instructions")
                L.append("")
                L.append("```text")
                L.append(clip(sysm.get("content", ""), 4000))
                L.append("```")
                L.append("")
            if usr:
                L.append("## Task")
                L.append("")
                L.append("> " + clip(usr.get("content", ""), 600).replace("\n", "\n> "))
                L.append("")
            break

    L.append("## Steps")
    L.append("")
    n = 0
    for step in d["steps"]:
        kind = step["kind"]
        if kind == "note":
            L.append(f"> ⚠️ **{step['response'].get('text', '')}**")
            L.append("")
            continue

        n += 1
        if kind == "llm":
            who = step.get("name", "")
            agent_tag = who.split("·")[-1].strip() if "·" in who else "model"
            L.append(f"### {n}. {agent_tag} turn  ·  {step['seconds']}s  ·  "
                     f"{step['completion_tokens']} tokens out")
            L.append("")
            if step.get("reasoning"):
                L.append("<details><summary>reasoning</summary>")
                L.append("")
                L.append("```text")
                L.append(clip(step["reasoning"], MAX_REASONING))
                L.append("```")
                L.append("")
                L.append("</details>")
                L.append("")
            resp = step.get("response") or {}
            calls = resp.get("tool_calls") or []
            if calls:
                for c in calls:
                    L.append(f"**calls** `{c['function']['name']}("
                             f"{clip(c['function']['arguments'], 200)})`")
                    L.append("")
            elif resp.get("content"):
                L.append(clip(resp["content"], 800))
                L.append("")

        elif kind == "tool":
            L.append(f"### {n}. tool result  ·  `{step['name']}`")
            L.append("")
            if step.get("request"):
                L.append(f"arguments: `{json.dumps(step['request'])[:200]}`")
                L.append("")
            L.append("```json")
            L.append(clip(json.dumps(step.get("response"), indent=2,
                                     default=str), MAX_RESULT))
            L.append("```")
            L.append("")

    # Human checkpoint is a property of the pipeline, not of any one step, but
    # a reader of the trajectory needs to know where it sits.
    L.append("## Human checkpoint")
    L.append("")
    L.append("This trajectory ends at a written report and a fix plan. Nothing "
             "in the dataset has been modified. Applying the plan requires "
             "`dsdoctor apply`, which refuses to proceed without an explicit "
             "confirmation typed by a person, and which never applies steps "
             "marked `requires_human_review`.")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--all", action="store_true",
                    help="treat path as a directory and render every .json in it")
    ap.add_argument("--out", default="", help="write .md files next to the source")
    args = ap.parse_args()

    p = Path(args.path)
    targets = sorted(p.glob("*.json")) if args.all else [p]
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for t in targets:
        md = render(t)
        if out_dir or args.all:
            dst = (out_dir or t.parent) / f"{t.stem}.md"
            dst.write_text(md)
            print(f"wrote {dst}")
        else:
            print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
