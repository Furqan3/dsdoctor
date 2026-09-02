"""A report you can look at.

Every judgement this tool makes is about label text and detector output, and
every artefact it produces is text as well. That is a real limitation for the
one question a person can answer instantly and a detector cannot: *is this
finding actually a problem?* A box flagged as tiny might be a distant traffic
light, correctly annotated, or a stray click. Nothing in a Markdown report
distinguishes them; two seconds of looking does.

So this renders the findings with the evidence drawn on the pixels - the
affected image, cropped to context, with the offending boxes outlined - into a
single self-contained HTML file with the images inlined as data URIs. No
server, no asset directory, no network: one file that can be attached to a
ticket or sent to whoever produced the dataset.

The embedded size is capped. A report that is a 400MB file is one nobody
opens, which is the same as no report.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .dataset import Dataset
from .findings import CRITICAL, MAJOR, GOVERNANCE

THUMBS_PER_FINDING = 8
THUMB_WIDTH = 320
JPEG_QUALITY = 72
MAX_EMBEDDED_BYTES = 12_000_000

SEVERITY_COLOR = {CRITICAL: "#b4232a", MAJOR: "#a4670e", "minor": "#4a5568"}


class _Budget:
    def __init__(self, limit: int):
        self.limit, self.used = limit, 0

    def take(self, n: int) -> bool:
        if self.used + n > self.limit:
            return False
        self.used += n
        return True


# Red is a claim: "this is the box the finding is about". Grey is context.
IMPLICATED = (220, 38, 38)
CONTEXT = (120, 120, 130)


def _thumbnail(ds: Dataset, key: str, budget: _Budget,
               lines: list[int] | None = None) -> str | None:
    """A data-URI thumbnail of one sample, with the implicated box outlined.

    `lines` are the label rows the finding is actually about. The first
    version of this outlined *every* box in the image in red, which reads as
    "all of these are wrong" for a finding about one box among twenty - the
    picture asserted something the finding did not. Where the rows are known
    only those are marked; everything else is drawn in grey as context. Where
    they are not known, nothing is drawn in red at all, because implying a
    precision the finding does not have is the same mistake in reverse.
    """
    from PIL import Image, ImageDraw

    sample = ds.get(key)
    if sample is None or sample.image_path is None:
        return None
    wanted = set(lines or ())
    try:
        with Image.open(sample.image_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            draw = ImageDraw.Draw(im)
            for b in (sample.label.boxes if sample.label else []):
                implicated = b.line_no in wanted
                x1, y1, x2, y2 = b.xyxy
                # Clamp for *drawing only*: an out-of-bounds box still has to
                # be visible somewhere on the canvas, and the finding text is
                # what states that it left the frame.
                box = [max(0.0, min(1.0, x1)) * W, max(0.0, min(1.0, y1)) * H,
                       max(0.0, min(1.0, x2)) * W, max(0.0, min(1.0, y2)) * H]
                if box[2] - box[0] < 1:
                    box[2] = box[0] + 1
                if box[3] - box[1] < 1:
                    box[3] = box[1] + 1
                draw.rectangle(box,
                               outline=IMPLICATED if implicated else CONTEXT,
                               width=max(2, W // 200) if implicated
                               else max(1, W // 400))
            im.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=JPEG_QUALITY)
    except Exception:
        return None
    data = buf.getvalue()
    if not budget.take(len(data)):
        return None
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


CSS = """
:root { color-scheme: light dark;
  --bg:#fbfbfa; --fg:#1a1a19; --muted:#6b6b68; --line:#e3e3e0; --card:#fff; }
@media (prefers-color-scheme: dark) { :root {
  --bg:#141414; --fg:#ececeb; --muted:#9a9a96; --line:#2c2c2b; --card:#1c1c1b; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.6 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif; }
.wrap { max-width:940px; margin:0 auto; padding:40px 24px 80px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }
h2 { font-size:18px; margin:44px 0 14px; padding-bottom:8px;
  border-bottom:1px solid var(--line); letter-spacing:-.01em; }
.verdict { font-size:17px; margin:16px 0 22px; padding:14px 16px;
  border-left:3px solid var(--accent,#b4232a); background:var(--card);
  border-radius:0 6px 6px 0; }
.meta { color:var(--muted); font-size:13px; }
.counts { display:flex; gap:10px; flex-wrap:wrap; margin:16px 0 0; }
.pill { padding:4px 11px; border-radius:999px; font-size:13px; font-weight:600;
  border:1px solid var(--line); background:var(--card); }
.finding { background:var(--card); border:1px solid var(--line);
  border-radius:8px; padding:18px 20px; margin:14px 0; }
.tag { display:inline-block; font-size:11px; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; padding:2px 8px; border-radius:4px; color:#fff; }
.title { font-size:16px; font-weight:650; margin:10px 0 6px; }
.detail { color:var(--fg); margin:0 0 12px; }
.thumbs { display:grid; gap:10px; margin:14px 0 6px;
  grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); }
.thumb img { width:100%; border-radius:5px; display:block; border:1px solid var(--line); }
.thumb figcaption { font-size:11px; color:var(--muted); margin-top:4px;
  overflow-wrap:anywhere; font-family:ui-monospace,monospace; }
pre { background:var(--bg); border:1px solid var(--line); border-radius:6px;
  padding:12px; overflow-x:auto; font-size:12.5px; margin:10px 0 0; }
details summary { cursor:pointer; color:var(--muted); font-size:13px; }
.empty { color:var(--muted); font-style:italic; }
"""


def render(ds: Dataset, findings: list, *, verdict: str = "",
           headline: str = "", detectors_run: list[str] | None = None,
           model: str = "", elapsed: float = 0.0,
           with_images: bool = True) -> str:
    s = ds.summary()
    budget = _Budget(MAX_EMBEDDED_BYTES)
    e = html.escape

    train = [f for f in findings if f.category != GOVERNANCE]
    gov = [f for f in findings if f.category == GOVERNANCE]
    n = {sev: sum(1 for f in train if f.severity == sev)
         for sev in (CRITICAL, MAJOR, "minor")}

    L = ["<!doctype html><html><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         f"<title>dsdoctor · {e(Path(s['root']).name)}</title>",
         f"<style>{CSS}</style></head><body><div class='wrap'>"]

    L.append(f"<h1>Trainability audit: <code>{e(Path(s['root']).name)}</code></h1>")
    files = sum(v["images"] for v in s["splits"].values())
    L.append(f"<div class='meta'>{files:,} images · {s['total_boxes']:,} boxes · "
             f"{s['nc']} classes · "
             + " · ".join(f"{e(k)} {v['images']:,}" for k, v in s["splits"].items())
             + "</div>")
    if verdict:
        color = SEVERITY_COLOR.get(CRITICAL if verdict == "blocked" else MAJOR)
        L.append(f"<div class='verdict' style='--accent:{color}'>"
                 f"<strong>{e(verdict.replace('_', ' '))}</strong>"
                 + (f"<br>{e(headline)}" if headline else "") + "</div>")
    L.append("<div class='counts'>"
             + "".join(f"<span class='pill' style='color:{SEVERITY_COLOR[k]}'>"
                       f"{v} {k}</span>" for k, v in n.items())
             + (f"<span class='pill'>{len(gov)} governance</span>" if gov else "")
             + "</div>")

    def block(f) -> None:
        color = SEVERITY_COLOR.get(f.severity, "#4a5568")
        L.append("<div class='finding'>")
        L.append(f"<span class='tag' style='background:{color}'>"
                 f"{e(f.severity)}</span> <span class='meta'>{e(f.type)} · "
                 f"{e(f.detector)}</span>")
        L.append(f"<div class='title'>{e(f.title)}</div>")
        L.append(f"<p class='detail'>{e(f.detail)}</p>")
        if with_images and f.items:
            thumbs = []
            locs = f.locations or {}
            for key in f.items[:THUMBS_PER_FINDING]:
                uri = _thumbnail(ds, key, budget, locs.get(key))
                if uri:
                    thumbs.append(f"<figure class='thumb'><img src='{uri}' "
                                  f"alt='{e(key)}'><figcaption>{e(key)}"
                                  f"</figcaption></figure>")
            if thumbs:
                L.append("<div class='thumbs'>" + "".join(thumbs) + "</div>")
                note = ("red outlines the annotation this finding is about; "
                        "grey is the rest of the image"
                        if f.locations else
                        "this finding is about the file rather than one "
                        "annotation; all boxes shown for context")
                L.append(f"<div class='meta'>{note}</div>")
                if f.n_items > THUMBS_PER_FINDING:
                    L.append(f"<div class='meta'>showing "
                             f"{len(thumbs)} of {f.n_items} affected file(s)</div>")
        if f.evidence:
            L.append("<details><summary>evidence</summary><pre>"
                     + e("\n".join(str(x) for x in f.evidence[:12])) + "</pre></details>")
        if f.fix:
            L.append(f"<div class='meta'>suggested fix: <code>"
                     f"{e(str(f.fix.get('action')))}</code></div>")
        L.append("</div>")

    L.append("<h2>What to fix, in order</h2>")
    if not train:
        L.append("<p class='empty'>Nothing. Every check that was run passed.</p>")
    for f in train:
        block(f)

    if gov:
        L.append("<h2>Governance and privacy</h2>")
        L.append("<p class='meta'>These do not affect whether the dataset "
                 "trains. They affect whether it may lawfully be trained on "
                 "or published.</p>")
        for f in gov:
            block(f)

    L.append("<h2>How this was produced</h2>")
    L.append("<p class='meta'>Every finding above comes from a deterministic "
             "check that read the files directly. Boxes are drawn as stored, "
             "clamped to the frame for display only; red marks the annotation "
             "the finding is about, where the check knows which one it is."
             + (f" Detectors run: {e(', '.join(detectors_run))}." if detectors_run else "")
             + (f" Model: <code>{e(model)}</code>." if model else "")
             + (f" Wall time: {elapsed:.0f}s." if elapsed else "")
             + f" dsdoctor {__version__}, "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.</p>")
    if budget.used >= budget.limit:
        L.append("<p class='meta'>Image budget reached; later findings are "
                 "listed without thumbnails.</p>")
    L.append("</div></body></html>")
    return "\n".join(L)
