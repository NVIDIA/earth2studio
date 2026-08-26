# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the scorecard documentation pages from per-model score exports.

A model page is built from two inputs and nothing else:

  docs/_static/scorecard/eval_scores_<model>.json   the numbers, exported by
                                                    the eval recipe's
                                                    ``export_scores.py --docs``
  docs/scorecard/config/<model>.md                  the prose: front matter
                                                    (label, category) plus a
                                                    description and optional
                                                    extra sections (Reference)

For every such pair this writes, under the git-ignored ``generated/`` folder:

  docs/scorecard/generated/<model>/index.md   the doc page, embedding the plot
  docs/scorecard/generated/index.md           the section index with cards
  docs/_static/scorecard/plot.html            ONE shared interactive plot

The Makefile ``docs-generate`` target runs this script before each site build, so
the pages always exist on the fly and are never committed. Adding a model is:
export its JSON, add its ``config/<model>.md``, list it in ``mkdocs.yml``.

The plot holds no data: it fetches eval_scores_<model>.json (selected by its
?model= query parameter) and parses it in the browser -- no CDN, and the
numbers exist exactly once, minified for compression. Run from anywhere::

    python docs/generate_scorecard.py
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import yaml

DOCS = Path(__file__).resolve().parent  # docs/
HERE = DOCS / "scorecard"
CONFIG = HERE / "config"  # per-model prose: front matter + description
GENERATED = HERE / "generated"  # output pages, git-ignored


def _load_defaults() -> dict:
    """Site-wide defaults (fallback labels, metric direction) from
    ``config/default.md`` front matter."""
    path = CONFIG / "default.md"
    if path.exists():
        text = path.read_text()
        if text.startswith("---"):
            return yaml.safe_load(text.split("---", 2)[1]) or {}
    return {}


DEFAULTS = _load_defaults()
LABELS = DEFAULTS.get("labels", {})
LOWER_IS_BETTER = DEFAULTS.get("metrics", {}).get("lower_is_better", [])
DATA_SOURCES = DEFAULTS.get("data_sources", {})
STATIC = DOCS / "_static" / "scorecard"  # data + shared plot live here

# Self-contained single-model skill plot. Same palette and idioms as the
# recipe's full scorecard so the docs and the recipe read as one system.
PLOT_HTML = r"""<!doctype html>
<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model skill</title>
<style>
  :root{color-scheme:light dark}
  body{
    --surface-1:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e;
    --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
    --s1:#2a78d6;
    margin:0;background:var(--plane);color:var(--ink);
    font:13px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
  @media (prefers-color-scheme:dark){body{
    --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink-2:#c3c2b7;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10); --s1:#3987e5}}
  .wrap{position:relative;max-width:980px;margin:0 auto;padding:12px 14px 16px}
  .badge{position:absolute;top:14px;right:14px;z-index:2;
    background:var(--s1);color:#fff;font-weight:600;font-size:12px;
    border-radius:20px;padding:4px 13px;letter-spacing:.02em;
    box-shadow:0 3px 10px rgba(0,0,0,.18)}
  .bar{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-end;
    background:var(--surface-1);border:1px solid var(--border);
    border-radius:10px;padding:10px 12px;margin-bottom:12px}
  .ctl{display:flex;flex-direction:column;gap:4px}
  .ctl label{font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
  select{font:inherit;color:var(--ink);background:var(--surface-1);
    border:1px solid var(--axis);border-radius:7px;padding:6px 9px}
  .card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;
    padding:12px 14px 8px}
  .card h2{font-size:13.5px;margin:0 0 2px} .card p{margin:0 0 8px;color:var(--ink-2);font-size:12px}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .gl{stroke:var(--grid);stroke-width:.8} .ax{stroke:var(--axis);stroke-width:1}
  .tk{fill:var(--ink-2);font-size:10.5px} .al{fill:var(--muted);font-size:10.5px}
  .tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .08s;
    background:var(--surface-1);border:1px solid var(--border);border-radius:7px;
    padding:6px 8px;font-size:11.5px;box-shadow:0 6px 20px rgba(0,0,0,.16);z-index:9}
  .tip b{font-weight:600} .tip .k{color:var(--muted)}
</style></head>
<body><div class="wrap">
<div class="badge"></div>
<div class="bar">
  <div class="ctl"><label>Metric</label><select id="m"></select></div>
  <div class="ctl"><label>Variable</label><select id="v"></select></div>
</div>
<div class="card"><h2 id="t"></h2><p id="s"></p><svg id="c" viewBox="0 0 900 330"></svg></div>
</div>
<div class="tip" id="tip"></div>
<script>
// No data here: the model is picked by ?model=, its JSON fetched below.
const $=s=>document.querySelector(s);
const Q=new URLSearchParams(location.search);
const MODEL=Q.get("model")||"";
const LABEL=Q.get("label")||MODEL;
document.title=LABEL+" skill";
$(".badge") && ($(".badge").textContent=LABEL);
let D=null,days=[];
const LOWER=__LOWER__;
const css=n=>getComputedStyle(document.body).getPropertyValue(n).trim();
const isFin=q=>q!=null&&isFinite(q);
const tip=$("#tip");
const ns="http://www.w3.org/2000/svg";
const mk=(svg,t,a)=>{const e=document.createElementNS(ns,t);
  for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e;};
const fmt=v=>{if(!isFin(v))return "–";const a=Math.abs(v);
  if(a!==0&&(a<1e-3||a>=1e5))return v.toExponential(2);
  return v.toFixed(a>=100?1:a>=10?2:3);};
const mSel=$("#m"),vSel=$("#v");
function fillVars(){
  const vs=Object.keys(D.metrics[mSel.value].values);
  vs.sort((a,b)=>D.variables.indexOf(a)-D.variables.indexOf(b));
  const prev=vSel.value; vSel.innerHTML="";
  let g=null,last=null;
  vs.forEach(v=>{const grp=(D.variable_groups||{})[v]||"Other";
    if(grp!==last){g=document.createElement("optgroup");g.label=grp;vSel.appendChild(g);last=grp;}
    g.appendChild(new Option(v+(D.units[v]?"  ("+D.units[v]+")":""),v));});
  if(vs.includes(prev))vSel.value=prev;
  else if(vs.includes("z500"))vSel.value="z500";
}
function draw(){
  const k=mSel.value,v=vSel.value,y=D.metrics[k].values[v]||[];
  const svg=$("#c");svg.innerHTML="";
  const unit=D.metrics[k].unit||D.units[v]||"";
  $("#t").textContent=`${D.metrics[k].label} — ${v}${unit?" ("+unit+")":""}`;
  $("#s").textContent=
    k==="spread_skill"?"Ensemble spread over ensemble-mean RMSE. 1.0 is calibrated; below 1 is over-confident."
    :LOWER.includes(k)?"Lower is better. Latitude-weighted, averaged over initial conditions."
    :"Higher is better.";
  const all=y.filter(isFin);
  if(!all.length){mk(svg,"text",{x:450,y:155,class:"al","text-anchor":"middle"}).textContent="no data";return;}
  const W=900,H=330,L=64,R=24,T=14,B=44;
  const div=k==="spread_skill";
  let lo,hi;
  if(div){const l=Math.max(...all.map(q=>Math.abs(Math.log2(q||1))))*1.15||.4;
    lo=Math.pow(2,-l);hi=Math.pow(2,l);}
  else if(k==="acc"){lo=Math.min(0,Math.min(...all));hi=1.02;}
  else{lo=0;hi=Math.max(...all)*1.08||1;}
  const px=d=>L+(W-L-R)*(d-days[0])/((days[days.length-1]-days[0])||1);
  const py=q=>T+(H-T-B)*(1-(q-lo)/((hi-lo)||1));
  for(let i=0;i<=4;i++){const val=lo+(hi-lo)*i/4,Y=py(val);
    mk(svg,"line",{x1:L,x2:W-R,y1:Y,y2:Y,class:"gl"});
    mk(svg,"text",{x:L-8,y:Y+4,class:"tk","text-anchor":"end"}).textContent=fmt(val);}
  const maxd=days[days.length-1],step=maxd<=3?0.5:maxd<=8?1:2;
  for(let d=0;d<=maxd+1e-9;d+=step){
    mk(svg,"line",{x1:px(d),x2:px(d),y1:T,y2:H-B,class:"gl"});
    mk(svg,"text",{x:px(d),y:H-B+16,class:"tk","text-anchor":"middle"})
      .textContent=(d%1?d.toFixed(1):d);}
  mk(svg,"line",{x1:L,x2:W-R,y1:H-B,y2:H-B,class:"ax"});
  mk(svg,"line",{x1:L,x2:L,y1:T,y2:H-B,class:"ax"});
  mk(svg,"text",{x:(L+W-R)/2,y:H-6,class:"al","text-anchor":"middle"}).textContent="lead time (days)";
  if(div){const Y=py(1);mk(svg,"line",{x1:L,x2:W-R,y1:Y,y2:Y,stroke:css("--muted"),
    "stroke-width":1,"stroke-dasharray":"4 4"});
    mk(svg,"text",{x:W-R,y:Y-6,class:"al","text-anchor":"end"}).textContent="calibrated (1.0)";}
  let d="",pen=false;
  y.forEach((q,i)=>{if(!isFin(q)){pen=false;return;}
    d+=(pen?"L":"M")+px(days[i])+" "+py(q);pen=true;});
  mk(svg,"path",{d,fill:"none",stroke:css("--s1"),"stroke-width":2,
    "stroke-linejoin":"round","stroke-linecap":"round"});
  const hl=mk(svg,"line",{x1:L,x2:L,y1:T,y2:H-B,stroke:css("--muted"),"stroke-width":1,opacity:0});
  const dot=mk(svg,"circle",{r:4.5,fill:css("--s1"),stroke:css("--surface-1"),"stroke-width":2,opacity:0});
  const hit=mk(svg,"rect",{x:L,y:T,width:W-L-R,height:H-T-B,fill:"transparent"});
  hit.style.cursor="crosshair";
  hit.addEventListener("mousemove",e=>{
    const bb=svg.getBoundingClientRect(),sx=(e.clientX-bb.left)*W/bb.width;
    let bi=0,bd=1e9;days.forEach((dd,i)=>{const q=Math.abs(px(dd)-sx);if(q<bd){bd=q;bi=i;}});
    hl.setAttribute("x1",px(days[bi]));hl.setAttribute("x2",px(days[bi]));hl.setAttribute("opacity",1);
    const q=y[bi];
    if(isFin(q)){dot.setAttribute("cx",px(days[bi]));dot.setAttribute("cy",py(q));
      dot.setAttribute("opacity",1);}
    else dot.setAttribute("opacity",0);
    tip.innerHTML=`<span class="k">lead</span> ${D.lead_hours[bi]} h (${days[bi]} d)<br><b>${fmt(q)}</b> ${D.metrics[k].unit||D.units[v]||""}`;
    tip.style.opacity=1;
    let x=e.clientX+14,yy=e.clientY-10;
    const r=tip.getBoundingClientRect();
    if(x+r.width+8>innerWidth)x=e.clientX-r.width-14;
    tip.style.left=Math.max(8,x)+"px";tip.style.top=Math.max(8,yy)+"px";});
  hit.addEventListener("mouseleave",()=>{tip.style.opacity=0;
    hl.setAttribute("opacity",0);dot.setAttribute("opacity",0);});
}
mSel.addEventListener("change",()=>{fillVars();draw();});
vSel.addEventListener("change",draw);
// The export is plain minified JSON.
fetch(`eval_scores_${MODEL}.json`)
  .then(r=>{if(!r.ok)throw new Error("HTTP "+r.status);return r.text();})
  .then(t=>{
    D=JSON.parse(t);
    days=D.lead_hours.map(h=>h/24);
    Object.keys(D.metrics).forEach(k=>mSel.appendChild(new Option(D.metrics[k].label,k)));
    fillVars();draw();})
  .catch(e=>{$("#t").textContent=`failed to load eval_scores_${MODEL}.json`;
    $("#s").textContent=String(e);});
</script></body></html>
"""

PAGE_MD = """\
<!-- Generated by docs/generate_scorecard.py from _static/scorecard/eval_scores_{model}.json -- do not hand-edit. -->

# {label}
{badges}{description}
## Skill

Pick a metric and variable; hover for exact values at each lead time.

<iframe src="../../../_static/scorecard/plot.html?model={model_q}&label={label_q}" title="{label} skill"
        style="width:100%;height:560px;border:1px solid rgba(128,128,128,.35);border-radius:10px;"
        loading="lazy"></iframe>

## Evaluation

{summary}

Scores are latitude-weighted (cos φ) and aggregated over the initial
conditions. Evaluation is done against ERA5 fetched from ARCO.

| | |
|---|---|
| Type | {kind} |
| Initial conditions | {n_ic} ({years}) |
| Initial condition source | {ic_source} |
| Verification (ground truth) | {verification_source} |
| Lead times | {lead_first} h to {lead_last_d} days |
| Variables scored | {n_var} |
| Metrics | {metric_list} |

## Variables

??? note "Scored output variables ({n_var})"

{variables_table}

All of the model's output variables that have ERA5 verification are scored.

## Data

The numbers behind the plot are in [`eval_scores_{model}.json`](../../_static/scorecard/eval_scores_{model}.json), exported by
the [eval recipe scorecard](https://github.com/NVIDIA/earth2studio/tree/main/recipes/eval)
(`scorecard/export_scores.py --docs`) -- one value per metric, variable
and lead time, in the variable's own units.

## Reproducibility

??? info "Run and environment details"

{provenance_table}
{reference}"""

INDEX_MD = """\
---
title: Scorecard - Index
---

<!-- Generated by docs/generate_scorecard.py -- do not hand-edit. -->

# Scorecards

!!! warning
    The scorecards are in beta. We are actively working on adding more models
    and improved evaluation.

Forecast skill of Earth2Studio models, one scorecard per model. These show
each model's own skill, not a comparison between models. Every model was evaluated on the
same campaign: {n_ic} initial conditions ({years}), 14-day horizon, ERA5
verification via ARCO. Pages are generated from per-model score (JSON) exports
produced by the
[scorecard recipe](https://github.com/NVIDIA/earth2studio/tree/main/recipes/eval/scorecard),
which documents how to generate a scorecard for any model; the
[eval recipe](https://github.com/NVIDIA/earth2studio/tree/main/recipes/eval)
provides the backend support (inference, scoring and metrics).

## Prognostic models

<div class="grid cards" markdown>

{entries}

</div>
"""

# Reproducibility fields, in display order: export key -> row label.
_PROV_ROWS = {
    "date_scored": "Date scored",
    "scores_written": "Scores written",
    "gpus": "GPUs",
    "torch": "PyTorch",
    "cuda": "CUDA",
    "python": "Python",
    "repo_commit": "Repo commit",
    "provenance_source": "Provenance source",
    "exported": "Exported",
}


REPO_URL = "https://github.com/NVIDIA/earth2studio"

# Human-readable variable descriptions, derived from the naming convention
# (prefix + pressure level or height) plus a few exact surface names.
_DESC_EXACT = {
    "msl": "Mean sea level pressure",
    "sp": "Surface pressure",
    "tcwv": "Total column water vapour",
    "t2m": "2-metre temperature",
    "u10m": "10-metre eastward (zonal) wind",
    "v10m": "10-metre northward (meridional) wind",
    "u100m": "100-metre eastward (zonal) wind",
    "v100m": "100-metre northward (meridional) wind",
}
_DESC_PREFIX = {
    "z": "Geopotential",
    "t": "Temperature",
    "u": "Eastward (zonal) wind",
    "v": "Northward (meridional) wind",
    "q": "Specific humidity",
    "r": "Relative humidity",
    "w": "Vertical velocity",
}


# Authoritative descriptions come from the package's own variable vocabulary
# (earth2studio.lexicon.base.E2STUDIO_VOCAB). Parse it out of the source file
# rather than importing it: importing the package pulls torch and friends,
# which a docs-only environment need not have, and the vocab is a pure dict
# literal. The pattern tables above remain the fallback.
def _load_vocab() -> dict:
    """Variable vocabulary parsed from earth2studio's lexicon source file."""
    import ast
    import contextlib

    src = DOCS.parent / "earth2studio" / "lexicon" / "base.py"
    # Any parsing hiccup just falls back to the pattern tables below.
    with contextlib.suppress(Exception):
        for node in ast.parse(src.read_text()).body:
            if (
                isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "E2STUDIO_VOCAB"
            ):
                return ast.literal_eval(node.value)
    return {}


E2STUDIO_VOCAB = _load_vocab()


def describe(var: str) -> str:
    """Human-readable description of a variable name like ``z500`` or ``t2m``."""
    import re

    if var in E2STUDIO_VOCAB:
        # Entries read "geopotential at 500 hPa (m2 s-2)": drop the trailing
        # unit parenthetical (the table has its own Unit column) and
        # capitalise the first letter.
        text = re.sub(r"\s*\([^()]*\)\s*$", "", E2STUDIO_VOCAB[var]).strip()
        return text[:1].upper() + text[1:]
    if var in _DESC_EXACT:
        return _DESC_EXACT[var]
    m = re.fullmatch(r"([a-z]+)(\d+)", var)
    if m and m.group(1) in _DESC_PREFIX:
        return f"{_DESC_PREFIX[m.group(1)]} at {m.group(2)} hPa"
    return ""


def variables_table(doc: dict) -> str:
    """Markdown table of scored variables with descriptions and units."""
    rows = [
        f"| `{v}` | {describe(v)} | {doc['units'].get(v, '')} "
        f"| {doc['variable_groups'].get(v, '')} |"
        for v in doc["variables"]
    ]
    table = "| Name | Description | Unit | Group |\n|---|---|---|---|\n" + "\n".join(
        rows
    )
    return "\n".join("    " + ln for ln in table.splitlines())


def _api_badges(px_class: str) -> str:
    """Badge set from the model's generated API catalog page, so the
    scorecard never disagrees with the catalog. generate_api.py runs before
    this script in the Makefile, so the page exists during a docs build."""
    if not px_class:
        return ""
    page = DOCS / "modules" / "generated" / "models" / "px" / f"{px_class}.md"
    if not page.exists():
        return ""
    text = page.read_text()
    if not text.startswith("---"):
        return ""
    meta = yaml.safe_load(text.split("---", 2)[1]) or {}
    badges = meta.get("badges", [])
    return " ".join(badges) if isinstance(badges, list) else str(badges)


def read_config(model: str) -> dict:
    """Parse ``config/<model>.md`` into label, description and extra sections.

    The file is ordinary markdown with YAML front matter. Everything before
    the first ``## `` heading is the model description; the headings and their
    content are appended verbatim at the bottom of the page (Reference etc.).
    """
    path = CONFIG / f"{model}.md"
    meta: dict = {}
    body = ""
    if path.exists():
        text = path.read_text()
        if text.startswith("---"):
            _, fm, body = text.split("---", 2)
            meta = yaml.safe_load(fm) or {}
        else:
            body = text
    else:
        print(f"!! no config/{model}.md -- page gets defaults; please add one")
    body = body.strip()
    idx = body.find("\n## ")
    description = body if idx < 0 else body[:idx].strip()
    extras = "" if idx < 0 else body[idx:].strip()
    short = meta.get("short") or (
        description.split(". ")[0].rstrip(".") + "." if description else ""
    )
    return {
        "label": meta.get("label", LABELS.get(model, model)),
        "badges": str(
            meta.get("badges", "") or _api_badges(meta.get("px_class", ""))
        ).strip(),
        "description": description,
        "extras": extras,
        "short": short.replace("\n", " "),
    }


def provenance_table(doc: dict) -> str:
    """Markdown table for the collapsible reproducibility section."""
    prov = doc.get("provenance") or {}
    if not prov:
        return (
            "    Not recorded for this run -- provenance capture was added "
            "after it was scored. Re-exporting with the current "
            "`export_scores.py` records it."
        )
    # The commit links to the exact tree, and uv.lock at that commit pins the
    # full dependency set a reader can `uv sync` from.
    if commit := prov.get("repo_commit"):
        prov = dict(prov)
        prov["repo_commit"] = f"[`{commit[:12]}`]({REPO_URL}/tree/{commit})"
    rows = [f"| {label} | {prov[k]} |" for k, label in _PROV_ROWS.items() if k in prov]
    if commit:
        rows.append(
            f"| Locked dependencies | [`uv.lock` @ `{commit[:12]}`]"
            f"({REPO_URL}/blob/{commit}/uv.lock) |"
        )
    extra = {k: v for k, v in prov.items() if k not in _PROV_ROWS}
    rows += [f"| {k} | {v} |" for k, v in sorted(extra.items())]
    table = "| | |\n|---|---|\n" + "\n".join(rows)
    return "\n".join("    " + ln for ln in table.splitlines())


def build_page(model: str, doc: dict, conf: dict) -> str:
    """Return the model's generated page."""
    label = conf["label"]
    years = sorted({t[:4] for t in doc["initial_conditions"]})
    kind = (
        f"{doc['members']}-member ensemble"
        if doc["kind"] == "prob"
        else "deterministic"
    )
    summary = (
        f"{kind.capitalize()} · {len(doc['initial_conditions'])} initial conditions · "
        f"{doc['lead_hours'][-1] // 24}-day horizon · "
        f"{len(doc['variables'])} variables"
    )
    md = PAGE_MD.format(
        model=model,
        model_q=quote(model),
        label=label,
        summary=summary,
        kind=kind,
        n_ic=len(doc["initial_conditions"]),
        years="/".join(years),
        lead_first=doc["lead_hours"][0],
        lead_last_d=doc["lead_hours"][-1] // 24,
        n_var=len(doc["variables"]),
        ic_source=DATA_SOURCES.get(doc.get("ic_source"), doc.get("ic_source", "—")),
        verification_source=DATA_SOURCES.get(
            doc.get("verification_source"), doc.get("verification_source", "—")
        ),
        metric_list=", ".join(m["label"] for m in doc["metrics"].values()),
        provenance_table=provenance_table(doc),
        variables_table=variables_table(doc),
        label_q=quote(label),
        badges=("\n{% badges " + conf["badges"] + " %}\n" if conf["badges"] else ""),
        description=("\n" + conf["description"] + "\n") if conf["description"] else "",
        reference=("\n" + conf["extras"] + "\n") if conf["extras"] else "",
    )
    return md


def _card(model: str, doc: dict, conf: dict) -> str:
    """One Material grid card for the section index."""
    label = conf["label"]
    short = conf["short"]
    kind = (
        f"{doc['members']}-member ensemble"
        if doc["kind"] == "prob"
        else "deterministic"
    )
    facts = (
        f"{kind} · {len(doc['variables'])} variables · "
        f"{doc['lead_hours'][-1] // 24}-day horizon"
    )
    tooltip = short or f"{label} scorecard"
    # The whole card is the link; the description appears only on hover.
    # <br> keeps the facts on their own line while the link text stays a
    # single paragraph (markdown links cannot span blank lines).
    return f"- [**{label}**<br>*{facts}*]({model}.md)" f'{{ title="{tooltip}" }}'


def main() -> int:
    """Generate all scorecard pages and the shared plot."""
    models = sorted(
        f.name[len("eval_scores_") : -len(".json")]
        for f in STATIC.glob("eval_scores_*.json")
    )
    if not models:
        raise SystemExit(
            f"no eval_scores_<model>.json under {STATIC} -- run the recipe's "
            "export_scores.py --docs first"
        )

    # Fully regenerated every run: clear stale pages from renamed models.
    import shutil

    shutil.rmtree(GENERATED, ignore_errors=True)
    GENERATED.mkdir(parents=True)

    # One shared plot for every model; it holds no data.
    (STATIC / "plot.html").write_text(
        PLOT_HTML.replace("__LOWER__", json.dumps(LOWER_IS_BETTER))
    )
    print("wrote _static/scorecard/plot.html (shared, data-free)")

    docs = {}
    confs = {}
    for model in models:
        doc = json.loads((STATIC / f"eval_scores_{model}.json").read_text())
        conf = read_config(model)
        (GENERATED / f"{model}.md").write_text(build_page(model, doc, conf))
        docs[model], confs[model] = doc, conf
        print(f"wrote scorecard/generated/{model}.md")

    any_doc = next(iter(docs.values()))
    years = sorted({t[:4] for d in docs.values() for t in d["initial_conditions"]})
    (GENERATED / "index.md").write_text(
        INDEX_MD.format(
            n_ic=len(any_doc["initial_conditions"]),
            years="/".join(years),
            entries="\n\n".join(_card(m, docs[m], confs[m]) for m in models),
        )
    )
    print(
        f"wrote scorecard/generated/index.md ({len(models)} model(s): {', '.join(models)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
