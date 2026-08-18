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

"""Generate the scorecard documentation pages from per-model YAML exports.

For every ``docs/_static/scorecard/eval_scores_<model>.yaml`` (produced by the
eval recipe's ``export_yaml.py --docs``) this writes:

  docs/scorecard/<model>/index.md        the doc page, embedding the plot
  docs/_static/scorecard/plot.html       ONE shared interactive plot
  docs/scorecard/index.md                the section index / toctree

The plot holds no data: it fetches eval_scores_<model>.yaml (selected by its
?model= query parameter) and parses it in the browser. The export writes that
YAML as JSON-formatted YAML precisely so JSON.parse can read it -- no JS YAML
library, no CDN, and the numbers exist exactly once. Run from anywhere::

    python docs/scorecard/make_pages.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import yaml

HERE = Path(__file__).resolve().parent  # docs/scorecard
STATIC = HERE.parent / "_static" / "scorecard"  # data + shared plot live here

# Display names for page titles and the plot's floating model badge.
LABELS = {
    "fcn3": "FCN3",
    "aurora": "Aurora",
    "sfno": "SFNO",
    "fengwu": "FengWu",
    "ucast": "UCast",
    "graphcast": "GraphCast",
    "graphcast_small": "GraphCast-small",
    "pangu3": "Pangu (3 h)",
    "pangu6": "Pangu (6 h)",
    "pangu24": "Pangu (24 h)",
}

LOWER_BETTER = {"rmse", "mae", "lsd", "ensemble_mean_mse", "crps", "ensemble_variance"}

# Short model description + citation shown on each page. Models without an
# entry simply get no description/reference section.
MODEL_INFO = {
    "fcn3": {
        "description": (
            "FourCastNet 3 is NVIDIA's probabilistic machine-learning weather "
            "model, built on spherical (geometric) signal processing with a "
            "hidden-Markov ensemble formulation: each member evolves its own "
            "calibrated stochastic state, so the ensemble spread is learned "
            "rather than imposed by initial-condition perturbations. It "
            "forecasts 72 atmospheric variables globally at 0.25° resolution "
            "with a 6-hour step."
        ),
        "citation": (
            "Bonev, B., Kurth, T., Mahesh, A., Bisson, M., Kossaifi, J., "
            "Kashinath, K., ... & Keller, A. (2025). FourCastNet 3: A "
            "geometric approach to probabilistic machine-learning weather "
            "forecasting at scale. arXiv preprint arXiv:2507.12144."
        ),
    },
    "aurora": {
        "description": (
            "Aurora is a foundation model of the atmosphere from Microsoft "
            "Research: a 1.3B-parameter Swin-transformer with Perceiver-style "
            "encoders pretrained on over a million hours of diverse weather "
            "and climate data. The version scored here is the 0.25° "
            "deterministic medium-range configuration, which consumes the two "
            "most recent analysis frames (t-6h and t0) and steps forward "
            "6 hours at a time on a 720x1440 grid (pole-padded onto ERA5's "
            "721x1440 for verification)."
        ),
        "citation": (
            "Bodnar, C., Bruinsma, W. P., Lucic, A., Stanley, M., "
            "Brandstetter, J., Garvan, P., ... & Perdikaris, P. (2024). "
            "Aurora: A foundation model of the atmosphere. arXiv preprint "
            "arXiv:2405.13063, 1(8)."
        ),
    },
}

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
// No data here: the model is picked by ?model=, its YAML fetched below.
const $=s=>document.querySelector(s);
const Q=new URLSearchParams(location.search);
const MODEL=Q.get("model")||"";
const LABEL=Q.get("label")||MODEL;
document.title=LABEL+" skill";
$(".badge") && ($(".badge").textContent=LABEL);
let D=null,days=[];
const LOWER=["rmse","mae","lsd","ensemble_mean_mse","crps","ensemble_variance"];
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
// The export writes JSON-formatted YAML exactly so this needs no YAML
// library: strip the leading comment lines and JSON.parse the rest.
fetch(`eval_scores_${MODEL}.yaml`)
  .then(r=>{if(!r.ok)throw new Error("HTTP "+r.status);return r.text();})
  .then(t=>{
    D=JSON.parse(t.replace(/^(?:#[^\n]*\n)+/,""));
    days=D.lead_hours.map(h=>h/24);
    Object.keys(D.metrics).forEach(k=>mSel.appendChild(new Option(D.metrics[k].label,k)));
    fillVars();draw();})
  .catch(e=>{$("#t").textContent=`failed to load eval_scores_${MODEL}.yaml`;
    $("#s").textContent=String(e);});
</script></body></html>
"""

PAGE_MD = """\
<!-- Generated by docs/scorecard/make_pages.py from _static/scorecard/eval_scores_{model}.yaml -- do not hand-edit. -->

# {label}
{description}
## Skill

Pick a metric and variable; hover for exact values at each lead time.

<iframe src="../../_static/scorecard/plot.html?model={model}&label={label_q}" title="{label} skill"
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
| Lead times | {lead_first} h to {lead_last_d} days |
| Variables scored | {n_var} |
| Metrics | {metric_list} |

## Data

The numbers behind the plot are in [`eval_scores_{model}.yaml`](../../_static/scorecard/eval_scores_{model}.yaml), exported by
the [eval recipe scorecard](https://github.com/NVIDIA/earth2studio/tree/main/recipes/eval)
(`scorecard/export_yaml.py --docs`) -- one value per metric, variable
and lead time, in the variable's own units.

## Reproducibility

??? info "Run and environment details"

{provenance_table}
{reference}"""

INDEX_MD = """\
<!-- Generated by docs/scorecard/make_pages.py -- do not hand-edit. -->

# Scorecards

Forecast skill of Earth2Studio models, one scorecard per model -- these show
each model's own skill, not a comparison. Every model was evaluated on the
same campaign: {n_ic} initial conditions ({years}), 14-day horizon, ERA5
verification via ARCO. Pages are generated from per-model YAML exports
produced by the
[scorecard recipe](https://github.com/NVIDIA/earth2studio/tree/main/recipes/eval/scorecard),
which documents how to generate a scorecard for any model; the
[eval recipe](https://github.com/NVIDIA/earth2studio/tree/main/recipes/eval)
provides the backend support (inference, scoring and metrics).

## Prognostic models

{entries}
"""

# Reproducibility fields, in display order: yaml key -> row label.
_PROV_ROWS = {
    "scores_written": "Scores written",
    "gpus": "GPUs",
    "torch": "PyTorch",
    "cuda": "CUDA",
    "python": "Python",
    "repo_commit": "Repo commit (at export)",
    "exported": "YAML exported",
}


def provenance_table(doc: dict) -> str:
    prov = doc.get("provenance") or {}
    if not prov:
        return (
            "    Not recorded for this run -- provenance capture was added "
            "after it was scored. Re-exporting with the current "
            "`export_yaml.py` records it."
        )
    rows = [f"| {label} | {prov[k]} |" for k, label in _PROV_ROWS.items() if k in prov]
    extra = {k: v for k, v in prov.items() if k not in _PROV_ROWS}
    rows += [f"| {k} | {v} |" for k, v in sorted(extra.items())]
    table = "| | |\n|---|---|\n" + "\n".join(rows)
    return "\n".join("    " + ln for ln in table.splitlines())


def build_page(model: str, doc: dict) -> str:
    """Return the model's index.md."""
    label = LABELS.get(model, model)
    years = sorted({t[:4] for t in doc["initial_conditions"]})
    kind = (
        f"{doc['members']}-member ensemble" if doc["kind"] == "prob" else "deterministic"
    )
    summary = (
        f"{kind.capitalize()} · {len(doc['initial_conditions'])} initial conditions · "
        f"{doc['lead_hours'][-1] // 24}-day horizon · "
        f"{len(doc['variables'])} variables"
    )
    md = PAGE_MD.format(
        model=model,
        label=label,
        summary=summary,
        kind=kind,
        n_ic=len(doc["initial_conditions"]),
        years="/".join(years),
        lead_first=doc["lead_hours"][0],
        lead_last_d=doc["lead_hours"][-1] // 24,
        n_var=len(doc["variables"]),
        metric_list=", ".join(m["label"] for m in doc["metrics"].values()),
        provenance_table=provenance_table(doc),
        label_q=quote(label),
        description=(
            "\n" + MODEL_INFO[model]["description"] + "\n"
            if model in MODEL_INFO
            else ""
        ),
        reference=(
            "\n## Reference\n\n" + MODEL_INFO[model]["citation"] + "\n"
            if model in MODEL_INFO
            else ""
        ),
    )
    return md


def main() -> int:
    models = sorted(
        f.name[len("eval_scores_") : -len(".yaml")]
        for f in STATIC.glob("eval_scores_*.yaml")
    )
    if not models:
        raise SystemExit(
            f"no eval_scores_<model>.yaml under {STATIC} -- run the recipe's "
            "export_yaml.py --docs first"
        )

    # One shared plot for every model; it holds no data.
    (STATIC / "plot.html").write_text(PLOT_HTML)
    print("wrote _static/scorecard/plot.html (shared, data-free)")

    docs = {}
    for model in models:
        # The YAML body is JSON (a YAML subset), so safe_load reads it too.
        doc = yaml.safe_load((STATIC / f"eval_scores_{model}.yaml").read_text())
        (HERE / model).mkdir(exist_ok=True)
        (HERE / model / "index.md").write_text(build_page(model, doc))
        docs[model] = doc
        print(f"wrote scorecard/{model}/index.md")

    any_doc = next(iter(docs.values()))
    years = sorted({t[:4] for d in docs.values() for t in d["initial_conditions"]})
    (HERE / "index.md").write_text(
        INDEX_MD.format(
            n_ic=len(any_doc["initial_conditions"]),
            years="/".join(years),
            entries="\n".join(
                f"- [{LABELS.get(m, m)}]({m}/index.md)" for m in models
            ),
        )
    )
    print(f"wrote scorecard/index.md ({len(models)} model(s): {', '.join(models)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
