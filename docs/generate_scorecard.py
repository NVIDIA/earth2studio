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
import os
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
# Baseline runs (config/default.md): exported like models, drawn by the
# plot as reference overlays, and excluded from page/card generation.
BASELINES = DEFAULTS.get("baselines", {}) or {}
DATA_SOURCES = DEFAULTS.get("data_sources", {})
STATIC = DOCS / "_static" / "scorecard"  # data + shared plot live here

# The score JSONs are not stored in the git repository: they live in the
# Earth2Studio assets dataset on Hugging Face, and the docs build fetches
# them into ``STATIC``. Local files always preferred, so a developer
# iterating on fresh exports never triggers a download.
# ``SCORECARD_DATA_REVISION`` selects a branch or PR ref of the dataset
# (for example ``refs/pr/2`` to build against a pending data update).
DATA_REPO = "nvidia/earth2studio-assets"
DATA_REVISION = os.environ.get("SCORECARD_DATA_REVISION", "main")


def _sync_data_from_hub() -> None:
    """Download the score JSONs from the assets dataset when absent.

    Snapshots the dataset's ``scorecard/`` folder through
    ``huggingface_hub`` — authenticated by ``HF_TOKEN`` when set, and
    served from the local hub cache on repeat builds — then copies every
    ``eval_scores_*.json`` into ``STATIC`` under its file name
    (per-model subfolders and a flat layout both work).  Runs only when
    ``STATIC`` holds no score files at all, so a checkout with local
    exports builds fully offline.
    """
    import shutil

    from huggingface_hub import snapshot_download

    if any(STATIC.glob("eval_scores_*.json")):
        return
    root = snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        revision=DATA_REVISION,
        allow_patterns=["scorecard/*"],
    )
    files = sorted(Path(root, "scorecard").rglob("eval_scores_*.json"))
    if not files:
        raise SystemExit(f"no score files found in the assets dataset: {DATA_REPO}")
    STATIC.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, STATIC / f.name)
    print(f"fetched {len(files)} score files from the assets dataset")


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
  select{font:inherit;color:var(--ink);
    border:1px solid var(--axis);border-radius:7px;
    padding:6px 30px 6px 10px;-webkit-appearance:none;appearance:none;
    background:var(--surface-1) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'><path d='M1 1l4 4 4-4' fill='none' stroke='%23898781' stroke-width='1.6' stroke-linecap='round'/></svg>") no-repeat right 10px center}
  select:disabled{opacity:.45}
  .card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;
    padding:12px 14px 8px}
  .card h2{font-size:13.5px;margin:0 0 2px} .card p{margin:0 0 8px;color:var(--ink-2);font-size:12px}
  .legend{display:flex;gap:14px;margin:0 0 6px;font-size:11.5px;color:var(--ink-2)}
  .legend .chip{display:inline-block;width:14px;height:0;border-top:2.5px solid var(--s1);
    vertical-align:middle;margin-right:5px;border-radius:2px}
  .legend .chip.ref{border-top-style:dashed;border-top-color:var(--muted)}
  .legend .chip.dot{border-top-style:dotted;border-top-color:var(--muted)}
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
  <div class="ctl" id="vwctl" hidden><label>View</label><select id="vw">
    <option value="curve">Skill curve</option>
    <option value="heat">IC heat map</option></select></div>
  <div class="ctl"><label>Metric</label><select id="m"></select></div>
  <div class="ctl"><label>Variable</label><select id="v"></select></div>
  <div class="ctl" id="lctl"><label>Level</label><select id="l"></select></div>
  <div class="ctl" id="rctl" hidden><label>Region</label><select id="r"></select></div>
  <div class="ctl" id="moctl" hidden><label>Month</label><select id="mo"></select></div>
  <div class="ctl" id="hctl" hidden><label>Init hour</label><select id="h"></select></div>
  <div class="ctl" id="bctl" hidden><label>Baseline</label><select id="b"></select></div>
</div>
<div class="card"><h2 id="t"></h2><p id="s"></p>
<div class="legend" id="lg" hidden></div><svg id="c" viewBox="0 0 900 330"></svg></div>
</div>
<div class="tip" id="tip"></div>
<script>
// No data here: the model is picked by ?model=, its JSON fetched below.
// Regional / monthly splits live in sibling eval_scores_<model>_*.json
// files and are fetched lazily the first time their control is used.
const $=s=>document.querySelector(s);
const Q=new URLSearchParams(location.search);
const MODEL=Q.get("model")||"";
const LABEL=Q.get("label")||MODEL;
document.title=LABEL+" skill";
$(".badge") && ($(".badge").textContent=LABEL);
let D=null,days=[],MONTHLY=null;
const RCACHE={};      // region name -> fetched split doc (null while loading)
const LOWER=__LOWER__;
const css=n=>getComputedStyle(document.body).getPropertyValue(n).trim();
const isFin=q=>q!=null&&isFinite(q);
const pretty=n=>n.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase());
const tip=$("#tip");
const ns="http://www.w3.org/2000/svg";
const mk=(svg,t,a)=>{const e=document.createElementNS(ns,t);
  for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e;};
const fmt=v=>{if(!isFin(v))return "–";const a=Math.abs(v);
  if(a!==0&&(a<1e-3||a>=1e5))return v.toExponential(2);
  return v.toFixed(a>=100?1:a>=10?2:3);};
const mSel=$("#m"),vSel=$("#v"),lSel=$("#l"),rSel=$("#r"),moSel=$("#mo"),
      hSel=$("#h"),bSel=$("#b"),vwSel=$("#vw");
let HEAT=null;   // lazily fetched eval_scores_<model>_heatmap.json
let HOURLY=null; // lazily fetched eval_scores_<model>_hourly.json
const BASELINES=__BASELINES__;
const BCACHE={}; // baseline name -> its main eval_scores_<name>.json
// The heatmap view swaps in its own (narrower) metric/variable sets.
function metricSet(){return vwSel.value==="heat"&&HEAT?HEAT.metrics:D.metrics;}
function fillMetrics(){
  const ms=metricSet(),prev=mSel.value; mSel.innerHTML="";
  Object.keys(ms).forEach(k=>mSel.appendChild(new Option(ms[k].label,k)));
  if(prev in ms)mSel.value=prev;
}
// The variable list splits into a quantity selector and a level selector,
// so neither dropdown carries all ~70 names.  Level variables follow the
// <base><hPa> convention (t500); everything else (t2m, msl, tcwv, ...) is
// a surface quantity with the level selector disabled.
let VBASES={};
function fillVars(){
  const vs=Object.keys(metricSet()[mSel.value].values);
  vs.sort((a,b)=>D.variables.indexOf(a)-D.variables.indexOf(b));
  const surface=[];VBASES={};
  // A var is a pressure-level entry only when the export groups it off
  // Surface — the name shape alone misfiles e.g. tp06 as "tp at 6 hPa".
  vs.forEach(v=>{const m=v.match(/^([a-z]+)(\d+)$/);
    if(m&&((D.variable_groups||{})[v]||"Surface")!=="Surface")
      (VBASES[m[1]]=VBASES[m[1]]||[]).push(+m[2]);
    else surface.push(v);});
  Object.values(VBASES).forEach(a=>a.sort((x,y)=>x-y));
  const prev=vSel.value; vSel.innerHTML="";
  if(surface.length){
    const g=document.createElement("optgroup");g.label="Surface";
    surface.forEach(v=>g.appendChild(
      new Option(v+(D.units[v]?"  ("+D.units[v]+")":""),v)));
    vSel.appendChild(g);
  }
  const bases=Object.keys(VBASES);
  if(bases.length){
    const g=document.createElement("optgroup");g.label="Pressure levels";
    bases.forEach(b=>{
      const grp=(D.variable_groups||{})[b+VBASES[b][0]]||b;
      g.appendChild(new Option(`${grp} (${b})`,b));});
    vSel.appendChild(g);
  }
  const opts=[...vSel.options].map(o=>o.value);
  if(opts.includes(prev))vSel.value=prev;
  else if(opts.includes("z"))vSel.value="z";
  fillLevels();
}
function fillLevels(){
  const levs=VBASES[vSel.value];
  const prev=lSel.value; lSel.innerHTML="";
  if(!levs){lSel.appendChild(new Option("surface",""));lSel.disabled=true;return;}
  lSel.disabled=false;
  levs.forEach(p=>lSel.appendChild(new Option(p+" hPa",p)));
  if(levs.map(String).includes(prev))lSel.value=prev;
  else if(levs.includes(500))lSel.value="500";
}
// The concrete variable name the data files use (t + 500 -> t500).
function varName(){
  return VBASES[vSel.value]?vSel.value+lSel.value:vSel.value;
}
// The active split curve (region or month) for the current metric/variable,
// or null when the whole-grid / all-IC curve is the only one to show.
// undefined y means the split has no data for this metric (e.g. LSD is
// spectral, hence global-only).
function activeSplit(){
  const k=mSel.value,v=varName();
  if(moSel.value&&moSel.value!=="all"){
    const mm=MONTHLY&&MONTHLY.metrics_by_month[moSel.value];
    return {y:mm&&mm[k]?mm[k].values[v]:undefined,label:moSel.value,ref:"All months"};
  }
  if(hSel.value&&hSel.value!=="all"){
    const hh=HOURLY&&HOURLY.metrics_by_hour[hSel.value];
    return {y:hh&&hh[k]?hh[k].values[v]:undefined,label:hSel.value+" ICs",ref:"All hours"};
  }
  if(rSel.value&&rSel.value!=="global"){
    const rd=RCACHE[rSel.value];
    return {y:rd&&rd.metrics[k]?rd.metrics[k].values[v]:undefined,
            label:pretty(rSel.value),ref:"Global"};
  }
  return null;
}
function legend(items){
  const lg=$("#lg");
  if(!items){lg.hidden=true;lg.innerHTML="";return;}
  lg.hidden=false;
  lg.innerHTML=items.map(i=>{
    const cls=i.dash==="2 5"?" dot":(i.ref?" ref":"");
    return `<span><span class="chip${cls}"></span>${i.label}</span>`;}).join("");
}
function drawLine(svg,y,px,py,color,dash){
  let d="",pen=false;
  y.forEach((q,i)=>{if(!isFin(q)){pen=false;return;}
    d+=(pen?"L":"M")+px(days[i])+" "+py(q);pen=true;});
  const a={d,fill:"none",stroke:color,"stroke-width":2,
    "stroke-linejoin":"round","stroke-linecap":"round"};
  if(dash){a["stroke-dasharray"]=dash;a["stroke-width"]=1.6;}
  mk(svg,"path",a);
}
function draw(){
  if(vwSel.value==="heat")drawHeat(); else drawCurve();
}
function drawHeat(){
  const k=mSel.value,v=varName(),svg=$("#c");svg.innerHTML="";
  legend(null);
  const rows=(HEAT.metrics[k]&&HEAT.metrics[k].values[v])||[];
  const unit=D.units[v]||"";
  $("#t").textContent=`${HEAT.metrics[k].label} by initial condition — ${v}${unit?" ("+unit+")":""}`;
  const flat=rows.flat().filter(isFin);
  if(!flat.length){$("#s").textContent="";
    mk(svg,"text",{x:450,y:155,class:"al","text-anchor":"middle"}).textContent="no data";return;}
  const lo=Math.min(...flat),hi=Math.max(...flat);
  $("#s").textContent=`One row per initial condition, one column per lead time (whole grid). `
    +`Range ${fmt(lo)} to ${fmt(hi)} ${unit}.`;
  const W=900,H=330,L=64,R=24,T=14,B=44;
  // Sequential single-hue ramp: chart surface -> accent.
  const rgb=s=>{const m=s.match(/#([0-9a-f]{6})/i);if(!m)return [128,128,128];
    const n=parseInt(m[1],16);return [n>>16&255,n>>8&255,n&255];};
  const c0=rgb(css("--surface-1")),c1=rgb(css("--s1"));
  const col=q=>{const t=(q-lo)/((hi-lo)||1);
    return `rgb(${c0.map((x,i)=>Math.round(x+(c1[i]-x)*t)).join(",")})`;};
  const ics=HEAT.initial_conditions,nr=rows.length,nc=days.length;
  const cw=(W-L-R)/nc,ch=(H-T-B)/nr;
  rows.forEach((row,ri)=>row.forEach((q,ci)=>{if(!isFin(q))return;
    mk(svg,"rect",{x:L+ci*cw,y:T+ri*ch,width:cw+0.4,height:ch+0.4,fill:col(q)});}));
  // y: one tick per month (4 ICs/month); x: lead-day ticks like the curve.
  for(let ri=0;ri<nr;ri+=4)
    mk(svg,"text",{x:L-8,y:T+(ri+0.5)*ch+3,class:"tk","text-anchor":"end"})
      .textContent=ics[ri].slice(0,7);
  const maxd=days[days.length-1],step=maxd<=3?0.5:maxd<=8?1:2;
  for(let d=step;d<=maxd+1e-9;d+=step){
    const X=L+(W-L-R)*(d-days[0])/((days[days.length-1]-days[0])||1);
    mk(svg,"text",{x:X,y:H-B+16,class:"tk","text-anchor":"middle"})
      .textContent=(d%1?d.toFixed(1):d);}
  mk(svg,"text",{x:(L+W-R)/2,y:H-6,class:"al","text-anchor":"middle"}).textContent="lead time (days)";
  // Color legend: a thin gradient bar with min/max, top right.
  const gx=W-R-130;
  for(let i=0;i<26;i++)
    mk(svg,"rect",{x:gx+i*4,y:2,width:4.4,height:7,fill:col(lo+(hi-lo)*i/25)});
  mk(svg,"text",{x:gx-6,y:9,class:"tk","text-anchor":"end"}).textContent=fmt(lo);
  mk(svg,"text",{x:gx+110,y:9,class:"tk"}).textContent=fmt(hi);
  // Hover: one overlay, cell resolved from the pointer.
  const hit=mk(svg,"rect",{x:L,y:T,width:W-L-R,height:H-T-B,fill:"transparent"});
  hit.style.cursor="crosshair";
  const box=mk(svg,"rect",{width:cw,height:ch,fill:"none",
    stroke:css("--ink"),"stroke-width":1,opacity:0});
  hit.addEventListener("mousemove",e=>{
    const bb=svg.getBoundingClientRect();
    const ci=Math.max(0,Math.min(nc-1,Math.floor(((e.clientX-bb.left)*W/bb.width-L)/cw)));
    const ri=Math.max(0,Math.min(nr-1,Math.floor(((e.clientY-bb.top)*H/bb.height-T)/ch)));
    box.setAttribute("x",L+ci*cw);box.setAttribute("y",T+ri*ch);box.setAttribute("opacity",1);
    const q=rows[ri]?rows[ri][ci]:null;
    tip.innerHTML=`<span class="k">IC</span> ${ics[ri]}<br>`
      +`<span class="k">lead</span> ${D.lead_hours[ci]} h (${days[ci]} d)<br><b>${fmt(q)}</b> ${unit}`;
    tip.style.opacity=1;
    let x=e.clientX+14,yy=e.clientY-10;
    const r=tip.getBoundingClientRect();
    if(x+r.width+8>innerWidth)x=e.clientX-r.width-14;
    tip.style.left=Math.max(8,x)+"px";tip.style.top=Math.max(8,yy)+"px";});
  hit.addEventListener("mouseleave",()=>{tip.style.opacity=0;box.setAttribute("opacity",0);});
}
function drawCurve(){
  const k=mSel.value,v=varName(),base=D.metrics[k].values[v]||[];
  const split=activeSplit();
  const svg=$("#c");svg.innerHTML="";
  const unit=D.metrics[k].unit||D.units[v]||"";
  const where=split?` — ${split.label}`:"";
  $("#t").textContent=`${D.metrics[k].label} — ${v}${unit?" ("+unit+")":""}${where}`;
  const bnote=activeBaselines().length
    ?" Baseline curves are whole-grid, all-IC references in every view.":"";
  $("#s").textContent=(
    k==="spread_skill"?"Ensemble spread over ensemble-mean RMSE. 1.0 is calibrated; below 1 is over-confident."
    :LOWER.includes(k)?"Lower is better. Latitude-weighted, averaged over initial conditions."
    :"Higher is better.")+bnote;
  if(split&&split.y===undefined){
    legend(null);
    mk(svg,"text",{x:450,y:155,class:"al","text-anchor":"middle"})
      .textContent=`no ${D.metrics[k].label} data for ${split.label}`+
        (k==="lsd"?" (spectral metrics are whole-grid only)":"");
    return;
  }
  // Baseline overlays: whole-grid all-IC reference runs, one dash pattern
  // per baseline so identity never rides on color alone.
  const dashes=["6 5","2 5"];
  const extras=activeBaselines().map(b=>{
    const mB=BCACHE[b].metrics[k];
    return {y:mB?mB.values[v]:null,label:BASELINES[b],ref:true,
            dash:dashes[Object.keys(BASELINES).indexOf(b)%dashes.length]};
  }).filter(s=>s.y);
  const series=split
    ?[{y:base,label:split.ref,ref:true},{y:split.y||[],label:split.label},...extras]
    :[{y:base,label:extras.length?LABEL:""},...extras];
  legend((split||extras.length)?series:null);
  const all=series.flatMap(s=>s.y.filter(isFin));
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
  series.forEach(s=>drawLine(svg,s.y,px,py,s.ref?css("--muted"):css("--s1"),
    s.dash||(s.ref?"5 5":null)));
  const hl=mk(svg,"line",{x1:L,x2:L,y1:T,y2:H-B,stroke:css("--muted"),"stroke-width":1,opacity:0});
  const dots=series.map(s=>mk(svg,"circle",{r:4.5,
    fill:s.ref?css("--muted"):css("--s1"),
    stroke:css("--surface-1"),"stroke-width":2,opacity:0}));
  const hit=mk(svg,"rect",{x:L,y:T,width:W-L-R,height:H-T-B,fill:"transparent"});
  hit.style.cursor="crosshair";
  hit.addEventListener("mousemove",e=>{
    const bb=svg.getBoundingClientRect(),sx=(e.clientX-bb.left)*W/bb.width;
    let bi=0,bd=1e9;days.forEach((dd,i)=>{const q=Math.abs(px(dd)-sx);if(q<bd){bd=q;bi=i;}});
    hl.setAttribute("x1",px(days[bi]));hl.setAttribute("x2",px(days[bi]));hl.setAttribute("opacity",1);
    const rows=series.map((s,si)=>{const q=s.y[bi];
      if(isFin(q)){dots[si].setAttribute("cx",px(days[bi]));dots[si].setAttribute("cy",py(q));
        dots[si].setAttribute("opacity",1);}
      else dots[si].setAttribute("opacity",0);
      const name=s.label?`<span class="k">${s.label}</span> `:"";
      return `${name}<b>${fmt(q)}</b>`;});
    tip.innerHTML=`<span class="k">lead</span> ${D.lead_hours[bi]} h (${days[bi]} d)<br>`+
      rows.join("<br>")+` ${D.metrics[mSel.value].unit||D.units[varName()]||""}`;
    tip.style.opacity=1;
    let x=e.clientX+14,yy=e.clientY-10;
    const r=tip.getBoundingClientRect();
    if(x+r.width+8>innerWidth)x=e.clientX-r.width-14;
    tip.style.left=Math.max(8,x)+"px";tip.style.top=Math.max(8,yy)+"px";});
  hit.addEventListener("mouseleave",()=>{tip.style.opacity=0;
    hl.setAttribute("opacity",0);dots.forEach(d=>d.setAttribute("opacity",0));});
}
// Region and month are mutually exclusive splits: the monthly breakdown is
// computed on the whole grid, so picking a month snaps the region back to
// Global (and vice versa the month back to All).
function syncControls(){
  // One split at a time: region, month/season, and init hour are computed
  // on independent axes, so combining them would need cross exports.  The
  // baseline overlays stay available in every curve view: they are always
  // whole-grid all-IC references (noted in the subtitle), so they never
  // depend on the split.  The heatmap is a per-IC view of one model.
  const heat=vwSel.value==="heat";
  const monthOn=moSel.value&&moSel.value!=="all";
  const regionOn=rSel.value&&rSel.value!=="global";
  const hourOn=hSel.value&&hSel.value!=="all";
  rSel.disabled=heat||monthOn||hourOn;
  moSel.disabled=heat||regionOn||hourOn;
  hSel.disabled=heat||regionOn||monthOn;
  bSel.disabled=heat;
}
function activeBaselines(){
  if(vwSel.value==="heat"||!bSel.value||bSel.value==="none")return [];
  const names=bSel.value==="both"?Object.keys(BASELINES):[bSel.value];
  return names.filter(b=>b!==MODEL&&BCACHE[b]);
}
function onBaseline(){
  const want=(bSel.value==="both"?Object.keys(BASELINES).filter(b=>b!==MODEL):
    (bSel.value&&bSel.value!=="none"?[bSel.value]:[]));
  want.filter(b=>!(b in BCACHE)).forEach(b=>{
    BCACHE[b]=null;
    fetchJSON(`eval_scores_${b}.json`)
      .then(d=>{BCACHE[b]=d;syncControls();draw();})
      .catch(()=>{delete BCACHE[b];syncControls();draw();});
  });
  syncControls();draw();
}
function onHour(){
  if(hSel.value!=="all"&&!HOURLY){
    fetchJSON(`eval_scores_${MODEL}_hourly.json`)
      .then(d=>{HOURLY=d;syncControls();draw();})
      .catch(()=>{hSel.value="all";syncControls();draw();});
    return;
  }
  syncControls();draw();
}
function onView(){
  if(vwSel.value==="heat"&&!HEAT){
    fetchJSON(`eval_scores_${MODEL}_heatmap.json`)
      .then(d=>{HEAT=d;fillMetrics();fillVars();syncControls();draw();})
      .catch(()=>{vwSel.value="curve";syncControls();draw();});
    return;
  }
  fillMetrics();fillVars();syncControls();draw();
}
function fetchJSON(name){
  return fetch(name).then(r=>{
    if(!r.ok)throw new Error("HTTP "+r.status);return r.text();}).then(JSON.parse);
}
function onRegion(){
  const reg=rSel.value;
  if(reg!=="global"&&!(reg in RCACHE)){
    RCACHE[reg]=null;
    fetchJSON(`eval_scores_${MODEL}_region_${reg}.json`)
      .then(d=>{RCACHE[reg]=d;if(rSel.value===reg){syncControls();draw();}})
      .catch(()=>{delete RCACHE[reg];rSel.value="global";syncControls();draw();});
  }
  syncControls();draw();
}
function onMonth(){
  if(moSel.value!=="all"&&!MONTHLY){
    fetchJSON(`eval_scores_${MODEL}_monthly.json`)
      .then(d=>{MONTHLY=d;syncControls();draw();})
      .catch(()=>{moSel.value="all";syncControls();draw();});
  }
  syncControls();draw();
}
mSel.addEventListener("change",()=>{fillVars();draw();});
vSel.addEventListener("change",()=>{fillLevels();draw();});
lSel.addEventListener("change",draw);
rSel.addEventListener("change",onRegion);
moSel.addEventListener("change",onMonth);
hSel.addEventListener("change",onHour);
bSel.addEventListener("change",onBaseline);
vwSel.addEventListener("change",onView);
// The export is plain minified JSON.
fetchJSON(`eval_scores_${MODEL}.json`)
  .then(d=>{
    D=d;
    days=D.lead_hours.map(h=>h/24);
    Object.keys(D.metrics).forEach(k=>mSel.appendChild(new Option(D.metrics[k].label,k)));
    if(D.regions&&D.regions.length>1){
      $("#rctl").hidden=false;
      D.regions.forEach(r=>rSel.appendChild(new Option(pretty(r),r)));
      rSel.value="global";
    }
    if(D.has_monthly){
      $("#moctl").hidden=false;
      moSel.appendChild(new Option("All months","all"));
      // Season blocks first, then the individual months.
      ["DJF","MAM","JJA","SON"]
        .forEach(s=>moSel.appendChild(new Option(s+" (season)",s)));
      ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        .forEach(m=>moSel.appendChild(new Option(m,m)));
      moSel.value="all";
    }
    if(D.has_hourly){
      $("#hctl").hidden=false;
      hSel.appendChild(new Option("All hours","all"));
      [...new Set(D.initial_conditions.map(t=>t.slice(11,13)+"Z"))].sort()
        .forEach(hh=>hSel.appendChild(new Option(hh,hh)));
      hSel.value="all";
    }
    if(D.has_heatmap){$("#vwctl").hidden=false;vwSel.value="curve";}
    const bnames=Object.keys(BASELINES).filter(b=>b!==MODEL);
    if(bnames.length){
      $("#bctl").hidden=false;
      bSel.appendChild(new Option("None","none"));
      bnames.forEach(b=>bSel.appendChild(new Option(BASELINES[b],b)));
      if(bnames.length>1)bSel.appendChild(new Option("Both","both"));
      // Climatology on by default: the reference every skill curve should
      // beat.  Persistence stays one click away.
      bSel.value=bnames.includes("climatology")?"climatology":bnames[0];
      onBaseline();
    }
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

Pick a metric and variable; hover for exact values at each lead time.{splits_hint}

<!-- The src is resolved in the browser against the FINAL page URL
     (.../scorecard/generated/<model>/), so it stays correct at any site
     depth — fork preview, versioned deploy, or local serve — and there is
     no static URL for the site build to rewrite. -->
<iframe id="skill-plot" title="{label} skill"
        style="width:100%;height:560px;border:1px solid rgba(128,128,128,.35);border-radius:10px;"
        loading="lazy"></iframe>
<script>
document.getElementById("skill-plot").src = new URL(
  "../../../_static/scorecard/plot.html?model={model_q}&label={label_q}",
  window.location.href);
</script>

## Evaluation

{summary}

Scores are latitude-weighted (cos φ) and aggregated over the initial
conditions. Evaluation is done against ERA5 fetched from ARCO.{ic_note}

| | |
|---|---|
| Type | {kind} |
| Initial conditions | {n_ic} ({years}) |
| Initial condition source | {ic_source} |
| Verification (ground truth) | {verification_source} |
| Lead times | {lead_first} h to {lead_last_d} days |
| Variables scored | {n_var} |
| Metrics | {metric_list} |{region_row}

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
title: Scorecards
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

<!-- The site's own landing-page button classes (nvidia-material.css), so
     these render identically to the Tutorial / Install / Examples buttons
     on the home page.  hrefs are relative to this page's final URL. -->
<div class="e2s-start-grid">
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
    regions = [r for r in doc.get("regions", []) if r != "global"]
    hints = []
    if regions:
        hints.append("the Region selector for continental splits")
    if doc.get("has_monthly"):
        hints.append(
            "the Month selector for seasonal (DJF/MAM/JJA/SON) and per-month "
            "skill against the all-month curve"
        )
    if doc.get("has_hourly"):
        hints.append("the Init hour selector for skill by initialization time")
    if doc.get("has_heatmap"):
        hints.append("the View selector for the skill of every initial condition")
    if BASELINES and model not in BASELINES:
        hints.append(
            "the Baseline selector to overlay "
            + " and ".join(v.lower() for v in BASELINES.values())
            + " reference forecasts"
        )
    if len(hints) > 2:
        joined = ", ".join(hints[:-1]) + ", and " + hints[-1]
    else:
        joined = " and ".join(hints)
    splits_hint = f" Use {joined}." if hints else ""
    region_row = (
        "\n| Regions | global, "
        + ", ".join(r.replace("_", " ") for r in regions)
        + " |"
        if regions
        else ""
    )
    hours = sorted({t[11:13] for t in doc["initial_conditions"]})
    ic_note = (
        "\nInitial conditions rotate through the "
        + "/".join(f"{h}Z" for h in hours)
        + " hours."
        if len(hours) > 1
        else ""
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
        splits_hint=splits_hint,
        region_row=region_row,
        ic_note=ic_note,
    )
    return md


def _card(model: str, doc: dict, conf: dict) -> str:
    """One landing-page-style button for the section index."""
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
    tooltip = f"{short + ' ' if short else ''}{facts}."
    # A landing-page style button: the model name is the label, the short
    # description and campaign facts appear on hover.  Raw HTML with a
    # directory-relative href so the link resolves against the final page
    # URL at any site depth.
    return (
        f'  <a class="e2s-home-button" href="{model}/" '
        f'title="{tooltip}">{label}</a>'
    )


def main() -> int:
    """Generate all scorecard pages and the shared plot."""
    _sync_data_from_hub()
    # Split exports (regional / monthly breakdowns fetched lazily by the
    # plot) sit beside the model files; they are data for a model's page,
    # not models of their own.
    models = sorted(
        name
        for f in STATIC.glob("eval_scores_*.json")
        if not (name := f.name[len("eval_scores_") : -len(".json")]).endswith(
            ("_monthly", "_heatmap", "_hourly")
        )
        and "_region_" not in name
        and name not in BASELINES
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
        PLOT_HTML.replace("__LOWER__", json.dumps(LOWER_IS_BETTER)).replace(
            "__BASELINES__", json.dumps(BASELINES)
        )
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
            entries="\n".join(_card(m, docs[m], confs[m]) for m in models),
        )
    )
    print(
        f"wrote scorecard/generated/index.md ({len(models)} model(s): {', '.join(models)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
