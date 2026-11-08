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

import matplotlib
from common import *
from PIL import Image, ImageDraw

W = 5400
M, G, BAND = 30, 24, 78
GL = 1500  # globe edge
LEG = 120  # legend / colourbar strip under globes
R1 = BAND + GL + LEG  # row-1 height
R2 = 1116  # row-2 height
H = M + R1 + G + R2 + M
canvas = Image.new("RGB", (W, H), (0, 0, 0))
d = ImageDraw.Draw(canvas)
WHITE, GREY, DIM, CY = (240, 244, 250), (128, 138, 156), (70, 78, 94), (60, 220, 255)


def crop_nonblack(im, thr=10, pad=6):
    """Crop to pixels brighter than the threshold, retaining a small border."""
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a > thr)
    return im.crop(
        (
            max(xs.min() - pad, 0),
            max(ys.min() - pad, 0),
            min(xs.max() + pad, a.shape[1]),
            min(ys.max() + pad, a.shape[0]),
        )
    )


def spaced(draw, xy, text, f, fill, track=0):
    """Draw text with extra character spacing and return the final x position."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=f, fill=fill)
        x += draw.textlength(ch, font=f) + track
    return x


def header(x, y, num, title, sub):
    """Draw a numbered panel heading and subtitle on the hero canvas."""
    spaced(d, (x, y + 4), num, font(mono=True, size=22), CY, 2)
    spaced(d, (x + 44, y + 2), title, font(bold=True, size=30), WHITE, 3)
    d.text((x + 44, y + 40), sub, font=font(mono=True, size=21), fill=GREY)


def place(path, box, crop=False, cover=False):
    """Fit or cover a canvas box with an image, optionally trimming black edges."""
    x, y, w, h = box
    im = Image.open(path).convert("RGB")
    if crop:
        im = crop_nonblack(im)
    if cover:
        s = max(w / im.width, h / im.height)
        im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1), Image.LANCZOS)
        l, t = (im.width - w) // 2, (im.height - h) // 2
        im = im.crop((l, t, l + w, t + h))
        canvas.paste(im, (x, y))
        return
    s = min(w / im.width, h / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    canvas.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))


def tag(x, y, text, color=GREY):
    """Draw a label over a translucent black background on the canvas."""
    f = font(mono=True, size=20)
    tw = d.textlength(text, font=f)
    ov = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.rectangle((x, y, x + tw + 20, y + 32), fill=(0, 0, 0, 170))
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), ov).convert("RGB"))
    d.text((x + 10, y + 5), text, font=f, fill=color)


# ---- row 1
y1 = M
x01 = M
x03 = W - M - GL
x02 = x01 + GL + G
w02 = x03 - G - x02
header(
    x01,
    y1,
    "01",
    "MEDIUM RANGE AND CLIMATE FORECASTING",
    "SFNO · 0.25° · +144 h · 10 m wind, streamlines, MSLP · init 2026-01-15 00Z",
)
place(P("panels/globe_raw.png"), (x01, y1 + BAND, GL, GL), crop=True)
tw_ = matplotlib.cm.twilight
stops = (
    ["#000000", "#07041a"]
    + [matplotlib.colors.to_hex(tw_(x)) for x in (0.47, 0.40, 0.32, 0.24, 0.16, 0.08)]
    + ["#ffd9f2", "#ff7ad0", "#ff9a3c", "#fff3b0"]
)
cm = LinearSegmentedColormap.from_list("twilight_wind", stops)
bw, bh = 520, 10
bx, by = x01 + 44, y1 + BAND + GL + 46
bar = (cm(np.linspace(0, 1, bw))[:, :3] * 255).astype(np.uint8)[None].repeat(bh, 0)
canvas.paste(Image.fromarray(bar), (bx, by))
d.text((bx, by + 16), "0", font=font(mono=True, size=18), fill=GREY)
d.text((bx + bw - 70, by + 16), "22 m/s", font=font(mono=True, size=18), fill=GREY)
d.text(
    (bx + bw + 40, by - 6), "10 m wind speed", font=font(mono=True, size=22), fill=DIM
)
# 02 nowcasting 2x2
header(
    x02,
    y1,
    "02",
    "NOWCASTING",
    "StormScope · GOES + MRMS (US) · MTG FCI (Europe) · 2 h ahead",
)
th = (R1 - BAND - G) // 2
tw2 = (w02 - G) // 2
tiles = [
    (
        P("panels/storm_rgb.png"),
        "US · StormScope GOES cloud phase RGB · +2 h · 2024-04-26 22Z",
        0,
        0,
    ),
    (
        P("panels/storm_raw.png"),
        "US · StormScope GOES IR + MRMS radar + GLM · +2 h",
        1,
        0,
    ),
    (
        P("panels/eu_conv.png"),
        "EU · MTG FCI Convective Storms RGB · +120 min · 2026-06-30 12Z",
        0,
        1,
    ),
    (P("panels/eu_cloud.png"), "EU · MTG FCI Cloud Type RGB · +120 min", 1, 1),
]
for p, lab, cx, cy in tiles:
    bx_, by_ = x02 + cx * (tw2 + G), y1 + BAND + cy * (th + G)
    place(p, (bx_, by_, tw2, th), cover=True)
    tag(bx_ + 14, by_ + th - 46, lab)
# 03 DA
header(
    x03,
    y1,
    "03",
    "DATA ASSIMILATION",
    "HealDA · NNJA observations · 2026-01-15 00Z ±3 h",
)
place(P("panels/obs_raw.png"), (x03, y1 + BAND, GL, GL), crop=True)
ly = y1 + BAND + GL + 22
items = [
    ("Satellite winds", "3.2 M", (26, 140, 255)),
    ("GPS radio occultation", "1.1 M", (255, 77, 190)),
    ("Conventional winds", "1.0 M", (255, 184, 38)),
    ("Temperature", "0.4 M", (38, 255, 166)),
]
colw = GL // 2
for i, (n, c, col) in enumerate(items):
    xx = x03 + (i % 2) * colw + 6
    yy = ly + (i // 2) * 40
    d.ellipse((xx, yy + 8, xx + 16, yy + 24), fill=col)
    d.text((xx + 30, yy + 2), n, font=font(mono=True, size=24), fill=WHITE)
    d.text((xx + 30 + 420, yy + 2), c, font=font(mono=True, size=24), fill=GREY)
d.text(
    (x03 + 6, ly + 82),
    "5.8 M observations in one 6 h window",
    font=font(mono=True, size=21),
    fill=DIM,
)
# ---- row 2
y2 = M + R1 + G
th2 = R2 - BAND
header(
    M,
    y2,
    "04",
    "AI-READY DATA SOURCES",
    "GOES · Himawari · ERA5 · GFS … analysis, forecast and observation sources behind one Python API",
)
gap = 12
xs = M
srcs = [
    (P("panels/goes.png"), "GOES-16 ABI · true colour + IR · 2024-04-26 22Z"),
    (P("panels/himawari.png"), "Himawari-9 AHI · true colour + IR · 2024-04-26 22Z"),
    (P("panels/era5_tcwv.png"), "ERA5 · total column water vapour · 2024-04-27 12Z"),
    (P("panels/gfs_jet.png"), "GFS · 250 hPa jet stream wind speed · 2024-04-27 00Z"),
]
for p, lab in srcs:
    place(p, (xs, y2 + BAND, th2, th2), crop=True)
    tag(xs + 4, y2 + R2 - 40, lab)
    xs += th2 + gap
# wordmark
tx = xs + 40
tw = W - M - tx
d.text((tx, y2 + 6), "NVIDIA", font=font(bold=True, size=30), fill=(118, 185, 0))
spaced(d, (tx, y2 + 52), "EARTH2STUDIO", font(bold=True, size=92), WHITE, 6)
d.text((tx, y2 + 190), "Next-generation AI weather", font=font(size=34), fill=GREY)
d.text((tx, y2 + 234), "and climate modeling.", font=font(size=34), fill=GREY)
import os

from PIL import ImageFont

OBL = ImageFont.truetype(os.path.join(MPL, "DejaVuSans-Oblique.ttf"), 30)
GREEN = (118, 185, 0)
# dictionary entry
SUB = (84, 122, 40)  # muted dark green accent
d.line((tx, y2 + 330, tx + 1000, y2 + 330), fill=(38, 44, 58), width=2)
hf = font(bold=True, size=48)
spaced(d, (tx, y2 + 372), "Earth-2", hf, (150, 160, 178), 3)
d.text((tx + 4, y2 + 446), "(urth-too)", font=font(mono=True, size=22), fill=DIM)
d.text(
    (tx + 220, y2 + 445),
    "noun",
    font=ImageFont.truetype(os.path.join(MPL, "DejaVuSans-Oblique.ttf"), 22),
    fill=SUB,
)
d.text((tx, y2 + 500), "1", font=font(bold=True, size=22), fill=SUB)
for i, l in enumerate(
    [
        "Open, production-ready AI models and tools",
        "for global forecasting and planetary resilience.",
    ]
):
    d.text((tx + 34, y2 + 496 + i * 36), l, font=font(size=24), fill=(150, 160, 178))
d.text(
    (tx, y2 + 600),
    "see also",
    font=ImageFont.truetype(os.path.join(MPL, "DejaVuSans-Oblique.ttf"), 20),
    fill=DIM,
)
d.text(
    (tx + 100, y2 + 602),
    "earth2studio, physicsnemo, earth-2 open models",
    font=font(mono=True, size=19),
    fill=DIM,
)
caps = [
    "models · SFNO · StormScope · HealDA",
    "data · ERA5 · GFS · GOES · MRMS · NNJA",
    "one Python API",
]
for i, c in enumerate(caps):
    d.text((tx, y2 + R2 - 118 + i * 38), c, font=font(mono=True, size=24), fill=DIM)
print(
    "wordmark width",
    d.textlength("EARTH2STUDIO", font=font(bold=True, size=92)) + 11 * 6,
)
print(
    "wordmark col width",
    tw,
    "right edge text",
    tx + d.textlength("Next-generation AI weather", font=font(size=34)),
)
canvas.save(P("hero.png"), optimize=True)
canvas.resize((W // 2, H // 2), Image.LANCZOS).save(P("hero_2700.png"), optimize=True)
a = np.asarray(canvas).astype(float) / 255
print(
    canvas.size,
    "mean lum %.3f" % a.mean(),
    "nonblack %.1f%%" % (100 * (a.mean(-1) > 0.03).mean()),
)
