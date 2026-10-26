#!/usr/bin/env python3
"""Draw Fig. 1, the workflow in three steps, as a vector figure.

The figure shows no corpus data. Its numbers come from a small worked example
that this module computes, so each number follows from what the figure shows.
It writes figures/fig1_workflow.pdf.

Step 1 reads a passage of the translation into a clause with its fields and its
two marks. The rule-based parser makes the clause, and the language model checks
it and adds the clauses that the parser missed.

Step 2 gives eight clauses a verdict in eight of the 701 candidate years. The
bits of each row follow from that row by Eq. 1. The tier follows from the type
of a clause, whatever its bits, so one supporting row carries 2.0 bits and one
decisive row 0.2 bits. The decisive rows all agree in one column, the year T.

Step 3 draws three clocks over 91 years. Mars agrees with many years, Jupiter
with one year in about 11.86 and Saturn with short runs about 29.46 years apart.
The three clocks meet at T alone. T leads on K and on the decisive bits I_D, so
it alone forms the strict set S (Eqs. 2 and 3). The supporting bits I_S use the
same scale as I_D, and they only order the years.

All parts are drawn on one axes in inches of the printed figure, so a length
in the code is a length on the page.
"""

from __future__ import annotations

import random
from math import cos, log2, pi, sin
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as Outline
from matplotlib.patches import (Arc, Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle)

import figure_style as fs

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "figures"

HEIGHT = 3.62                        # inches. The width is fs.FULL_WIDTH
LABEL, SMALL, HEAD = 7.5, 7.0, 9.0   # font sizes in points
PANEL_FILL = "#F2F3F5"
AGREE, CONTRADICT, UNTESTABLE = fs.GREEN, "#A8394F", "#D3D7DC"
CLAY, CLAY_EDGE, CLAY_INK = "#D9C4A6", "#9C7E5A", "#86684A"
SYMBOLS = "DejaVu Sans"              # the astronomical signs, from figure_style's fallback font
DASHED = (0, (2.5, 1.5))
SIGN_LINE, SIGN_RUN = 2, 3            # the highlight holds three signs of the third line
SIGN_MARGIN = 0.010                  # inches between those signs and the highlight

# Panel edges in inches, as (left, right), and the common bottom and top.
PANELS = [(0.03, 2.22), (2.39, 4.60), (4.77, 6.94)]
BOTTOM, TOP = 0.46, 3.22

# Step 2. Rows are clauses and columns are eight candidate years. Each verdict
# is a (agree), c (contradict) or u (untestable). The flag marks a decisive row.
VERDICTS = [
    ("aaaaaaca", False),
    ("cuucacuc", True),
    ("ccuaucuu", False),
    ("aacaauaa", True),
    ("acaauaaa", False),
    ("aucaacuc", True),
    ("uuauauuu", False),
    ("uuucauuu", True),
]
T_COLUMN = 4                         # the column of the year T
OUTLINED = 1                         # the row worked out in the panel

# Step 3. Years shown on the clock rows, and the year T among them.
YEARS, T = 90, 45


def row_bits(verdicts: str) -> tuple[int, int, float]:
    """C_j, A_j and u_j = log2(C_j / A_j) of one row (Eq. 1)."""
    testable = sum(v != "u" for v in verdicts)
    agreeing = verdicts.count("a")
    return testable, agreeing, log2(testable / agreeing)


def check_grid() -> None:
    """Check that the decisive rows share column T alone, and that a supporting
    row has more bits than a decisive row."""
    decisive = [v for v, d in VERDICTS if d]
    shared = [c for c in range(8) if all(v[c] == "a" for v in decisive)]
    assert shared == [T_COLUMN]
    high_support = max(row_bits(v)[2] for v, d in VERDICTS if not d)
    low_decisive = min(row_bits(v)[2] for v, d in VERDICTS if d)
    assert high_support > low_decisive


def clocks() -> dict[str, set[int]]:
    """The years that each clock agrees with. The three clocks meet at T alone."""
    jupiter = {T + round(11.86 * k) for k in range(-4, 5)}
    jupiter = {y for y in jupiter if 0 <= y <= YEARS}
    saturn = set()
    for k in range(-2, 3):
        centre = T + round(29.46 * k)
        for step in ((-1, 0, 1) if k % 2 == 0 else (0, 1)):
            if 0 <= centre + step <= YEARS:
                saturn.add(centre + step)
    draw = random.Random(7)
    mars = {y for y in range(YEARS + 1) if draw.random() < 0.36} | {T}
    mars -= (jupiter & saturn) - {T}
    assert mars & jupiter & saturn == {T}
    return {"Mars": mars, "Jupiter": jupiter, "Saturn": saturn}


def scores(clock_years: dict[str, set[int]]):
    """K, I_D and I_S of each shown year, and the strict set S (Eqs. 2 and 3)."""
    span = YEARS + 1
    bits = {name: log2(span / len(ys)) for name, ys in clock_years.items()}
    k = {y: sum(y in ys for ys in clock_years.values()) for y in range(span)}
    i_d = {y: sum(b for name, b in bits.items() if y in clock_years[name])
           for y in range(span)}
    draw = random.Random(11)         # five broad supporting clauses, such as the Moon by a star
    support = [{y for y in range(span) if draw.random() < p}
               for p in (0.9, 0.85, 0.8, 0.75, 0.7)]
    i_s = {y: sum(log2(span / len(s)) for s in support if y in s)
           for y in range(span)}
    strict = {y for y in range(span)
              if not any(k[z] >= k[y] and i_d[z] >= i_d[y]
                         and (k[z] > k[y] or i_d[z] > i_d[y]) for z in range(span))}
    assert strict == {T} and k[T] == 3
    return k, i_d, i_s, strict


# Drawing helpers. Every length is in inches.

def text(ax, x, y, s, size=LABEL, **kw):
    kw.setdefault("color", fs.INK)
    kw.setdefault("va", "center_baseline")
    return ax.text(x, y, s, fontsize=size, **kw)


def box(ax, x, y, w, h, face, edge="none", lw=0.0, radius=0.04, **kw):
    patch = FancyBboxPatch((x, y), w, h, facecolor=face, edgecolor=edge,
                           linewidth=lw,
                           boxstyle=f"round,pad=0,rounding_size={radius}", **kw)
    ax.add_patch(patch)
    return patch


def line(ax, xs, ys, **kw):
    kw.setdefault("color", fs.GREY)
    kw.setdefault("linewidth", 0.9)
    artist = Line2D(xs, ys, **kw)
    ax.add_line(artist)
    return artist


def arrow(ax, start, end, color=fs.GREY, lw=0.9, style="-|>", **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=7,
                                 color=color, linewidth=lw, shrinkA=0, shrinkB=0,
                                 **kw))


def header(ax, x, y, number, title):
    ax.add_patch(Circle((x + 0.09, y), 0.085, facecolor=fs.INK, edgecolor="none"))
    text(ax, x + 0.09, y, str(number), color="white", fontweight="bold",
         ha="center")
    text(ax, x + 0.22, y, title, size=HEAD, fontweight="bold")


def tablet(ax, x, y, w, h):
    """A broken clay tablet with rows of wedges.

    Returns the signs line by line. Each sign is (x0, y0, x1, y1, whole), its
    box in inches and whether it lies wholly inside the tablet's outline.
    """
    outline = [(0.04, 0.88), (0.14, 0.98), (0.55, 1.0), (0.86, 0.97), (0.97, 0.86),
               (0.98, 0.58), (0.90, 0.50), (0.83, 0.34), (0.70, 0.27), (0.64, 0.08),
               (0.52, 0.02), (0.18, 0.0), (0.05, 0.07), (0.0, 0.40)]
    corners = [(x + u * w, y + v * h) for u, v in outline]
    shape = Polygon(corners, closed=True, facecolor=CLAY, edgecolor=CLAY_EDGE,
                    linewidth=0.7, zorder=2)
    ax.add_patch(shape)
    inside = Outline(corners + [corners[0]], closed=True)
    draw = random.Random(3)
    stroke = 0.5 / 72 / 2             # half the width of a tail, in inches
    lines = []
    for r in range(6):
        top = y + h * (0.93 - r / 6 * 0.9)
        line(ax, [x, x + w], [top - h * 0.15 + 0.004] * 2, color=CLAY_EDGE,
             linewidth=0.35, zorder=3).set_clip_path(shape)
        signs = []
        u = 0.05 + draw.random() * 0.05
        while u < 0.95:
            cx, cy, s = x + u * w, top - h * 0.075, 0.026
            if draw.random() < 0.6:  # a horizontal wedge, head to the left
                head = [(cx, cy + s / 2), (cx, cy - s / 2), (cx + s * 0.8, cy)]
                tail = ([cx + s * 0.8, cx + s * 1.9], [cy, cy])
            else:                    # a vertical wedge, head at the top
                head = [(cx - s / 2, cy + s / 2), (cx + s / 2, cy + s / 2),
                        (cx, cy - s * 0.2)]
                tail = ([cx, cx], [cy - s * 0.2, cy - s * 1.0])
            patch = Polygon(head, closed=True, facecolor=CLAY_INK, edgecolor="none",
                            zorder=3)
            ax.add_patch(patch)
            patch.set_clip_path(shape)
            line(ax, *tail, color=CLAY_INK, linewidth=0.5, zorder=3).set_clip_path(shape)
            # The head has no edge. The tail is a stroke with projecting caps.
            x0 = min([p[0] for p in head] + [v - stroke for v in tail[0]])
            x1 = max([p[0] for p in head] + [v + stroke for v in tail[0]])
            y0 = min([p[1] for p in head] + [v - stroke for v in tail[1]])
            y1 = max([p[1] for p in head] + [v + stroke for v in tail[1]])
            whole = bool(inside.contains_points(
                [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]).all())
            signs.append((x0, y0, x1, y1, whole))
            u += 0.09 + draw.random() * 0.08
        lines.append(signs)
    return lines, inside


def apart(a, b) -> bool:
    """Whether two boxes (x0, y0, x1, y1, ...) share no point."""
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def highlight(lines, inside, margin=SIGN_MARGIN):
    """The box around a run of whole signs on one line, with an even margin.

    The run is the rightmost one whose box stays inside the tablet and cuts no
    other sign. The zoom to the translation then crosses little of the tablet.
    """
    row = lines[SIGN_LINE]
    others = [sign for signs in lines for sign in signs]
    for first in range(len(row) - SIGN_RUN, -1, -1):
        run = row[first:first + SIGN_RUN]
        if not all(sign[4] for sign in run):
            continue
        frame = (min(s[0] for s in run) - margin, min(s[1] for s in run) - margin,
                 max(s[2] for s in run) + margin, max(s[3] for s in run) + margin)
        corners = [(frame[0], frame[1]), (frame[0], frame[3]),
                   (frame[2], frame[1]), (frame[2], frame[3])]
        if not inside.contains_points(corners).all():
            continue
        if all(apart(frame, sign) for sign in others if sign not in run):
            return run, frame
    raise ValueError("no run of whole signs can be framed on this line")


def word_bars(ax, x0, x1, y, seed, height=0.034):
    """A line of text drawn as grey word bars."""
    draw = random.Random(seed)
    x = x0
    while x < x1 - 0.05:
        w = min(0.05 + draw.random() * 0.14, x1 - x)
        ax.add_patch(Rectangle((x, y - height / 2), w, height,
                               facecolor=fs.FAINT, edgecolor="none"))
        x += w + 0.035


def dial(ax, cx, cy, r):
    """A sky dial of twelve signs, with the Sun, the Moon and the three planets."""
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=fs.GREY,
                        linewidth=0.9))
    ax.add_patch(Circle((cx, cy), r * 0.88, facecolor="none", edgecolor=fs.FAINT,
                        linewidth=0.5))
    for i in range(12):
        a = i * pi / 6
        line(ax, [cx, cx + r * 0.88 * cos(a)], [cy, cy + r * 0.88 * sin(a)],
             color=fs.FAINT, linewidth=0.5)
    for segment, sign in ((1, "☉"), (3, "☽"), (6, "♂"), (8, "♃"),
                          (10, "♄")):
        a = (segment + 0.5) * pi / 6
        text(ax, cx + r * 0.6 * cos(a), cy + r * 0.6 * sin(a), sign, size=SMALL,
             family=SYMBOLS, ha="center", va="center")


def padlock(ax, cx, cy, color=fs.GREY):
    ax.add_patch(Arc((cx, cy + 0.025), 0.07, 0.09, theta1=0, theta2=180,
                     color=color, linewidth=1.1))
    for dx in (-0.035, 0.035):
        line(ax, [cx + dx, cx + dx], [cy + 0.005, cy + 0.025], color=color,
             linewidth=1.1)
    box(ax, cx - 0.055, cy - 0.065, 0.11, 0.075, color, radius=0.012)
    ax.add_patch(Circle((cx, cy - 0.028), 0.011, facecolor="white", edgecolor="none"))


def scroll(ax, cx, cy):
    """A written protocol, a page between two rolls."""
    ax.add_patch(Rectangle((cx - 0.065, cy - 0.095), 0.13, 0.19, facecolor="white",
                           edgecolor=fs.GREY, linewidth=0.6))
    for i in range(4):
        y = cy + 0.05 - i * 0.033
        line(ax, [cx - 0.04, cx + 0.04], [y, y], color=fs.FAINT, linewidth=0.7)
    for y in (cy + 0.095, cy - 0.095):
        box(ax, cx - 0.085, y - 0.017, 0.17, 0.034, fs.PALE, fs.GREY, 0.6,
            radius=0.017)


def pill(ax, cx, cy, label, width):
    box(ax, cx - width / 2, cy - 0.075, width, 0.15, "white", fs.INK, 0.8,
        radius=0.075)
    text(ax, cx, cy, label, ha="center")


# The three panels.

def panel_read(ax, x0, x1):
    header(ax, x0 + 0.08, TOP - 0.17, 1, "Read")
    # A broken tablet, and a passage of its translation.
    lines, inside = tablet(ax, x0 + 0.09, 2.18, 0.62, 0.70)
    run, (bx0, by0, bx1, by1) = highlight(lines, inside)
    for sign in run:                 # the box holds its signs with the full margin
        assert (bx0 + SIGN_MARGIN <= sign[0] + 1e-9 and sign[2] <= bx1 - SIGN_MARGIN + 1e-9
                and by0 + SIGN_MARGIN <= sign[1] + 1e-9
                and sign[3] <= by1 - SIGN_MARGIN + 1e-9)
    assert all(apart((bx0, by0, bx1, by1), sign)
               for signs in lines for sign in signs if sign not in run)
    px0, px1 = x0 + 0.89, x1 - 0.08
    cx = (px0 + px1) / 2
    # The zoom to the translation starts at the box.
    ax.add_patch(Polygon([(bx1, by1), (bx1, by0), (px0 - 0.03, 2.28),
                          (px0 - 0.03, 2.92)], closed=True, facecolor=fs.PALE,
                         edgecolor="none", zorder=1))
    for corner, end in (((bx1, by1), (px0 - 0.03, 2.92)),
                        ((bx1, by0), (px0 - 0.03, 2.28))):
        line(ax, [corner[0], end[0]], [corner[1], end[1]], color=fs.FAINT,
             linewidth=0.5, zorder=4)
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, facecolor="none",
                           edgecolor=fs.INK, linewidth=0.6, zorder=5))
    for i, y in enumerate((2.88, 2.76)):
        word_bars(ax, px0, px1, y, seed=20 + i)
    box(ax, px0, 2.535, px1 - px0, 0.15, "white", fs.INK, 0.7, radius=0.02)
    text(ax, cx, 2.61, "Mars was in Sagittarius", ha="center")
    for i, y in enumerate((2.45, 2.33)):
        word_bars(ax, px0, px1, y, seed=30 + i)
    text(ax, cx, 2.20, "passage", size=SMALL, color=fs.GREY, ha="center")
    # The parser makes the clause, and the language model checks it and adds
    # the clauses that the parser missed.
    arrow(ax, (cx, 2.12), (cx, 1.97))
    box(ax, px0, 1.73, px1 - px0, 0.24, "white", fs.INK, 0.8, radius=0.03)
    text(ax, cx, 1.85, "rule-based parser", ha="center")
    arrow(ax, (cx, 1.73), (cx, 1.58))
    top, bottom = 1.58, 0.66
    box(ax, px0, bottom, px1 - px0, top - bottom, "white", fs.INK, 0.8, radius=0.03)
    line(ax, [px0, px1], [top - 0.19] * 2, color=fs.INK, linewidth=0.6)
    text(ax, cx, top - 0.095, "clause", fontweight="bold", ha="center")
    fx = px0 + 0.08
    text(ax, fx, 1.25, "body: Mars")
    text(ax, fx, 1.10, "sign: Sagittarius")
    text(ax, fx, 0.95, "marks:")
    for y, mark in ((0.95, "restored"), (0.80, "unobserved")):
        mx = fx + 0.37
        ax.add_patch(Rectangle((mx, y - 0.038), 0.075, 0.075, facecolor="white",
                               edgecolor=fs.INK, linewidth=0.6))
        text(ax, mx + 0.11, y, mark)
    lm_x0, lm_x1 = x0 + 0.12, x0 + 0.70
    lmc = (lm_x0 + lm_x1) / 2
    box(ax, lm_x0, 1.69, lm_x1 - lm_x0, 0.32, "white", fs.GREY, 0.8, radius=0.03,
        linestyle=DASHED)
    # One two-line label, so that no font's metrics can make the lines overlap.
    text(ax, lmc, 1.85, "language\nmodel", ha="center", va="center", linespacing=1.25)
    line(ax, [lmc, lmc], [1.69, 1.18], linestyle=DASHED)
    arrow(ax, (lmc, 1.18), (px0, 1.18), linestyle=DASHED)
    text(ax, lmc + 0.06, 1.37, "checks,", size=SMALL, color=fs.GREY)
    text(ax, lmc + 0.06, 1.25, "adds", size=SMALL, color=fs.GREY)


def panel_compute(ax, x0, x1):
    header(ax, x0 + 0.08, TOP - 0.17, 2, "Compute")
    check_grid()
    c, a, u = row_bits(VERDICTS[OUTLINED][0])
    text(ax, x0 + 0.12, 2.89, r"$u_j=\log_2(C_j/A_j)$", size=8.0)
    text(ax, x0 + 0.12, 2.735, r"$C_j$ testable years, $A_j$ agreeing years",
         size=SMALL, color=fs.GREY)
    text(ax, x0 + 0.12, 2.585, rf"outlined row: $\log_2({c}/{a})={u:.1f}$",
         size=SMALL)
    # The verdict grid, eight clauses by eight candidate years.
    pitch, cell = 0.118, 0.096
    gap = pitch - cell
    gx0, gtop = x0 + 0.84, 2.33
    gx1 = gx0 + 8 * pitch - gap
    gbottom = gtop - 8 * pitch + gap
    bx = x1 - 0.20
    xt = gx0 + T_COLUMN * pitch
    text(ax, (gx0 + gx1) / 2, gtop + 0.10, "verdicts", ha="center")
    text(ax, bx, gtop + 0.10, "bits $u_j$", ha="center")
    colours = {"a": AGREE, "c": CONTRADICT, "u": UNTESTABLE}
    for r, (verdicts, decisive) in enumerate(VERDICTS):
        y = gtop - (r + 1) * pitch + gap
        for col, v in enumerate(verdicts):
            ax.add_patch(Rectangle((gx0 + col * pitch, y), cell, cell,
                                   facecolor=colours[v], edgecolor="none"))
        text(ax, bx, y + cell / 2, f"{row_bits(verdicts)[2]:.1f}", fontweight="bold",
             color=fs.ORANGE if decisive else fs.BLUE, ha="center")
        if r == OUTLINED:
            box(ax, gx0 - 0.03, y - 0.022, bx + 0.16 - gx0, cell + 0.044, "none",
                fs.INK, 0.7, radius=0.02, zorder=3)
    ax.add_patch(Rectangle((xt - gap / 2, gbottom - gap / 2), cell + gap,
                           gtop - gbottom + gap, facecolor="none",
                           edgecolor=fs.ORANGE, linewidth=1.1, zorder=4))
    text(ax, xt + cell / 2, gbottom - 0.10, "T", color=fs.ORANGE, fontweight="bold",
         ha="center")
    arrow(ax, (gx0, gbottom - 0.21), (gx1, gbottom - 0.21), style="<|-|>",
          color=fs.INK, lw=0.7)
    text(ax, (gx0 + gx1) / 2, gbottom - 0.34, "701 candidate years", ha="center")
    # The astronomical tools that give the verdicts.
    dial(ax, x0 + 0.40, 1.93, 0.27)
    arrow(ax, (x0 + 0.70, 1.93), (gx0 - 0.06, 1.93))
    text(ax, x0 + 0.40, 1.52, "astronomical", ha="center")
    text(ax, x0 + 0.40, 1.39, "tools (MCP)", ha="center")
    # Legend, the verdicts on one row and the tiers on the next.
    ly, x = 0.83, x0 + 0.14
    for colour, word in ((AGREE, "agree"), (CONTRADICT, "contradict"),
                         (UNTESTABLE, "untestable")):
        ax.add_patch(Rectangle((x, ly - 0.045), 0.09, 0.09, facecolor=colour,
                               edgecolor="none"))
        text(ax, x + 0.13, ly, word)
        x += 0.62 if word == "agree" else 0.68
    x = x0 + 0.14
    for colour, word in ((fs.ORANGE, "decisive"), (fs.BLUE, "supporting")):
        line(ax, [x, x + 0.09], [ly - 0.17] * 2, color=colour, linewidth=2.6,
             solid_capstyle="butt")
        text(ax, x + 0.13, ly - 0.17, word)
        x += 0.62


def panel_decide(ax, x0, x1, clock_years):
    header(ax, x0 + 0.08, TOP - 0.17, 3, "Decide")
    padlock(ax, x0 + 0.20, 2.82)
    text(ax, x0 + 0.34, 2.87, "published dates withheld", size=SMALL)
    text(ax, x0 + 0.34, 2.75, "until results are final", size=SMALL)
    k, i_d, i_s, strict = scores(clock_years)
    lx = x0 + 0.70                   # right edge of the row labels
    tx0, tx1 = x0 + 0.78, x1 - 0.10

    def xof(year):
        return tx0 + (tx1 - tx0) * year / YEARS

    xt = xof(T)
    ax.add_patch(Rectangle((xt - 0.025, 1.07), 0.05, 2.50 - 1.07,
                           facecolor=fs.ORANGE, alpha=0.22, edgecolor="none"))
    text(ax, lx, 2.57, "clocks", size=SMALL, color=fs.GREY, ha="right")
    text(ax, xt + 0.05, 2.57, "strict set $S$", color=fs.ORANGE, fontweight="bold")
    for name, base in (("Mars", 2.33), ("Jupiter", 2.12), ("Saturn", 1.91)):
        text(ax, lx, base + 0.05, name, ha="right")
        line(ax, [tx0, tx1], [base, base], color=fs.FAINT, linewidth=0.8)
        for year in sorted(clock_years[name]):
            on_t = year in strict
            line(ax, [xof(year)] * 2, [base, base + 0.13],
                 color=fs.ORANGE if on_t else fs.GREY,
                 linewidth=1.2 if on_t else 0.6, solid_capstyle="butt")
    text(ax, xt + 0.05, 1.80, f"$K$ = {k[T]}", color=fs.ORANGE, fontweight="bold")
    top = max(i_d.values())
    for (first, second), base, values, colour in (
            (("decisive", "bits $I_D$"), 1.48, i_d, fs.GREY),
            (("supporting", "bits $I_S$"), 1.15, i_s, fs.BLUE)):
        text(ax, lx, base + 0.14, first, ha="right")
        text(ax, lx, base + 0.02, second, ha="right")
        line(ax, [tx0, tx1], [base, base], color=fs.FAINT, linewidth=0.8)
        for year, v in values.items():
            if v > 0:
                on_t = year in strict and colour == fs.GREY
                ax.add_patch(Rectangle((xof(year) - 0.006, base), 0.012,
                                       0.22 * v / top,
                                       facecolor=fs.ORANGE if on_t else colour,
                                       edgecolor="none"))
    text(ax, xt, 0.98, "T", color=fs.ORANGE, fontweight="bold", ha="center")
    mid = (x0 + x1) / 2
    for dx, word, width in ((-0.60, "commit", 0.52), (0.0, "rank", 0.40),
                            (0.58, "decline", 0.54)):
        pill(ax, mid + dx, 0.68, word, width)


def main() -> Path:
    fs.use()
    width = fs.FULL_WIDTH
    fig = plt.figure(figsize=(width, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, HEIGHT)
    ax.axis("off")

    for x0, x1 in PANELS:
        box(ax, x0, BOTTOM, x1 - x0, TOP - BOTTOM, PANEL_FILL, radius=0.07, zorder=0)
    for (_, left), (right, _) in zip(PANELS, PANELS[1:]):
        mid, y = (left + right) / 2, (BOTTOM + TOP) / 2
        ax.add_patch(Polygon([(mid - 0.055, y + 0.14), (mid + 0.065, y),
                              (mid - 0.055, y - 0.14)], closed=True,
                             facecolor=fs.GREY, edgecolor="none"))

    panel_read(ax, *PANELS[0])
    panel_compute(ax, *PANELS[1])
    panel_decide(ax, *PANELS[2], clocks())

    # The written protocol above the panels, with arrows to steps 1 and 3, and a
    # note on the held-out test (SI Appendix, section S8).
    cy = 3.41
    title = text(ax, 0, cy + 0.065, "written protocol", fontweight="bold")
    note = text(ax, 0, cy - 0.065,
                "held-out test: rules were set on two volumes and tested on the third",
                size=SMALL, color=fs.GREY)
    renderer = fig.canvas.get_renderer()
    span = max(t.get_window_extent(renderer).width for t in (title, note)) / fig.dpi
    left = (width - span - 0.17) / 2
    scroll(ax, left + 0.085, cy)
    for t in (title, note):
        t.set_x(left + 0.20)
    right = left + 0.20 + span
    dotted = dict(color=fs.GREY, linewidth=0.8, linestyle=(0, (1, 1.5)))
    for start, end in ((left - 0.05, sum(PANELS[0]) / 2),
                       (right + 0.06, sum(PANELS[2]) / 2)):
        line(ax, [start, end], [cy, cy], **dotted)
        arrow(ax, (end, cy), (end, TOP + 0.01), linestyle=(0, (1, 1.5)))

    # The evidence kept from passage to decision.
    arrow(ax, (PANELS[0][0] + 0.05, 0.33), (PANELS[2][1] - 0.05, 0.33),
          color=fs.FAINT, lw=1.4)
    text(ax, width / 2, 0.15,
         "Evidence kept at each step: passage, clause, verdict, decision",
         ha="center")
    return fs.save(fig, OUT / "fig1_workflow")


if __name__ == "__main__":
    print(main())
