#!/usr/bin/env python3
"""Generate six vector panels, one legend header, and two shared axis titles.

Run from the repository root with the hsmc environment:

    MPLBACKEND=Agg python -B plotting/plot_two_stage_total_cost_comparison_panels.py

Outputs in plotting/output/ (default --output-dir):
    two_stage_total_cost_comparison_T{24,37,50}_cap{100,1000}.pdf
    two_stage_total_cost_comparison_header.pdf
    two_stage_total_cost_comparison_xlabel.pdf
    two_stage_total_cost_comparison_ylabel.pdf
and matching 200 dpi PNG previews. Each panel is 2.15 x 2.15 inches; the
shared legend header is 6.45 x 0.62 inches. Match panel widths in LaTeX.
Axis titles are omitted from the panels; numeric tick labels are retained.
The x-title strip is 6.45 x 0.28 inches; the y-title strip is 0.28 x 4.3
inches, matching the width of three panels and height of two panel rows.

The cap-100 and cap-1000 panels for the same T have identical x/y limits.
Limits differ between T values. Source archives and original plots are
unchanged. T=24/T=37 use the July archives in the original paper figures;
T=50 uses the October 22 archives in the updated paper figure.

The panels are checked against two_stage_total_cost_comparison_T{24,37,50}.pdf,
read from the output directory: run plot_two_stage_total_cost_comparison.py
(T24, T37) and plot_two_stage_total_cost_comparison_T50.py (T50) first.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullFormatter
import numpy as np

import plot_two_stage_total_cost_comparison as reference
import plot_two_stage_total_cost_comparison_T50 as t50

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
STAGES = (24, 37, 50)
CAPS = (100, 1000)
WIDTH, HEIGHT = 2.15, 2.15
HEADER_WIDTH, HEADER_HEIGHT = WIDTH * 3, 0.62


def load_sources(data_dir: Path):
    # Pin the archives rather than choosing newest folders or reading plot pixels.
    specs = (
        reference.RunSpec("run_20260714-144004_2s_paper", 24, 100),
        reference.RunSpec("run_20260714-143914_2s_paper", 24, 1000),
        reference.RunSpec("run_20260714-151644_2s_paper", 37, 100),
        reference.RunSpec("run_20260714-151715_2s_paper", 37, 1000),
        reference.RunSpec("run_20261022-161115_", 50, 100),
        reference.RunSpec("run_20261022-161132_", 50, 1000),
    )
    runs = [(t50.load_run if spec.stages == 50 else reference.load_run)(data_dir, spec)
            for spec in specs]
    # The paper source figures display these two particle counts only.
    rows = [row for row in reference.build_rows(runs)
            if row["n_particles"] in reference.DISPLAY_PARTICLES]
    assert len(rows) == 180
    return runs, rows


def source_pdf_geometry(path: Path):
    """Read vector paths and tick positions in these Matplotlib source PDFs.

    This checks the actual existing artifacts, independently of their scripts.
    It is deliberately limited to the PDF structure emitted by Matplotlib here.
    No source PDF is cropped, rasterized, or used to draw the new figure.
    """
    content = None
    for stream in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", path.read_bytes(), re.S):
        try:
            decoded = zlib.decompress(stream).decode("latin1")
        except zlib.error:
            continue
        if "BT" in decoded and " re W" in decoded:
            content = decoded
            break
    if content is None:
        raise ValueError(f"Cannot find vector content in {path}")
    number = r"-?\d+(?:\.\d+)?"
    boxes = list(dict.fromkeys(re.findall(
        rf"q ({number}) ({number}) ({number}) ({number}) re W", content,
    )))
    boxes = np.array(boxes, dtype=float)
    assert boxes.shape == (2, 4), (path, boxes)
    x0, y0, width, height = boxes[0]
    xticks, yticks, bars = [], [], []
    segment = re.compile(rf"({number}) ({number}) m\n({number}) ({number}) l\n\nS\n")
    for match in segment.finditer(content):
        x1, y1, x2, y2 = map(float, match.groups())
        if x1 == x2:
            bars.append((x1, y1, y2))
        suffix = content[match.end():].split(" re W", 1)[0][:450]
        label = re.search(r"\[ \(([\d.]+)\) \] TJ", suffix)
        if label:
            if (x1 == x2 and x0 <= x1 <= x0 + width
                    and abs(y1 - y0) < 1e-5 and abs(y2 - y0 - height) < 1e-5):
                xticks.append((float(label[1]), x1))
            if (y1 == y2 and abs(x1 - x0) < 1e-5
                    and abs(x2 - x0 - width) < 1e-5):
                yticks.append((float(label[1]), y1))
    curve = re.compile(rf"^({number} {number} m\n(?:{number} {number} l\n){{4}})\nS$", re.M)
    curves = np.array([
        [list(map(float, line.split()[:2])) for line in match[1].splitlines()]
        for match in curve.finditer(content)
    ])
    assert curves.shape == (12, 5, 2), (path, curves.shape)
    return boxes, np.array(xticks), np.array(yticks), curves, np.array(bars)


def verify_source_pdfs(runs, rows, source_dir: Path):
    """Verify every displayed mean, cost, and CI against the existing PDFs."""
    for stages in STAGES:
        path = source_dir / f"two_stage_total_cost_comparison_T{stages}.pdf"
        boxes, xticks, yticks, curves, bars = source_pdf_geometry(path)
        multiplier = 1e8 if stages == 50 else 1e6
        xfit = np.polyfit(np.log10(xticks[:, 0] * multiplier), xticks[:, 1], 1)
        yfit = np.polyfit(yticks[:, 0], yticks[:, 1], 1)
        np.testing.assert_allclose(np.polyval(xfit, np.log10(xticks[:, 0] * multiplier)),
                                   xticks[:, 1], atol=1e-6, rtol=0)
        np.testing.assert_allclose(np.polyval(yfit, yticks[:, 0]), yticks[:, 1], atol=1e-6, rtol=0)
        for col, cap in enumerate(CAPS):
            run = next(r for r in runs if (r.spec.stages, r.spec.coarse_cap) == (stages, cap))
            for ni, n in enumerate(reference.DISPLAY_PARTICLES):
                for mi, method in enumerate(reference.METHODS):
                    points = [r for r in rows if r["run"] == run.spec.directory
                              and r["n_particles"] == n and r["method"] == method.key]
                    expected = np.array([
                        [np.polyval(xfit, np.log10(p["total_cost"])) + boxes[col, 0] - boxes[0, 0],
                         np.polyval(yfit, p["mean_error"])] for p in points
                    ])
                    np.testing.assert_allclose(curves[col * 6 + ni * 3 + mi], expected,
                                               atol=2e-6, rtol=0, err_msg=str(path))
                    for point, row in zip(expected, points, strict=True):
                        expected_bar = (point[0], point[1] - row["ci95_half_width"] * yfit[0],
                                        point[1] + row["ci95_half_width"] * yfit[0])
                        assert np.any(np.all(np.isclose(bars, expected_bar, atol=2e-6, rtol=0), axis=1)), path
        print(f"Matched all 60 means, costs, and confidence intervals in {path.name}.")


def create_panel(run, points, limits):
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    left, bottom, right, top = 0.35, 0.30, 0.10, 0.34
    axis = fig.add_axes((left / WIDTH, bottom / HEIGHT,
                         (WIDTH - left - right) / WIDTH,
                         (HEIGHT - bottom - top) / HEIGHT))
    curves = []
    for n in reference.DISPLAY_PARTICLES:
        for method in reference.METHODS:
            selected = [point for point in points if point["n_particles"] == n
                        and point["method"] == method.key]
            x = [point["total_cost"] for point in selected]
            y = [point["mean_error"] for point in selected]
            error = axis.errorbar(x, y, yerr=[point["ci95_half_width"] for point in selected],
                                  fmt="none", color=run.colors[n], capsize=1.8,
                                  elinewidth=0.7, capthick=0.7)
            line, = axis.plot(x, y, color=run.colors[n], marker=method.marker,
                              linestyle=method.linestyle, linewidth=1.05,
                              markersize=3.0, markeredgewidth=0.6)
            curves.append((selected, line, error))
    axis.set(xscale="log", xlim=limits[0], ylim=limits[1])
    axis.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
    axis.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5), numticks=30))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_locator(MaxNLocator(nbins=7, steps=(1, 2, 5, 10)))
    axis.tick_params(which="major", labelsize=8.5, width=0.45, length=2.3, pad=2)
    axis.tick_params(axis="x", which="minor", width=0.35, length=1.3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#777777")
    axis.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.4)
    annotations = [
        axis.text(0.97, 0.96, rf"$M_c^{{\max}}={run.spec.coarse_cap}$",
                  transform=axis.transAxes, ha="right", va="top", fontsize=8.5,
                  bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1)),
        fig.text((left + WIDTH - right) / (2 * WIDTH), 2.01 / HEIGHT,
                 rf"$T={run.spec.stages}$", ha="center", va="center", fontsize=11),
    ]
    return fig, axis, curves, annotations


def create_header(colors):
    fig = plt.figure(figsize=(HEADER_WIDTH, HEADER_HEIGHT))
    legends = [
        fig.legend(handles=[
            Line2D([], [], color=colors[n], linewidth=1.3, label=rf"$N={n}$")
            for n in reference.DISPLAY_PARTICLES
        ], loc="center", bbox_to_anchor=(0.5, 0.45 / HEADER_HEIGHT), frameon=False,
           ncol=2, fontsize=8.5, handlelength=2.0, columnspacing=2.5),
        fig.legend(handles=[
            Line2D([], [], color="black", marker=method.marker, linestyle=method.linestyle,
                   linewidth=1.05, markersize=3.0, label=method.label)
            for method in reversed(reference.METHODS)
        ], loc="center", bbox_to_anchor=(0.5, 0.17 / HEADER_HEIGHT), frameon=False,
           ncol=3, fontsize=8.5, handlelength=2.2, columnspacing=1.4, handletextpad=0.55),
    ]
    return fig, legends


def create_axis_title(direction):
    if direction == "x":
        fig = plt.figure(figsize=(HEADER_WIDTH, 0.28))
        title = fig.text(0.5, 0.5, "Total cost", ha="center", va="center", fontsize=9)
    else:
        fig = plt.figure(figsize=(0.28, HEIGHT * 2))
        title = fig.text(0.5, 0.5, r"Mean marginal $W_1$ error", rotation=90,
                         ha="center", va="center", fontsize=9)
    return fig, [title]


def validate_text(fig, artists):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]), "Text overlap"
    assert all(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1
               for box in boxes), "Text outside page"


def validate_panel(fig, axis, curves, annotations):
    assert len(curves) == 6 and not fig.legends and axis.get_legend() is None
    assert axis.get_xscale() == "log" and axis.get_yscale() == "linear"
    assert not axis.get_xlabel() and not axis.get_ylabel()
    assert not any("Total cost" in text.get_text() or "Mean marginal" in text.get_text()
                   for text in fig.texts + axis.texts)
    visible_labels = []
    for positions, labels, limits in (
        (axis.get_xticks(), axis.get_xticklabels(), axis.get_xlim()),
        (axis.get_yticks(), axis.get_yticklabels(), axis.get_ylim()),
    ):
        visible_labels.extend(label for pos, label in zip(positions, labels)
                              if limits[0] <= pos <= limits[1] and label.get_visible())
    validate_text(fig, visible_labels + annotations)
    for points, line, error in curves:
        np.testing.assert_array_equal(line.get_xdata(), [point["total_cost"] for point in points])
        np.testing.assert_array_equal(line.get_ydata(), [point["mean_error"] for point in points])
        method = next(m for m in reference.METHODS if m.key == points[0]["method"])
        assert line.get_marker() == method.marker and line.get_linestyle() == method.linestyle
        for point, segment in zip(points, error.lines[2][0].get_segments(), strict=True):
            lo = point["mean_error"] - point["ci95_half_width"]
            hi = point["mean_error"] + point["ci95_half_width"]
            np.testing.assert_array_equal(segment, [[point["total_cost"], lo], [point["total_cost"], hi]])
            assert axis.get_xlim()[0] < point["total_cost"] < axis.get_xlim()[1]
            assert axis.get_ylim()[0] < lo < hi < axis.get_ylim()[1]


def save_figure(fig, output, expected_curves):
    fig.savefig(output.with_suffix(".pdf"), metadata={"Title": output.name})
    fig.savefig(output.with_suffix(".png"), dpi=200)
    pdf = output.with_suffix(".pdf").read_bytes()
    assert b"/Subtype /Image" not in pdf, "Unexpected raster content"
    media = re.search(rb"/MediaBox\s*\[([^]]+)\]", pdf)[1].split()
    width, height = fig.get_size_inches()
    np.testing.assert_allclose(list(map(float, media)), [0, 0, width * 72, height * 72], atol=1e-6)
    number = rb"-?\d+(?:\.\d+)?"
    curve = rb"^" + number + rb" " + number + rb" m\n(?:" + number + rb" " + number + rb" l\n){4}\nS$"
    count = 0
    for stream in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", pdf, re.S):
        try:
            content = zlib.decompress(stream)
        except zlib.error:
            continue
        if b"BT" in content:
            count += len(re.findall(curve, content, re.M))
    assert count == expected_curves, (output, count)
    plt.close(fig)
    print(f"Wrote {output.with_suffix('.pdf')} and .png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=SOURCE_DIR / "output")
    args = parser.parse_args()
    # The source PDFs are outputs of the two scripts below, in the same output directory.
    source_pdfs = [args.output_dir / f"two_stage_total_cost_comparison_T{t}.pdf" for t in STAGES]
    missing = [str(path) for path in source_pdfs if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing source PDF(s): " + ", ".join(missing) + "\n"
            "Run plotting/plot_two_stage_total_cost_comparison.py (writes T24, T37) and "
            "plotting/plot_two_stage_total_cost_comparison_T50.py (writes T50) first, "
            "with the same --output-dir.")
    # Protect every existing original plot and script while allowing repeat runs.
    output_names = {f"two_stage_total_cost_comparison_T{t}_cap{c}.pdf" for t in STAGES for c in CAPS}
    output_names.update(f"two_stage_total_cost_comparison_{part}.pdf"
                        for part in ("header", "xlabel", "ylabel"))
    protected = {path: hashlib.sha256(path.read_bytes()).digest() for path in SOURCE_DIR.iterdir()
                 if path.suffix in (".py", ".pdf") and path.name not in output_names}
    protected.update({path: hashlib.sha256(path.read_bytes()).digest() for path in source_pdfs})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "axes.formatter.useoffset": False}):
        runs, rows = load_sources(args.data_dir)
        verify_source_pdfs(runs, rows, args.output_dir)
        for stages in STAGES:
            pair_rows = [row for row in rows if row["stages"] == stages]
            limits = reference.common_limits(pair_rows)
            pair_limits = []
            for cap in CAPS:
                run = next(run for run in runs if (run.spec.stages, run.spec.coarse_cap) == (stages, cap))
                points = [row for row in pair_rows if row["coarse_cap"] == cap]
                fig, axis, curves, annotations = create_panel(run, points, limits)
                validate_panel(fig, axis, curves, annotations)
                pair_limits.append((axis.get_xlim(), axis.get_ylim()))
                output = args.output_dir / f"two_stage_total_cost_comparison_T{stages}_cap{cap}"
                save_figure(fig, output, expected_curves=6)
            assert pair_limits[0] == pair_limits[1]
        header, legends = create_header(runs[0].colors)
        validate_text(header, legends)
        save_figure(header, args.output_dir / "two_stage_total_cost_comparison_header", expected_curves=0)
        for direction in ("x", "y"):
            title_figure, titles = create_axis_title(direction)
            validate_text(title_figure, titles)
            save_figure(title_figure, args.output_dir / f"two_stage_total_cost_comparison_{direction}label",
                        expected_curves=0)
    assert all(hashlib.sha256(path.read_bytes()).digest() == before for path, before in protected.items())
    print("Verified 180 points and confidence intervals, per-T matching axes, "
          "and separate legends and axis titles. Wrote nine vector PDFs.")


if __name__ == "__main__":
    main()
