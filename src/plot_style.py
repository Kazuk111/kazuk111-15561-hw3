"""Shared matplotlib style + helpers so every figure looks like one system."""
from __future__ import annotations

import io
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from load_data import OUT_FIG, ensure_output_dirs  # noqa: E402

# Categorical slots, used in this fixed order (never cycled past 8).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#b5b3ab"      # de-emphasised marks ("everything else")
# Single-hue sequential ramp (blue, light -> dark) for magnitude.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
# Diverging blue <-> red with a neutral gray midpoint.
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c23a39"])

DPI = 200


def apply_style() -> None:
    """Set the shared matplotlib rcParams for every figure.

    Args:
        None.

    Returns:
        None. Updates ``matplotlib.rcParams`` globally.
    """
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 22,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "text.color": INK,
    })


def subtitle(ax, text: str) -> None:
    """Draw a grey one-line subtitle above the axes.

    Args:
        ax (matplotlib.axes.Axes): Axes whose title gets a subtitle.
        text (str): Subtitle text.

    Returns:
        None.
    """
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9.5,
            color=INK_2, va="bottom", ha="left")


def footnote(fig, text: str) -> None:
    """Add a small grey footnote at the bottom left of a figure.

    Args:
        fig (matplotlib.figure.Figure): Figure to annotate.
        text (str): Footnote text.

    Returns:
        None.
    """
    fig.text(0.01, 0.005, text, fontsize=7.5, color=MUTED, ha="left",
             va="bottom", wrap=True)


def save(fig, name: str) -> str:
    """Save a figure as a PNG in the figures folder and close it.

    Renders to memory first, then writes the file, retrying up to
    four times with growing waits (OneDrive folders can time out).

    Args:
        fig (matplotlib.figure.Figure): Figure to save and close.
        name (str): File name without the ``.png`` extension.

    Returns:
        str: Path of the PNG written.

    Raises:
        OSError: If the write still fails on the fourth attempt.
    """
    ensure_output_dirs()
    path = OUT_FIG / f"{name}.png"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight",
                pad_inches=0.15)
    plt.close(fig)
    for attempt in range(4):
        try:
            path.write_bytes(buf.getvalue())
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return str(path)


def money(x: float, _pos=None) -> str:
    """Format a USD amount compactly for axis ticks.

    Args:
        x (float): Amount in USD.
        _pos (Any): Tick position; unused, required by matplotlib.

    Returns:
        str: Text such as "$1.2M" (>= 1e6), "$35k" (>= 1e3) or "$7".
    """
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:.0f}k"
    return f"${x:.0f}"


def place_labels(ax, xs, ys, labels, fontsize=8, color=INK_2, avoid_pts=True,
                 obstacles=()):
    """Place point labels greedily without overlaps.

    Tries several offsets (in points) around each point and keeps the
    first that avoids placed labels and, optionally, other markers. If
    none is free, the first offset is used.

    Args:
        ax (matplotlib.axes.Axes): Axes holding the points.
        xs (Sequence[float]): X data coordinates of the points.
        ys (Sequence[float]): Y data coordinates of the points.
        labels (Sequence[str]): Label for each point.
        fontsize (float): Label font size. Default 8.
        color (str): Label color. Default ``INK_2``.
        avoid_pts (bool): Also avoid other points. Default True.
        obstacles (Iterable[matplotlib.text.Text]): Existing texts that
            labels must not overlap.

    Returns:
        None. Adds annotations to ``ax``.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    placed = [a.get_window_extent(renderer) for a in obstacles]
    offsets = [(6, 3), (6, -11), (-6, 3), (-6, -11), (0, 8), (0, -16),
               (12, -4), (-12, -4), (14, 10), (-14, 10), (14, -18), (-14, -18),
               (0, 20), (0, -26), (24, -4), (-24, -4)]
    for x, y, lab in zip(xs, ys, labels):
        chosen = None
        for dx, dy in offsets:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            t = ax.annotate(lab, (x, y), xytext=(dx, dy),
                            textcoords="offset points", fontsize=fontsize,
                            color=color, ha=ha, va="bottom")
            bb = t.get_window_extent(renderer).expanded(1.04, 1.1)
            hit = any(bb.overlaps(b) for b in placed)
            if avoid_pts and not hit:
                hit = any(bb.contains(qx, qy) for qx, qy in pts)
            if not hit:
                chosen = t
                placed.append(bb)
                break
            t.remove()
        if chosen is None:  # nothing free: fall back to the first offset
            t = ax.annotate(lab, (x, y), xytext=offsets[0],
                            textcoords="offset points", fontsize=fontsize,
                            color=color, ha="left", va="bottom")
            placed.append(t.get_window_extent(renderer))
