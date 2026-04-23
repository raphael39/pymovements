# Copyright (c) 2026 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Classical vertical fixation-correction algorithms.

All functions share the same low-level signature::

    func(fixation_xy, midlines, **kwargs) -> NDArray[float64]

where

* ``fixation_xy`` : shape ``[N, 2]`` — ``(x, y)`` per fixation, not modified in place.
* ``midlines``    : sequence of float — y-coordinate of each text line, top-to-bottom.
* return          : shape ``[N]`` — corrected y-value per fixation (one of the midline values).

``warp`` and ``compare`` additionally require ``word_centers`` instead of
``midlines``. ``slice`` additionally requires a ``line_height`` argument.

Adapted from the implementation in
https://github.com/Gittingthehubbing/DIST-Dual_Input_Stream_Transformer/blob/master/classic_correction_algos.py
(itself adapted from https://github.com/jwcarr/eyekit/blob/master/eyekit/_snap.py),
which consolidates the algorithm set benchmarked in Carr et al. (2022)
:cite:p:`CarrEtAl2022`.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


FixationArray = NDArray[np.float64]
Midlines = Sequence[float]
WordCenters = Sequence[tuple[float, float]]


CLASSICAL_METHODS: tuple[str, ...] = (
    'attach',
    'chain',
    'cluster',
    'compare',
    'merge',
    'regress',
    'segment',
    'slice',
    'split',
    'stretch',
    'warp',
)


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------

def attach(
        fixation_xy: FixationArray,
        midlines: Midlines,
) -> NDArray[np.float64]:
    """Assign each fixation to the nearest text line by Euclidean y-distance.

    The simplest baseline algorithm.
    """
    fixation_xy = np.array(fixation_xy, dtype=float)
    line_y = np.array(midlines, dtype=float)
    corrected = fixation_xy[:, 1].copy()
    for i in range(len(corrected)):
        line_i = int(np.argmin(np.abs(line_y - corrected[i])))
        corrected[i] = line_y[line_i]
    return corrected


# ---------------------------------------------------------------------------
# chain
# ---------------------------------------------------------------------------

def chain(
        fixation_xy: FixationArray,
        midlines: Midlines,
        x_thresh: float = 192,
        y_thresh: float = 32,
) -> NDArray[np.float64]:
    """Chain consecutive fixations, then assign each chain to the nearest text line.

    Fixations are chained if they are within ``(x_thresh, y_thresh)`` of their
    predecessor. Each chain is assigned to the text line nearest its mean y.

    Default params: ``x_thresh=192``, ``y_thresh=32``.

    Original method implemented in popEye :cite:p:`Schroeder2019`.
    """
    fixation_xy = np.array(fixation_xy, dtype=float)
    line_y = np.array(midlines, dtype=float)
    dist_x = np.abs(np.diff(fixation_xy[:, 0]))
    dist_y = np.abs(np.diff(fixation_xy[:, 1]))
    boundaries = list(np.where(np.logical_or(dist_x > x_thresh, dist_y > y_thresh))[0] + 1)
    boundaries.append(len(fixation_xy))
    start = 0
    for end in boundaries:
        mean_y = float(np.mean(fixation_xy[start:end, 1]))
        line_i = int(np.argmin(np.abs(line_y - mean_y)))
        fixation_xy[start:end, 1] = line_y[line_i]
        start = end
    return fixation_xy[:, 1]


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

def cluster(
        fixation_xy: FixationArray,
        midlines: Midlines,
) -> NDArray[np.float64]:
    """K-means cluster fixations by y-coordinate and map clusters to text lines in order.

    ``k`` equals the number of text lines.

    Original method implemented in popEye :cite:p:`Schroeder2019`.
    """
    fixation_xy = np.array(fixation_xy, dtype=float)
    line_y = np.array(midlines, dtype=float)
    m = len(line_y)
    fixation_y = fixation_xy[:, 1].reshape(-1, 1)
    clusters = KMeans(n_clusters=m, n_init=100, max_iter=300).fit_predict(fixation_y)
    centers = np.array([float(fixation_y[clusters == i].mean()) for i in range(m)])
    ordered = np.argsort(centers)
    result = fixation_xy[:, 1].copy()
    for fixation_i, cluster_i in enumerate(clusters):
        line_i = int(np.where(ordered == cluster_i)[0][0])
        result[fixation_i] = line_y[line_i]
    return result


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def compare(
        fixation_xy: FixationArray,
        word_centers: WordCenters,
        x_thresh: float = 512,
        n_nearest_lines: int = 3,
) -> NDArray[np.float64]:
    """Compare fixation subsequences against candidate text lines using DTW.

    Fixations are split into reading-line segments at large leftward x-jumps,
    then each segment is assigned to the best-matching line by dynamic
    time warping.

    :cite:p:`LimaSanchesEtAl2015`.
    """
    fixation_xy = np.array(fixation_xy, dtype=float)
    word_xy = np.array(word_centers, dtype=float)
    line_y = np.unique(word_xy[:, 1])
    n_nearest_lines = min(n_nearest_lines, len(line_y) - 1)

    n = len(fixation_xy)
    diff_x = np.diff(fixation_xy[:, 0])
    end_indices = list(np.where(diff_x < -x_thresh)[0] + 1)
    end_indices.append(n)

    start = 0
    for end in end_indices:
        gaze_line = fixation_xy[start:end]
        mean_y = float(np.mean(gaze_line[:, 1]))
        nearest_line_i = np.argsort(np.abs(line_y - mean_y))[:n_nearest_lines]
        costs = np.zeros(len(nearest_line_i))
        for k, candidate_line_i in enumerate(nearest_line_i):
            text_line = word_xy[word_xy[:, 1] == line_y[candidate_line_i]]
            cost, _ = _dynamic_time_warping(gaze_line[:, 0:1], text_line[:, 0:1])
            costs[k] = cost
        best = nearest_line_i[int(np.argmin(costs))]
        fixation_xy[start:end, 1] = line_y[best]
        start = end
    return fixation_xy[:, 1]


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def merge(
        fixation_xy: FixationArray,
        midlines: Midlines,
        text_right_to_left: bool = False,
        y_thresh: float = 32,
        gradient_thresh: float = 0.1,
        error_thresh: float = 20,
) -> NDArray[np.float64]:
    """Merge progressive fixation sequences onto the same line by regression.

    Default params: ``y_thresh=32``, ``gradient_thresh=0.1``, ``error_thresh=20``.

    :cite:p:`SpakovEtAl2019`.
    """
    fixation_xy = np.array(fixation_xy, dtype=float)
    line_y = np.array(midlines, dtype=float)
    diff_x = np.diff(fixation_xy[:, 0])
    dist_y = np.abs(np.diff(fixation_xy[:, 1]))
    if text_right_to_left:
        boundaries = list(np.where(np.logical_or(diff_x > 0, dist_y > y_thresh))[0] + 1)
    else:
        boundaries = list(np.where(np.logical_or(diff_x < 0, dist_y > y_thresh))[0] + 1)
    seq_starts = [0] + boundaries
    seq_ends = boundaries + [len(fixation_xy)]
    sequences = [list(range(s, e)) for s, e in zip(seq_starts, seq_ends)]

    phases: list[tuple[int, int, bool]] = [
        (3, 3, False), (1, 3, False), (1, 1, False), (1, 1, True),
    ]
    for min_i, min_j, remove_constraints in phases:
        while len(sequences) > len(line_y):
            best_merger: tuple[int, int] | None = None
            best_error: float = np.inf
            for i in range(len(sequences) - 1):
                if len(sequences[i]) < min_i:
                    continue
                for j in range(i + 1, len(sequences)):
                    if len(sequences[j]) < min_j:
                        continue
                    candidate = fixation_xy[sequences[i] + sequences[j]]
                    grad, intercept = np.polyfit(candidate[:, 0], candidate[:, 1], 1)
                    residuals = candidate[:, 1] - (grad * candidate[:, 0] + intercept)
                    error = float(np.sqrt(np.sum(residuals ** 2) / len(candidate)))
                    ok = remove_constraints or (
                        abs(grad) < gradient_thresh and error < error_thresh
                    )
                    if ok and error < best_error:
                        best_merger = (i, j)
                        best_error = error
            if best_merger is None:
                break
            mi, mj = best_merger
            sequences.append(sequences[mi] + sequences[mj])
            del sequences[mj], sequences[mi]

    mean_y_seq = [float(fixation_xy[seq, 1].mean()) for seq in sequences]
    for line_i, seq_i in enumerate(np.argsort(mean_y_seq)):
        fixation_xy[sequences[seq_i], 1] = line_y[line_i]
    return fixation_xy[:, 1]


# ---------------------------------------------------------------------------
# regress
# ---------------------------------------------------------------------------

def regress(
        fixation_xy: FixationArray,
        midlines: Midlines,
        slope_bounds: tuple[float, float] = (-0.1, 0.1),
        offset_bounds: tuple[float, float] = (-50.0, 50.0),
        std_bounds: tuple[float, float] = (1.0, 20.0),
) -> NDArray[np.float64]:
    """Fit *m* regression lines that best explain fixation y-values.

    Each fixation is assigned to its best-fitting line.

    :cite:p:`Cohen2013`.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    fixation_xy = np.array(fixation_xy, dtype=float)
    line_y = np.array(midlines, dtype=float)
    density = np.zeros((len(fixation_xy), len(line_y)))

    def fit_lines(params: np.ndarray) -> float:
        k = slope_bounds[0] + (slope_bounds[1] - slope_bounds[0]) * norm.cdf(params[0])
        o = offset_bounds[0] + (offset_bounds[1] - offset_bounds[0]) * norm.cdf(params[1])
        s = std_bounds[0] + (std_bounds[1] - std_bounds[0]) * norm.cdf(params[2])
        predicted_y = fixation_xy[:, 0] * k
        for line_i in range(len(line_y)):
            fit_y = predicted_y + line_y[line_i] + o
            density[:, line_i] = norm.logpdf(fixation_xy[:, 1], fit_y, s)
        return -float(density.max(axis=1).sum())

    best_fit = minimize(fit_lines, [0.0, 0.0, 0.0], method='powell')
    fit_lines(best_fit.x)
    return line_y[density.argmax(axis=1)]


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------

def segment(
        fixation_xy: FixationArray,
        midlines: Midlines,
        text_right_to_left: bool = False,
) -> NDArray[np.float64]:
    """Split the scanpath at the *m*−1 most-likely return sweeps, one segment per line.

    :cite:p:`AbdulinKomogortsev2015`.
    """
    fixation_xy = np.array(fixation_xy, dtype=float)
    line_y = np.array(midlines, dtype=float)
    diff_x = np.diff(fixation_xy[:, 0])
    ordered = np.argsort(diff_x)
    if text_right_to_left:
        change_indices = set(ordered[-(len(line_y) - 1):].tolist())
    else:
        change_indices = set(ordered[: len(line_y) - 1].tolist())
    current_line = 0
    for i in range(len(fixation_xy)):
        fixation_xy[i, 1] = line_y[current_line]
        if i in change_indices:
            current_line += 1
    return fixation_xy[:, 1]


# ---------------------------------------------------------------------------
# slice  (Glandorf-Schroeder / "proto-line" method)
# ---------------------------------------------------------------------------

def slice(                # noqa: A001 — name matches Carr et al. algorithm set
        fixation_xy: FixationArray,
        midlines: Midlines,
        line_height: float,
        x_thresh: float = 192,
        y_thresh: float = 32,
        w_thresh: float = 32,
        n_thresh: float = 90,
) -> NDArray[np.float64]:
    """Form runs, group them into proto-lines, then map proto-lines onto text lines.

    Default params: ``x_thresh=192``, ``y_thresh=32``, ``w_thresh=32``, ``n_thresh=90``.

    :cite:p:`GlandorfSchroeder2021`.
    """
    fixation_xy = np.array(fixation_xy, dtype=float)
    line_y = np.array(midlines, dtype=float)
    proto_lines: dict[int, list[int]] = {}
    phantom_proto_lines: dict[int, np.ndarray] = {}

    dist_x = np.abs(np.diff(fixation_xy[:, 0]))
    dist_y = np.abs(np.diff(fixation_xy[:, 1]))
    run_ends = list(np.where(np.logical_or(dist_x > x_thresh, dist_y > y_thresh))[0] + 1)
    run_starts = [0] + run_ends
    run_ends_full = run_ends + [len(fixation_xy)]
    runs = [list(range(s, e)) for s, e in zip(run_starts, run_ends_full)]

    longest = int(np.argmax([fixation_xy[r[-1], 0] - fixation_xy[r[0], 0] for r in runs]))
    proto_lines[0] = runs.pop(longest)

    while runs:
        merged = False
        for pl_i, direction in [(min(proto_lines), -1), (max(proto_lines), 1)]:
            proto_lines[pl_i + direction] = []
            if proto_lines[pl_i]:
                pl_xy = fixation_xy[proto_lines[pl_i]]
            else:
                pl_xy = phantom_proto_lines[pl_i]

            run_diffs = np.zeros(len(runs))
            for ri, run in enumerate(runs):
                y_diffs = [
                    fixation_xy[idx, 1] - pl_xy[np.argmin(np.abs(pl_xy[:, 0] - fixation_xy[idx, 0])), 1]
                    for idx in run
                ]
                run_diffs[ri] = float(np.mean(y_diffs))

            into_current = list(np.where(np.abs(run_diffs) < w_thresh)[0])
            into_adjacent = list(np.where(
                np.logical_and(run_diffs * direction >= w_thresh, run_diffs * direction < n_thresh),
            )[0])

            for idx in into_current:
                proto_lines[pl_i].extend(runs[idx])
            for idx in into_adjacent:
                proto_lines[pl_i + direction].extend(runs[idx])

            if not into_adjacent:
                ax, ay = float(np.mean(pl_xy, axis=0)[0]), float(np.mean(pl_xy, axis=0)[1])
                phantom_proto_lines[pl_i + direction] = np.array(
                    [[ax, ay + line_height * direction]],
                )

            for idx in sorted(into_current + into_adjacent, reverse=True):
                del runs[idx]
                merged = True
        if not merged:
            break

    for run in runs:
        best_dist, best_pl = np.inf, None
        for pl_i in proto_lines:
            pl_xy = fixation_xy[proto_lines[pl_i]] if proto_lines[pl_i] else phantom_proto_lines[pl_i]
            y_diffs = [
                fixation_xy[idx, 1] - pl_xy[np.argmin(np.abs(pl_xy[:, 0] - fixation_xy[idx, 0])), 1]
                for idx in run
            ]
            d = float(np.abs(np.mean(y_diffs)))
            if d < best_dist:
                best_dist, best_pl = d, pl_i
        proto_lines[best_pl].extend(run)  # type: ignore[index]

    while len(proto_lines) > len(line_y):
        top, bot = min(proto_lines), max(proto_lines)
        if len(proto_lines[top]) < len(proto_lines[bot]):
            proto_lines[top + 1].extend(proto_lines[top])
            del proto_lines[top]
        else:
            proto_lines[bot - 1].extend(proto_lines[bot])
            del proto_lines[bot]

    for line_i, pl_i in enumerate(sorted(proto_lines)):
        fixation_xy[proto_lines[pl_i], 1] = line_y[line_i]
    return fixation_xy[:, 1]


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def split(
        fixation_xy: FixationArray,
        midlines: Midlines,
        text_right_to_left: bool = False,
) -> NDArray[np.float64]:
    """Detect return sweeps via k-means on x-differences, split at those points.

    :cite:p:`CarrEtAl2022`.
    """
    from scipy.cluster.vq import kmeans2

    fixation_xy = np.array(fixation_xy, dtype=float)
    line_y = np.array(midlines, dtype=float)
    diff_x = np.diff(fixation_xy[:, 0]).reshape(-1, 1).astype(float)
    centers, clusters = kmeans2(diff_x, 2, iter=100, minit='++', missing='raise')
    sweep_marker = int(np.argmax(centers)) if text_right_to_left else int(np.argmin(centers))
    end_indices = list(np.where(clusters == sweep_marker)[0] + 1)
    end_indices.append(len(fixation_xy))
    start = 0
    for end in end_indices:
        mean_y = float(np.mean(fixation_xy[start:end, 1]))
        line_i = int(np.argmin(np.abs(line_y - mean_y)))
        fixation_xy[start:end, 1] = line_y[line_i]
        start = end
    return fixation_xy[:, 1]


# ---------------------------------------------------------------------------
# stretch
# ---------------------------------------------------------------------------

def stretch(
        fixation_xy: FixationArray,
        midlines: Midlines,
        stretch_bounds: tuple[float, float] = (0.9, 1.1),
        offset_bounds: tuple[float, float] = (-50.0, 50.0),
) -> NDArray[np.float64]:
    """Optimise a global y-stretch + offset so fixations align best to text lines.

    :cite:p:`Lohmeier2015`.
    """
    from scipy.optimize import minimize

    fixation_y = np.array(fixation_xy, dtype=float)[:, 1]
    line_y = np.array(midlines, dtype=float)
    corrected_y = np.zeros(len(fixation_y))

    def fit(params: np.ndarray) -> float:
        candidate = fixation_y * params[0] + params[1]
        for i in range(len(candidate)):
            corrected_y[i] = line_y[int(np.argmin(np.abs(line_y - candidate[i])))]
        return float(np.sum(np.abs(candidate - corrected_y)))

    best = minimize(fit, [1.0, 0.0], method='powell', bounds=[stretch_bounds, offset_bounds])
    fit(best.x)
    return corrected_y


# ---------------------------------------------------------------------------
# warp
# ---------------------------------------------------------------------------

def warp(
        fixation_xy: FixationArray,
        word_centers: WordCenters,
) -> NDArray[np.float64]:
    """Map fixations onto word centers using DTW; assign each to its matched word's line.

    :cite:p:`CarrEtAl2022`.
    """
    fixation_xy = np.array(fixation_xy, dtype=float)
    word_xy = np.array(word_centers, dtype=float)
    n1, n2 = len(fixation_xy), len(word_xy)
    cost = np.full((n1 + 1, n2 + 1), np.inf)
    cost[0, 0] = 0.0
    for fi in range(n1):
        for wi in range(n2):
            d = float(np.sqrt(np.sum((fixation_xy[fi] - word_xy[wi]) ** 2)))
            cost[fi + 1, wi + 1] = d + min(cost[fi, wi + 1], cost[fi + 1, wi], cost[fi, wi])
    cost = cost[1:, 1:]
    path: list[list[int]] = [[] for _ in range(n1)]
    fi, wi = n1 - 1, n2 - 1
    while fi > 0 or wi > 0:
        path[fi].append(wi)
        moves = [np.inf, np.inf, np.inf]
        if fi > 0 and wi > 0:
            moves[0] = cost[fi - 1, wi - 1]
        if fi > 0:
            moves[1] = cost[fi - 1, wi]
        if wi > 0:
            moves[2] = cost[fi, wi - 1]
        best_move = int(np.argmin(moves))
        if best_move == 0:
            fi -= 1
            wi -= 1
        elif best_move == 1:
            fi -= 1
        else:
            wi -= 1
    path[0].append(0)
    for fi, words in enumerate(path):
        candidate_y = list(word_xy[words, 1])
        fixation_xy[fi, 1] = max(set(candidate_y), key=candidate_y.count)
    return fixation_xy[:, 1]


# ---------------------------------------------------------------------------
# wisdom_of_the_crowd (ensemble)
# ---------------------------------------------------------------------------

def wisdom_of_the_crowd(
        assignments: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Ensemble vote across several per-algorithm correction arrays.

    For each fixation the y-value with the most votes wins; ties are broken
    by the left-most (earliest) algorithm in the list.
    """
    stacked = np.column_stack(assignments)
    result = np.zeros(stacked.shape[0])
    for i, row in enumerate(stacked):
        counts = {y: int(np.sum(row == y)) for y in np.unique(row)}
        best_count = max(counts.values())
        best_candidates = {y for y, c in counts.items() if c == best_count}
        for y in row:
            if y in best_candidates:
                result[i] = y
                break
    return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _dynamic_time_warping(
        seq1: NDArray[np.float64],
        seq2: NDArray[np.float64],
) -> tuple[float, list[list[int]]]:
    """Minimal DTW used by :func:`compare`."""
    n1, n2 = len(seq1), len(seq2)
    cost = np.full((n1 + 1, n2 + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(n1):
        for j in range(n2):
            d = float(np.sqrt(np.sum((seq1[i] - seq2[j]) ** 2)))
            cost[i + 1, j + 1] = d + min(cost[i, j + 1], cost[i + 1, j], cost[i, j])
    cost = cost[1:, 1:]
    path: list[list[int]] = [[] for _ in range(n1)]
    i, j = n1 - 1, n2 - 1
    while i > 0 or j > 0:
        path[i].append(j)
        moves = [np.inf, np.inf, np.inf]
        if i > 0 and j > 0:
            moves[0] = cost[i - 1, j - 1]
        if i > 0:
            moves[1] = cost[i - 1, j]
        if j > 0:
            moves[2] = cost[i, j - 1]
        best_move = int(np.argmin(moves))
        if best_move == 0:
            i -= 1
            j -= 1
        elif best_move == 1:
            i -= 1
        else:
            j -= 1
    path[0].append(0)
    return float(cost[-1, -1]), path
