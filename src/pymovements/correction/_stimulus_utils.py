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
"""Utilities for deriving midlines and word centers from a ``TextStimulus``.

``TextStimulus`` exposes AOIs (areas of interest) as bounding boxes — each row
has a ``start_x``, ``start_y`` and either ``width/height`` or ``end_x/end_y``.
The classical correction algorithms need a flatter representation:

* ``midlines``      : one y-coordinate per text line (the vertical midpoint),
  ordered top-to-bottom.
* ``word_centers``  : one ``(x, y)`` per AOI (the geometric center),
  in reading order.

The functions below perform that derivation. They make a minimal set of
assumptions:

1. One AOI per word (or per character, or per syllable — anything at a
   granularity below "line" will work).
2. AOIs on the same line share the same ``start_y`` (up to the configured
   tolerance). If your AOIs are not line-aligned, pre-group them yourself.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pymovements.stimulus.text import TextStimulus


def _end_coordinates(
        stimulus: TextStimulus,
) -> tuple[pl.Expr, pl.Expr]:
    """Return polars expressions for the AOI's ``end_x`` and ``end_y`` columns.

    Falls back to ``start_* + width/height`` when explicit end columns are not set.

    Raises
    ------
    ValueError
        If neither end columns nor width/height columns are configured.
    """
    if stimulus.end_x_column is not None and stimulus.end_y_column is not None:
        return pl.col(stimulus.end_x_column), pl.col(stimulus.end_y_column)
    if stimulus.width_column is not None and stimulus.height_column is not None:
        return (
            pl.col(stimulus.start_x_column) + pl.col(stimulus.width_column),
            pl.col(stimulus.start_y_column) + pl.col(stimulus.height_column),
        )
    raise ValueError(
        'TextStimulus must have either (end_x_column, end_y_column) or '
        '(width_column, height_column) configured to compute line midlines / '
        'word centers.',
    )


def line_midlines(
        stimulus: TextStimulus,
        *,
        y_tolerance: float = 0.5,
) -> list[float]:
    """Derive the y-coordinate of each text line's vertical midpoint.

    Groups AOIs by rounded ``start_y`` (the line baseline, effectively), then
    for each group computes the mean of ``(start_y + end_y) / 2``. Results are
    returned sorted top-to-bottom.

    Parameters
    ----------
    stimulus : TextStimulus
        The text stimulus whose AOIs will be analysed.
    y_tolerance : float
        AOIs whose ``start_y`` differ by at most this amount are treated as the
        same line. Set higher if your AOIs have slight vertical jitter.
        (default: 0.5)

    Returns
    -------
    list[float]
        Midline y-values, one per text line, ordered top-to-bottom.
    """
    end_x_expr, end_y_expr = _end_coordinates(stimulus)
    df = stimulus.aois.select(
        start_y=pl.col(stimulus.start_y_column),
        mid_y=(pl.col(stimulus.start_y_column) + end_y_expr) / 2.0,
    )
    # bucketise start_y to the tolerance
    buckets = (df['start_y'] / y_tolerance).round() * y_tolerance
    df = df.with_columns(buckets.alias('_bucket'))
    grouped = df.group_by('_bucket').agg(pl.col('mid_y').mean()).sort('_bucket')
    return grouped['mid_y'].to_list()


def word_centers(
        stimulus: TextStimulus,
) -> list[tuple[float, float]]:
    """Derive the ``(x, y)`` center of each AOI, in reading order.

    "Reading order" is defined as the current row order of
    ``stimulus.aois`` — it is the caller's responsibility to ensure this
    order matches the intended reading order.

    Returns
    -------
    list[tuple[float, float]]
        One ``(x, y)`` per AOI.
    """
    end_x_expr, end_y_expr = _end_coordinates(stimulus)
    df = stimulus.aois.select(
        center_x=(pl.col(stimulus.start_x_column) + end_x_expr) / 2.0,
        center_y=(pl.col(stimulus.start_y_column) + end_y_expr) / 2.0,
    )
    xs = df['center_x'].to_numpy()
    ys = df['center_y'].to_numpy()
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def line_height(
        stimulus: TextStimulus,
) -> float:
    """Return the median vertical distance between consecutive line midlines.

    Useful for algorithms that require an explicit ``line_height`` parameter
    (notably :func:`pymovements.correction.slice`).

    Raises
    ------
    ValueError
        If the stimulus has fewer than two lines.
    """
    midlines = line_midlines(stimulus)
    if len(midlines) < 2:
        raise ValueError(
            f'Cannot derive line_height from {len(midlines)} line(s); need at least 2.',
        )
    return float(np.median(np.diff(np.array(midlines, dtype=float))))
