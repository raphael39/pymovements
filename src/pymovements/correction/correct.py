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
"""Unified entry point for vertical fixation-drift correction."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pymovements.correction import _classical
from pymovements.correction._classical import CLASSICAL_METHODS
from pymovements.correction._classical import FixationArray
from pymovements.correction._classical import Midlines
from pymovements.correction._classical import WordCenters
from pymovements.correction._stimulus_utils import line_height as _derive_line_height
from pymovements.correction._stimulus_utils import line_midlines as _derive_line_midlines
from pymovements.correction._stimulus_utils import word_centers as _derive_word_centers
from pymovements.stimulus.text import TextStimulus


_ML_METHODS: tuple[str, ...] = ('dist',)
METHODS: tuple[str, ...] = CLASSICAL_METHODS + _ML_METHODS


def correct_vertical_drift(
        fixation_xy: FixationArray,
        *,
        stimulus: TextStimulus | None = None,
        midlines: Midlines | None = None,
        word_centers: WordCenters | None = None,
        line_height: float | None = None,
        method: str = 'warp',
        **algo_kwargs: Any,
) -> NDArray[np.float64]:
    """Assign each fixation to the correct text line.

    Supply text-layout information either via a :class:`TextStimulus` (the
    stimulus's AOIs are used to derive midlines / word centers / line height)
    or directly via the ``midlines`` / ``word_centers`` / ``line_height``
    keyword arguments. Directly-supplied values take precedence over stimulus-
    derived values when both are provided.

    Parameters
    ----------
    fixation_xy: array-like, shape [N, 2]
        Raw fixation coordinates. Each row is ``(x, y)``.
    stimulus: TextStimulus | None
        Text stimulus whose AOIs define line midlines and word centers.
        If provided, line midlines / word centers / line height are derived
        from it as needed. (default: None)
    midlines: sequence of float | None
        y-coordinate of each text line's midpoint, in top-to-bottom order.
        Required if ``stimulus`` is not provided and the method needs
        midlines. (default: None)
    word_centers: sequence of (float, float) | None
        ``(x, y)`` center of each word in reading order.
        Required for ``method='warp'`` and ``method='compare'`` (unless a
        ``stimulus`` is provided). (default: None)
    line_height: float | None
        Pixel height between consecutive text lines.
        Required for ``method='slice'`` (unless a ``stimulus`` is provided).
        (default: None)
    method: str
        Algorithm to use. One of ``'attach'``, ``'chain'``, ``'cluster'``,
        ``'compare'``, ``'merge'``, ``'regress'``, ``'segment'``, ``'slice'``,
        ``'split'``, ``'stretch'``, ``'warp'`` (classical), or ``'dist'``
        (ML — reserved, not yet implemented). (default: ``'warp'``)
    **algo_kwargs
        Any additional keyword arguments are forwarded to the chosen
        algorithm (e.g. ``x_thresh``, ``y_thresh`` for ``'chain'``).

    Returns
    -------
    np.ndarray, shape [N]
        Corrected y-value per fixation, one of the values in ``midlines``.

    Raises
    ------
    ValueError
        If ``method`` is unknown or a required argument is missing.
    NotImplementedError
        If ``method='dist'`` is requested.

    Examples
    --------
    With explicit midlines (the lightweight path):

    >>> import numpy as np
    >>> fixations = np.array([[100, 102], [200, 99], [300, 151], [400, 198]])
    >>> midlines = [100.0, 150.0, 200.0]
    >>> correct_vertical_drift(fixations, midlines=midlines, method='attach')
    array([100., 100., 150., 200.])

    With a TextStimulus (line midlines + word centers derived automatically):

    >>> import polars as pl
    >>> from pymovements.stimulus import TextStimulus
    >>> aois = pl.DataFrame({
    ...     'word': ['a', 'b', 'c', 'd'],
    ...     'x0': [0, 50, 0, 50], 'y0': [0, 0, 100, 100],
    ...     'x1': [40, 90, 40, 90], 'y1': [20, 20, 120, 120],
    ... })
    >>> stim = TextStimulus(
    ...     aois, aoi_column='word',
    ...     start_x_column='x0', start_y_column='y0',
    ...     end_x_column='x1', end_y_column='y1',
    ... )
    >>> fixations = np.array([[10, 9], [60, 11], [20, 108]])
    >>> correct_vertical_drift(fixations, stimulus=stim, method='attach')
    array([ 10.,  10., 110.])
    """
    fixation_xy = np.asarray(fixation_xy, dtype=float)
    method = method.lower()

    if method not in METHODS:
        raise ValueError(
            f'Unknown method {method!r}. Choose from: '
            + ', '.join(sorted(METHODS)),
        )

    if method == 'dist':
        raise NotImplementedError(
            "method='dist' is not yet available. "
            'It will be added once the DistCorrector wrapper around the '
            'Mercier et al. (2023) DIST model is in place. Use one of the '
            'classical methods in the meantime: '
            + ', '.join(sorted(CLASSICAL_METHODS)),
        )

    # --- derive layout info from stimulus if needed ---
    if midlines is None and stimulus is not None:
        midlines = _derive_line_midlines(stimulus)
    if word_centers is None and stimulus is not None and method in ('warp', 'compare'):
        word_centers = _derive_word_centers(stimulus)
    if line_height is None and stimulus is not None and method == 'slice':
        line_height = _derive_line_height(stimulus)

    # --- early-out for trivial cases (methods using midlines) ---
    if method not in ('warp', 'compare'):
        if midlines is None:
            raise ValueError(
                f"method={method!r} requires either 'stimulus' or 'midlines' to be provided.",
            )
        midlines_list = list(midlines)
        n_lines = len(midlines_list)
        if n_lines == 0:
            raise ValueError('midlines must not be empty.')
        if n_lines == 1 or len(fixation_xy) <= 2:
            return np.full(len(fixation_xy), float(midlines_list[0]))
    else:
        # warp / compare use word_centers (midlines derived from them internally)
        if word_centers is None:
            raise ValueError(
                f"method={method!r} requires either 'stimulus' or 'word_centers' to be provided.",
            )
        midlines_list = None  # unused

    if method == 'slice' and line_height is None:
        raise ValueError(
            "method='slice' requires either 'stimulus' or 'line_height' to be provided.",
        )

    # --- dispatch ---
    if method == 'attach':
        return _classical.attach(fixation_xy, midlines_list, **algo_kwargs)
    if method == 'chain':
        return _classical.chain(fixation_xy, midlines_list, **algo_kwargs)
    if method == 'cluster':
        return _classical.cluster(fixation_xy, midlines_list, **algo_kwargs)
    if method == 'compare':
        return _classical.compare(fixation_xy, word_centers, **algo_kwargs)
    if method == 'merge':
        return _classical.merge(fixation_xy, midlines_list, **algo_kwargs)
    if method == 'regress':
        return _classical.regress(fixation_xy, midlines_list, **algo_kwargs)
    if method == 'segment':
        return _classical.segment(fixation_xy, midlines_list, **algo_kwargs)
    if method == 'slice':
        return _classical.slice(
            fixation_xy, midlines_list, line_height=float(line_height), **algo_kwargs,
        )
    if method == 'split':
        return _classical.split(fixation_xy, midlines_list, **algo_kwargs)
    if method == 'stretch':
        return _classical.stretch(fixation_xy, midlines_list, **algo_kwargs)
    if method == 'warp':
        return _classical.warp(fixation_xy, word_centers)
    # unreachable — kept for type checker
    raise ValueError(f'Unhandled method: {method!r}')
