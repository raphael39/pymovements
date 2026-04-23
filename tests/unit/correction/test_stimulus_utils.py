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
"""Tests for the TextStimulus-derived midline/word-center helpers."""
from __future__ import annotations

import polars as pl
import pytest

from pymovements.correction import line_height
from pymovements.correction import line_midlines
from pymovements.correction import word_centers
from pymovements.stimulus import TextStimulus


def _stim_with_end_cols() -> TextStimulus:
    """3 lines × 2 words, with end_x/end_y columns."""
    aois = pl.DataFrame({
        'w': ['a', 'b', 'c', 'd', 'e', 'f'],
        'x0': [0.0, 50.0, 0.0, 50.0, 0.0, 50.0],
        'y0': [90.0, 90.0, 140.0, 140.0, 190.0, 190.0],
        'x1': [40.0, 90.0, 40.0, 90.0, 40.0, 90.0],
        'y1': [110.0, 110.0, 160.0, 160.0, 210.0, 210.0],
    })
    return TextStimulus(
        aois,
        aoi_column='w',
        start_x_column='x0',
        start_y_column='y0',
        end_x_column='x1',
        end_y_column='y1',
    )


def _stim_with_width_height() -> TextStimulus:
    """Same layout, specified via width/height instead of end_*."""
    aois = pl.DataFrame({
        'w': ['a', 'b', 'c', 'd'],
        'x0': [0.0, 50.0, 0.0, 50.0],
        'y0': [90.0, 90.0, 140.0, 140.0],
        'width': [40.0, 40.0, 40.0, 40.0],
        'height': [20.0, 20.0, 20.0, 20.0],
    })
    return TextStimulus(
        aois,
        aoi_column='w',
        start_x_column='x0',
        start_y_column='y0',
        width_column='width',
        height_column='height',
    )


def _stim_missing_bounds() -> TextStimulus:
    """Neither end_* nor width/height configured — helpers should raise."""
    aois = pl.DataFrame({
        'w': ['a', 'b'],
        'x0': [0.0, 50.0],
        'y0': [90.0, 90.0],
    })
    return TextStimulus(
        aois,
        aoi_column='w',
        start_x_column='x0',
        start_y_column='y0',
    )


class TestLineMidlines:

    def test_end_cols(self) -> None:
        stim = _stim_with_end_cols()
        assert line_midlines(stim) == [100.0, 150.0, 200.0]

    def test_width_height(self) -> None:
        stim = _stim_with_width_height()
        assert line_midlines(stim) == [100.0, 150.0]

    def test_missing_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match='end_x_column'):
            line_midlines(_stim_missing_bounds())


class TestWordCenters:

    def test_end_cols(self) -> None:
        stim = _stim_with_end_cols()
        centers = word_centers(stim)
        assert centers == [
            (20.0, 100.0), (70.0, 100.0),
            (20.0, 150.0), (70.0, 150.0),
            (20.0, 200.0), (70.0, 200.0),
        ]

    def test_width_height(self) -> None:
        stim = _stim_with_width_height()
        centers = word_centers(stim)
        assert centers == [
            (20.0, 100.0), (70.0, 100.0),
            (20.0, 150.0), (70.0, 150.0),
        ]


class TestLineHeight:

    def test_uniform_spacing(self) -> None:
        stim = _stim_with_end_cols()
        assert line_height(stim) == 50.0

    def test_single_line_raises(self) -> None:
        aois = pl.DataFrame({
            'w': ['a'],
            'x0': [0.0], 'y0': [0.0],
            'x1': [10.0], 'y1': [10.0],
        })
        stim = TextStimulus(
            aois, aoi_column='w',
            start_x_column='x0', start_y_column='y0',
            end_x_column='x1', end_y_column='y1',
        )
        with pytest.raises(ValueError, match='at least 2'):
            line_height(stim)
