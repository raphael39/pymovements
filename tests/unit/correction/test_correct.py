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
"""Integration tests for ``correct_vertical_drift``."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pymovements.correction import CLASSICAL_METHODS
from pymovements.correction import correct_vertical_drift
from pymovements.correction import METHODS
from pymovements.stimulus import TextStimulus


MIDLINES = [100.0, 150.0, 200.0]


def _scanpath(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for line_y in MIDLINES:
        for i in range(5):
            rows.append([100.0 + i * 100.0, line_y + rng.uniform(-3.0, 3.0)])
    return np.array(rows, dtype=float)


def _stimulus_3_lines() -> TextStimulus:
    """TextStimulus with 3 lines × 5 word-AOIs."""
    aois = pl.DataFrame({
        'word': [f'w{i}' for i in range(15)],
        'x0': [75.0 + i * 100.0 for _ in range(3) for i in range(5)],
        'y0': [y - 10.0 for y in MIDLINES for _ in range(5)],
        'x1': [125.0 + i * 100.0 for _ in range(3) for i in range(5)],
        'y1': [y + 10.0 for y in MIDLINES for _ in range(5)],
    })
    return TextStimulus(
        aois,
        aoi_column='word',
        start_x_column='x0',
        start_y_column='y0',
        end_x_column='x1',
        end_y_column='y1',
    )


class TestMethodsConstants:

    def test_classical_has_eleven(self) -> None:
        assert len(CLASSICAL_METHODS) == 11

    def test_methods_includes_dist(self) -> None:
        assert 'dist' in METHODS

    def test_methods_is_superset_of_classical(self) -> None:
        assert set(CLASSICAL_METHODS).issubset(set(METHODS))


class TestDispatcherWithExplicitMidlines:

    @pytest.mark.parametrize(
        'method',
        ['attach', 'chain', 'cluster', 'merge', 'regress',
         'segment', 'split', 'stretch'],
    )
    def test_midline_method_roundtrip(self, method: str) -> None:
        result = correct_vertical_drift(
            _scanpath(), midlines=MIDLINES, method=method,
        )
        assert result.shape == (15,)
        for y in result:
            assert y in MIDLINES

    def test_slice_requires_line_height(self) -> None:
        with pytest.raises(ValueError, match='line_height'):
            correct_vertical_drift(
                _scanpath(), midlines=MIDLINES, method='slice',
            )

    def test_slice_with_line_height(self) -> None:
        result = correct_vertical_drift(
            _scanpath(), midlines=MIDLINES, method='slice', line_height=50.0,
        )
        for y in result:
            assert y in MIDLINES


class TestDispatcherWithWordCenters:

    def _word_centers(self) -> list[tuple[float, float]]:
        return [(100.0 + i * 100.0, y) for y in MIDLINES for i in range(5)]

    def test_warp(self) -> None:
        result = correct_vertical_drift(
            _scanpath(), word_centers=self._word_centers(), method='warp',
        )
        for y in result:
            assert y in MIDLINES

    def test_compare(self) -> None:
        result = correct_vertical_drift(
            _scanpath(), word_centers=self._word_centers(), method='compare',
        )
        for y in result:
            assert y in MIDLINES

    def test_warp_requires_word_centers(self) -> None:
        with pytest.raises(ValueError, match='word_centers'):
            correct_vertical_drift(_scanpath(), midlines=MIDLINES, method='warp')


class TestDispatcherWithTextStimulus:

    def test_attach_derives_midlines_from_stimulus(self) -> None:
        stim = _stimulus_3_lines()
        result = correct_vertical_drift(
            _scanpath(), stimulus=stim, method='attach',
        )
        for y in result:
            assert y in MIDLINES

    def test_warp_derives_word_centers_from_stimulus(self) -> None:
        stim = _stimulus_3_lines()
        result = correct_vertical_drift(
            _scanpath(), stimulus=stim, method='warp',
        )
        for y in result:
            assert y in MIDLINES

    def test_slice_derives_line_height_from_stimulus(self) -> None:
        stim = _stimulus_3_lines()
        result = correct_vertical_drift(
            _scanpath(), stimulus=stim, method='slice',
        )
        for y in result:
            assert y in MIDLINES


class TestValidation:

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match='Unknown method'):
            correct_vertical_drift(_scanpath(), midlines=MIDLINES, method='bogus')

    def test_dist_requires_corrector(self) -> None:
        with pytest.raises(ValueError, match='dist_corrector'):
            correct_vertical_drift(_scanpath(), midlines=MIDLINES, method='dist')

    def test_dist_requires_dffix_and_trial(self) -> None:
        class _StubCorrector:
            def predict(self, dffix, trial):  # pragma: no cover - not called
                raise AssertionError('should not be reached')

        with pytest.raises(ValueError, match='dffix'):
            correct_vertical_drift(
                _scanpath(), midlines=MIDLINES, method='dist',
                dist_corrector=_StubCorrector(),
            )

    def test_midlines_required_when_no_stimulus(self) -> None:
        with pytest.raises(ValueError, match='midlines'):
            correct_vertical_drift(_scanpath(), method='attach')

    def test_empty_midlines_rejected(self) -> None:
        with pytest.raises(ValueError, match='must not be empty'):
            correct_vertical_drift(_scanpath(), midlines=[], method='attach')

    def test_single_line_short_circuit(self) -> None:
        result = correct_vertical_drift(
            _scanpath(), midlines=[150.0], method='attach',
        )
        np.testing.assert_array_equal(result, np.full(15, 150.0))


class TestMethodCaseInsensitive:

    def test_uppercase(self) -> None:
        result = correct_vertical_drift(
            _scanpath(), midlines=MIDLINES, method='ATTACH',
        )
        for y in result:
            assert y in MIDLINES
