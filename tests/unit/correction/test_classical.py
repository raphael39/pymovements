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
"""Per-algorithm smoke tests for the classical correction algorithms.

Each test constructs a small synthetic scanpath where the correct line
assignment is obvious, then asserts the algorithm produces it (or the
closest permissible approximation).
"""
from __future__ import annotations

import numpy as np
import pytest

from pymovements.correction import _classical


# Three horizontal text lines at y = 100, 150, 200.
MIDLINES = [100.0, 150.0, 200.0]


def _reading_scanpath(jitter: float = 3.0, seed: int = 0) -> np.ndarray:
    """Simulate a 3-line reading pattern with small vertical drift.

    Five fixations per line, moving left-to-right, with ±jitter noise on y.
    Returns shape [15, 2].
    """
    rng = np.random.default_rng(seed)
    rows: list[list[float]] = []
    for line_y in MIDLINES:
        for i in range(5):
            x = 100.0 + i * 100.0
            y = line_y + rng.uniform(-jitter, jitter)
            rows.append([x, y])
    return np.array(rows, dtype=float)


def _word_centers_grid() -> list[tuple[float, float]]:
    """Word centers on the same 3-line grid: 5 words per line."""
    return [
        (100.0 + i * 100.0, line_y)
        for line_y in MIDLINES
        for i in range(5)
    ]


class TestAttach:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.attach(fixations, MIDLINES)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)

    def test_does_not_mutate_input(self) -> None:
        fixations = _reading_scanpath()
        original = fixations.copy()
        _classical.attach(fixations, MIDLINES)
        np.testing.assert_array_equal(fixations, original)


class TestChain:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.chain(fixations, MIDLINES)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestCluster:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.cluster(fixations, MIDLINES)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestCompare:

    def test_clean_signal_maps_each_line(self) -> None:
        # synthetic scanpath has 400-px return sweeps; lower x_thresh accordingly
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.compare(fixations, _word_centers_grid(), x_thresh=300)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestMerge:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.merge(fixations, MIDLINES)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestRegress:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.regress(fixations, MIDLINES)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestSegment:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.segment(fixations, MIDLINES)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestSlice:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.slice(fixations, MIDLINES, line_height=50.0)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestSplit:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.split(fixations, MIDLINES)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestStretch:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.stretch(fixations, MIDLINES)
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestWarp:

    def test_clean_signal_maps_each_line(self) -> None:
        fixations = _reading_scanpath(jitter=3.0)
        result = _classical.warp(fixations, _word_centers_grid())
        expected = np.repeat(MIDLINES, 5)
        np.testing.assert_array_equal(result, expected)


class TestWisdomOfTheCrowd:

    def test_unanimous_vote(self) -> None:
        a = np.array([100.0, 150.0, 200.0])
        result = _classical.wisdom_of_the_crowd([a, a.copy(), a.copy()])
        np.testing.assert_array_equal(result, a)

    def test_majority_vote(self) -> None:
        a = np.array([100.0, 150.0])
        b = np.array([100.0, 200.0])
        c = np.array([100.0, 150.0])
        result = _classical.wisdom_of_the_crowd([a, b, c])
        np.testing.assert_array_equal(result, np.array([100.0, 150.0]))

    def test_tie_broken_by_first_voter(self) -> None:
        a = np.array([100.0])
        b = np.array([200.0])
        result = _classical.wisdom_of_the_crowd([a, b])
        assert result[0] == 100.0


class TestClassicalMethodsConstant:

    def test_has_all_eleven(self) -> None:
        assert set(_classical.CLASSICAL_METHODS) == {
            'attach', 'chain', 'cluster', 'compare', 'merge',
            'regress', 'segment', 'slice', 'split', 'stretch', 'warp',
        }


@pytest.mark.parametrize(
    'name',
    ['attach', 'chain', 'cluster', 'merge', 'regress', 'segment',
     'split', 'stretch'],
)
def test_midline_based_algos_return_only_midline_values(name: str) -> None:
    """Every returned y must be exactly one of the configured midlines."""
    fixations = _reading_scanpath(jitter=8.0, seed=1)
    func = getattr(_classical, name)
    if name == 'slice':
        result = func(fixations, MIDLINES, line_height=50.0)
    else:
        result = func(fixations, MIDLINES)
    for y in result:
        assert y in MIDLINES, f'{name} returned non-midline value {y}'
