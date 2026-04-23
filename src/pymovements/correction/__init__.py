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
"""Vertical fixation-drift correction.

Provides classical algorithms from Carr et al. (2022) :cite:p:`CarrEtAl2022`
and a reserved slot for the ML-based DIST method :cite:p:`MercierEtAl2023`.

Quick start
-----------
>>> from pymovements.correction import correct_vertical_drift
>>> corrected_y = correct_vertical_drift(
...     fixation_xy, stimulus=text_stimulus, method='warp',
... )
"""
from pymovements.correction._classical import attach
from pymovements.correction._classical import chain
from pymovements.correction._classical import CLASSICAL_METHODS
from pymovements.correction._classical import cluster
from pymovements.correction._classical import compare
from pymovements.correction._classical import merge
from pymovements.correction._classical import regress
from pymovements.correction._classical import segment
from pymovements.correction._classical import slice
from pymovements.correction._classical import split
from pymovements.correction._classical import stretch
from pymovements.correction._classical import warp
from pymovements.correction._classical import wisdom_of_the_crowd
from pymovements.correction._stimulus_utils import line_height
from pymovements.correction._stimulus_utils import line_midlines
from pymovements.correction._stimulus_utils import word_centers
from pymovements.correction.correct import correct_vertical_drift
from pymovements.correction.correct import METHODS


__all__ = [
    # high-level dispatcher
    'correct_vertical_drift',
    # individual classical algorithms
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
    'wisdom_of_the_crowd',
    # stimulus helpers
    'line_midlines',
    'word_centers',
    'line_height',
    # constants
    'CLASSICAL_METHODS',
    'METHODS',
]
