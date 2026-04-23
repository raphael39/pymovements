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
"""Tests for the DIST ML-based corrector wrapper.

Most of these tests run without the ``[dist]`` extras installed — the module
itself must be import-clean. Tests that need a working torch are marked
with ``pytest.importorskip``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymovements.correction import correct_vertical_drift
from pymovements.correction import DistCorrector
from pymovements.correction import dist as dist_module


class TestImportIsAlwaysClean:
    """Importing the module must not fail even without the heavy deps."""

    def test_module_imports(self) -> None:
        # Importing at test collection already succeeded if we got here.
        assert DistCorrector is dist_module.DistCorrector


class TestOptionalDependencyHandling:

    def test_instantiation_without_torch_raises(
            self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without torch, DistCorrector.__init__ must fail with ImportError."""
        monkeypatch.setattr(dist_module, '_TORCH_AVAILABLE', False)
        with pytest.raises(ImportError, match=r"pymovements\[dist\]"):
            DistCorrector(model=None, model_cfg={})

    def test_from_checkpoint_without_torch_raises(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(dist_module, '_TORCH_AVAILABLE', False)
        ckpt = tmp_path / 'fake.ckpt'
        ckpt.touch()
        with pytest.raises(ImportError, match=r"pymovements\[dist\]"):
            DistCorrector.from_checkpoint(ckpt)


class TestFromCheckpointValidation:

    def test_missing_checkpoint_raises(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        torch = pytest.importorskip('torch')  # noqa: F841 — need guard
        monkeypatch.setattr(dist_module, '_TORCH_AVAILABLE', True)
        bogus = tmp_path / 'does-not-exist.ckpt'
        with pytest.raises(FileNotFoundError, match='DIST checkpoint not found'):
            DistCorrector.from_checkpoint(bogus)


class TestDispatcherDistBranch:
    """Dispatcher behaviour for method='dist'."""

    def _scanpath(self) -> np.ndarray:
        return np.array([[100.0, 102.0], [200.0, 99.0], [300.0, 152.0]])

    def test_returns_midline_values_when_midlines_provided(self) -> None:
        midlines = [100.0, 150.0, 200.0]

        class _Stub:
            def predict(self, dffix, trial):  # noqa: ARG002
                return np.array([0, 0, 1], dtype=int)

        result = correct_vertical_drift(
            self._scanpath(),
            midlines=midlines,
            method='dist',
            dist_corrector=_Stub(),
            dffix='<ignored-stub>',
            trial={'ignored': True},
        )
        np.testing.assert_array_equal(result, np.array([100.0, 100.0, 150.0]))

    def test_returns_indices_without_midlines(self) -> None:
        class _Stub:
            def predict(self, dffix, trial):  # noqa: ARG002
                return np.array([0, 1, 2], dtype=int)

        result = correct_vertical_drift(
            self._scanpath(),
            method='dist',
            dist_corrector=_Stub(),
            dffix='<ignored-stub>',
            trial={'ignored': True},
        )
        np.testing.assert_array_equal(result, np.array([0.0, 1.0, 2.0]))

    def test_clamps_out_of_range_indices(self) -> None:
        midlines = [100.0, 150.0]

        class _Stub:
            def predict(self, dffix, trial):  # noqa: ARG002
                return np.array([0, 5, 1], dtype=int)  # 5 is out of range

        result = correct_vertical_drift(
            self._scanpath(),
            midlines=midlines,
            method='dist',
            dist_corrector=_Stub(),
            dffix='<ignored-stub>',
            trial={'ignored': True},
        )
        np.testing.assert_array_equal(result, np.array([100.0, 150.0, 150.0]))


class TestPredictStillNotImplemented:
    """`.predict()` on a real DistCorrector is still a TODO (tracked)."""

    def test_predict_raises_not_implemented(
            self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(dist_module, '_TORCH_AVAILABLE', True)
        # Build via direct constructor (bypassing from_checkpoint) to avoid
        # needing torch to be installed.
        corrector = DistCorrector(model=object(), model_cfg={})
        with pytest.raises(NotImplementedError, match='preprocessing'):
            corrector.predict(dffix=None, trial={})
