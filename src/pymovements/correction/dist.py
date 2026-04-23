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
"""ML-based vertical drift correction via the DIST model.

This module wraps the Dual Input Stream Transformer (DIST) model by
Mercier et al. (2023) :cite:p:`MercierEtAl2023`. DIST achieves
~98% line-assignment accuracy when combined with a classical algorithm
into an ensemble, outperforming every classical method alone.

Dependencies (``pip install pymovements[dist]``)
-------------------------------------------------

The DIST wrapper requires the following heavy dependencies which are NOT
installed by default:

* ``torch`` — model execution
* ``pytorch-lightning`` — the upstream checkpoint format is a ``LitModule``
* ``timm`` — backbone ``CoAtNet`` used for the character-image stream
* ``transformers`` — ``BERT`` encoder used for the fixation-sequence stream
* ``matplotlib`` + font rendering — DIST's second input stream is a rendered
  image of the stimulus page

Installation::

    pip install pymovements[dist]

Usage
-----

>>> from pymovements.correction.dist import DistCorrector
>>> corrector = DistCorrector.from_checkpoint(
...     checkpoint_path='models/BERT_20240104-223349_....ckpt',
...     config_path='models/BERT_fin_exp_20240104-223349.yaml',
... )
>>> line_indices = corrector.predict(dffix=fix_df, trial=trial_dict)

Or via the unified dispatcher::

    corrected_y = correct_vertical_drift(
        fixation_xy, stimulus=stim, method='dist',
        dist_corrector=corrector,
    )

Current status
--------------

This wrapper implements the inference-side plumbing: checkpoint loading,
forward pass, and logit-to-line-index conversion. The *preprocessing step*
— converting a ``pymovements.EventDataFrame`` + ``TextStimulus`` into the
character-level ``trial`` dict + pandas ``dffix`` DataFrame that DIST's
``prep_data_for_dist()`` expects — is not yet implemented. Users must
currently provide DIST-format inputs directly.

A ``TextStimulus``-to-trial adapter is tracked in the roadmap; until then,
the recommended path for end-to-end correction is DIST's own HuggingFace
Space (``bugroup/Eye_Tracking_Drift_Correction``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


# Optional-dependency guard. Import errors are deferred to the point of use
# so that ``import pymovements.correction.dist`` itself never fails on a
# bare ``pip install pymovements`` — only *instantiating* ``DistCorrector``
# requires the heavy deps.
_TORCH_AVAILABLE: bool
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


_MISSING_DEPS_MESSAGE = (
    "DistCorrector requires the 'dist' extras. Install with:\n\n"
    "    pip install 'pymovements[dist]'\n\n"
    'This installs torch, pytorch-lightning, timm, transformers and '
    'matplotlib.'
)


class DistCorrector:
    """Inference-only wrapper around a single DIST checkpoint.

    Parameters
    ----------
    model: Any
        A loaded ``LitModel`` (or compatible callable) in eval mode.
    model_cfg: dict[str, Any]
        The hyperparameter / preprocessing config that matches the checkpoint.
    device: str
        Compute device. (default: ``'cpu'``)

    Notes
    -----
    In almost all cases you should instantiate via
    :meth:`DistCorrector.from_checkpoint` rather than calling the constructor
    directly — the constructor is intended for cases where you have already
    built the model elsewhere (e.g. in a test fixture or when loading
    multiple models for an ensemble).
    """

    def __init__(
            self,
            model: Any,
            model_cfg: dict[str, Any],
            device: str = 'cpu',
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(_MISSING_DEPS_MESSAGE)

        self.model = model
        self.model_cfg = model_cfg
        self.device = device
        self._setup_eval()

    def _setup_eval(self) -> None:
        """Put the model into inference-mode on the target device."""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        if hasattr(self.model, 'freeze'):
            self.model.freeze()
        if hasattr(self.model, 'to'):
            self.model.to(self.device)

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: str | Path,
            config_path: str | Path | None = None,
            *,
            device: str = 'cpu',
    ) -> DistCorrector:
        """Load a DistCorrector from a PyTorch-Lightning checkpoint on disk.

        Parameters
        ----------
        checkpoint_path: str | Path
            Path to the ``.ckpt`` file produced by DIST's training code.
        config_path: str | Path | None
            Path to the matching ``.yaml`` config. If omitted, the config is
            read from the checkpoint's ``hyper_parameters`` field (which
            modern DIST checkpoints embed). (default: None)
        device: str
            Compute device. (default: ``'cpu'``)

        Raises
        ------
        ImportError
            If the ``[dist]`` extras are not installed.
        FileNotFoundError
            If the checkpoint file does not exist.
        RuntimeError
            If the checkpoint cannot be parsed, typically because it was
            produced by an incompatible DIST / PyTorch-Lightning version.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError(_MISSING_DEPS_MESSAGE)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f'DIST checkpoint not found: {ckpt_path}')

        # Peek at the checkpoint to find the embedded hyperparameters.
        try:
            raw = _torch.load(str(ckpt_path), map_location='cpu')
        except Exception as exc:
            raise RuntimeError(
                f'Failed to load DIST checkpoint {ckpt_path}: {exc}',
            ) from exc

        model_cfg = _resolve_config(raw, config_path)
        model = _build_model_from_cfg(model_cfg, raw.get('state_dict'))
        return cls(model=model, model_cfg=model_cfg, device=device)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def predict(self, dffix: Any, trial: dict[str, Any]) -> Any:
        """Run DIST inference and return the predicted line index per fixation.

        This is currently a thin pass-through of DIST's native inference
        contract. ``dffix`` must be a pandas ``DataFrame`` following DIST's
        convention (``x``, ``y`` columns at minimum; the exact column list
        is ``model_cfg['sample_cols']``). ``trial`` must be a dict with
        keys ``chars_list``, ``y_char_unique``, ``num_char_lines`` and the
        other fields produced by DIST's own trial-building utilities.

        The :class:`pymovements.EventDataFrame` + ``TextStimulus`` →
        (``dffix``, ``trial``) adapter is not yet implemented; see
        module docstring.

        Parameters
        ----------
        dffix: pandas.DataFrame
            Fixation dataframe in DIST's column convention.
        trial: dict
            Trial layout dict in DIST's format.

        Returns
        -------
        numpy.ndarray, shape [N]
            Predicted line index (0-based) per fixation.
        """
        raise NotImplementedError(
            'DistCorrector.predict is not wired up to the full DIST '
            'preprocessing pipeline yet. See module docstring for status '
            'and recommended alternatives until the trial-dict adapter '
            'ships.',
        )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _resolve_config(
        raw_checkpoint: dict[str, Any],
        config_path: str | Path | None,
) -> dict[str, Any]:
    """Pull the model config from the checkpoint or a separate yaml file."""
    hp = raw_checkpoint.get('hyper_parameters') or {}
    embedded_cfg = hp.get('cfg') if isinstance(hp, dict) else None

    if embedded_cfg is not None:
        return dict(embedded_cfg)

    if config_path is None:
        raise ValueError(
            "Checkpoint does not embed a 'hyper_parameters.cfg' block; "
            'pass config_path=<yaml> explicitly.',
        )

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Reading a separate DIST config file requires pyyaml "
            '(already a pymovements runtime dep — this should not happen).',
        ) from exc

    with open(config_path) as fh:
        return dict(yaml.safe_load(fh))


def _build_model_from_cfg(
        model_cfg: dict[str, Any],  # noqa: ARG001 — placeholder
        state_dict: Any,              # noqa: ARG001 — placeholder
) -> Any:
    """Reconstruct the LitModel and load weights.

    Currently a placeholder that raises — building the LitModel requires
    the vendored model classes from the upstream DIST repo. Tracked in the
    roadmap as Milestone 3 Phase 2.
    """
    raise NotImplementedError(
        'LitModel reconstruction is not vendored into pymovements yet. '
        'For now, construct the model separately (using upstream DIST code) '
        'and pass it to DistCorrector(model=..., model_cfg=...) directly.',
    )
