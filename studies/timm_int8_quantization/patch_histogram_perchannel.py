"""Monkey-patch HistogramCalibrator to support per-channel (axis != None).

Import this module BEFORE using HistogramCalibrator with axis=1:

    import patch_histogram_perchannel  # noqa: F401 — applies patch on import

Original HistogramCalibrator raises NotImplementedError when axis is not None.
This patch adds:
  - Batched per-channel histogram collection via scatter_add_ (GPU-friendly)
  - Vectorized percentile compute_amax (no Python loop)
  - ProcessPoolExecutor-parallel entropy/mse compute_amax

No other modelopt code is modified.
"""

import numpy as np
import torch

from modelopt.torch.quantization.calib.histogram import (
    HistogramCalibrator,
    _compute_amax_entropy,
    _compute_amax_mse,
)

# ------------------------------------------------------------------ #
# Save originals
# ------------------------------------------------------------------ #
_orig_init = HistogramCalibrator.__init__
_orig_collect = HistogramCalibrator.collect
_orig_reset = HistogramCalibrator.reset
_orig_compute_amax = HistogramCalibrator.compute_amax


# ------------------------------------------------------------------ #
# Top-level helper for ProcessPoolExecutor (must be picklable)
# ------------------------------------------------------------------ #
def _compute_amax_one_channel(args):
    hist, edges, num_bits, unsigned, stride, start_bin, method = args
    if method == "entropy":
        return _compute_amax_entropy(hist, edges, num_bits, unsigned, stride, start_bin)
    elif method == "mse":
        return _compute_amax_mse(hist, edges, num_bits, unsigned, stride, start_bin)
    raise TypeError(f"Unknown method {method}")


# ------------------------------------------------------------------ #
# Patched __init__: remove the axis check, add per-channel state
# ------------------------------------------------------------------ #
def _patched_init(self, num_bits=8, axis=None, unsigned=False, num_bins=2048,
                  grow_method=None, skip_zeros=False, torch_hist=True):
    _orig_init.__wrapped__(self, num_bits=num_bits, axis=None, unsigned=unsigned,
                           num_bins=num_bins, grow_method=grow_method,
                           skip_zeros=skip_zeros, torch_hist=torch_hist)
    # Override axis (original forces None)
    self._axis = axis
    self._per_channel = axis is not None
    if self._per_channel:
        self._pc_hists_2d = None   # (C, num_bins) float tensor
        self._pc_max = None        # (C,) per-channel running max
        self._pc_ndim = None


# We need to bypass the original __init__ which raises on axis!=None.
# Store the raw original before we patch it.
_patched_init.__wrapped__ = _orig_init


def _new_init(self, num_bits=8, axis=None, unsigned=False, num_bins=2048,
              grow_method=None, skip_zeros=False, torch_hist=True):
    if axis is not None:
        # Per-channel path: call base _Calibrator.__init__ directly,
        # then set up our state.
        from modelopt.torch.quantization.calib.calibrator import _Calibrator
        _Calibrator.__init__(self, num_bits, axis, unsigned)
        self._num_bins = num_bins
        self._skip_zeros = skip_zeros
        self._torch_hist = torch_hist
        self._calib_bin_edges = None
        self._calib_hist = None
        self._per_channel = True
        self._pc_hists_2d = None
        self._pc_max = None
        self._pc_ndim = None
    else:
        _orig_init(self, num_bits=num_bits, axis=axis, unsigned=unsigned,
                   num_bins=num_bins, grow_method=grow_method,
                   skip_zeros=skip_zeros, torch_hist=torch_hist)
        self._per_channel = False


# ------------------------------------------------------------------ #
# Patched collect: batched per-channel via scatter_add_
# ------------------------------------------------------------------ #
@torch.no_grad()
def _collect_per_channel(self, x):
    axis = self._axis
    self._pc_ndim = x.ndim
    x = x.movedim(axis, 0).flatten(start_dim=1)  # (C, M)
    C, M = x.shape
    nb = self._num_bins

    x_max = x.amax(dim=1)  # (C,)

    if self._pc_hists_2d is None:
        self._pc_max = x_max.clone()
        self._pc_hists_2d = torch.zeros(C, nb, device=x.device)
    else:
        grew = x_max > self._pc_max
        if grew.any():
            old_max = self._pc_max.clone()
            self._pc_max = torch.maximum(self._pc_max, x_max)
            for c in grew.nonzero(as_tuple=True)[0]:
                if old_max[c] == 0:
                    continue
                old_w = old_max[c] / nb
                new_w = self._pc_max[c] / nb
                old_hist = self._pc_hists_2d[c].clone()
                centres = torch.arange(nb, device=x.device).float() * old_w + old_w * 0.5
                new_idx = (centres / (new_w + 1e-20)).long().clamp(0, nb - 1)
                self._pc_hists_2d[c].zero_()
                self._pc_hists_2d[c].scatter_add_(0, new_idx, old_hist)

    widths = (self._pc_max / nb).unsqueeze(1)  # (C, 1)
    bin_idx = (x / (widths + 1e-20)).long().clamp(0, nb - 1)  # (C, M)
    batch_h = torch.zeros(C, nb, device=x.device)
    batch_h.scatter_add_(1, bin_idx, torch.ones_like(x))
    self._pc_hists_2d += batch_h


def _new_collect(self, x):
    if torch.min(x) < 0.0:
        x = x.abs()
    x = x.float()
    if self._per_channel:
        _collect_per_channel(self, x)
    else:
        _orig_collect(self, x)


# ------------------------------------------------------------------ #
# Patched reset
# ------------------------------------------------------------------ #
def _new_reset(self):
    _orig_reset(self)
    if self._per_channel:
        self._pc_hists_2d = None
        self._pc_max = None
        self._pc_ndim = None


# ------------------------------------------------------------------ #
# Patched compute_amax: vectorized percentile, parallel entropy/mse
# ------------------------------------------------------------------ #
def _compute_amax_per_channel(self, method, *, stride=1, start_bin=128,
                              percentile=99.99):
    if self._pc_hists_2d is None:
        return None

    C, nb = self._pc_hists_2d.shape
    widths = self._pc_max / nb

    if method == "percentile":
        # Fully vectorized — no Python loop
        cdf = self._pc_hists_2d.cumsum(dim=1)
        total = cdf[:, -1:].clamp(min=1)
        idx = (cdf / total >= percentile / 100.0).float().argmax(dim=1)
        amax = idx.float() * widths
    else:
        # entropy / mse: parallel via ProcessPoolExecutor
        from concurrent.futures import ProcessPoolExecutor
        hists_np = self._pc_hists_2d.int().cpu().numpy()
        maxes_np = self._pc_max.cpu().numpy()
        args = []
        for c in range(C):
            edges = np.linspace(0, float(maxes_np[c]), nb + 1)
            args.append((hists_np[c], edges, self._num_bits,
                         self._unsigned, stride, start_bin, method))
        workers = min(C, 32)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            raw = list(pool.map(_compute_amax_one_channel, args))
        amax = torch.stack([r if r is not None else torch.tensor(0.0)
                            for r in raw])

    if self._pc_ndim is not None:
        shape = [1] * self._pc_ndim
        shape[self._axis] = C
        amax = amax.reshape(shape)
    return amax


def _new_compute_amax(self, method, *, stride=1, start_bin=128,
                      percentile=99.99):
    if self._per_channel:
        return _compute_amax_per_channel(self, method, stride=stride,
                                         start_bin=start_bin,
                                         percentile=percentile)
    return _orig_compute_amax(self, method, stride=stride, start_bin=start_bin,
                              percentile=percentile)


# ------------------------------------------------------------------ #
# Apply patches
# ------------------------------------------------------------------ #
HistogramCalibrator.__init__ = _new_init
HistogramCalibrator.collect = _new_collect
HistogramCalibrator.reset = _new_reset
HistogramCalibrator.compute_amax = _new_compute_amax
