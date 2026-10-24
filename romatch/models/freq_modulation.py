"""Frequency-domain feature enhancement for cross-domain retinal matching.

Idea (adapted from FAMNet's FAM/MSFM): different retinal modalities (fundus / FFA /
SLO / OCTA-CFP) share the same underlying structure (vessels, optic disc) but differ
hugely in appearance. Decomposing a feature map into low / mid / high frequency bands
isolates the *stable* mid band (vessel skeleton) from the noisier low/high bands. We
use the mid band to gate the others (a lightweight, channel-wise version of MSFM), so
the matcher sees a more domain-invariant feature.

Designed for very small data (~30-40 pairs):
  - frequency split is parameter-free;
  - fusion is channel gating (SE), not the O(N^2) spatial attention of FAMNet/MSFM;
  - a learnable ``alpha`` residual gates the contribution (``x + alpha * out``); it starts
    small (0.1) so the module begins close to the pretrained RoMa behaviour but still
    receives gradient (alpha=0 left it inert on short runs).

The same module instance is applied to both images (im_A and im_B) -> shared weights,
symmetric, and it nudges both modalities toward a common structural representation.
"""

import torch
import torch.nn as nn


class FreqBandFilter(nn.Module):
    """Radial low/mid/high frequency decomposition of a 2D feature map (no params).

    Unlike FAMNet (which works on 1D foreground point sequences and reshapes them into
    a square), RoMa features are already dense 2D maps, so we FFT them directly.

    ``cutoff`` splits both edges symmetrically: ``low_r = max_r * cutoff`` and
    ``high_r = max_r * (1 - cutoff)``. ``max_r`` is the distance to the *corner* of the
    Fourier plane, i.e. ``sqrt(2)`` times the axis Nyquist, so the high band only reaches
    the axes once ``(1 - cutoff) * sqrt(2) <= 1``, that is ``cutoff >= 1 - 1/sqrt(2) =
    0.2929``. Below that the high band is a set of four corners holding no purely
    horizontal or vertical frequency at all -- an orientation bias with nothing behind it,
    and at the old 0.25 default it covered only 17% of the plane. 0.35 clears the
    threshold with room to spare (high_r / Nyquist = 0.92, 37% of the plane).

    Do not push it much further: the bands are annuli, feature spectra are red, and a
    narrow band carries almost no energy. At 0.45 the mid band is 15% of the plane and
    holds 2-15% of a channel's energy, so it contributes essentially nothing to the sum.
    """

    def __init__(self, cutoff: float = 0.35):
        super().__init__()
        self.cutoff = cutoff
        self._cache = {}  # (H, W, device) -> [3, 1, 1, H, W] low/mid/high masks

    def _masks(self, H, W, device):
        key = (H, W, str(device))
        if key in self._cache:
            return self._cache[key]
        cy, cx = H // 2, W // 2
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        dist = torch.sqrt((y - cy).float() ** 2 + (x - cx).float() ** 2)
        max_r = (cy ** 2 + cx ** 2) ** 0.5
        low_r, high_r = max_r * self.cutoff, max_r * (1 - self.cutoff)
        low = (dist <= low_r).float()
        high = (dist >= high_r).float()
        mid = ((dist > low_r) & (dist < high_r)).float()
        masks = torch.stack([low, mid, high])[:, None, None]  # [3, 1, 1, H, W]
        self._cache[key] = masks
        return masks

    def forward(self, x):  # x: [B, C, H, W]
        x = x.float()  # FFT needs float32 (unstable / unsupported in fp16)
        xf = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        low_m, mid_m, high_m = self._masks(x.shape[-2], x.shape[-1], x.device)

        low = torch.fft.ifft2(torch.fft.ifftshift(xf * low_m, dim=(-2, -1)), dim=(-2, -1)).real
        mid = torch.fft.ifft2(torch.fft.ifftshift(xf * mid_m, dim=(-2, -1)), dim=(-2, -1)).real
        high = torch.fft.ifft2(torch.fft.ifftshift(xf * high_m, dim=(-2, -1)), dim=(-2, -1)).real

        return low, mid, high  # each [B, C, H, W], float32


class FreqFeatureModulation(nn.Module):
    """Lightweight per-image frequency-domain feature enhancement.

    Splits the feature map into low/mid/high bands; uses the stable mid band to produce
    SE channel gates that modulate the low and high bands; fuses, then adds back with a
    learnable ``alpha`` residual. Apply the *same* instance to both im_A and im_B so the
    weights are shared.

    The gates read the mid band's *relative power* rather than the mid band itself. The
    band is exactly zero-mean -- the DC term is in the low band -- so the leading
    ``AdaptiveAvgPool2d`` used to hand both gates an all-zero input, which made them
    emit an input-independent constant and collapsed the whole module to a fixed linear
    filter. See :meth:`forward`.

    The gate activation is ``Softplus``, not ``Sigmoid``: a gate capped at 1 can only
    ever *attenuate* the low and high bands, hard-coding "the mid band matters most" into
    the architecture. On modalities whose landmarks live in the high band that prior is
    exactly backwards, and unbounded gates let the module say so.

    ``alpha_init`` defaults to a small positive value (0.1) rather than 0: with alpha=0
    the residual ``out`` gets almost no gradient and the module never engages on short
    fine-tuning runs (observed alpha staying ~0 after 800 steps). A small non-zero start
    breaks that chicken-and-egg. Set ``alpha_init=0`` to recover the exact identity start.
    """

    def __init__(self, dim: int, cutoff: float = 0.35, reduction: int = 8,
                 alpha_init: float = 0.1):
        super().__init__()
        self.filt = FreqBandFilter(cutoff)

        self.gate_low = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Softplus(),
        )  # mid-band power -> channel gate for the low band

        self.gate_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Softplus(),
        )  # mid-band power -> channel gate for the high band

        self.fuse = nn.Conv2d(dim, dim, 1)                    # fuse the three bands
        self.alpha = nn.Parameter(torch.full((1,), alpha_init))  # residual gate

    def forward(self, x):  # x: [B, C, H, W]
        in_dtype = x.dtype
        x = x.float()
        low, mid, high = self.filt(x)
        power = mid.pow(2) / x.pow(2).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        refined = mid + self.gate_low(power) * low + self.gate_high(power) * high
        out = self.fuse(refined)
        return (x + self.alpha * out).to(in_dtype)
