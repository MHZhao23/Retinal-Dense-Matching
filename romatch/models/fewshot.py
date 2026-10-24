"""
Few-shot cross-domain registration on top of RoMa.

Setup (mirrors few-shot segmentation): one *support* image pair carries a GT warp and
acts as the modality prior; a *query* image pair is the one we actually register. The
support correspondences yield a per-episode "prototype" -- a cross-modal feature
alignment transform (PATNet-style) -- that conditions the query features before RoMa's
decoder. See ``FEWSHOT_REGISTRATION.md`` / the plan for the full rationale.

``SupportAlignmentModule`` frequency-filters the query and support features together in
one batched call. The support features estimate the per-episode ridge map ``M``, which
is then applied to the filtered query im_B features as a residual scaled by a fixed
scalar ``gamma``. The filtered query pair is passed to RoMa's decoder. Since ``gamma``
is not stored in the model state dict, evaluation must use the same value as training.

Ablation switches
-----------------
Three independent switches carve the module into the arms of the ablation, and each has
to be set the same way for training and for scoring -- a training-time augmentation and
a module that co-adapts with the decoder cannot be flipped after the fact without
measuring something other than the arm one means to report:

``use_support``
    The support-derived ridge residual. Off is the "gamma = 0" arm: ``phi`` / ``psi`` /
    ``gamma`` are not built at all, so the residual is dead for the whole run rather
    than amputated afterwards, and the support pair is never encoded. What is left
    still runs over the query pair, which is the point -- the frequency filter is
    support-*independent*, so a gain that survives here never needed a support.
``use_perturb``
    The training-only channel-stat perturbation of the im_B features.
``use_freq``
    The shared low/mid/high frequency modulation. Off, the filter is not built and the
    features pass through untouched.

With every switch off the module is a no-op; that arm is built with no alignment scales
at all instead, which leaves the graph and the state dict identical to vanilla RoMa's.
See :mod:`experiments.utils.variants` for the arm table.

``M`` is the ridge solution of ``min_M ||A - M B||^2 + lambda ||M||^2``, whose normal
equations are

    M = (A B^T) (B B^T + lambda I)^-1

Control-point neighborhood (fine scales only)
-----------------------------------------------
The support pair's dense GT warp is itself a TPS fit over a handful of annotated
control points (see ``romatch.datasets.homography_octa``); away from them it is
extrapolation, measurably less accurate than near them. On the scales fine enough to
still have a usable pool after restricting to that neighborhood (``scale <=
CTRL_MASK_MAX_SCALE``, i.e. 1/2/4 at the usual pyramid), ``solve_maps`` intersects the
in-bounds mask with a ``ctrl_mask`` the loader provides, trading pool size for
correspondence quality. The coarse scales (8/16) keep the full in-bounds pool
unconditionally -- their feature grid is coarse enough that a handful of points'
neighborhoods can collapse to only a few, or even zero, distinct cells. ``ctrl_mask``
is optional throughout: a support pair with no annotated control points (MeDAL,
MegaDepth) simply falls back to the full pool everywhere.

``FewShotRegressionMatcher`` subclasses ``RegressionMatcher``: it reuses the encoder,
decoder and every match/eval helper unchanged, only wrapping ``forward`` to encode the
support pair and condition the query im_B feature pyramid. A support-conditioned arm is
support-conditioned everywhere -- training and scoring both hand it a prior -- so a batch
that carries none is a bug rather than a fallback, and the forward says so. Use
``match_fewshot()``, or stash the prior on ``_support``, rather than plain ``match()``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from romatch.models.freq_modulation import FreqFeatureModulation
from romatch.models.matcher import RegressionMatcher

# Raw encoder feature-pyramid channel counts (pre-proj), keyed by RoMa scale.
# {16: ViT patch tokens, 8/4/2/1: VGG19 features}. See encoders.py / roma_models.py.
# The scale-16 width is the ViT's embedding dim; all three supported backbones are
# ViT-L, so 1024 covers them. ``roma_fewshot_model`` passes the built encoder's actual
# width rather than reading this, so a future non-1024 backbone needs no edit here.
PYRAMID_DIMS = {16: 1024, 8: 512, 4: 256, 2: 128, 1: 64}

# Align scales at or finer than this restrict the support correspondence pool to the
# ctrl_mask neighborhood (see SupportAlignmentModule.solve_maps); coarser scales always
# use the full in-bounds pool regardless of ctrl_mask, because their feature grid is so
# coarse (e.g. 40x40 at scale 16, 560px input) that a handful of annotated points'
# neighborhoods can collapse to only a few, or even zero, distinct cells.
CTRL_MASK_MAX_SCALE = 4


class SupportAlignmentModule(nn.Module):
    """Per-episode cross-modal feature alignment estimated from one support pair.

    Given the support pair features ``f_sA``/``f_sB`` and the support GT warp, sample
    matched (im_A, im_B) feature pairs, solve a ridge regression map ``M`` (im_B -> im_A
    space), and apply it to the query im_B features as a gated residual.
    """

    def __init__(self, in_dim, proj_dim=256, ridge_lambda=1e-2, max_samples=2048,
                 min_samples=16, gamma_init=0.0, freq_cutoff=0.35,
                 freq_reduction=8, freq_alpha_init=0.1, perturb_noise_std=0.25,
                 use_support=True, use_perturb=True, use_freq=True,
                 use_ctrl_mask=False):
        super().__init__()
        self.use_ctrl_mask = bool(use_ctrl_mask)
        self.proj_dim = min(in_dim, proj_dim)
        self.ridge_lambda = ridge_lambda
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.use_support = bool(use_support)
        self.use_perturb = bool(use_perturb)
        self.use_freq = bool(use_freq)
        self.perturb_noise_std = float(perturb_noise_std)
        if self.perturb_noise_std < 0.0:
            raise ValueError("perturb_noise_std must be non-negative")

        if self.use_support:
            self.phi = nn.Conv2d(in_dim, self.proj_dim, 1, 1, 0)  # pre-projection C -> d
            self.psi = nn.Conv2d(self.proj_dim, in_dim, 1, 1, 0)  # back-projection d -> C
            self.gamma = float(gamma_init)
        else:
            self.phi = self.psi = self.gamma = None

        self.support_filter = FreqFeatureModulation(
            in_dim,
            cutoff=freq_cutoff,
            reduction=freq_reduction,
            alpha_init=freq_alpha_init,
        ) if self.use_freq else None

    @staticmethod
    def _apply_perturb(f_qB, f_sB, scale_std, shift_std):
        r"""Apply one episode-level style perturbation to query/support B features.

        For each channel, the perturbation is

        .. math::
            x' = s(x - \mu_x) + \mu_x + \delta\sigma_x,
            \quad s = \exp(\epsilon_s),
            \quad \epsilon_s \sim \mathcal{N}(0, \mathrm{scale\_std}^2),
            \quad \delta \sim \mathcal{N}(0, \mathrm{shift\_std}^2).

        ``scale_std`` is the standard deviation of the sampled log-scale. It controls
        multiplicative contrast/style variation; ``exp`` keeps the scale positive.
        ``shift_std`` is the standard deviation of the dimensionless mean shift
        ``delta``. The actual additive shift is ``delta * sigma_x``, so it also affects
        zero-mean channels and follows each channel's feature magnitude.

        The sampled ``s`` and ``delta`` are shared by query B and the support shot,
        making the whole episode one synthetic modality. Each tensor keeps its own
        spatial mean ``mu_x`` and standard deviation ``sigma_x``.

        The support tensor is optional: the gamma = 0 arms encode no support, and the
        perturbation is then a plain augmentation of the query im_B features. The draw
        is taken from the query, so it is the same distribution either way.

        Args:
            f_qB: Query im_B features, shape ``(B, C, h, w)``.
            f_sB: Support im_B features, shape ``(B, C, h, w)``, or ``None``.
            scale_std: Standard deviation of the sampled log-scale.
            shift_std: Standard deviation of the dimensionless mean shift.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Perturbed query features and the
            perturbed support features, or ``None`` when none were given.
        """
        B, C = f_qB.shape[:2]
        shape = (B, C, 1, 1)  # one draw per episode, shared by query and support
        log_scale = torch.randn(shape, device=f_qB.device, dtype=f_qB.dtype) * scale_std
        mean_shift = torch.randn(shape, device=f_qB.device, dtype=f_qB.dtype) * shift_std
        scale = log_scale.exp()

        q_mean = f_qB.mean(dim=(-2, -1), keepdim=True)
        q_std = f_qB.std(dim=(-2, -1), keepdim=True, unbiased=False)
        perturbed_qB = scale * (f_qB - q_mean) + q_mean + mean_shift * q_std

        if f_sB is None:
            perturbed_sB = None
        else:
            s_mean = f_sB.mean(dim=(-2, -1), keepdim=True)
            s_std = f_sB.std(dim=(-2, -1), keepdim=True, unbiased=False)
            perturbed_sB = scale * (f_sB - s_mean) + s_mean + mean_shift * s_std

        return perturbed_qB, perturbed_sB

    def _apply_filter(self, branches):
        """Frequency-filter every feature branch in one batched call.

        The filter is branch-agnostic, so the maps are concatenated along the batch
        axis and split back afterwards rather than filtered one call at a time.
        Returns the branches untouched when the arm has no filter.

        Args:
            branches: Feature tensors, each ``(N_i, C, h, w)`` with a common ``C``.

        Returns:
            list[torch.Tensor]: One filtered tensor per input, in the same order.
        """
        if self.support_filter is None:
            return list(branches)
        sizes = [f.shape[0] for f in branches]

        return list(self.support_filter(torch.cat(list(branches), dim=0)).split(sizes, dim=0))

    def solve_maps(self, f_sA, f_sB, supp_warp, ctrl_mask=None, filtered=False):
        """Solve one ridge map per episode from the support features alone.

        Split out of :meth:`forward` because the result depends on nothing but the
        support: at eval the perturbation is off (it is gated on ``self.training``) and
        the frequency filter is per-sample -- no BatchNorm, an FFT and a 1x1 conv, with
        the batch axis only there to save kernel launches -- so filtering the support
        here rather than alongside the query is numerically the same tensor. That makes
        the maps cacheable across every query that shares a prior, which is what
        ``FewShotRegressionMatcher`` does; see its ``_support`` property.

        Each episode's map is the closed-form ridge solution
        ``argmin_M ||A - M B||^2 + lambda ||M||^2`` over its correspondence pool. Too
        few of them (or a solve that does not come out finite) leaves that episode's
        slot ``None``, and the residual falls back to the identity. The pool is every
        in-bounds pixel of the (downsampled) warp, further restricted to the
        ``ctrl_mask`` neighborhood on the scales that use it (see
        :data:`CTRL_MASK_MAX_SCALE`) -- the dense warp away from an annotated point is
        TPS-extrapolated and measurably less accurate, so those scales trade pool size
        for correspondence quality. Oversized pools are cut down to ``max_samples`` by
        an evenly spaced (not random) selection, so a run's correspondence set is a
        function of the support alone, not of the RNG state at the moment it is solved.

        Args:
            f_sA, f_sB: support im_A / im_B features, ``(B, C, h, w)``.
            supp_warp: support GT warp im_A->im_B, ``(B, H, W, 2)``.
            ctrl_mask: Optional ``(B, H, W)`` boolean mask, true near an annotated
                support control point (full im_A resolution; downsampled here to match
                this scale). Read only when ``self.use_ctrl_mask`` is set; ignored --
                the full in-bounds pool is used -- when ``None`` or on a scale built
                with ``use_ctrl_mask=False``.
            filtered: The features have already been through the frequency filter, so
                do not filter them again. ``forward`` passes them that way, having run
                all four branches through one batched call; see there for why.

        Returns:
            list: One ``(d, d)`` map per episode, or ``None`` in that slot when the
            episode carried no usable correspondences and the residual falls back to
            the identity.
        """
        d = self.proj_dim
        f_sA = f_sA.float()
        f_sB = f_sB.float()
        f_sA_filtered, f_sB_filtered = (f_sA, f_sB) if filtered else self._apply_filter([f_sA, f_sB])

        za = F.normalize(self.phi(f_sA_filtered), dim=1)  # (B, d, h, w)
        zb = F.normalize(self.phi(f_sB_filtered), dim=1)

        h, w = za.shape[-2:]
        warp = supp_warp.permute(0, 3, 1, 2).float()       # (B, 2, H, W)
        warp = F.interpolate(warp, size=(h, w), mode="nearest")
        warp = warp.permute(0, 2, 3, 1)                    # (B, h, w, 2)
        valid = (warp.abs() <= 1).all(dim=-1)              # (B, h, w)

        if self.use_ctrl_mask and ctrl_mask is not None:
            near = F.interpolate(
                ctrl_mask[:, None].float(), size=(h, w), mode="nearest"
            )[:, 0] > 0.5                                   # (B, h, w)
            valid = valid & near

        # Resample support im_B feats onto im_A's grid: za[..,y,x] ~ zb_al[..,y,x] match.
        zb_al = F.grid_sample(zb, warp, align_corners=False, mode="bilinear")  # (B, d, h, w)

        maps = []
        for b in range(za.shape[0]):
            v = valid[b].reshape(-1)                            # (h*w,)
            A = za[b].reshape(d, -1)[:, v]                       # (d, Nv)
            Bm = zb_al[b].reshape(d, -1)[:, v]                   # (d, Nv)
            Nv = A.shape[1]
            if Nv < self.min_samples:  # no usable prior
                maps.append(None)
                continue
            if Nv > self.max_samples:
                # Evenly spaced rather than random: the correspondence set a run solves
                # from should depend only on (support, scale), not on training's ongoing
                # RNG state or an eval-only seed switch. May land on the same index
                # twice when rounding collides, same convention as select_support_indices.
                idx = torch.unique(
                    torch.linspace(0, Nv - 1, self.max_samples, device=A.device).round().long()
                )
                A, Bm = A[:, idx], Bm[:, idx]

            abt, bbt = A @ Bm.t(), Bm @ Bm.t()
            eye = torch.eye(d, device=bbt.device, dtype=bbt.dtype)
            reg = self.ridge_lambda * bbt.diagonal().mean().clamp_min(1e-6) * eye
            m = torch.linalg.solve(bbt + reg, abt.t()).t()
            maps.append(m if torch.isfinite(m).all() else None)

        return maps

    def forward(self, f_qA, f_qB, f_sA=None, f_sB=None, supp_warp=None, ctrl_mask=None,
                maps=None):
        """Condition the query features on a one-shot support prior.

        Support features are optional. Without them -- either because the arm sets
        ``use_support=False`` or because the caller has no prior -- only the query pair
        is processed, and the ridge residual is skipped entirely rather than computed
        and multiplied by zero.

        Args:
            f_qA, f_qB: query im_A / im_B features, ``(B, C, h, w)``.
            f_sA, f_sB: support im_A / im_B features, ``(B, C, h, w)``, or ``None``.
            supp_warp:  support GT warp im_A->im_B, ``(B, H, W, 2)``, normalized
                        ``[-1, 1]`` xy (OOB pixels carry the sentinel 100.0 from the
                        dataset loader). ``None`` alongside ``f_sA``.
            ctrl_mask:  Optional ``(B, H, W)`` control-point neighborhood mask, passed
                        through to :meth:`solve_maps` when ``maps`` is not already
                        solved. See there.
            maps:       Pre-solved ridge maps from :meth:`solve_maps`, one per episode.
                        Given, the support features are not needed and are ignored.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: filtered query im_A features and
            conditioned query im_B features.
        """
        qA_dtype, qB_dtype = f_qA.dtype, f_qB.dtype
        d = self.proj_dim

        if self.use_support and maps is None:
            episode_sizes = [f.shape[0] for f in (f_qA, f_qB, f_sA, f_sB)]
            if len(set(episode_sizes)) != 1:
                raise ValueError(
                    "query and support features must cover the same episodes, got batch "
                    f"sizes {episode_sizes}"
                )
            f_sA, f_sB = f_sA.float(), f_sB.float()
        else:
            f_sA = f_sB = None

        f_qA, f_qB = f_qA.float(), f_qB.float()
        if self.use_perturb and self.training and self.perturb_noise_std > 0.0:
            f_qB, f_sB = self._apply_perturb(
                f_qB, f_sB, scale_std=self.perturb_noise_std,
                shift_std=self.perturb_noise_std,
            )

        if not self.use_support:
            f_qA_filtered, f_qB_filtered = self._apply_filter([f_qA, f_qB])
            return f_qA_filtered.to(qA_dtype), f_qB_filtered.to(qB_dtype)

        if maps is None:
            # All four branches through one batched filter call, which is what this did
            # before the maps became cacheable. The filter's convs are shared, so their
            # weight gradient is one reduction over the concatenated batch here and two
            # partial reductions if the call is split -- the same number to within float32
            # rounding, but not the same bits. Keeping the single call keeps a training
            # run bit-identical to the pre-cache one.
            f_qA_filtered, f_qB_filtered, f_sA_filtered, f_sB_filtered = self._apply_filter(
                [f_qA, f_qB, f_sA, f_sB]
            )
            maps = self.solve_maps(
                f_sA_filtered, f_sB_filtered, supp_warp, ctrl_mask=ctrl_mask, filtered=True
            )
        else:
            f_qA_filtered, f_qB_filtered = self._apply_filter([f_qA, f_qB])

        B = f_qB_filtered.shape[0]
        if len(maps) == 1 and B > 1:
            maps = maps * B  # one stashed prior, shared by every query in the batch
        if len(maps) != B:
            raise ValueError(
                f"got {len(maps)} ridge maps for a batch of {B} queries; a prior must "
                "cover either one episode (shared) or each episode in the batch"
            )

        h, w = f_qB_filtered.shape[-2:]
        zbq = F.normalize(self.phi(f_qB_filtered), dim=1)  # (B, d, h, w)

        out = torch.empty_like(f_qB_filtered)
        for b in range(B):
            M = maps[b]
            if M is None:  # no usable correspondences -> identity
                out[b] = f_qB_filtered[b]
                continue
            zq = zbq[b].reshape(d, -1)                   # (d, h*w)
            aligned = M @ zq                             # (d, h*w)
            residual = self.psi((aligned - zq).reshape(1, d, h, w))[0]  # (C, h, w)
            out[b] = f_qB_filtered[b] + self.gamma * residual

        return f_qA_filtered.to(qA_dtype), out.to(qB_dtype)


class FewShotRegressionMatcher(RegressionMatcher):
    """RoMa matcher conditioned on a one-shot support prior via ``SupportAlignmentModule``.

    Adds ``support_align`` (one module per conditioned scale). ``forward`` encodes both
    the support pair and the query pair, jointly filters every feature branch, and
    aligns query im_B before the decoder.

    ``use_support`` decides what a batch without ``supp_*`` keys means:

    * ``True`` (the few-shot arms) -- a mistake. These arms are trained on episodes and
      scored on episodes, so the prior is part of the model's input contract; a batch
      without one would silently score a different model than the one that was trained,
      and the forward raises instead.
    * ``False`` (the gamma = 0 arms, and the no-adaptation arms built with no align
      scales at all) -- expected: there is no prior *by construction*, so the alignment
      path runs over the query pair alone. Skipping it would bypass the trained
      frequency filter and score a different model.
    """

    def __init__(self, *args, align_scales=(16,), align_kwargs=None, pyramid_dims=None, **kwargs):
        super().__init__(*args, **kwargs)
        align_kwargs = dict(align_kwargs or {})
        self.align_scales = [int(s) for s in align_scales]
        self.use_support = bool(align_kwargs.get("use_support", True))
        dims = {**PYRAMID_DIMS, **{int(s): int(d) for s, d in (pyramid_dims or {}).items()}}
        self.support_align = nn.ModuleDict(
            {
                str(s): SupportAlignmentModule(
                    dims[int(s)], use_ctrl_mask=(s <= CTRL_MASK_MAX_SCALE), **align_kwargs
                )
                for s in self.align_scales
            }
        )
        self._support_tensors = None
        self._support_maps = {}

    @property
    def _support(self):
        """The stashed inference prior, or ``None``.

        A plain attribute in everything but name; the property exists so that assigning
        a new prior -- or clearing it -- drops the ridge maps solved from the old one.
        """
        return self._support_tensors

    @_support.setter
    def _support(self, support):
        self._support_tensors = support
        self._support_maps = {}

    def _encode_pair(self, im_A, im_B, batched, upsample):
        feats = self.extract_backbone_features(
            {"im_A": im_A, "im_B": im_B}, batched=batched, upsample=upsample
        )
        f_A = {s: f.chunk(2)[0] for s, f in feats.items()}
        f_B = {s: f.chunk(2)[1] for s, f in feats.items()}
        return f_A, f_B

    def _resolve_support(self, batch):
        """The episode's prior, and whether it is the cacheable stashed one.

        Returns:
            tuple: ``(support, cacheable)``. ``support`` is the ``(B, ...)`` prior or
            ``None``; ``cacheable`` says it came from ``_support`` rather than from the
            batch, so the maps solved from it hold for every later query.
        """
        if "supp_im_A" in batch:
            support = (batch["supp_im_A"], batch["supp_im_B"],
                       batch["supp_dense_warp_A_to_B"], batch.get("supp_ctrl_mask"))
            cacheable = False  # a training batch: a fresh episode every step
        else:
            support = self._support  # set by match_fewshot(), else None
            cacheable = support is not None
        return support, cacheable

    def _cached_maps(self, supp, upsample, batched):
        """Ridge maps for the stashed prior, solved once and reused.

        The maps depend on the support alone: the perturbation is gated on
        ``self.training`` and the frequency filter is per-sample, so a map solved here
        is the tensor ``solve_maps`` would have produced inside ``forward`` (see its
        docstring). Every query sharing the prior therefore reuses it, and the support
        pair goes through the encoder once per prior rather than once per query.

        Keyed on ``upsample`` because the two passes see different feature-map sizes,
        and so solve different maps, and on inference mode because a tensor made inside
        it cannot be used outside. Training never lands here: it takes the batch path,
        where the episode changes every step and the residual needs its graph.

        Args:
            supp: The prior, ``(B, ...)``.
            upsample: Which decoder pass is running.
            batched: Passed through to the encoder.

        Returns:
            dict: ``{scale: maps}`` for the align scales present at this pass.
        """
        key = (upsample, torch.is_inference_mode_enabled())
        cached = self._support_maps.get(key)
        if cached is not None:
            return cached

        supp_im_A, supp_im_B, supp_warp, supp_ctrl_mask = supp
        f_sA, f_sB = self._encode_pair(supp_im_A, supp_im_B, batched, upsample)
        cached = {}
        for s in self.align_scales:
            if s not in f_sA:
                continue
            cached[s] = self.support_align[str(s)].solve_maps(
                f_sA[s], f_sB[s], supp_warp, ctrl_mask=supp_ctrl_mask
            )
        self._support_maps[key] = cached
        return cached

    def forward(self, batch, batched=True, upsample=False, scale_factor=1):
        supp, cacheable = self._resolve_support(batch) if self.use_support else (None, False)
        if self.use_support and supp is None:
            raise ValueError(
                "this model was built with use_support=True and is conditioned on a "
                "support prior in training as well as at inference, but the batch "
                "carries no supp_* keys and no prior is stashed on _support. Call "
                "match_fewshot(), or set _support, rather than plain match()."
            )

        f_qA, f_qB = self._encode_pair(batch["im_A"], batch["im_B"], batched, upsample)
        f_sA = f_sB = supp_warp = supp_ctrl_mask = None
        maps = {}
        if cacheable and not self.training:
            maps = self._cached_maps(supp, upsample, batched)
        elif supp is not None:
            supp_im_A, supp_im_B, supp_warp, supp_ctrl_mask = supp
            f_sA, f_sB = self._encode_pair(supp_im_A, supp_im_B, batched, upsample)

        for s in self.align_scales:
            if s not in f_qB:
                continue
            supp_A = supp_B = None
            if f_sA is not None:
                supp_A, supp_B = f_sA[s], f_sB[s]
            f_qA[s], f_qB[s] = self.support_align[str(s)](
                f_qA[s], f_qB[s], supp_A, supp_B, supp_warp,
                ctrl_mask=supp_ctrl_mask, maps=maps.get(s)
            )

        corresps = self.decoder(
            f_qA,
            f_qB,
            upsample=upsample,
            **(batch["corresps"] if "corresps" in batch else {}),
            scale_factor=scale_factor,
        )
        return corresps

    @torch.inference_mode()
    def match_fewshot(self, im_A, im_B, supp_im_A, supp_im_B, supp_warp,
                       supp_ctrl_mask=None, **kwargs):
        """FSS-style one-shot inference: register (im_A, im_B) conditioned on a support
        pair with a GT warp. Reuses the full ``match()`` pipeline by stashing the
        support prior on the module. Requires ``symmetric=False`` (match() routes
        through the few-shot ``forward``; ``forward_symmetric`` is not support-aware).
        ``supp_ctrl_mask`` is optional -- omit it for a support pair with no annotated
        control points (e.g. drawn from a MeDAL/MegaDepth source); the fine scales then
        fall back to the same full in-bounds pool the coarse scales always use."""
        assert not self.symmetric, "match_fewshot requires the model be built with symmetric=False"
        assert self.use_support, (
            "match_fewshot needs a support-conditioned model; this one was built with "
            "use_support=False (a gamma = 0 or no-adaptation arm) and ignores the prior"
        )
        self._support = (supp_im_A, supp_im_B, supp_warp, supp_ctrl_mask)
        try:
            return self.match(im_A, im_B, **kwargs)
        finally:
            self._support = None
