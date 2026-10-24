"""Episodic wrapper over RoMa's own scene datasets, for few-shot training on MegaDepth.

This is the MegaDepth/ScanNet twin of :mod:`romatch.datasets.fewshot_homography_octa`:
it turns a list of ``MegadepthScene`` / ``ScanNetScene`` objects into episodes, each
carrying one **query** pair plus one **support** pair.

One *scene* is one domain
-------------------------
The supports and the query are always drawn from the *same scene*. A scene is the only
thing on MegaDepth that plays the role the modality plays on the retinal benchmarks:
shared capture conditions, illumination and structure. Drawing supports across scenes
would hand the alignment module a prior for a scene the query is not in, which is not
the FSS setup the retinal datasets implement.

What the query carries
----------------------
Nothing new. The query keeps the scene dataset's own keys -- ``im_A`` / ``im_B`` /
``im_A_depth`` / ``im_B_depth`` / ``K1`` / ``K2`` / ``T_1to2`` -- so ``RobustLosses``
takes its **depth** branch, byte-for-byte the supervision official RoMa trains under.
No ``dense_warp_A_to_B`` or ``H_A_to_B`` key is emitted, which is exactly what keeps
the loss on that branch.

What the supports carry
-----------------------
``SupportAlignmentModule`` reads a dense warp, and depth+pose is not one, so the warp is
baked here with :func:`romatch.utils.utils.get_gt_warp`: the same function the loss uses,
with the invalid pixels (occluded, depth-inconsistent, out of bounds) overwritten by the
``100.0`` sentinel the module tests for. It is baked at a *coarse* resolution -- the
module nearest-resamples it to feature size anyway, so a full-resolution field would be
float64 work thrown away in the loader.

Episodes are random by default (the index is ignored, as in the retinal twin). Pass
``episode_seed`` to make ``__getitem__(idx)`` deterministic, which is what a validation
split needs: the same episodes at every evaluation, so the metric moves with the model
rather than with the draw.

This module adds no dependency on the retinal code path, and modifies nothing that ships
with official RoMa -- it is a new file beside ``megadepth.py`` and ``scannet.py``.
"""

import random

import torch

from romatch.utils.utils import get_gt_warp

OOB = 100.0

_QUERY_KEYS = ("im_A", "im_B", "im_A_depth", "im_B_depth", "K1", "K2", "T_1to2")


class FewShotSceneDataset(torch.utils.data.Dataset):
    """Episodes of (query pair, one support pair) drawn from one scene at a time.

    Args:
        scenes: ``MegadepthScene`` / ``ScanNetScene`` objects, as built by
            ``MegadepthBuilder.build_scenes`` or ``ScanNetBuilder.build_scenes``.
        ht: Image height the scenes were built at; the support warp grid follows it.
        wt: Image width, likewise.
        samples_per_epoch: Length reported to the sampler. Episodes are drawn fresh, so
            this only sets how many the loader may hand out before the sampler restarts.
        scene_weight_exponent: Scenes are drawn with probability proportional to
            ``len(scene) ** scene_weight_exponent``. The default ``0.25`` reproduces the
            official scene-level marginal: ``weight_scenes(alpha=0.75)`` gives each of a
            scene's ``n`` samples weight ``n**-0.75``, so the scene's total mass under
            the official flat sampler is ``n * n**-0.75 == n**0.25``. An episode never
            crosses scenes, so that weighting has to be applied here rather than by a
            flat sampler over the concatenated pairs.
        warp_res: ``(h, w)`` the support warp is baked at. Defaults to ``(ht//4, wt//4)``
            -- fine enough for alignment at scales 16/8/4, and 16x cheaper than full
            resolution. Raise it when aligning at scale 2 or 1.
        min_valid_frac: An episode is redrawn while the support warp has a smaller
            fraction of valid correspondences than this.
        max_retries: How many support redraws to spend on that before accepting the
            last one. Each redraw costs one image+depth load, so this trades loader
            throughput for prior quality; the query is never redrawn.
        depth_interpolation_mode: Passed to ``get_gt_warp`` when baking a support warp.
        relative_depth_error_threshold: Likewise; the depth-consistency tolerance that
            decides which support pixels count as correspondences.
        episode_seed: ``None`` for fresh random episodes (training). An int makes
            ``__getitem__(idx)`` a pure function of ``idx``, which is what a validation
            split needs.
    """

    def __init__(self, scenes, ht, wt, samples_per_epoch=100_000,
                 scene_weight_exponent=0.25, warp_res=None,
                 min_valid_frac=0.05, max_retries=4,
                 depth_interpolation_mode="bilinear",
                 relative_depth_error_threshold=0.05,
                 episode_seed=None):
        super().__init__()
        self.scenes = [s for s in scenes if len(s) >= 2]
        if not self.scenes:
            raise ValueError(
                f"no scene holds the 2 pairs a 1-shot episode needs "
                f"(checked {len(scenes)} scenes)"
            )
        self.scene_weights = [len(s) ** scene_weight_exponent for s in self.scenes]
        self.ht, self.wt = ht, wt
        self._length = samples_per_epoch
        self.warp_res = tuple(warp_res) if warp_res else (ht // 4, wt // 4)
        self.min_valid_frac = min_valid_frac
        self.max_retries = max_retries
        self.depth_interpolation_mode = depth_interpolation_mode
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.episode_seed = episode_seed
        print(
            f"[FewShotScene] {len(self.scenes)} scenes, "
            f"{sum(len(s) for s in self.scenes)} pairs, 1-shot episodes, "
            f"support warp baked at {self.warp_res}, epoch length {self._length}"
            + ("" if episode_seed is None else f", deterministic (seed {episode_seed})")
        )

    def __len__(self):
        return self._length

    def _rng(self, idx):
        """Per-item RNG: the shared one while training, a fixed one for a val split."""
        if self.episode_seed is None:
            return random
        return random.Random(self.episode_seed + idx)

    @torch.no_grad()
    def _support_warp(self, sample):
        """Bake one support pair's depth + pose into a dense warp with an OOB sentinel.

        Args:
            sample: A scene sample carrying ``im_A_depth`` / ``im_B_depth`` / ``T_1to2``
                / ``K1`` / ``K2``.

        Returns:
            tuple[torch.Tensor, float]: The ``(h, w, 2)`` warp in normalized ``[-1, 1]``
            coordinates, with invalid pixels set to :data:`OOB`, and the fraction of
            pixels that carry a valid correspondence.
        """
        h, w = self.warp_res
        x2, prob = get_gt_warp(
            sample["im_A_depth"][None], sample["im_B_depth"][None],
            sample["T_1to2"][None], sample["K1"][None], sample["K2"][None],
            depth_interpolation_mode=self.depth_interpolation_mode,
            relative_depth_error_threshold=self.relative_depth_error_threshold,
            H=h, W=w,
        )
        warp, valid = x2[0].float(), prob[0] > 0
        warp = torch.where(valid[..., None], warp, torch.full_like(warp, OOB))
        return warp, valid.float().mean().item()

    def _draw_support_index(self, scene, rng, exclude):
        """Pick one pair index from a scene, avoiding the query.

        Rejection sampling rather than ``rng.sample`` over a filtered range: a MegaDepth
        scene holds up to 100k pairs, and materialising that list on every redraw would
        cost more than the redraw saves. ``__init__`` guarantees the scene holds at
        least 2 pairs, so this terminates.

        Args:
            scene: The scene to draw from.
            rng: The episode's random source.
            exclude: The query's index, which the support may not repeat.

        Returns:
            int: An index distinct from ``exclude``.
        """
        while True:
            candidate = rng.randrange(len(scene))
            if candidate != exclude:
                return candidate

    def __getitem__(self, idx):
        rng = self._rng(idx)
        scene = rng.choices(self.scenes, weights=self.scene_weights, k=1)[0]
        query_idx = rng.randrange(len(scene))
        query = scene[query_idx]
        for _ in range(self.max_retries):
            pick = self._draw_support_index(scene, rng, query_idx)
            support = scene[pick]
            warp, frac = self._support_warp(support)
            if frac >= self.min_valid_frac:
                break

        episode = {key: query[key] for key in _QUERY_KEYS}
        episode["supp_im_A"] = support["im_A"]
        episode["supp_im_B"] = support["im_B"]
        episode["supp_dense_warp_A_to_B"] = warp
        return episode


class SceneQueryDataset(torch.utils.data.Dataset):
    """The same draw as :class:`FewShotSceneDataset`, minus the support pair.

    The arms that read no prior -- ``baseline`` and the ``gamma0`` family -- must see
    the same query distribution as the arms that do, or the ablation compares data as
    well as models. This draws the query exactly as the episodic dataset draws its own
    (scene picked by ``len ** scene_weight_exponent``, pair uniform within it) and stops
    there, so
    the loader never pays for shots nothing will encode.

    Args:
        scenes: As for :class:`FewShotSceneDataset`.
        samples_per_epoch: Length reported to the sampler.
        scene_weight_exponent: Scene-sampling exponent; must match the episodic
            dataset's, or the arms would differ in their data as well as their model.
        episode_seed: ``None`` for random draws, an int for a deterministic split.
    """

    def __init__(self, scenes, samples_per_epoch=100_000, scene_weight_exponent=0.25,
                 episode_seed=None):
        super().__init__()
        self.scenes = [s for s in scenes if len(s) >= 1]
        if not self.scenes:
            raise ValueError(f"no usable scene among the {len(scenes)} given")
        self.scene_weights = [len(s) ** scene_weight_exponent for s in self.scenes]
        self._length = samples_per_epoch
        self.episode_seed = episode_seed
        print(f"[SceneQuery] {len(self.scenes)} scenes, "
              f"{sum(len(s) for s in self.scenes)} pairs, epoch length {self._length}")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        rng = random if self.episode_seed is None else random.Random(self.episode_seed + idx)
        scene = rng.choices(self.scenes, weights=self.scene_weights, k=1)[0]
        sample = scene[rng.randrange(len(scene))]
        return {key: sample[key] for key in _QUERY_KEYS}
