"""
Self-supervised synthetic pair dataset for RoMa, built from MeDAL-Retina.

MeDAL-Retina ships single, unpaired fundus images: ``Auxilliary_Training``
(1920 unlabeled images, used as the training pool) and
``SuperRetina_training_data`` (268 images with 39 vessel-keypoint annotations
each, used as the validation pool). There is no real second view of either
image, so a pair is *synthesized*: two independent random transforms (an
affine plus a control-point elastic/TPS field) are applied to the same source
image to produce ``im_A`` and ``im_B``, and the dense correspondence between
them is known analytically -- not fit from noisy sparse annotations -- which
avoids the GT-limited ceiling that ``homography_octa.py``'s real-pair TPS
supervision runs into (see the ``roma-homography-training`` project notes:
sparse-landmark leave-one-out noise there exceeds model error).

Expected on-disk layout (``data_root``, e.g. ``.../datasets/Retina``)::

    data_root/MeDAL-Retina/Auxilliary_Training/*.png                       # train pool, unlabeled
    data_root/MeDAL-Retina/SuperRetina_training_data/ImageData/*.jpg       # val/test pool
    data_root/MeDAL-Retina/SuperRetina_training_data/Annotations/<id>.txt  # optional keypoints

Annotation format: one keypoint per line, ``<x> <y>`` normalized to ``[0, 1]``
(a leading index column, if a release ships one, is ignored; a trailing blank
line is tolerated).

This module also hosts the affine-warp primitives shared with
``homography_octa.py`` (:func:`rand_affine`, :func:`apply_affine_px`,
:func:`warp_tensor`) as free
functions taking explicit parameters instead of dataset-instance methods --
``homography_octa.py`` imports them from here rather than the other way
around, since MeDAL is the base synthetic-pair generator and the real-pair
loader reuses its primitives.
"""

import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.interpolate import RBFInterpolator
from torchvision import transforms

from romatch.utils.utils import get_tuple_transform_ops

# Validation/test pairs must be reproducible across calls -- training-time
# checkpoint selection compares the same synthesized query pair epoch to
# epoch -- so their random transform is seeded from the sample index instead
# of drawn fresh.
EVAL_PAIR_SEED = 2026


# ***************** Shared affine primitives (reused by homography_octa.py) *****************

def rand_affine(wt, ht, aug_rot, aug_scale, aug_trans, flip=False, rng=None):
    """Random 3x3 affine: rotation/scale about the image center + shift.

    Returned in ``(wt, ht)`` pixel coords as a forward (src -> dst) map. The
    caller decides ``flip`` (shared across a pair, or independent -- shared
    is required when the two images are real same-eye photos, since an
    independent per-image flip would mirror one relative to the other, which
    never occurs in a real fundus/FFA pair).

    Args:
        rng: Optional ``numpy.random.Generator``/``RandomState`` for
            reproducible sampling; defaults to the global ``numpy.random``
            state.
    """
    rand = np.random if rng is None else rng
    cx, cy = wt / 2.0, ht / 2.0
    angle = np.deg2rad(rand.uniform(-aug_rot, aug_rot))
    scale = rand.uniform(1.0 - aug_scale, 1.0 + aug_scale)
    tx = rand.uniform(-aug_trans, aug_trans) * wt
    ty = rand.uniform(-aug_trans, aug_trans) * ht
    cos, sin = np.cos(angle), np.sin(angle)
    R = scale * np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    if flip:
        R = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ R  # flip x
    center = np.array([cx, cy])
    M = np.eye(3, dtype=np.float64)
    M[:2, :2] = R
    M[:2, 2] = center - R @ center + np.array([tx, ty])
    return M


def apply_affine_px(M, pts):
    """Apply a 3x3 affine matrix ``M`` to ``(N, 2)`` x-y pixel coords."""
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ M.T
    return ph[:, :2] / ph[:, 2:3]


def warp_tensor(im, M, wt, ht, border_value):
    """Affine-warp a ``(3, ht, wt)`` image tensor by pixel matrix ``M``.

    Uses cv2 semantics (``out(p) = in(M^-1 p)``), filling exposed corners
    with the (normalized) ``border_value``.
    """
    arr = im.permute(1, 2, 0).contiguous().numpy()
    warped = cv2.warpAffine(
        arr, M[:2].astype(np.float32), (wt, ht),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return torch.from_numpy(warped).permute(2, 0, 1).contiguous()


# ***************** MeDAL-specific: synthetic elastic (TPS) field *****************

def sample_elastic_ctrl_points(wt, ht, grid_size=6, max_disp_frac=0.06, rng=None):
    """Sample a regular control-point grid with random interior displacement.

    Points on the outer ring of the grid are pinned (zero displacement) so
    the deformation is anchored at the image border: every pixel stays
    inside the convex hull of the control points (no TPS extrapolation), and
    the field cannot by itself introduce black padding at the canvas edge.

    Args:
        wt, ht: Canvas size in pixels.
        grid_size: Control points per axis (``grid_size ** 2`` total).
        max_disp_frac: Max interior displacement as a fraction of ``min(wt, ht)``.
        rng: Optional ``numpy.random.Generator``/``RandomState``; defaults to
            the global ``numpy.random`` state.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(src, dst)``, each ``(grid_size**2, 2)``.
    """
    rand = np.random if rng is None else rng
    xs = np.linspace(0, wt - 1, grid_size)
    ys = np.linspace(0, ht - 1, grid_size)
    gx, gy = np.meshgrid(xs, ys)
    src = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    max_disp = max_disp_frac * min(wt, ht)
    disp = rand.uniform(-max_disp, max_disp, size=src.shape)
    is_border = (
        (src[:, 0] <= 0) | (src[:, 0] >= wt - 1)
        | (src[:, 1] <= 0) | (src[:, 1] >= ht - 1)
    )
    disp[is_border] = 0.0
    dst = src + disp
    return src, dst


def invert_deformation_points(deform, targets, n_iter=30, tol=1e-6):
    """Solve ``deform(x) = t`` for x, by fixed-point iteration.

    ``deform`` is a near-identity smooth map (identity plus a small
    displacement), so ``x <- x + (t - deform(x))`` is a contraction and
    converges geometrically. Used only for the handful of annotated
    keypoints -- the dense field never needs an inverse, because the
    deformation is defined in the direction both the rendered image and the
    GT warp consume (see :class:`MedalPairDataset`).

    Args:
        deform: Callable mapping ``(N, 2)`` points to ``(N, 2)`` points.
        targets: ``(N, 2)`` points in the deformed frame.
        n_iter: Maximum iterations.
        tol: Stop once the largest residual falls below this, in pixels.

    Returns:
        tuple[np.ndarray, float]: The ``(N, 2)`` pre-images and the final
        maximum residual in pixels.
    """
    x = np.asarray(targets, dtype=np.float64).copy()
    residual = np.zeros_like(x)
    for _ in range(n_iter):
        residual = targets - deform(x)
        x = x + residual
        if np.abs(residual).max() < tol:
            break
    return x, float(np.abs(residual).max())


class MedalPairDataset(torch.utils.data.Dataset):
    """Synthesizes RoMa training/eval pairs from single MeDAL-Retina images.

    Every ``__getitem__`` draws one source image and renders two views of it:
    ``im_A`` through a random affine composed with a random elastic (TPS)
    field, ``im_B`` through an independent random affine alone.

    The elastic field sits on the ``im_A`` branch on purpose, because that
    makes the ground truth *exact*. Rendering ``im_A`` needs the map from its
    own pixel grid back to canonical-image coordinates, and the dense GT --
    which is indexed by ``im_A``'s grid -- needs that same map before pushing
    the result forward through ``im_B``'s (closed-form invertible) affine. So
    one TPS fit, in one direction, serves both: no numerical inverse enters
    the correspondence at all. Putting the elastic on ``im_B`` instead would
    force the GT to invert it, and fitting a second TPS the other way round
    from the same control points is only an approximate inverse -- measured
    at ~1 px mean error for a 6% displacement, i.e. the same GT-noise floor
    this dataset exists to avoid. Only the sparse annotated keypoints need an
    inverse, and those are solved to ~1e-6 px by
    :func:`invert_deformation_points`.

    For ``split="train"`` a fresh random pair is drawn every call. For
    ``split in ("val", "test")`` the transform is seeded from the sample
    index (:data:`EVAL_PAIR_SEED`) so repeated evaluation compares the same
    synthesized pairs.
    """

    def __init__(
        self,
        data_root="data",
        subset="MeDAL-Retina",
        split="train",
        ht=560,
        wt=560,
        normalize=True,
        photometric=True,
        jitter_params=(0.3, 0.3, 0.2, 0.05),  # brightness, contrast, saturation, hue
        aug_rot=15.0,
        aug_scale=0.2,
        aug_trans=0.15,
        aug_hflip=0.5,
        elastic_grid_size=6,
        elastic_max_disp_frac=0.06,
    ):
        super().__init__()
        self.data_root = data_root
        self.subset = subset
        self.split = split
        self.ht, self.wt = ht, wt
        self.aug_rot, self.aug_scale = aug_rot, aug_scale
        self.aug_trans, self.aug_hflip = aug_trans, aug_hflip
        self.elastic_grid_size = elastic_grid_size
        self.elastic_max_disp_frac = elastic_max_disp_frac

        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            self._border_value = tuple((-mean / std).tolist())
        else:
            self._border_value = (0.0, 0.0, 0.0)

        self.im_transform_ops = get_tuple_transform_ops(resize=(ht, wt), normalize=normalize)
        # Only jitter on the training pool: val/test pairs must be
        # reproducible, and ColorJitter draws from torch's global RNG which
        # the per-index numpy seeding below does not control.
        self.photo_jitter = (
            transforms.ColorJitter(*jitter_params) if (photometric and split == "train") else None
        )

        self.image_paths, self.keypoints = self._gather_samples(subset, split)
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found for subset={subset} split={split} under {data_root}. "
                "Expected Auxilliary_Training/ (train) or "
                "SuperRetina_training_data/ImageData/ (val/test)."
            )

    def _gather_samples(self, subset, split):
        subset_dir = os.path.join(self.data_root, subset)
        if split == "train":
            image_dir = os.path.join(subset_dir, "Auxilliary_Training")
            paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
            return paths, {}
        image_dir = os.path.join(subset_dir, "SuperRetina_training_data", "ImageData")
        ann_dir = os.path.join(subset_dir, "SuperRetina_training_data", "Annotations")
        paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        keypoints = {}
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            kp = self._read_keypoints(os.path.join(ann_dir, stem + ".txt"))
            if kp is not None:
                keypoints[p] = kp
        return paths, keypoints

    @staticmethod
    def _read_keypoints(ann_path):
        """Parse ``<x> <y>`` normalized ``[0,1]`` keypoints, one per line.

        The last two whitespace-separated fields are taken as ``x, y``, so a
        release that prefixes an index column parses the same way. Blank (or
        index-only) trailing lines are skipped.

        Returns:
            np.ndarray | None: ``(N, 2)`` normalized coords, or ``None`` if
            the annotation file is absent or empty.
        """
        if not os.path.exists(ann_path):
            return None
        pts = []
        with open(ann_path, encoding="utf-8-sig") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue  # blank or index-only trailing line
                x, y = float(parts[-2]), float(parts[-1])
                pts.append((x, y))
        return np.array(pts, dtype=np.float64) if pts else None

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        im_pil = self._load_image(path)
        w0, h0 = im_pil.size

        rand = np.random if self.split == "train" else np.random.RandomState(EVAL_PAIR_SEED + idx)

        if self.photo_jitter is not None:
            im_A_pil, im_B_pil = self.photo_jitter(im_pil), self.photo_jitter(im_pil)
        else:
            im_A_pil, im_B_pil = im_pil, im_pil
        # (3, ht, wt) canonical tensors: same geometry, independently jittered.
        im_A_canon, im_B_canon = self.im_transform_ops((im_A_pil, im_B_pil))

        flip_A = self.aug_hflip > 0 and rand.uniform() < self.aug_hflip
        flip_B = self.aug_hflip > 0 and rand.uniform() < self.aug_hflip
        A = rand_affine(self.wt, self.ht, self.aug_rot, self.aug_scale, self.aug_trans, flip_A, rand)
        B = rand_affine(self.wt, self.ht, self.aug_rot, self.aug_scale, self.aug_trans, flip_B, rand)

        src_ctrl, dst_ctrl = sample_elastic_ctrl_points(
            self.wt, self.ht, self.elastic_grid_size, self.elastic_max_disp_frac, rand
        )
        # The one elastic fit: im_A's own pixel grid -> the pre-elastic (affine-A)
        # frame. Both the im_A rendering and the dense GT read it in this
        # direction, so neither pays an inversion error.
        elastic = RBFInterpolator(src_ctrl, dst_ctrl, kernel="thin_plate_spline")

        gy, gx = np.mgrid[0:self.ht, 0:self.wt].astype(np.float64)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (ht*wt, 2)

        # phi: im_A pixel grid -> canonical-image pixel coords (elastic, then affine A^-1).
        phi = apply_affine_px(np.linalg.inv(A), elastic(grid))

        # im_A: resampled through phi in one pass.
        arr = im_A_canon.permute(1, 2, 0).contiguous().numpy()
        map_x = phi[:, 0].reshape(self.ht, self.wt).astype(np.float32)
        map_y = phi[:, 1].reshape(self.ht, self.wt).astype(np.float32)
        warped = cv2.remap(
            arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=self._border_value,
        )
        im_A = torch.from_numpy(warped).permute(2, 0, 1).contiguous()

        # im_B: pure affine on the canonical (jittered) tensor -- exact, closed-form.
        im_B = warp_tensor(im_B_canon, B, self.wt, self.ht, self._border_value)

        # Dense GT: im_A grid -> canonical coords (phi) -> im_B grid (affine B).
        oob = (
            (phi[:, 0] < 0) | (phi[:, 0] > self.wt - 1)
            | (phi[:, 1] < 0) | (phi[:, 1] > self.ht - 1)
        )
        dst = apply_affine_px(B, phi)

        dense_warp = np.stack([
            2.0 * dst[:, 0] / self.wt - 1.0,
            2.0 * dst[:, 1] / self.ht - 1.0,
        ], axis=-1).astype(np.float32)
        dense_warp[oob] = 100.0
        dense_warp = dense_warp.reshape(self.ht, self.wt, 2)

        identifier = os.path.splitext(os.path.basename(path))[0]
        out = {
            "im_A": im_A,
            "im_B": im_B,
            "dense_warp_A_to_B": torch.from_numpy(dense_warp),
            "im_A_size": torch.tensor([w0, h0], dtype=torch.float32),
            "im_B_size": torch.tensor([w0, h0], dtype=torch.float32),
            "im_A_path": path,
            "im_B_path": path,
            "im_A_identifier": f"{identifier}_A",
            "im_B_identifier": f"{identifier}_B",
        }

        kp = self.keypoints.get(path)
        if kp is not None:
            kp_px = kp * np.array([self.wt, self.ht])  # normalized [0,1] -> canonical pixel coords
            # im_B side is the plain affine, exact. The im_A side is the one
            # place an inverse is needed -- solved on these few points only.
            ctrl_B = apply_affine_px(B, kp_px)
            ctrl_A, _ = invert_deformation_points(elastic, apply_affine_px(A, kp_px))
            out["ctrl_pts_A"] = torch.from_numpy(ctrl_A.astype(np.float32))
            out["ctrl_pts_B"] = torch.from_numpy(ctrl_B.astype(np.float32))

        return out


def build_medal(data_root="data", subset="MeDAL-Retina", split="train", ht=560, wt=560, **kwargs):
    """Convenience builder. Wrap the returned dataset in a standard DataLoader.

    Example:
        ds = build_medal("data", split="train", ht=h, wt=w)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True,
                                             num_workers=8, drop_last=True)
    """
    return MedalPairDataset(
        data_root=data_root, subset=subset, split=split, ht=ht, wt=wt, **kwargs,
    )
