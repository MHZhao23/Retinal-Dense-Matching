"""
TPS-supervised retinal image-pair dataset for RoMa.

Loads cross-modal pairs (OCTA/MIP <-> CFP, or CALLISTO frames) annotated with
sparse control-point correspondences, fits a thin-plate spline (TPS) warp per
pair, and yields a dense warp field in the format RoMa's training loop /
``RobustLosses`` expect.

Expected on-disk layout (data_root = "data"):

    data/<subset>/homography/<entry>.json       # keys: cfp_path, octa_path, pairs (+ optionally homography)
    data/<subset>/fold<fold>/<split>_list.txt   # one <entry> per line (no .json)
    data/<cfp_path>, data/<octa_path>           # image paths as stored in the json

Convention: im_A := octa/MIP image, im_B := cfp image.  The "pairs" field in the
json contains {"octa": [x,y], "cfp": [x,y]} correspondences in original pixel
coords; these are rescaled to (ht, wt) and used to fit a TPS that defines the
dense forward warp im_A -> im_B stored as ``dense_warp_A_to_B`` (H, W, 2) in
normalized [-1,1] coords. Every annotated subset carries at least four
correspondences, so TPS is the only supervision path; a json without them is
an error rather than a fall back to the "homography" field.

Online geometric augmentation (ported from Attention-Reg's
``retina_tps_dataset``): on the TPS path each image is independently warped by a
random affine (rotation / scale / flip / shift) and that transform is folded
into the dense GT warp, so the network sees fresh geometry every epoch on the
tiny per-fold splits. See ``_getitem_tps``.
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from torchvision import transforms

from romatch.datasets.medal import apply_affine_px, rand_affine, warp_tensor
from romatch.utils.utils import get_tuple_transform_ops


class HomographyOctaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root="data",
        subsets=("OCTA", "MISP", "MEMO", "FIRE", "FA-SLO", "CALLISTOeye"),
        fold=0,
        split="train",
        ht=560,
        wt=560,
        normalize=True,
        homography_dir="homography",
        photometric=True,
        jitter_params=(0.3, 0.3, 0.2, 0.05),  # brightness, contrast, saturation, hue
        geometric_aug=True,
        aug_rot=10.0,       # +/- deg per image (relative rot mostly <= ~12 deg, max 30)
        aug_scale=0.15,     # scale in [1 - s, 1 + s] (on top of real FOV diff from TPS)
        aug_trans=0.1,      # +/- fraction of image size
        aug_hflip=0.5,      # prob of a SHARED h-flip applied to BOTH im_A & im_B
        ctrl_radius_px=48,  # neighborhood radius (px, at ht/wt) for ctrl_mask
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.ht, self.wt = ht, wt
        self.ctrl_radius_px = ctrl_radius_px
        self.geometric_aug = geometric_aug and split == "train"
        self.aug_rot, self.aug_scale = aug_rot, aug_scale
        self.aug_trans, self.aug_hflip = aug_trans, aug_hflip
        if normalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            self._border_value = tuple((-mean / std).tolist())
        else:
            self._border_value = (0.0, 0.0, 0.0)

        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt), normalize=normalize
        )
        self.photo_jitter = (
            transforms.ColorJitter(*jitter_params)
            if (photometric and split == "train")
            else None
        )

        self.samples = self._gather_samples(subsets, fold, split, homography_dir)
        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found for subsets={subsets} fold={fold} split={split} "
                f"under {data_root}. Check the *_list.txt files and homography dir."
            )

    def _gather_samples(self, subsets, fold, split, homography_dir):
        samples = []
        for subset in subsets:
            subset_dir = os.path.join(self.data_root, subset)
            hom_dir = os.path.join(subset_dir, homography_dir)
            entries = self._read_entries(subset_dir, fold, split, hom_dir)
            for entry in entries:
                json_path = os.path.join(hom_dir, entry + ".json")
                if not os.path.exists(json_path):
                    continue
                samples.append(json_path)
        return samples

    def _read_entries(self, subset_dir, fold, split, hom_dir):
        """Return the list of <entry> ids for this subset/split.

        Prefers the fold list file; if absent, falls back to every json found
        under the homography dir (relative path without extension).
        """
        list_path = os.path.join(subset_dir, f"fold{fold}", f"{split}_list.txt") if fold is not None \
            else os.path.join(subset_dir, f"{split}_list.txt")
        if os.path.exists(list_path):
            with open(list_path, encoding="utf-8-sig") as f:
                return [ln.strip() for ln in f if ln.strip()]
        # fallback: glob all jsons
        entries = []
        for root, _, files in os.walk(hom_dir):
            for fn in files:
                if fn.endswith(".json"):
                    rel = os.path.relpath(os.path.join(root, fn), hom_dir)
                    entries.append(os.path.splitext(rel)[0].replace(os.sep, "/"))
        return sorted(entries)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, rel_path):
        im = Image.open(os.path.join(self.data_root, rel_path)).convert("RGB")
        return im

    def __getitem__(self, idx):
        json_path = self.samples[idx]
        with open(json_path) as f:
            meta = json.load(f)

        # im_A = octa/MIP, im_B = cfp.
        im_A_pil = self._load_image(meta["octa_path"])
        im_B_pil = self._load_image(meta["cfp_path"])
        w0, h0 = im_A_pil.size  # original octa (im_A) size
        w1, h1 = im_B_pil.size  # original cfp  (im_B) size

        if self.photo_jitter is not None:
            im_A_pil = self.photo_jitter(im_A_pil)
            im_B_pil = self.photo_jitter(im_B_pil)

        im_A, im_B = self.im_transform_ops((im_A_pil, im_B_pil))  # (3, ht, wt)

        pairs = meta.get("pairs")
        if not pairs or len(pairs) < 4:
            raise ValueError(
                f"{json_path} has {0 if not pairs else len(pairs)} control-point "
                "correspondences; TPS supervision needs at least 4. Re-run this "
                "subset's converter to annotate it."
            )
        out = self._getitem_tps(meta, im_A, im_B, w0, h0, w1, h1)

        # Scoring a pair in a frame other than the one it was annotated in needs the
        # per-pair json itself (ANHIR keeps the original slide size and the
        # original -> installed matrix under "meta", which this loader ignores).
        # A string collates fine and costs nothing when unused.
        out["json_path"] = json_path
        return out

    def _getitem_tps(self, meta, im_A, im_B, w0, h0, w1, h1):
        """Build supervision from TPS fitted on annotated control points."""
        pairs = meta["pairs"]
        # Control points in original pixel coords, (x, y) convention.
        octa_pts = np.array([p["octa"] for p in pairs], dtype=np.float64)  # (N, 2)
        cfp_pts  = np.array([p["cfp"]  for p in pairs], dtype=np.float64)

        # Scale to training resolution.
        octa_pts = octa_pts * np.array([self.wt / w0, self.ht / h0])
        cfp_pts  = cfp_pts  * np.array([self.wt / w1, self.ht / h1])

        # Fit forward TPS: octa (im_A) pixel coords -> cfp (im_B) pixel coords.
        rbf_fwd = RBFInterpolator(octa_pts, cfp_pts, kernel="thin_plate_spline")

        # Full-resolution im_A pixel grid (x-first); the query lattice for the warp.
        gy, gx = np.mgrid[0:self.ht, 0:self.wt].astype(np.float64)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (ht*wt, 2)

        oob = None
        A = B = None
        if self.geometric_aug:
            do_flip = self.aug_hflip > 0 and np.random.rand() < self.aug_hflip
            A = rand_affine(self.wt, self.ht, self.aug_rot, self.aug_scale, self.aug_trans, do_flip)
            B = rand_affine(self.wt, self.ht, self.aug_rot, self.aug_scale, self.aug_trans, do_flip)
            im_A = warp_tensor(im_A, A, self.wt, self.ht, self._border_value)
            im_B = warp_tensor(im_B, B, self.wt, self.ht, self._border_value)
            src = apply_affine_px(np.linalg.inv(A), grid)
            cfp_coords = apply_affine_px(B, rbf_fwd(src))
            oob = (
                (src[:, 0] < 0) | (src[:, 0] > self.wt - 1)
                | (src[:, 1] < 0) | (src[:, 1] > self.ht - 1)
            )
        else:
            cfp_coords = rbf_fwd(grid)  # (ht*wt, 2)

        # Convert pixel coords to normalized [-1, 1].
        dense_warp = np.stack([
            2.0 * cfp_coords[:, 0] / self.wt - 1.0,
            2.0 * cfp_coords[:, 1] / self.ht - 1.0,
        ], axis=-1).astype(np.float32)  # (ht*wt, 2)
        if oob is not None:
            dense_warp[oob] = 100.0
        dense_warp = dense_warp.reshape(self.ht, self.wt, 2)

        # Dense-warp pixels near an annotated point are the TPS fit's interpolation
        # region; everything else is extrapolation, measurably less accurate (see the
        # retina GT-floor diagnostic: leave-one-out ctrl-pt error runs 1.6-2x the
        # model's own dense EPE). ctrl_mask marks the im_A-grid pixels within
        # ctrl_radius_px of an annotated point, in whichever frame this sample's grid
        # actually is -- the aug-mapped original frame (``src``) when geometric_aug put
        # im_A through a random affine, otherwise the untouched grid. It is attached to
        # every sample, query or support, so the few-shot wrapper can forward the
        # support's copy under ``supp_ctrl_mask`` without this loader knowing which role
        # a given draw will play.
        ref_grid = src if self.geometric_aug else grid
        near_dist, _ = cKDTree(octa_pts).query(ref_grid, k=1)
        ctrl_mask = (near_dist <= self.ctrl_radius_px).reshape(self.ht, self.wt)

        out = {
            "im_A": im_A,
            "im_B": im_B,
            "dense_warp_A_to_B": torch.from_numpy(dense_warp),
            "ctrl_mask": torch.from_numpy(ctrl_mask),
            "im_A_size": torch.tensor([w0, h0], dtype=torch.float32),
            "im_B_size": torch.tensor([w1, h1], dtype=torch.float32),
            "im_A_path": os.path.join(self.data_root, meta["octa_path"]),
            "im_B_path": os.path.join(self.data_root, meta["cfp_path"]),
            "im_A_identifier": os.path.splitext(os.path.basename(meta["octa_path"]))[0],
            "im_B_identifier": os.path.splitext(os.path.basename(meta["cfp_path"]))[0],
        }

        # Control points at the model's output resolution (im_A source -> im_B dest), in pixel coords.
        if self.split != "train":
            out["ctrl_pts_A"] = torch.from_numpy(octa_pts.astype(np.float32))
            out["ctrl_pts_B"] = torch.from_numpy(cfp_pts.astype(np.float32))

        return out


def build_homography_octa(
    data_root="data",
    subsets=("OCTA", "MISP", "CALLISTOeye"),
    fold=0,
    split="train",
    ht=560,
    wt=560,
    **kwargs,
):
    """Convenience builder. Wrap the returned dataset in a standard DataLoader.

    Example:
        ds = build_homography_octa("data", fold=0, split="train", ht=h, wt=w)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True,
                                             num_workers=8, drop_last=True)
    """
    return HomographyOctaDataset(
        data_root=data_root,
        subsets=subsets,
        fold=fold,
        split=split,
        ht=ht,
        wt=wt,
        **kwargs,
    )
