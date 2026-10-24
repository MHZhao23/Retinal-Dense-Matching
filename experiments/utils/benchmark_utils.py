"""Benchmark metrics, match cache, and dense evaluation for retinal RoMa.

Shared scoring machinery for retinal RoMa evaluation. Depends only on stdlib +
numpy/torch + romatch, so it imports cleanly from training and evaluation
scripts (no cycles).
"""
import json
import os
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ******************** Metric constants **********************

# ``grid_sample(align_corners=False)`` reads index x - 0.5 for the normalised
# coordinate of pixel x, while this project's GT warp is built on ``np.mgrid``
# (field index i holds the target of source pixel i). Sampling a landmark
# therefore has to be nudged forward half a pixel to land on the intended cell.
GRID_SAMPLE_OFFSET_PX = 0.5

# Control-point AUC threshold as a fraction of the original im_B diagonal. One
# rule, instantiated per dataset: 0.6% of the diagonal is 25 px on FIRE's
# 2912x2912 frame, i.e. the threshold the retinal registration literature uses.
# A pair counts as an outright failure when its worst point exceeds FAIL_TAU_MULT
# times that.
TAU_DIAG_FRAC = 0.006
FAIL_TAU_MULT = 3.0

# Headline metrics: control-point errors in the *original* image frame.
CTRL_METRIC_KEYS = ["med_px", "max_px", "auc", "succ_rate", "fail_rate"]

# Supplementary diagnostics. ``ctrl_epe`` is the mean of the same per-point
# errors as the headline metrics (median/max instead of mean), also rescaled to
# the *original* image frame here -- but the ``ctrl_epe`` logged by training-time
# validation (train_roma_fewshot.py / train_roma_medal.py) stays in *model* frame
# (the evaluation resolution), since its synthetic samples have no original size
# to rescale to. The two are not the same unit; do not compare them directly. The
# rest score the dense field against the TPS warp interpolated from those control
# points, so they are GT-limited and are not comparable across datasets or
# against published numbers -- ablation use only.
DIAG_METRIC_KEYS = ["ctrl_epe", "val_epe_hull", "hull_cov", "val_epe", "val_pck_3",
                    "val_pck_5"]

SPEED_METRIC_KEYS = ["inference_time_ms"]

# Complete metric order used when displaying full-suite RoMa results.
METRIC_ORDER = CTRL_METRIC_KEYS + DIAG_METRIC_KEYS + SPEED_METRIC_KEYS

# Pair-level error fields whose spread over the test set is worth saving next to
# the aggregate. Set-level metrics (auc/succ_rate/fail_rate) have no per-pair std by construction;
# their uncertainty comes from the support choice and is measured by the LOO sweep.
PAIR_STD_KEYS = {"epe": "val_epe_std", "ctrl_epe": "ctrl_epe_std", "med_px": "med_px_std"}


DENSE_DIAGNOSTIC_MEANS = (("ctrl_epe", "ctrl_epe"), ("epe", "val_epe"),
                          ("epe_hull", "val_epe_hull"), ("hull_cov", "hull_cov"),
                          ("pck_3", "val_pck_3"), ("pck_5", "val_pck_5"))


# ****** Generic statistics and serialization helpers ********

def is_finite_number(value):
    """Return whether ``value`` is a finite real number.

    ``int`` and ``float`` qualify; booleans, NaN, both infinities and
    non-numeric objects do not.

    Args:
        value: Value to test.

    Returns:
        bool: True when the value is a finite real number.
    """
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value))


def filter_finite_numbers(values):
    """Return finite real numbers from an iterable as Python floats.

    Args:
        values: Values to filter.

    Returns:
        list[float]: The finite real numbers converted to Python floats.
    """
    return [float(v) for v in values if is_finite_number(v)]


def compute_mean_std_count(values):
    """Summarize the finite numeric values as ``(mean, std, n)``.

    Invalid values are removed with :func:`filter_finite_numbers`. The standard
    deviation uses NumPy's population definition with ``ddof=0``.

    Args:
        values: Numeric values to summarize.

    Returns:
        tuple[float, float, int] | None: The mean, population standard
        deviation, and count, or ``None`` if no finite values remain.
    """
    finite_values = filter_finite_numbers(values)
    if not finite_values:
        return None
    return (float(np.mean(finite_values)), float(np.std(finite_values)), len(finite_values))


def metric_stats_to_dict(metric_stats):
    """Convert metric-stat tuples into a JSON-friendly named-field mapping.

    Optional tuple fields after ``(mean, std, n)`` are ignored.

    Args:
        metric_stats: Metric names mapped to tuples beginning with
            ``(mean, std, n)``.

    Returns:
        dict: Metric names mapped to ``mean``, ``std``, and ``n`` fields.
    """
    stats_by_metric = {}
    for metric_name, statistics in metric_stats.items():
        mean_value, std_value, sample_count = statistics[:3]
        stats_by_metric[metric_name] = {"mean": mean_value, "std": std_value, "n": sample_count}
    return stats_by_metric


def save_json(path, payload):
    """Write a payload as indented JSON and create its parent directory.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable data to write.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ************* Metric aggregation (control-point + dense diagnostics) *************

def cal_error_auc(errors, thresholds):
    """Compute normalized AUCs of the cumulative error curve.

    The curve plots the fraction of samples whose error is below a given value.
    For each threshold ``t``, this function integrates the curve over ``[0, t]``
    and divides the area by ``t``. The resulting score is normally in ``[0, 1]``
    and is higher when more samples have small errors. This matches
    gluefactory's ``AUCMetric`` calculation.

    Args:
        errors: Per-sample errors; lower is better.
        thresholds: Maximum error thresholds at which to compute the AUC.

    Returns:
        A list of AUC values in the same order as ``thresholds``. If ``errors``
        is empty, returns one NaN for each threshold.
    """
    if len(errors) == 0:
        return [np.nan] * len(thresholds)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(float(np.round(np.trapezoid(r, x=e) / t, 4)))
    return aucs


def aggregate_ctrl_metrics(pair_rows, tau_frac=TAU_DIAG_FRAC, fail_mult=FAIL_TAU_MULT):
    """Reduce per-pair control-point errors to the five headline metrics.

    Every pair contributes its median point error (``med_px``) and worst point error
    (``max_px``), both in original-image pixels and as a fraction of that image's
    diagonal. The absolute columns stay in native pixels because each dataset gets
    its own table; only the threshold is shared, expressed relative to the diagonal
    so one rule instantiates to the right pixel value per dataset.

    Args:
        pair_rows: Per-pair rows carrying ``med_px``, ``max_px``, ``med_rel``,
            ``max_rel`` and ``diag_px``. A row missing any of them, or holding a
            non-finite value, is dropped from every metric at once.
        tau_frac: Success threshold as a fraction of the image diagonal.
        fail_mult: Multiple of ``tau_frac`` above which a pair counts as failed.

    Returns:
        dict: ``med_px``, ``max_px``, ``auc``, ``succ_rate``, ``fail_rate`` plus ``tau_px``
        (the threshold in this dataset's pixels, for the table caption).
    """
    # Filter once, on whole rows: the five metrics below must describe the same
    # set of pairs, or e.g. succ_rate and fail_rate would silently be scored over different ones.
    fields = ("med_px", "max_px", "med_rel", "max_rel", "diag_px")
    rows = [row for row in pair_rows if all(is_finite_number(row.get(k)) for k in fields)]
    if not rows:
        return {}

    med_px, max_px, med_rel, max_rel, diag_px = (np.array([float(row[k]) for row in rows]) for k in fields)
    
    return {
        "med_px": float(np.median(med_px)),
        "max_px": float(np.median(max_px)),
        "auc": cal_error_auc(med_rel, [tau_frac])[0],
        "succ_rate": float((med_rel <= tau_frac).mean() * 100.0),
        "fail_rate": float((max_rel > fail_mult * tau_frac).mean() * 100.0),
        "tau_px": float(np.mean(diag_px) * tau_frac),
    }


def compute_testset_pair_std(pair_rows):
    """Compute population std across all image pairs in one test set.

    Args:
        pair_rows: Per-pair result rows from the dense benchmark.

    Returns:
        dict: ``<metric>_std`` entries for every field in :data:`PAIR_STD_KEYS`
        that has at least one finite value, computed with NumPy ``ddof=0``.
    """
    std_by_metric = {}
    for row_key, out_key in PAIR_STD_KEYS.items():
        values = filter_finite_numbers(row.get(row_key) for row in pair_rows)
        if values:
            std_by_metric[out_key] = float(np.std(values))
    return std_by_metric


def dense_diagnostic_means(pair_rows):
    """Mean of each dense diagnostic over the pairs that carry it.

    Args:
        pair_rows: Per-pair result rows from the dense benchmark.

    Returns:
        dict: One entry per diagnostic present in the rows.
    """
    out = {}
    for row_key, out_key in DENSE_DIAGNOSTIC_MEANS:
        values = filter_finite_numbers(row.get(row_key) for row in pair_rows)
        if values:
            out[out_key] = float(np.mean(values))
    return out


def aggregate_pair_rows(pair_rows):
    """Everything a benchmark run reports that is a function of its per-pair rows.

    This is every metric ``evaluate_full_suite`` produces except the timings, which
    measure the run rather than the pairs. Splitting it out lets a caller re-aggregate
    over a *subset* of the pairs without a second forward pass -- which is how the
    unswept arms are scored on the same query sets as the swept ones. Because both
    paths call this one function, the two cannot drift into aggregating differently.

    Args:
        pair_rows: Per-pair result rows, or any subset of them.

    Returns:
        dict: ``num_pairs``, the headline control-point metrics, the dense
        diagnostics, and the across-pair spreads.
    """
    out = {"num_pairs": len(pair_rows)}
    out.update(aggregate_ctrl_metrics(pair_rows))
    out.update(dense_diagnostic_means(pair_rows))
    out.update(compute_testset_pair_std(pair_rows))
    return out


# ************ Coordinate and geometry helpers ***************

def norm_to_px(p, W, H):
    """Convert RoMa ``[-1, 1]`` normalized coordinates to pixel coordinates.

    ``p`` may have any leading dimensions but its last dimension must be
    ``[x, y]``. The conversion follows the project's ``align_corners=False``
    convention: ``(-1, -1) -> (0, 0)`` and ``(1, 1) -> (W, H)``.
    Args:
        p: Normalized coordinates whose last dimension is ``[x, y]``.
        W: Image width in pixels.
        H: Image height in pixels.

    Returns:
        torch.Tensor: Pixel coordinates with the same shape as ``p``.
    """
    return torch.stack((W * (p[..., 0] + 1) / 2, H * (p[..., 1] + 1) / 2), dim=-1)


def px_to_norm(p, W, H):
    """Convert pixel ``[x, y]`` coordinates to RoMa's ``[-1, 1]`` space.

    This is the inverse of :func:`norm_to_px` and accepts any leading tensor
    dimensions. It is used when pixel locations must be passed to RoMa or
    ``grid_sample`` under the project's ``align_corners=False`` convention.

    Args:
        p: Pixel coordinates whose last dimension is ``[x, y]``.
        W: Image width in pixels.
        H: Image height in pixels.

    Returns:
        torch.Tensor: Normalized coordinates with the same shape as ``p``.
    """
    return torch.stack((2.0 * p[..., 0] / W - 1.0, 2.0 * p[..., 1] / H - 1.0), dim=-1)


def sample_field_at_points(field, points_px, W, H, offset_px=GRID_SAMPLE_OFFSET_PX):
    """Bilinearly read a normalized target-position field at pixel landmarks.

    Shared by every landmark-based benchmark so they all sample the field the
    same way; see :data:`GRID_SAMPLE_OFFSET_PX` for the half-pixel nudge.

    Args:
        field: Normalized target-position field with shape ``(1, 2, H, W)``.
        points_px: Query landmarks with shape ``(N, 2)``, in the field's frame.
        W: Field width in pixels.
        H: Field height in pixels.
        offset_px: Half-pixel sampling correction; pass ``0.0`` to disable.

    Returns:
        torch.Tensor: Sampled target positions with shape ``(N, 2)``, in pixels.
    """
    grid = px_to_norm(points_px + offset_px, W, H).view(1, 1, -1, 2)  # (1, 1, N, 2)
    sampled = F.grid_sample(field, grid, mode="bilinear", align_corners=False)
    return norm_to_px(sampled[0, :, 0, :].permute(1, 0), W, H)  # (N, 2)


def convex_hull_mask(points_px, src_wh, out_hw):
    """Rasterise the convex hull of the control points at the match resolution.

    Only inside this hull is the TPS ground truth an interpolation; outside it
    the GT is extrapolated and scoring against it measures the spline, not the
    model. Used to restrict the diagnostic dense metrics to the trustworthy area.

    Args:
        points_px: ``(N, 2)`` control points in the loader's resized im_A frame.
        src_wh: ``(width, height)`` of that resized frame.
        out_hw: ``(height, width)`` of the predicted match grid.

    Returns:
        torch.Tensor: Boolean ``out_hw`` mask, all-True when the hull cannot be
        built (fewer than three points).
    """
    out_h, out_w = out_hw
    src_w, src_h = src_wh
    pts = np.asarray(points_px, dtype=np.float64)
    if len(pts) < 3:
        return torch.ones(out_h, out_w, dtype=torch.bool)
    scaled = pts * np.array([out_w / src_w, out_h / src_h])
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(scaled).astype(np.int32)), 1)
    return torch.from_numpy(mask.astype(bool))


# ***************** Ground-truth dense warp ******************

def get_gt_dense_warp(sample, ht, wt):
    """Return a sample's image-A-to-B GT warp as an ``(ht, wt, 2)`` grid.

    Every loader supplies ``dense_warp_A_to_B`` -- TPS-fitted for the real
    annotated pairs, analytic for the synthesized MeDAL ones -- so this is
    now a plain accessor, kept as a named seam for the scoring callers.

    Args:
        sample: Dataset sample containing ``dense_warp_A_to_B``.
        ht: Unused; kept so callers need not know the field is precomputed.
        wt: Unused; see ``ht``.

    Returns:
        torch.Tensor: The normalized image-A-to-B target-position grid.
    """
    return sample["dense_warp_A_to_B"]


# ********************* Inference timing *********************

class TimedMatcher:
    """Run model matching and record inference latency."""

    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.times_ms = []
        model.eval()

    @torch.inference_mode()
    def run(self, compute):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = perf_counter()
        output = compute()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.times_ms.append((perf_counter() - start) * 1000.0)
        return output

    def speed_metrics(self):
        """Return timing statistics for actual model inferences."""
        if not self.times_ms:
            return {}
        mean_ms = float(np.mean(self.times_ms))
        return {
            "inference_time_ms": mean_ms,
            "inference_time_ms_std": float(np.std(self.times_ms)),
            "inference_time_ms_median": float(np.median(self.times_ms)),
            "inference_FPS": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
            "inference_num_pairs": len(self.times_ms),
        }


# *************** Benchmark implementations ******************

class DenseWarpBenchmark:
    """Evaluate RoMa's dense warp output directly against a GT dense warp.

    Compares every valid predicted image-B position with its ground-truth
    position, without fitting a homography. Two families come out of the same
    forward pass: control-point errors sampled at the annotations and reported in
    the *original* image frame (the headline metrics), and dense EPE/PCK against
    the TPS warp in the model's frame (GT-limited diagnostics). Per-pair rows are
    stored in ``pair_rows``.

    Args:
        dataset: Dataset containing image pairs and ground-truth dense warps.
        batch_size: Number of image pairs per batch.
        num_workers: Number of data-loading workers.
        timed_matcher: Optional wrapper for timing model matching.
    """

    def __init__(self, dataset, batch_size=1, num_workers=2, timed_matcher=None):
        """Initialize the dense-warp benchmark.

        Args:
            dataset: Dataset containing image pairs and ground-truth dense
                warps.
            batch_size: Number of image pairs per batch.
            num_workers: Number of data-loading workers.
            timed_matcher: Optional wrapper for timing model matching.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.timed_matcher = timed_matcher

    def dense_errors(self, dense_warp_A_to_B, dense_matches):
        """Compute the pixel EPE map of a predicted dense match grid.

        Args:
            dense_warp_A_to_B: Ground-truth normalized target coordinates with
                shape ``(B, H, W, 2)``.
            dense_matches: Predicted source and target coordinates with shape
                ``(B, H, W, 4)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The ``(B, H, W)`` pixel EPE map and
            the ``(B, H, W)`` boolean mask of locations whose GT target is in frame.
        """
        # dense_matches: (b, h1, w1, 4); [...,:2]=im_A grid, [...,2:]=pred im_B (normalized)
        # dense_warp_A_to_B: (b, ht, wt, 2) precomputed TPS warp in normalized coords
        b, h1, w1, _ = dense_matches.shape
        x2_hat = dense_matches[..., 2:].reshape(b, h1 * w1, 2)
        with torch.no_grad():
            # Resample precomputed TPS warp to match resolution.
            warp_bchw = dense_warp_A_to_B.permute(0, 3, 1, 2).float()
            warp_scaled = F.interpolate(warp_bchw, size=(h1, w1), mode="bilinear", align_corners=False)
            x2 = warp_scaled.permute(0, 2, 3, 1).reshape(b, h1 * w1, 2)
            valid = (
                (x2[..., 0] > -1) & (x2[..., 0] < 1)
                & (x2[..., 1] > -1) & (x2[..., 1] < 1)
            )
        gd = (norm_to_px(x2_hat, w1, h1) - norm_to_px(x2, w1, h1)).norm(dim=-1)
        return gd.reshape(b, h1, w1), valid.reshape(b, h1, w1)

    def geometric_dist(self, dense_warp_A_to_B, dense_matches):
        """Compute EPE for valid locations in a predicted dense match grid.

        Thin wrapper over :meth:`dense_errors` kept for the training loops, which
        want a flat tensor of the in-frame errors.

        Args:
            dense_warp_A_to_B: Ground-truth normalized target coordinates with
                shape ``(B, H, W, 2)``.
            dense_matches: Predicted source and target coordinates with shape
                ``(B, H, W, 4)``.

        Returns:
            torch.Tensor: Pixel EPE values at valid ground-truth locations.
        """
        gd, valid = self.dense_errors(dense_warp_A_to_B, dense_matches)
        return gd[valid]

    def control_point_epe(self, pred_field, src_px, dst_px, W, H, scale=None,
                          offset_px=GRID_SAMPLE_OFFSET_PX):
        """Compute EPE at annotated control points in the predicted warp.

        The predicted target field is bilinearly sampled at each source control
        point, nudged by :data:`GRID_SAMPLE_OFFSET_PX` so the query lands on the
        cell the GT warp indexes.

        Args:
            pred_field: Normalized image-B target field with shape
                ``(1, 2, H, W)``.
            src_px: Source control points with shape ``(N, 2)``.
            dst_px: Target control points with shape ``(N, 2)``.
            W: Image width in pixels.
            H: Image height in pixels.
            scale: Optional ``(sx, sy)`` factor applied to the displacement before
                taking its norm, to report the error in the original image frame
                rather than the model's; ``None`` leaves it in model pixels.
            offset_px: Half-pixel sampling correction; pass ``0.0`` to disable.

        Returns:
            torch.Tensor: Per-control-point EPE values in pixels.
        """
        delta = sample_field_at_points(pred_field, src_px, W, H, offset_px) - dst_px
        if scale is not None:
            delta = delta * scale
        return delta.norm(dim=-1)  # (N,)

    @torch.inference_mode()
    def benchmark(self, model):
        """Evaluate a model against the dataset's ground-truth dense warps.

        Fills ``self.pair_rows`` with one row per image pair, carrying the
        headline control-point errors and the control-point mean (``ctrl_epe``),
        both in the original image frame, plus the diagnostic dense errors
        (model frame).

        Args:
            model: RoMa model to evaluate.

        Returns:
            dict: Aggregate control-point metrics plus the dense diagnostics.
        """
        was_training = model.training
        model.train(False)
        dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=False,
        )
        self.pair_rows = []
        for data in dataloader:
            im_A, im_B = data["im_A"].cuda(), data["im_B"].cuda()
            dense_warp = data["dense_warp_A_to_B"].cuda()
            id_A = data.get("im_A_identifier", ["?"] * im_A.shape[0])
            id_B = data.get("im_B_identifier", ["?"] * im_A.shape[0])
            if self.timed_matcher is not None:
                matches, _ = self.timed_matcher.run(
                    lambda: model.match(im_A, im_B, batched=True)
                )
            else:
                matches, _ = model.match(im_A, im_B, batched=True)
            ws, hs = im_A.shape[-1], im_A.shape[-2]
            matches = matches[:, :, :ws, :]  # A->B half (robust if symmetric=True)
            gd, valid = self.dense_errors(dense_warp, matches)
            cp_A, cp_B = data.get("ctrl_pts_A"), data.get("ctrl_pts_B")
            size_B = data.get("im_B_size")
            mw, mh = matches.shape[-2], matches.shape[-3]

            for b in range(im_A.shape[0]):
                gd_b, valid_b = gd[b], valid[b]
                in_frame = gd_b[valid_b]
                row = {
                    "pair": f"{id_A[b]}__{id_B[b]}",
                    "epe": round(in_frame.mean().item(), 4),
                    "pck_3": round((in_frame < 3.0).float().mean().item(), 4),
                    "pck_5": round((in_frame < 5.0).float().mean().item(), 4),
                }
                if cp_A is not None:
                    # Trustworthy GT is in-frame AND inside the control-point hull;
                    # outside it the TPS extrapolates and scores the spline, not the model.
                    hull = convex_hull_mask(cp_A[b].cpu().numpy(), (ws, hs), (mh, mw))
                    scored = valid_b & hull.to(valid_b.device)
                    row["hull_cov"] = round(scored.float().mean().item(), 4)
                    if scored.any():
                        row["epe_hull"] = round(gd_b[scored].mean().item(), 4)
                    pred_field = matches[b:b+1, ..., 2:].permute(0, 3, 1, 2).float()
                    scale = None
                    if size_B is not None:
                        scale = size_B[b].cuda() / torch.tensor([ws, hs], device="cuda").float()
                    cp_err = self.control_point_epe(
                        pred_field, cp_A[b].cuda(), cp_B[b].cuda(), ws, hs, scale=scale
                    )
                    row["ctrl_epe"] = round(cp_err.mean().item(), 4)
                    diag = float(np.hypot(ws, hs)) if size_B is None else \
                        float(torch.linalg.vector_norm(size_B[b]).item())
                    row["med_px"] = round(cp_err.median().item(), 4)
                    row["max_px"] = round(cp_err.max().item(), 4)
                    row["diag_px"] = round(diag, 2)
                    row["med_rel"] = row["med_px"] / diag
                    row["max_rel"] = row["max_px"] / diag
                self.pair_rows.append(row)

        model.train(was_training)
        out = aggregate_ctrl_metrics(self.pair_rows)
        out.update(dense_diagnostic_means(self.pair_rows))
        return out


# ******************* Full-suite runner **********************

def evaluate_full_suite(model, dataset, num_workers=2):
    """Run dense-warp evaluation on a dataset.

    Computes the headline control-point metrics (original image frame), the
    GT-limited dense diagnostics, inference speed, and test-set standard
    deviations. The per-pair rows behind the aggregates are returned alongside
    them so a caller can re-aggregate over a subset or run a paired test without
    a second forward pass.

    Any support prior must already be stashed on ``model._support`` by the caller.

    Args:
        model: RoMa model to evaluate.
        dataset: Dataset used by the dense benchmark.
        num_workers: Number of dense benchmark data-loading workers.

    Returns:
        tuple[dict, list[dict]]: Combined aggregate metrics and the per-pair rows.
    """
    model.eval()
    timed_matcher = TimedMatcher(model)
    combined_metrics = {"num_pairs": len(dataset)}
    dense_benchmark = DenseWarpBenchmark(
        dataset, batch_size=1, num_workers=num_workers,
        timed_matcher=timed_matcher,
    )
    combined_metrics.update(dense_benchmark.benchmark(model))
    combined_metrics.update(timed_matcher.speed_metrics())
    combined_metrics.update(compute_testset_pair_std(dense_benchmark.pair_rows))
    return combined_metrics, dense_benchmark.pair_rows


# ******************** Few-shot helpers **********************

def select_support_indices(n, num_support):
    """Select deterministic, approximately evenly spaced support-pair indices.

    Rounded indices are deduplicated, so the result may contain fewer than
    ``num_support`` entries.

    Args:
        n: Total number of available pairs.
        num_support: Requested number of support pairs.

    Returns:
        list[int]: Sorted support-pair indices in ``[0, n)``.
    """
    return sorted({int(round(t)) for t in np.linspace(0, n - 1, num_support)})


def prepare_support_pair(sample, ht, wt, device):
    """Prepare one dataset sample as a one-shot few-shot support prior.

    Homography and TPS samples are converted to the same dense-warp representation,
    with a leading batch dimension of ``1`` because a prior conditions a whole
    evaluation run.

    Args:
        sample: One dataset sample.
        ht: Ground-truth warp height.
        wt: Ground-truth warp width.
        device: Device to which the tensors are moved.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]: Image A
        ``(1, C, ht, wt)``, image B with the same shape, the dense warp
        ``(1, ht, wt, 2)``, and the control-point neighborhood mask ``(1, ht, wt)`` --
        or ``None`` when the sample's dataset does not provide one (MeDAL/MegaDepth
        sources have no annotated control points).
    """
    im_A = sample["im_A"][None].to(device)
    im_B = sample["im_B"][None].to(device)
    warp = get_gt_dense_warp(sample, ht, wt)[None].to(device)
    ctrl_mask = sample["ctrl_mask"][None].to(device) if "ctrl_mask" in sample else None
    return im_A, im_B, warp, ctrl_mask


def score_dense_match(model, bench, sample, ht, wt, device, support=None):
    """Run dense matching for one query and compute its registration errors.

    Uses standard matching when no support pair is provided and few-shot
    matching otherwise.

    Args:
        model: RoMa model used for matching.
        bench: Dense benchmark providing the EPE helper methods.
        sample: Query sample containing images and registration ground truth.
        ht: Ground-truth warp height.
        wt: Ground-truth warp width.
        device: Device used for matching and evaluation.
        support: Optional few-shot support tuple.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: Dense EPE values and optional
        control-point EPE values.
    """
    im_A = sample["im_A"][None].to(device)
    im_B = sample["im_B"][None].to(device)
    if support is None:
        warp, _ = model.match(im_A, im_B, batched=True)
    else:
        warp, _ = model.match_fewshot(im_A, im_B, *support, batched=True)
    matches = warp[:, :, :im_B.shape[-1], :]

    gt_warp = get_gt_dense_warp(sample, ht, wt)[None].to(device)
    dense_epe = bench.geometric_dist(gt_warp, matches)

    control_epe = None
    if "ctrl_pts_A" in sample:
        width, height = matches.shape[-2], matches.shape[-3]
        pred_field = matches[..., 2:].permute(0, 3, 1, 2).float()
        control_epe = bench.control_point_epe(
            pred_field,
            sample["ctrl_pts_A"].to(device),
            sample["ctrl_pts_B"].to(device),
            width,
            height,
        )

    return dense_epe, control_epe
