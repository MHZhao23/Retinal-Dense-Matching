"""Figure rendering for the few-shot dense-warp evaluation.

One figure per query pair, laid out as two rows: the support prior on top and the
evaluated query below. The registration panels reuse the checkerboard style of the
retinal converters (``Data_Processing/Retina/*_converter.py``): image A and image B
are interleaved in alternating tiles, so a correct warp lets the vessels cross the
tile borders unbroken.

Both the GT and the predicted warp are forward maps im_A -> im_B, so im_B is
resampled onto im_A's grid (a direct ``remap``, no field inversion is needed) and
every checkerboard is drawn in im_A's frame.

Depends only on numpy/torch/cv2/matplotlib plus ``benchmark_utils``, so it imports
cleanly from any evaluation script.
"""
import os

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")  # evaluation runs headless on the training nodes
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

from experiments.utils.benchmark_utils import DenseWarpBenchmark, get_gt_dense_warp, norm_to_px


# ******************** Rendering constants *******************

# The loaders normalize with the ImageNet statistics; undo them before drawing.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CTRL_CMAP = "turbo"     # control points are coloured by index so A and B panels match
CTRL_MARKER_SIZE = 34
MIN_TILE = 16           # checkerboard tiles never go below this many pixels


# ****************** Image and warp helpers ******************

def tensor_to_rgb(im):
    """Undo the ImageNet normalization of a dataset image tensor.

    Args:
        im: Normalized image tensor with shape ``(3, H, W)``.

    Returns:
        np.ndarray: ``(H, W, 3)`` float32 RGB image with values in ``[0, 1]``.
    """
    arr = im.detach().cpu().float().permute(1, 2, 0).numpy()
    return np.clip(arr * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)


def checkerboard(a, b, tile=70):
    """Combine two equally sized RGB images in alternating square tiles.

    Args:
        a: First image, shown in the even tiles.
        b: Second image, shown in the odd tiles.
        tile: Tile edge length in pixels.

    Returns:
        np.ndarray: The interleaved image.
    """
    rows, cols = np.indices(a.shape[:2])
    take_b = ((rows // tile) + (cols // tile)) % 2 == 1
    return np.where(take_b[..., None], b, a)


def warp_im_B_to_A_frame(image_b, warp):
    """Resample im_B onto im_A's pixel grid with a forward A-to-B warp field.

    The field stores, for every im_A pixel, its normalized position in im_B, which
    is exactly the sampling grid ``remap`` needs; a field coarser than the image is
    upsampled first. Targets outside im_B are filled with black by ``remap`` itself
    and flagged in the returned mask, which uses the same bounds as the dense EPE.

    Args:
        image_b: ``(H, W, 3)`` float RGB image B.
        warp: Forward warp field with shape ``(h, w, 2)`` in normalized coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: The warped image in im_A's frame and its
        boolean validity mask.
    """
    h, w = image_b.shape[:2]
    field = warp.detach().cpu().float()
    if field.shape[:2] != (h, w):
        field = F.interpolate(
            field.permute(2, 0, 1)[None], size=(h, w), mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0)
    px = norm_to_px(field, w, h).numpy().astype(np.float32)
    warped = cv2.remap(
        np.ascontiguousarray(image_b, dtype=np.float32), px[..., 0], px[..., 1],
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0),
    )
    valid = (field.abs() < 1).all(dim=-1).numpy()
    return warped, valid


def pair_checkerboard(sample, warp, tile=None):
    """Render one pair as an im_A / warped-im_B checkerboard in im_A's frame.

    Args:
        sample: Dataset sample holding ``im_A`` and ``im_B``.
        warp: Forward A-to-B warp field in normalized coordinates.
        tile: Tile edge length in pixels; defaults to an eighth of the image.

    Returns:
        tuple[np.ndarray, float]: The checkerboard image and the fraction of im_A
        pixels whose target falls inside im_B.
    """
    image_a, image_b = tensor_to_rgb(sample["im_A"]), tensor_to_rgb(sample["im_B"])
    warped, valid = warp_im_B_to_A_frame(image_b, warp)
    if tile is None:
        tile = max(MIN_TILE, image_a.shape[0] // 8)
    return checkerboard(image_a, warped, tile=tile), float(valid.mean())


def pair_identifier(sample):
    """Return the ``<im_A>__<im_B>`` id used for pair rows and figure file names.

    Args:
        sample: Dataset sample carrying the per-image identifiers.

    Returns:
        str: The pair identifier.
    """
    return f"{sample.get('im_A_identifier', '?')}__{sample.get('im_B_identifier', '?')}"


# *********************** Panel drawing **********************

def draw_image(ax, image, title):
    """Show one panel image with its title and no axes.

    Args:
        ax: Target matplotlib axis.
        image: RGB image to display.
        title: Panel title.
    """
    ax.imshow(image)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def draw_ctrl_points(ax, points):
    """Scatter annotated control points, colour-coded by index.

    The same colour marks the same landmark in the A and B panels, so a reader can
    follow one correspondence across the pair. Missing annotations are skipped.

    Args:
        ax: Target matplotlib axis.
        points: ``(N, 2)`` pixel control points, or ``None``.
    """
    if points is None:
        return
    pts = points.detach().cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 1], s=CTRL_MARKER_SIZE, marker="+", linewidths=1.3,
               c=np.arange(len(pts)), cmap=CTRL_CMAP)


def draw_info(ax, lines):
    """Fill a panel slot with a monospaced key/value block.

    Args:
        ax: Target matplotlib axis.
        lines: Text lines to render top to bottom.
    """
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11,
            family="monospace", transform=ax.transAxes)
    ax.axis("off")


def render_fewshot_figure(out_path, subset, support_sample, support_warp,
                          query_sample, query_gt_warp, query_pred_warp, stats):
    """Draw and save the support/query panel figure for one query pair.

    Layout (2 x 4): the support row shows the prior the model was conditioned on
    (pair, control points, GT checkerboard) plus a metric block; the query row shows
    the same pair panels followed by the GT and the predicted checkerboard, which are
    the two panels to compare.

    Args:
        out_path: Destination PNG path; parent directories are created.
        subset: Name of the evaluated dataset subset.
        support_sample: Support dataset sample.
        support_warp: Support GT forward warp field.
        query_sample: Query dataset sample.
        query_gt_warp: Query GT forward warp field.
        query_pred_warp: Predicted query forward warp field.
        stats: Per-pair metrics to print in the info panel.
    """
    support_gt_cb, support_valid = pair_checkerboard(support_sample, support_warp)
    query_gt_cb, query_gt_valid = pair_checkerboard(query_sample, query_gt_warp)
    query_pred_cb, query_pred_valid = pair_checkerboard(query_sample, query_pred_warp)
    support_id, query_id = pair_identifier(support_sample), pair_identifier(query_sample)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    draw_image(axes[0, 0], tensor_to_rgb(support_sample["im_A"]), "support im_A + ctrl pts")
    draw_ctrl_points(axes[0, 0], support_sample.get("ctrl_pts_A"))
    draw_image(axes[0, 1], tensor_to_rgb(support_sample["im_B"]), "support im_B + ctrl pts")
    draw_ctrl_points(axes[0, 1], support_sample.get("ctrl_pts_B"))
    draw_image(axes[0, 2], support_gt_cb,
               f"support GT checkerboard (valid {support_valid:.1%})")
    draw_info(axes[0, 3], [
        f"subset      : {subset}",
        f"support     : {support_id}",
        f"query       : {query_id}",
        "",
        f"dense EPE   : {stats['epe']:.3f} px",
        f"ctrl EPE    : {stats['ctrl_epe']:.3f} px" if "ctrl_epe" in stats
        else "ctrl EPE    : n/a",
        f"PCK@1/3/5   : {stats['pck_1']:.3f} / {stats['pck_3']:.3f} / {stats['pck_5']:.3f}",
        "",
        f"GT valid    : {query_gt_valid:.1%}",
        f"pred valid  : {query_pred_valid:.1%}",
    ])

    draw_image(axes[1, 0], tensor_to_rgb(query_sample["im_A"]), "query im_A + ctrl pts")
    draw_ctrl_points(axes[1, 0], query_sample.get("ctrl_pts_A"))
    draw_image(axes[1, 1], tensor_to_rgb(query_sample["im_B"]), "query im_B + ctrl pts")
    draw_ctrl_points(axes[1, 1], query_sample.get("ctrl_pts_B"))
    draw_image(axes[1, 2], query_gt_cb, f"query GT checkerboard (valid {query_gt_valid:.1%})")
    draw_image(axes[1, 3], query_pred_cb,
               f"query pred checkerboard ({stats['mode']}, EPE {stats['epe']:.2f} px)")

    fig.suptitle(f"{subset}  |  query {query_id}  conditioned on support {support_id} "
                 f"({stats['mode']})", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ******************* Few-shot visualization *****************

@torch.inference_mode()
def visualize_fewshot_subset(model, dataset, support_index, support, out_dir, subset,
                             ht, wt, device="cuda"):
    """Render one figure per query pair of a subset into ``<out_dir>/<subset>``.

    This is a separate pass over the query pairs, so the scored run stays untouched
    (its inference timings in particular). Every pair is matched again with the same
    support prior that was used for the few-shot metrics.

    Args:
        model: Few-shot RoMa model, already in eval mode.
        dataset: Full dataset of the subset, the support pair included.
        support_index: Index of the support pair inside ``dataset``. Skipped as a query.
        support: Prepared support tuple, or ``None`` for support-free matching.
        out_dir: Root visualization directory; the subset folder is created below it.
        subset: Dataset subset name, used for the folder name and the titles.
        ht: Evaluation image height.
        wt: Evaluation image width.
        device: Device used for matching.

    Returns:
        int: Number of figures written.
    """
    bench = DenseWarpBenchmark(dataset)
    support_sample = dataset[support_index]
    support_warp = get_gt_dense_warp(support_sample, ht, wt)
    subset_dir = os.path.join(out_dir, subset)
    mode = "1-shot" if support is not None else "no support"

    n_written = 0
    for idx in range(len(dataset)):
        if idx == support_index:
            continue
        sample = dataset[idx]
        im_A, im_B = sample["im_A"][None].to(device), sample["im_B"][None].to(device)
        if support is None:
            warp, _ = model.match(im_A, im_B, batched=True)
        else:
            warp, _ = model.match_fewshot(im_A, im_B, *support, batched=True)
        matches = warp[:, :, :im_B.shape[-1], :]  # A->B half (robust if symmetric=True)

        gt_warp = get_gt_dense_warp(sample, ht, wt)
        gd = bench.geometric_dist(gt_warp[None].to(device), matches)
        stats = {
            "mode": mode,
            "epe": gd.mean().item(),
            "pck_1": (gd < 1.0).float().mean().item(),
            "pck_3": (gd < 3.0).float().mean().item(),
            "pck_5": (gd < 5.0).float().mean().item(),
        }
        if "ctrl_pts_A" in sample:
            pred_field = matches[..., 2:].permute(0, 3, 1, 2).float()
            stats["ctrl_epe"] = bench.control_point_epe(
                pred_field, sample["ctrl_pts_A"].to(device), sample["ctrl_pts_B"].to(device),
                matches.shape[-2], matches.shape[-3],
            ).mean().item()

        render_fewshot_figure(
            os.path.join(subset_dir, f"{pair_identifier(sample)}.png"), subset,
            support_sample, support_warp, sample, gt_warp, matches[0, ..., 2:], stats,
        )
        n_written += 1

    print(f"wrote {n_written} {subset} visualizations to {subset_dir}")
    return n_written
