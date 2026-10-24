"""Retinal evaluation of one ablation arm of the few-shot RoMa matcher.

For a (held-out) subset the support-conditioned arms take a support pair *with GT warp*
as the modality prior and register every remaining query pair conditioned on it; metrics
use the query GT warp for scoring only. A 1-shot number depends heavily on which pair
served as the prior, so ``--num_support`` supports are swept leave-one-out and the result
carries the mean and the spread over that sweep rather than a single draw.

Each run scores exactly one configuration -- the one the checkpoint was trained as. There
is no support-free row underneath it: an arm built with ``use_support=True`` is
conditioned on a prior in training and at inference alike, so scoring it without one would
report a model that was never trained. The comparison the ablation is after is *between*
arms, and it is assembled by the summaries from the JSONs of the separate runs; per-pair
``med_px`` is saved alongside the aggregates so that comparison can be paired over exactly
the same pairs.

``--variant`` names the arm (see :mod:`experiments.utils.variants`) and has to match how
the checkpoint was trained; the label is written into ``_meta`` so the summaries can group
the runs. ``zero`` takes no checkpoint at all -- it is off-the-shelf RoMa, scored through
this same harness at the same resolution as every other arm. The arms without a prior
(``zero``, ``baseline``, ``gamma0*``) go through the same leave-one-out sweep as the rest,
so every arm reports on identical query sets; on a ``gamma0`` arm those passes are not
plain RoMa, since the model runs its alignment path with or without a prior and the
trained frequency filter is in it.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import Subset

from romatch.datasets.homography_octa import HomographyOctaDataset
from romatch.models.encoders import VIT_BACKBONES
from romatch.models.model_zoo import roma_fewshot
from experiments.utils.benchmark_utils import (METRIC_ORDER, TAU_DIAG_FRAC,
                                               aggregate_pair_rows, evaluate_full_suite,
                                               filter_finite_numbers, is_finite_number,
                                               prepare_support_pair, save_json,
                                               select_support_indices)
from experiments.utils.vis_utils import visualize_fewshot_subset
from experiments.utils.variants import (add_variant_arg, align_kwargs_for,
                                        align_scales_for, describe, is_finetuned,
                                        uses_support)
from experiments.Retina.train_roma_fewshot import resolutions


# ***************** Full metric evaluation *******************

@torch.inference_mode()
def evaluate_variant(model, data_root, subset, fold=0, split="test", ht=560, wt=560,
                     num_support=1, num_workers=2, device="cuda",
                     vis_dir=None, use_support=True):
    """Run the retinal metric suite for one ablation arm over one subset.

    A support-conditioned arm is scored the way it was trained -- with a prior. Which
    pair serves as the prior moves a 1-shot number more than most effects being measured,
    so ``num_support`` support sets are swept leave-one-out, each scoring the pairs it did
    not consume, and the reported metrics are the mean over that sweep with the spread
    saved beside them as ``<metric>_supp_std``.

    An arm *without* a prior is put through the same sweep, so that the two are compared
    on identical query sets under identical aggregation. Scoring it in one pass over all
    ``n`` pairs instead -- which is what this used to do -- silently gave it the pairs the
    swept arms had spent on their priors, and reduced a mean-of-medians to a single
    median. It needs no extra inference: without a prior a pair's result does not depend
    on which other pair was held out, so one pass over every pair is run and then
    re-aggregated over each repetition's queries. The ``_supp_std`` that falls out is the
    null baseline for the swept arms' own spread -- it is what leave-one-out alone
    contributes, with no support effect in it.

    Per-pair ``med_px`` comes back alongside the aggregates: the arms are separate runs in
    separate processes, so pairing them for a signed-rank test later is only possible if
    each run saves its own per-pair rows. For a swept arm a pair's value is averaged over
    the support sets that scored it, which keeps one value per pair and so keeps the
    deltas independent.

    Args:
        model: The arm's RoMa model.
        data_root: Root directory containing the retinal datasets.
        subset: Dataset subset to evaluate.
        fold: Fold index, or ``-1`` to evaluate the complete subset.
        split: Dataset split to evaluate.
        ht: Evaluation image height.
        wt: Evaluation image width.
        num_support: Number of sweep repetitions, spread evenly over the subset;
            ``0`` sweeps every pair. Ignored without a prior.
        num_workers: Number of data-loading workers.
        device: Device used for matching and metric computation.
        vis_dir: Root directory for the per-pair figures, or ``None`` to skip them.
            Figures land in ``<vis_dir>/<subset>``.
        use_support: Whether this arm conditions on a support prior.

    Returns:
        tuple[dict, dict, dict]: Aggregate metrics, the per-pair ``med_px`` map, and the
        sweep description (support indices; empty without a prior).
    """
    ds = HomographyOctaDataset(
        data_root=data_root, subsets=(subset,), fold=fold, split=split, ht=ht, wt=wt
    )
    n = len(ds)
    # A repetition needs its one support held out plus at least one query left to score.
    # This binds the unswept arms too: they are scored over the same query sets, so a
    # subset too small to sweep is one neither kind of arm reports on.
    if n < 2:
        return {}, {}, {}
    was_training = model.training
    model.train(False)

    # The same draw for every arm -- it depends only on (n, num_support), never on the
    # model -- so the swept and unswept arms score identical query sets.
    support_indices = select_support_indices(n, num_support or n)
    runs = []
    med_px_per_pair = {}

    if not use_support:
        model._support = None
        # One pass over every pair. Without a prior a pair's row does not depend on
        # which other pair was held out, so each repetition's numbers are a
        # re-aggregation of these rows rather than a second forward pass. The rows come
        # back in dataset order (the benchmark loads with batch_size=1 and no shuffle),
        # which is what lets the support *index* select them.
        run_metrics, pair_rows = evaluate_full_suite(model, ds, num_workers=num_workers)
        if len(pair_rows) != n:
            raise RuntimeError(
                f"expected one row per pair to re-aggregate over, got {len(pair_rows)} "
                f"rows for {n} pairs"
            )
        for held_out in support_indices:
            rows = [row for i, row in enumerate(pair_rows) if i != held_out]
            # Start from the full-run metrics so the timings, which measure the run and
            # not the pairs, survive; everything pair-derived is then overwritten.
            metrics_r = dict(run_metrics)
            metrics_r.update(aggregate_pair_rows(rows))
            runs.append(metrics_r)
            for row in rows:
                if is_finite_number(row.get("med_px")):
                    med_px_per_pair.setdefault(row["pair"], []).append(float(row["med_px"]))
        if vis_dir:
            # The figure is laid out around a support row, so there is nothing to draw
            # for an arm that has no prior. Say so rather than writing half a figure.
            print(f"--- {subset}: skipping figures, this arm has no support prior")
    else:
        for held_out in support_indices:
            query_ids = [i for i in range(n) if i != held_out]
            model._support = prepare_support_pair(ds[held_out], ht, wt, device)
            run_metrics, run_rows = evaluate_full_suite(
                model, Subset(ds, query_ids), num_workers=num_workers
            )
            model._support = None
            runs.append(run_metrics)
            for row in run_rows:
                if is_finite_number(row.get("med_px")):
                    med_px_per_pair.setdefault(row["pair"], []).append(float(row["med_px"]))

    metrics = summarize_support_sweep(runs)
    # One value per pair, averaged over the supports that scored it, so a later paired
    # test between arms compares independent deltas rather than repeated measurements.
    pair_med_px = {pair: float(np.mean(values)) for pair, values in med_px_per_pair.items()}

    if vis_dir and use_support:
        held_out = support_indices[0]
        support = prepare_support_pair(ds[held_out], ht, wt, device)
        visualize_fewshot_subset(
            model, ds, held_out, support, vis_dir, subset, ht, wt, device=device
        )
    model.train(was_training)
    return metrics, pair_med_px, {"support_indices": support_indices}


def summarize_support_sweep(runs):
    """Average one arm's metrics over the support sweep.

    Set-level metrics such as AUC have no per-pair spread, so their uncertainty is
    taken from the sweep: every numeric key gets a ``<key>_supp_std`` sibling holding
    the population std across supports. Keys that are themselves a pair-level std are
    averaged but not given a std of their own.

    Args:
        runs: One metric dictionary per support pair.

    Returns:
        dict: Mean of every numeric key, plus the across-support std.
    """
    if not runs:
        return {}
    summary = {}
    for key in runs[0]:
        values = filter_finite_numbers(run.get(key) for run in runs)
        if not values:
            continue
        summary[key] = float(np.mean(values))
        if not key.endswith("_std"):
            summary[f"{key}_supp_std"] = float(np.std(values))
    return summary


# ************** CLI helpers and entry point *****************

def _print_report(subset, variant, metrics, sweep):
    """Print one arm's metric table for one subset.

    Args:
        subset: Name of the evaluated dataset subset.
        variant: Ablation arm the model was built as.
        metrics: The arm's aggregate metrics.
        sweep: Support-sweep description, or an empty mapping without a prior.
    """
    tau_px = metrics.get("tau_px", float("nan"))
    support_indices = sweep.get("support_indices", [])
    if support_indices:
        scope = f"{len(support_indices)} supports swept; +/- is across supports"
    else:
        scope = "no support prior; +/- is across the test set"
    print(f"\n=== {subset}: {variant} ===")
    print(f"    {int(metrics.get('num_pairs', 0))} pairs, {scope}, "
          f"tau = {100 * TAU_DIAG_FRAC:.1f}% diag = {tau_px:.1f} px")
    print(f"{'metric':<16} {'value':>11} {'+/-':>14}")
    print("-" * 44)
    for k in METRIC_ORDER:
        if k not in metrics:
            continue
        std = metrics.get(f"{k}_supp_std", metrics.get(f"{k}_std"))
        std_s = f"{std:>14.4f}" if isinstance(std, (int, float)) else f"{'':>14}"
        print(f"{k:<16} {metrics[k]:>11.4f} {std_s}")
    print("-" * 44)


def load_model(checkpoint, resolution, variant, align_scales, perturb_noise_std, device,
               gamma_init=0.0,
               backbone="dinov2", backbone_tuning="frozen", backbone_weights=None,
               lora_rank=8, lora_alpha=16.0):
    """Build one ablation arm and load its checkpoint.

    Args:
        checkpoint: Path to the checkpoint file, or ``None`` for the ``zero`` arm, which
            is scored on the official RoMa outdoor weights with no fine-tuning.
        resolution: Resolution preset name.
        variant: Ablation arm; decides which components the model is built with.
        align_scales: Feature scales the alignment modules are inserted at.
        perturb_noise_std: Channel-stat perturbation std; use 0.0 for testing.
        device: Device on which to construct the model.
        backbone: Scale-16 ViT the checkpoint was trained with. It is part of the graph,
            so naming it wrong here loads a checkpoint into a different network.
        backbone_tuning: How that backbone was trained. It decides whether the ViT is a
            state-dict key at all, so it too has to match the training run.
        backbone_weights: Checkpoint for a non-default backbone. Still needed at test
            time for a ``frozen`` arm, whose ViT the run's own checkpoint does not carry.
        lora_rank, lora_alpha: Adapter shape, for a ``lora`` arm.

    Returns:
        tuple: The loaded model, evaluation height, and evaluation width.
    """
    h, w = resolutions[resolution]
    model = roma_fewshot(
        device, load_pretrained=checkpoint is None,
        coarse_res=(h, w), upsample_res=(h, w),
        upsample_preds=False, symmetric=False, attenuate_cert=False,
        align_scales=align_scales_for(variant, align_scales),
        align_kwargs=align_kwargs_for(
            variant, gamma_init=gamma_init, perturb_noise_std=perturb_noise_std
        ),
        backbone=backbone, backbone_tuning=backbone_tuning,
        backbone_weights=backbone_weights, lora_rank=lora_rank, lora_alpha=lora_alpha,
    )
    if checkpoint is None:
        # The "zero" arm resolves to no alignment scales, so roma_fewshot has just built
        # vanilla RoMa's graph and loaded the official outdoor weights into it: this is
        # off-the-shelf RoMa, scored through the same harness -- same pairs, same
        # resolution, same metrics -- as every fine-tuned arm.
        print(f"{variant}: official RoMa outdoor weights, no fine-tuning")
        model.eval()
        return model, h, w

    states = torch.load(checkpoint, map_location="cpu")
    state_dict = states["model"] if "model" in states else states
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {checkpoint} at step {states.get('n')}: "
          f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    # A variant mismatch shows up here and nowhere else: loading a freq checkpoint into a
    # no-freq arm silently drops the filter, and the reverse silently leaves it at init.
    # Every such key lives under support_align., so name them rather than only counting.
    stray = [k for k in list(missing) + list(unexpected) if k.startswith("support_align.")]
    if stray:
        raise SystemExit(
            f"checkpoint/variant mismatch: {len(stray)} unmatched alignment keys "
            f"(e.g. {stray[:3]}). Check --variant {variant} against how this checkpoint "
            "was trained."
        )
    model.eval()
    return model, h, w


def resolve_checkpoint(parser, args):
    """Check that the checkpoint and the arm agree, and return the one to load.

    Args:
        parser: The parser to raise usage errors through.
        args: Parsed arguments carrying ``--variant`` and ``--checkpoint``.

    Returns:
        str | None: The checkpoint path, or ``None`` for the un-finetuned arm.
    """
    if is_finetuned(args.variant):
        if not args.checkpoint:
            parser.error(f"--variant {args.variant} is fine-tuned; pass --checkpoint")
        return args.checkpoint
    if args.checkpoint:
        parser.error(f"--variant {args.variant} is off-the-shelf RoMa; drop --checkpoint")
    return None


def main():
    """Parse CLI arguments, run evaluation, and optionally save the results."""
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="fine-tuned checkpoint to score; omit only for the 'zero' "
                             "arm, which scores off-the-shelf RoMa")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--test_set", required=True, help="single dataset to evaluate")
    parser.add_argument("--fold", default=-1, type=int, choices=[-1, 0, 1, 2, 3, 4], help="fold index, or '-1' to score the whole subset (every pair)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--train_resolution", default="vlow")
    parser.add_argument("--num_support", default=0, type=int, help="sweep repetitions, spread evenly over the subset; 0 sweeps every pair")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--seed", default=0, type=int, help="torch RNG seed. The support-correspondence subsampling in SupportAlignmentModule draws max_samples points at random whenever a scale has more, so without a fixed seed two runs of the same config differ by more than a small A/B effect")
    add_variant_arg(parser)
    parser.add_argument("--backbone", default="dinov2", choices=sorted(VIT_BACKBONES),
                        help="scale-16 ViT the checkpoint was trained with; it is part "
                             "of the graph, so it has to match the training run")
    parser.add_argument("--backbone_tuning", default="frozen",
                        choices=("frozen", "unfrozen", "lora"),
                        help="how that backbone was trained; decides whether its weights "
                             "are in the checkpoint at all, so it has to match too")
    parser.add_argument("--backbone_weights", default=None,
                        help="checkpoint for a non-default backbone. Needed even for a "
                             "frozen arm, whose ViT its own checkpoint does not carry")
    parser.add_argument("--lora_rank", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16.0, type=float)
    parser.add_argument("--align_scales", nargs="+", default=["16"])
    parser.add_argument("--gamma_init", default=None, type=float,
                        help="fixed support-residual coefficient used during training; "
                             "required for support-conditioned variants")
    parser.add_argument("--perturb_noise_std", default=0.0, type=float, help="channel-stat perturbation std (test default: 0.0/off)")
    parser.add_argument("--vis_dir", default=None, help="write one support/query figure per query pair under <vis_dir>/<test_set>; omit to skip visualization")
    parser.add_argument("--out_json", default=None, help="write per-subset metrics (incl. over-test-set std) to this JSON, for the 5-fold summary aggregator")
    args = parser.parse_args()

    # The alignment module subsamples correspondences at random on any scale with
    # more than ``max_samples`` of them, so an unseeded run is not reproducible and
    # an A/B comparison would be reading that noise. Seeds CUDA too.
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = resolve_checkpoint(parser, args)
    support_conditioned = uses_support(args.variant)
    if support_conditioned and args.gamma_init is None:
        parser.error("support-conditioned variants require --gamma_init matching training")
    gamma_init = args.gamma_init if support_conditioned else 0.0
    print(f"[fewshot] variant={describe(args.variant)}")
    if support_conditioned:
        print(f"[fewshot] fixed support gamma={gamma_init}")
    model, h, w = load_model(
        checkpoint,
        args.train_resolution,
        args.variant,
        args.align_scales,
        args.perturb_noise_std,
        device,
        gamma_init=gamma_init,
        backbone=args.backbone,
        backbone_tuning=args.backbone_tuning,
        backbone_weights=args.backbone_weights,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    payload = {"_meta": {"checkpoint": checkpoint, "fold": args.fold,
                         "split": args.split, "variant": args.variant,
                         "use_support": support_conditioned,
                         "gamma_init": gamma_init if support_conditioned else None,
                         "align_scales": list(args.align_scales),
                         "num_support": args.num_support,
                         # Part of the graph, so it belongs in the label the summaries
                         # group by, not just in the command line that produced the run.
                         "backbone": args.backbone,
                         "backbone_tuning": args.backbone_tuning,
                         "seed": args.seed}}

    metrics, pair_med_px, sweep = evaluate_variant(
        model, data_root=args.data_root, subset=args.test_set, fold=args.fold,
        split=args.split, ht=h, wt=w, num_support=args.num_support,
        num_workers=args.num_workers,
        device=device,
        vis_dir=args.vis_dir,
        use_support=support_conditioned,
    )
    payload[args.test_set] = {
        "metrics": metrics, "pair_med_px": pair_med_px, "sweep": sweep,
    }
    _print_report(args.test_set, args.variant, metrics, sweep)

    if args.out_json:
        save_json(args.out_json, payload)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    main()
