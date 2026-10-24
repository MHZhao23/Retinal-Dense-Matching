"""Few-shot registration training on self-supervised synthetic MeDAL-Retina pairs.

Mirrors ``train_roma_fewshot.py`` (episodic few-shot RoMa, same model / loss /
optimizer / checkpoint conventions -- see that file's docstring), swapping in
the MeDAL-Retina datasets:
  - dataset  -> ``FewShotMedalDataset`` (episodic; support + query per item)
  - eval     -> ``evaluate_medal_fewshot`` against ``MedalPairDataset(split="val")``

MeDAL-Retina is unpaired, unannotated fundus imagery: ``Auxilliary_Training``
(1920 images) is the training pool, ``SuperRetina_training_data`` (268 images,
39 keypoints each) is the validation pool. Neither pool has real image pairs,
so every training/validation pair is *synthesized* from a single source image
via an independent random affine + elastic (TPS) warp per branch -- see
``romatch/datasets/medal.py``. Because the ground-truth warp is generated
analytically rather than fit from sparse real-pair annotations, it does not
carry the GT-limited noise ceiling ``roma-homography-training`` found in the
real cross-modal TPS pairs; MeDAL's keypoint annotations still give an
additional ``ctrl_epe`` sanity metric on the validation pool.

``--variant`` selects the ablation arm (see :mod:`experiments.utils.variants`),
exactly as in ``train_roma_fewshot.py``.
"""
import math
import os
import random
from argparse import ArgumentParser
from warnings import warn

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import wandb

import romatch
from romatch.datasets.fewshot_medal import FewShotMedalDataset
from romatch.datasets.medal import MedalPairDataset
from romatch.losses.robust_loss import RobustLosses
from romatch.train.train import train_k_steps
from romatch.checkpointing import CheckPoint
from romatch.models.model_zoo import roma_fewshot
from experiments.utils.benchmark_utils import DenseWarpBenchmark, compute_mean_std_count, prepare_support_pair, score_dense_match, select_support_indices
from experiments.utils.variants import (add_variant_arg, align_kwargs_for,
                                        align_scales_for, describe, is_finetuned,
                                        uses_support)

resolutions = {"tiny":(14*8*2, 14*8*2), "vlow":(14*8*3, 14*8*3), "low":(448, 448), "medium":(14*8*5, 14*8*5), "high":(14*8*6, 14*8*6)}

# ***************** Validation *******************
@torch.inference_mode()
def evaluate_medal_fewshot(model, data_root, subset, ht=560, wt=560,
                           num_support=1, device="cuda", use_support=True):
    """Run lightweight dense evaluation of one ablation arm on MeDAL-Retina.

    Same protocol as ``train_roma_fewshot.evaluate_fewshot``, scored on
    ``MedalPairDataset(split="val")`` (the ``SuperRetina_training_data`` pool)
    instead of a real cross-modal dataset. Each index there returns a
    reproducible synthesized pair, so this comparison is stable across steps.

    That pool holds 268 images, an order of magnitude more than the real
    retinal val splits, and every pair is synthesized on the fly -- so the
    whole pass is not cheap. Use a coarser ``--save_every`` here than the
    FIRE sweep uses rather than scoring fewer pairs.

    Args:
        model: Few-shot RoMa model to evaluate.
        data_root: Root directory containing the retinal datasets.
        subset: MeDAL-Retina subset directory name.
        ht: Evaluation image height.
        wt: Evaluation image width.
        num_support: Number of deterministic support pairs to use.
        device: Device used for matching and metric computation.
        use_support: Whether the arm conditions on a support prior.

    Returns:
        dict: Dense metrics for the single configuration used by this arm.
    """
    ds = MedalPairDataset(data_root=data_root, subset=subset, split="val", ht=ht, wt=wt)
    n = len(ds)
    if n < 2:
        return {"epe": float("nan")}
    nan_stats = (float("nan"), float("nan"), 0)
    bench = DenseWarpBenchmark(ds)
    was_training = model.training
    model.train(False)
    samples = [ds[i] for i in range(n)]  # val split is deterministic (seeded per index)

    # The support-conditioned arms hold one pair out as the prior and score the rest;
    # the others score every pair with no prior at all.
    support, held_out = None, -1
    if use_support:
        held_out = select_support_indices(n, num_support)[0]
        support = prepare_support_pair(samples[held_out], ht, wt, device)

    epe, pck1, pck3, ctrl_epe = [], [], [], []
    for i, sample in enumerate(samples):
        if i == held_out:
            continue
        gd, ctrl = score_dense_match(
            model, bench, sample, ht, wt, device, support=support
        )
        epe.append(gd.mean().item())
        pck1.append((gd < 1.0).float().mean().item())
        pck3.append((gd < 3.0).float().mean().item())
        if ctrl is not None:
            ctrl_epe.append(ctrl.mean().item())

    model.train(was_training)

    epe_mean, epe_std, _ = compute_mean_std_count(epe) or nan_stats
    pck1_mean, _, _ = compute_mean_std_count(pck1) or nan_stats
    pck3_mean, _, _ = compute_mean_std_count(pck3) or nan_stats
    out = {
        "epe": epe_mean,
        "epe_std": epe_std,  # spread over the query set
        "pck_1": pck1_mean,
        "pck_3": pck3_mean,
    }
    if ctrl_epe:
        ctrl_mean, ctrl_std, _ = compute_mean_std_count(ctrl_epe) or nan_stats
        out["ctrl_epe"] = ctrl_mean
        out["ctrl_epe_std"] = ctrl_std
    return out


def _fold_arg(x):
    return None if str(x).strip().lower() == "none" else int(x)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    romatch.RANK = 0
    romatch.LOCAL_RANK = 0
    if device == "cuda":
        torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data_generator = torch.Generator().manual_seed(args.seed)

    resolution = args.train_resolution
    h, w = resolutions[resolution]
    if not is_finetuned(args.variant):
        raise SystemExit(
            f"variant {args.variant!r} is off-the-shelf RoMa; there is nothing to train. "
            "Score it directly with the test entry point."
        )
    support_conditioned = uses_support(args.variant)
    # Print the *resolved* scales: the arms with no adaptation resolve to none at all,
    # whatever --align_scales asked for, and the header should say what was built.
    print(f"[medal] variant={describe(args.variant)} "
          f"scales={align_scales_for(args.variant, args.align_scales)} "
          f"perturb_noise_std={args.perturb_noise_std}")
    wandb_mode = "disabled" if args.dont_log_wandb else "online"
    experiment_name = f"roma_fewshot_{args.train_set}"
    wandb.init(project="romatch", entity=args.wandb_entity, name=experiment_name,
               reinit=False, mode=wandb_mode)
    # MeDAL has no fold concept (train/val are two disjoint pools, Auxilliary_Training
    # vs SuperRetina_training_data, not a k-fold split) -- --fold only feeds the
    # checkpoint-directory tag, kept for layout parity with the FIRE/OCTA sweep scripts.
    fold_tag = "none" if args.fold is None else args.fold
    checkpoint_dir = f"{args.ckpt_dir}/{args.train_set}/fold{fold_tag}/"
    assert args.num_support == 1, "Only single-support training is currently supported"

    # Few-shot matcher: symmetric=False so match()/match_fewshot route through the
    # support-aware forward. Loads official RoMa outdoor weights (strict=False) unless
    # --from_scratch; the only fresh params are the alignment modules. The ``baseline``
    # arm resolves to no alignment scales, which leaves this graph identical to
    # ``roma_model``'s -- plain RoMa fine-tuning through the same code path.
    model = roma_fewshot(
        device,
        load_pretrained=args.from_pretrained,
        coarse_res=(h, w),
        upsample_res=(h, w),
        upsample_preds=False,
        symmetric=False,
        attenuate_cert=False,
        align_scales=align_scales_for(args.variant, args.align_scales),
        align_kwargs=align_kwargs_for(
            args.variant,
            gamma_init=args.gamma_init,
            perturb_noise_std=args.perturb_noise_std,
        ),
    )

    # DINOv2 is never a parameter (list-hack in the encoder). Freeze the CNN encoder
    # unless training end-to-end.
    if not args.train_cnn:
        for p in model.encoder.parameters():
            p.requires_grad = False
        # Fully freeze the backbone: keep the (VGG19_BN) encoder in eval so its BN
        # running stats don't drift, and stop train_k_steps' per-step model.train(True)
        # from flipping it back. (DINOv2 is already eval via the list-hack.)
        model.encoder.eval()
        model.encoder.train = lambda mode=True: model.encoder

    batch_size = args.gpu_batch_size
    romatch.STEP_SIZE = 1
    effective_bs = batch_size
    N = args.num_steps
    k = min(args.save_every, N)

    if support_conditioned:
        train_dataset = FewShotMedalDataset(
            data_root=args.data_root,
            subset=args.train_set,
            split="train",
            ht=h, wt=w,
        )
    else:
        # No prior is read, so the episode collapses to its query pair -- which the
        # episodic dataset draws uniformly at random anyway, ignoring the index. Loading
        # the shots as well would only cost time.
        train_dataset = MedalPairDataset(
            data_root=args.data_root,
            subset=args.train_set,
            split="train",
            ht=h, wt=w,
        )

    objective = RobustLosses(
        ce_weight=0.01,
        local_dist={1: 4, 2: 4, 4: 8, 8: 8},
        local_largest_scale=8,
        depth_interpolation_mode="bilinear",
        alpha=0.5,
        c=1e-4,
    )

    # Pretrained decoder gets the small RoMa lr; the fresh alignment modules get a
    # higher lr. (CNN, if trained, gets the small lr too.)
    lr = effective_bs * 1e-4 / 8
    parameters = [{"params": model.decoder.parameters(), "lr": lr}]
    # Empty on the arms built with no alignment scales; an empty param group is legal
    # but pointless, and leaving it out keeps the optimizer state readable.
    align_params = list(model.support_align.parameters())
    if align_params:
        parameters.append({"params": align_params, "lr": 4 * lr})
    if args.train_cnn:
        parameters.append({"params": model.encoder.parameters(), "lr": lr})
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(9 * N) // 10])

    checkpointer = CheckPoint(checkpoint_dir, experiment_name)
    grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
    # best_epe comes back with the rest of the run state: a resume that reset it to inf
    # would overwrite a good _best.pth with the first score it happened to measure.
    model, optimizer, lr_scheduler, global_step, best_epe = checkpointer.load(
        model, optimizer, lr_scheduler, 0, grad_scaler=grad_scaler,
        data_generator=data_generator,
    )
    romatch.GLOBAL_STEP = global_step
    # Stamp the progress file up front, so an arm that has nothing left to do still
    # leaves the sweep script something to skip on next time.
    checkpointer.record_progress(romatch.GLOBAL_STEP, total_steps=N)
    grad_clip_norm = 1.0
    if global_step:
        print(f"[medal] resuming at step {global_step}/{N}"
              + (f", best epe {best_epe:.4f}" if math.isfinite(best_epe) else ""))

    for n in range(romatch.GLOBAL_STEP, N, k):
        kk = min(k, N - n)
        sampler = RandomSampler(train_dataset, replacement=True,
                                num_samples=batch_size * kk, generator=data_generator)
        dataloader = iter(
            DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler,
                num_workers=args.num_workers, drop_last=True, pin_memory=True,
                generator=data_generator,
            )
        )
        train_k_steps(
            n, kk, dataloader, model, objective, optimizer, lr_scheduler,
            grad_scaler, grad_clip_norm=grad_clip_norm,
        )
        checkpointer.save(model, optimizer, lr_scheduler, romatch.GLOBAL_STEP,
                          grad_scaler=grad_scaler, data_generator=data_generator,
                          best_metric=best_epe,
                          total_steps=N)

        # Few-shot validation on the SuperRetina_training_data pool.
        if args.train_set:
            evaluation_metrics = evaluate_medal_fewshot(
                model, data_root=args.data_root, subset=args.train_set,
                ht=h, wt=w, num_support=args.num_support,
                device=device, use_support=support_conditioned,
            )
            logged_metrics = {
                f"{args.train_set}/{metric_name}": metric_value
                for metric_name, metric_value in evaluation_metrics.items()
            }
            print(f"[step {romatch.GLOBAL_STEP}] " +
                  "  ".join(
                      f"{metric_name}={metric_value:.4f}"
                      for metric_name, metric_value in logged_metrics.items()
                  ))
            wandb.log(logged_metrics, step=romatch.GLOBAL_STEP)
            # Each arm reports only the validation pass matching its training setup.
            selection_key = f"{args.train_set}/epe"
            mean_epe = logged_metrics[selection_key]
            if mean_epe < best_epe:
                best_epe = mean_epe
                checkpointer.save_best(model, romatch.GLOBAL_STEP, metric=mean_epe)


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True
    warn("Current version of romatch is not tested for training, use at your own risk.")
    parser = ArgumentParser()
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--dont_log_wandb", action="store_true")
    parser.add_argument("--train_resolution", default="low")
    parser.add_argument("--gpu_batch_size", default=2, type=int)
    parser.add_argument("--wandb_entity", required=False)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--from_scratch", dest="from_pretrained", action="store_false",
                        help="train without loading official roma_outdoor weights")
    parser.add_argument("--train_cnn", action="store_true",
                        help="also fine-tune the CNN encoder (DINOv2 stays frozen)")
    parser.add_argument("--train_set", default="MeDAL-Retina", type=str,
                        help="MeDAL-Retina subset directory name under --data_root")
    parser.add_argument("--num_support", default=1, type=int)
    add_variant_arg(parser)
    parser.add_argument("--align_scales", nargs="+", default=["16"])
    parser.add_argument("--gamma_init", default=0.0, type=float,
                        help="residual-gate init; 0=exact identity (ReZero bootstrap), "
                             "try 0.1 for a faster cold start")
    parser.add_argument("--perturb_noise_std", default=0.25, type=float,
                        help="training-only support/query B channel-stat perturbation "
                             "std (default: 0.25; 0 disables it)")
    parser.add_argument("--seed", default=0, type=int,
                        help="RNG seed for approximate reproducibility (model init, "
                             "sampler order, per-worker aug). Vary it for independent runs.")
    parser.add_argument("--fold", default="none", type=_fold_arg,
                        help="checkpoint-directory tag only; MeDAL has no fold split "
                             "(train/val are the Auxilliary_Training/SuperRetina_training_data "
                             "pools), so this stays 'none'")
    parser.add_argument("--num_steps", default=20000, type=int)
    parser.add_argument("--save_every", default=500, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--ckpt_dir", default="workspace/checkpoints/fewshot")
    args = parser.parse_args()
    romatch.DEBUG_MODE = args.debug_mode
    train(args)
