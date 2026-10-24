"""Aggregate few-shot RoMa retina JSON results into in/cross-domain tables.

Reads the ``<tag>.json`` files that experiments/Retina/test_roma_fewshot.py writes (via
``--out_json``) under ``--eval_dir``. One directory holds one ablation arm -- the sweep
writes ``<OUT_DIR>/<GROUP>/Retina/<variant>/`` -- so this produces that arm's two
cross-domain results; comparing the arms against each other is
``experiments/summarize_ablation.py``'s job. One FIRE ``foldnone`` model is evaluated
on complete target subsets. Per-pair EPE metrics use their saved whole-test-set std.

Filenames encode train subset / fold / regime (written by test_roma_fewshot_retina.sh):
    FIRE_foldnone_on_<subset>_whole.json

Run (from RoMa/, in the uv env):
    uv run python experiments/Retina/test_roma_fewshot_retina_summary.py \
        --eval_dir workspace/eval_fewshot/baseline
"""

import argparse
import csv
import json
import os
import re

from experiments.utils.benchmark_utils import (CTRL_METRIC_KEYS, METRIC_ORDER,
                                               compute_mean_std_count,
                                               metric_stats_to_dict, save_json)

# A 1-shot number depends on which pair served as the prior, so the test script sweeps
# supports leave-one-out and saves the spread as "<metric>_supp_std". Set-level metrics
# (auc/succ_rate/fail_rate) have no per-pair std at all, and that sweep is their only honest error bar,
# so the cross-domain table reports it as the run's spread.
SUPPORT_STD_SUFFIX = "_supp_std"
SUPPORT_STD_KEYS = frozenset(CTRL_METRIC_KEYS)

# Result filename format:
#   <train>_fold<fold>_on_<rest>.json
#   fold=none identifies cross-domain results; each file contains one target dataset.
NAME_RE = re.compile(r"^(?P<train>.+)_fold(?P<fold>none|-?\d+)_on_(?P<rest>.+)\.json$")


def order_subsets(found, cross_test_order=()):
    """Select and order subsets according to the requested cross-test list.

    Args:
        found: An iterable containing the subset names to order.
        cross_test_order: Cross-domain subset names to report, in the exact
            desired output order. If empty, report every found subset in
            alphabetical order.

    Returns:
        list[str]: The ordered subset names.
    """
    found = set(found)
    if cross_test_order:
        return [subset for subset in cross_test_order if subset in found]
    return sorted(found)


def parse_result_filename(filename):
    """Parse the fold and evaluation regime from a result filename.

    Args:
        filename: The result filename to parse.

    Returns:
        dict | None: A dictionary containing ``fold``, or ``None`` if the filename is
        not a cross-domain result.
    """
    # Only parse JSON files that are not summary files
    if not filename.endswith(".json") or filename.endswith("_summary.json"):
        return None

    # Parse <train>_fold<fold>_on_<rest>.json from the filename
    match = NAME_RE.match(filename)
    if not match:
        return None

    fold_text = match.group("fold")
    return {"fold": -1} if fold_text == "none" else None


def collect(eval_dir):
    """Load cross-domain result JSON files.

    Args:
        eval_dir: The directory containing result JSON files.

    Returns:
        dict: ``{test_subset: {fold: {"metrics": {...}}}}``.
    """
    assert os.path.isdir(eval_dir), f"Directory not found: {eval_dir}"

    results = {}

    for filename in sorted(os.listdir(eval_dir)):
        parsed = parse_result_filename(filename)
        if parsed is None:
            continue

        with open(os.path.join(eval_dir, filename)) as file:
            result_payload = json.load(file)

        for subset, entry in result_payload.items():
            if subset != "_meta":
                results.setdefault(subset, {})[parsed["fold"]] = {
                    "metrics": entry.get("metrics", {}),
                }

    return results


def _metric_values(runs, metric):
    """Extract available metric values from the requested result branch.

    Args:
        runs: Results grouped by run, with the structure
            ``{fold: {"metrics": {...}}}``.
        metric: The metric name, such as ``"val_epe"`` or ``"val_pck_3"``.
    Returns:
        list: Metric values ordered by fold, excluding unavailable values.
    """
    return [runs[key]["metrics"][metric] for key in sorted(runs)
            if metric in runs[key]["metrics"]]


def aggregate_metric(runs, metric, saved_std_keys=SUPPORT_STD_KEYS,
                     std_suffix=SUPPORT_STD_SUFFIX):
    """Aggregate one cross-domain metric.

    For the control-point metrics, use the saved support-sweep
    ``<metric>_supp_std`` when available.

    Args:
        runs: Evaluation results grouped by run.
        metric: The metric to aggregate.
        saved_std_keys: Metrics whose spread is read from a saved
            ``<metric><std_suffix>``.
        std_suffix: Suffix under which that spread was saved.

    Returns:
        tuple[float, float, int, bool] | None: The mean, standard deviation,
        count, and whether the saved test-set standard deviation was used, or
        ``None`` if no values are available.
    """
    stats = compute_mean_std_count(_metric_values(runs, metric))
    if stats is None:
        return None

    mean, std, count = stats

    saved = _metric_values(runs, metric + std_suffix)
    uses_saved = metric in saved_std_keys and bool(saved)
    std = saved[0] if uses_saved else std

    return mean, std, count, uses_saved


def print_table(title, stats_by_subset, subsets, report_keys=METRIC_ORDER,
                metric_width=22, precision=3):
    """Print a formatted metric table for the requested subsets.

    Metrics without a saved per-pair standard deviation are marked with ``*``.

    Args:
        title: The table title.
        stats_by_subset: Metric statistics grouped by subset, with the
            structure
            ``{subset: {metric: (mean, std, n[, uses_testset_std])}}``.
        subsets: The subset display order.
        report_keys: Metric display order; defaults to the same column set the
            test script prints, so the two never drift apart. Benchmarks with their
            own metric set pass theirs instead.
        metric_width: Width of the metric-name column; widen it for longer names.
        precision: Decimals shown. rTRE-style metrics are fractions of an image
            diagonal and need more than the retinal default.
    """
    print(f"\n===== {title} =====")

    header = f"{'metric':<{metric_width}}" + "".join(f"{subset:>24}" for subset in subsets)
    print("\n" + header)
    print("-" * len(header))

    for metric_name in report_keys:
        row = f"{metric_name:<{metric_width}}"
        for subset in subsets:
            metric_stats = stats_by_subset.get(subset, {}).get(metric_name)
            if metric_stats is None:
                row += f"{'n/a':>24}"
                continue

            mean, std = metric_stats[0], metric_stats[1]
            uses_saved_std = len(metric_stats) > 3 and metric_stats[3]
            std_marker = "" if uses_saved_std else "*"
            row += f"{mean:>12.{precision}f} +/-{std:<7.{precision}f}{std_marker:<1}"
        print(row)

    print("\n  * no per-pair std; others = std over the whole test set.")
    print()


def build_agg(results, report_keys=METRIC_ORDER,
              saved_std_keys=SUPPORT_STD_KEYS, std_suffix=SUPPORT_STD_SUFFIX):
    """Aggregate all report metrics for each subset.

    Each value is ``(mean, std, count, uses_saved_std)``.

    Args:
        results: Evaluation results grouped by subset and run.
        report_keys: Metrics to aggregate; defaults to the shared RoMa metric
            order. Benchmarks with their own metric set
            pass theirs instead.
        saved_std_keys: Metrics that saved their own spread.
        std_suffix: Suffix under which that spread was saved.

    Returns:
        dict: Aggregated metric statistics grouped by subset.
    """
    summary = {}
    for subset, runs in results.items():
        summary[subset] = {}
        for metric in report_keys:
            stats = aggregate_metric(runs, metric, saved_std_keys, std_suffix)
            if stats is not None:
                summary[subset][metric] = stats
    return summary


def save_combined(eval_dir, name, agg, subsets, report_keys=METRIC_ORDER):
    """Save aggregated statistics to matching JSON and CSV files.

    Args:
        eval_dir: The output directory.
        name: The output basename without a file extension.
        agg: Aggregated metric statistics grouped by subset.
        subsets: The subsets to write, in output order.
        report_keys: Metric row order; defaults to the shared RoMa metric order.
            Benchmarks with their own metric set pass
            theirs instead.
    """
    json_path = os.path.join(eval_dir, f"{name}.json")
    save_json(json_path, {subset: metric_stats_to_dict(agg[subset]) for subset in subsets})

    csv_path = os.path.join(eval_dir, f"{name}.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        header = ["metric"]
        for subset in subsets:
            header += [f"{subset}_mean", f"{subset}_std", f"{subset}_n"]
        writer.writerow(header)
        for metric_name in report_keys:
            row = [metric_name]
            for subset in subsets:
                if metric_name in agg[subset]:
                    row += list(agg[subset][metric_name][:3])
                else:
                    row += ["", "", ""]
            writer.writerow(row)
    print(f"wrote {json_path} and {csv_path}")


def main():
    """Parse arguments, aggregate evaluation results, and write summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="workspace/eval/fewshot", help="directory holding <tag>.json from test_roma_fewshot.py --out_json")
    parser.add_argument("--cross_test", nargs="*", default=(), help="cross-domain test dataset names passed from CROSS_TEST by the sweep script")
    args = parser.parse_args()

    results = collect(args.eval_dir)
    if not results:
        print(f"No aggregatable few-shot JSONs under {args.eval_dir}; nothing to do. ")
        return

    subsets = order_subsets(results.keys(), args.cross_test)
    agg = build_agg(results)
    print_table("FIRE -> cross-domain whole set: metric mean +/- std OVER the test set",
                agg, subsets)
    save_combined(args.eval_dir, "fewshot_cross_summary", agg, subsets)


if __name__ == "__main__":
    main()
