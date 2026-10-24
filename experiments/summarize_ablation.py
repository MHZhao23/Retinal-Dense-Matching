"""Collect the whole ablation -- every arm, every benchmark -- into one CSV.

The per-benchmark summaries each answer "how good is this arm on this benchmark". This
one answers the question the ablation was actually run to answer: how the arms compare,
side by side, over every benchmark at once. It reads the JSONs the test entry points
write and produces

  ``ablation_summary.csv``   one row per (benchmark, arm, test set, metric): the mean,
                             spread, and number of runs. Long format on purpose -- a
                             benchmark is free to report its own metric set, and forcing
                             several of them into shared columns would only produce a
                             sparse matrix with a misleading common header.
  ``ablation_headline.csv``  the reading table: one row per (benchmark, test set,
                             metric), one column per arm, restricted to each benchmark's
                             headline metrics.
  ``ablation_summary.json``  the same numbers, nested, for anything downstream.

It also prints those headline tables and a paired signed-rank test of every arm against
the reference, which a per-arm mean cannot show. Pairing matters -- the arms are scored
on identical pairs, and per-pair difficulty otherwise swamps the effect being measured.
It needs per-pair errors, so it appears only where a run saved them; the retinal runs
save per-pair ``med_px``.

Directory layout, written by the sweep scripts::

    <eval_dir>/<benchmark>/<variant>/<train>_fold<f>_on_<rest>.json

``<benchmark>`` is Retina; ``<variant>`` is one of the arms. The arm is read from each
file's ``_meta`` rather than from its path, so a moved or renamed directory cannot
silently relabel a result; a file whose ``_meta`` disagrees with its directory is
reported and skipped.

``--benchmarks`` restricts the scan to some of those subdirectories, and ``--out_dir``
says where the files land. A sweep script uses both, passing its own benchmark and
writing into ``<eval_dir>/<benchmark>/``: sweeps share one ``<eval_dir>``, so without
that they would take turns overwriting one set of files at its root, each time with a
table describing whichever benchmarks happened to have finished.

Run (from RoMa/, in the uv env)::

    # one benchmark, beside its own results -- what the sweep scripts run
    uv run python experiments/summarize_ablation.py --eval_dir workspace/eval/fewshot \
        --benchmarks Retina --out_dir workspace/eval/fewshot/Retina

    # every benchmark present, for the cross-benchmark reading
    uv run python experiments/summarize_ablation.py --eval_dir workspace/eval/fewshot
"""

import argparse
import csv
import json
import os
import re

from scipy import stats

from experiments.utils.benchmark_utils import compute_mean_std_count, save_json
from experiments.utils.variants import REFERENCE_VARIANT, VARIANTS

# Each benchmark's headline metrics, in reading order, and whether a smaller value is
# better. The long CSV carries everything a run saved; these are what the printed tables
# and the wide CSV are restricted to, because a nine-column table of every diagnostic is
# not a table anyone reads.
HEADLINES = {
    "Retina": ["med_px", "max_px", "auc", "succ_rate", "fail_rate"],
}

# Metrics where a smaller number is the better one; everything else reads upwards. A
# benchmark that brings its own metric set adds its lower-is-better names here.
LOWER_BETTER = {"med_px", "max_px", "fail_rate", "ctrl_epe", "val_epe", "val_epe_hull",
                "inference_time_ms"}

# Result filename format, shared by every benchmark:
#   <train>_fold<fold>_on_<rest>.json
FILE_RE = re.compile(r"^(?P<train>.+)_fold(?P<fold>none|-?\d+)_on_.+\.json$")

# Suffixes a run may have saved its own spread under, most specific first. A swept arm
# saves the across-support spread; an unswept one saves the across-test-set spread.
STD_SUFFIXES = ("_supp_std", "_std")


def parse_name(filename):
    """Split a result filename into its train subset and fold.

    Args:
        filename: Basename of a result JSON written by one of the sweep scripts.

    Returns:
        dict | None: ``{"train", "fold"}``, or ``None`` when the name is not a sweep
        result (a summary file, say).
    """
    if not filename.endswith(".json") or "summary" in filename:
        return None
    found = FILE_RE.match(filename)
    if not found:
        return None
    return {"train": found.group("train"), "fold": found.group("fold")}


def collect(eval_dir, benchmarks=None):
    """Load every arm's results under ``eval_dir``.

    Args:
        eval_dir: Directory holding ``<benchmark>/<variant>/<tag>.json``.
        benchmarks: Benchmark directory names to read, or ``None`` for all of them.
            A sweep passes its own so that its tables describe its own benchmark and
            nothing else.

    Returns:
        tuple[dict, list[str]]: The results, keyed
        ``{benchmark: {variant: {(train, test): {fold: entry}}}}`` where an entry
        is ``{"metrics": {...}, "pair_med_px": {...}}``, and the warnings raised
        while loading.
    """
    results, warnings = {}, []
    if not os.path.isdir(eval_dir):
        raise SystemExit(f"--eval_dir not found: {eval_dir}")

    for bench in sorted(os.listdir(eval_dir)):
        bench_dir = os.path.join(eval_dir, bench)
        if not os.path.isdir(bench_dir):
            continue
        if benchmarks is not None and bench not in benchmarks:
            continue
        for dir_arm in sorted(os.listdir(bench_dir)):
            arm_dir = os.path.join(bench_dir, dir_arm)
            if not os.path.isdir(arm_dir):
                continue
            for filename in sorted(os.listdir(arm_dir)):
                parsed = parse_name(filename)
                if parsed is None:
                    continue
                path = os.path.join(arm_dir, filename)
                with open(path, encoding="utf-8") as file:
                    payload = json.load(file)

                # The arm is a property of the model that was built, so it is read from
                # the file. A directory renamed by hand must not relabel a result.
                arm = payload.get("_meta", {}).get("variant")
                if arm is None:
                    warnings.append(f"{path}: no _meta.variant, skipped")
                    continue
                if arm != dir_arm:
                    warnings.append(
                        f"{path}: _meta.variant={arm!r} but directory={dir_arm!r}; skipped")
                    continue

                for test_set, entry in payload.items():
                    if test_set == "_meta" or not isinstance(entry, dict):
                        continue
                    if parsed["train"] == test_set:
                        warnings.append(f"{path}: same-domain result skipped")
                        continue
                    metrics = entry.get("metrics") or {}
                    if not metrics:
                        continue
                    key = (parsed["train"], test_set)
                    (results.setdefault(bench, {}).setdefault(arm, {})
                            .setdefault(key, {})[parsed["fold"]]) = {
                        "metrics": metrics,
                        "pair_med_px": entry.get("pair_med_px") or {},
                    }
    return results, warnings


def aggregate(runs, metric):
    """Aggregate one metric over the folds of a single (arm, test set).

    When only one run is available, its saved spread is used -- the across-support
    spread for a swept arm, the across-test-set spread otherwise.

    Args:
        runs: ``{fold: entry}`` for one benchmark, arm, train set and test set.
        metric: Metric key to aggregate.

    Returns:
        dict | None: ``{"mean", "std", "count", "std_source"}``, or ``None`` when no
        fold carries the metric.
    """
    values = [run["metrics"].get(metric) for run in runs.values()]
    summary = compute_mean_std_count(values)
    if summary is None:
        return None
    mean, std, count = summary
    std_source = "across_folds"

    if count < 2:
        # One fold: fall back to whatever spread the run itself saved, and say which.
        for suffix in STD_SUFFIXES:
            saved = compute_mean_std_count(
                [run["metrics"].get(metric + suffix) for run in runs.values()])
            if saved is not None:
                std, std_source = saved[0], suffix.strip("_")
                break
        else:
            std, std_source = float("nan"), "none"
    return {"mean": mean, "std": std, "count": count, "std_source": std_source}


def build_table(results):
    """Aggregate every (benchmark, arm, train, test, metric).

    Args:
        results: Output of :func:`collect`.

    Returns:
        dict: ``{benchmark: {(train, test): {metric: {variant: stats}}}}``.
    """
    table = {}
    for bench, arms in results.items():
        for arm, groups in arms.items():
            for key, runs in groups.items():
                metrics = set()
                for run in runs.values():
                    metrics.update(k for k, v in run["metrics"].items()
                                   if isinstance(v, (int, float)) and not _is_std_key(k))
                for metric in metrics:
                    summary = aggregate(runs, metric)
                    if summary is None:
                        continue
                    (table.setdefault(bench, {}).setdefault(key, {})
                          .setdefault(metric, {})[arm]) = summary
    return table


def _is_std_key(metric):
    """Whether a metric key is itself a saved spread rather than a value."""
    return any(metric.endswith(suffix) for suffix in STD_SUFFIXES)


def order_variants(found):
    """Order the arms canonically, with anything unrecognised appended.

    Args:
        found: An iterable of arm names present in the results.

    Returns:
        list[str]: The arms in the order the ablation table reads best.
    """
    found = set(found)
    ordered = [variant for variant in VARIANTS if variant in found]
    return ordered + sorted(found - set(ordered))


def pick_metrics(bench, available):
    """The metrics a printed table for one benchmark is restricted to.

    Args:
        bench: Benchmark directory name.
        available: The metric keys the benchmark's results actually carry.

    Returns:
        list[str]: Headline metrics present in the results, in reading order.
    """
    wanted = HEADLINES.get(bench)
    if wanted is None:  # an unrecognised benchmark directory: show what it has
        return sorted(available)
    return [metric for metric in wanted if metric in available]


def print_tables(table, arms):
    """Print one arms-across table per benchmark and metric.

    Args:
        table: Output of :func:`build_table`.
        arms: Arm display order.
    """
    for bench in sorted(table):
        groups = table[bench]
        available = {metric for metrics in groups.values() for metric in metrics}
        metrics = pick_metrics(bench, available)
        if not metrics:
            continue
        for metric in metrics:
            direction = "lower is better" if metric in LOWER_BETTER else "higher is better"
            title = f"{bench}: {metric} ({direction})"
            print(f"\n===== {title} =====")
            header = f"{'train -> test':<36}" + "".join(
                f"{arm:>24}" for arm in arms)
            print(header)
            print("-" * len(header))
            for key in sorted(groups):
                scores = groups[key].get(metric)
                if not scores:
                    continue
                train, test = key
                row = f"{f'{train} -> {test}':<36}"
                for arm in arms:
                    entry = scores.get(arm)
                    if entry is None:
                        row += f"{'-':>24}"
                        continue
                    # A single-fold result whose run saved no spread of its own has no
                    # honest error bar; print the value alone rather than "+/- nan".
                    cell = (f"{entry['mean']:.4f}" if entry["std"] != entry["std"]
                            else f"{entry['mean']:.4f} +/- {entry['std']:.4f}")
                    row += f"{cell:>24}"
                print(row)


def paired_tests(results, arm_order, reference=REFERENCE_VARIANT):
    """Signed-rank test of every arm against the reference, over shared pairs.

    The arms are scored on identical pairs, so a paired test is the right one: per-pair
    difficulty is far larger than the effects being compared, and an unpaired test would
    mostly measure it. Only runs that saved per-pair errors take part.

    Args:
        results: Output of :func:`collect`.
        arm_order: Arm display order.
        reference: The arm every other one is compared against.

    Returns:
        dict: ``{benchmark: {(train, test): {variant: {...}}}}``.
    """
    out = {}
    for bench, arms in results.items():
        if reference not in arms:
            continue
        for key, ref_runs in arms[reference].items():
            ref_pairs = _merge_pairs(ref_runs)
            if not ref_pairs:
                continue
            for arm in arm_order:
                if arm == reference or key not in arms.get(arm, {}):
                    continue
                arm_pairs = _merge_pairs(arms[arm][key])
                shared = sorted(set(ref_pairs) & set(arm_pairs))
                if len(shared) < 2:
                    continue
                # Positive = the arm is more accurate than the reference on that pair.
                deltas = [ref_pairs[p] - arm_pairs[p] for p in shared]
                if all(d == 0 for d in deltas):
                    p_value = 1.0
                else:
                    p_value = float(stats.wilcoxon(deltas).pvalue)
                ordered = sorted(deltas)
                mid = len(ordered) // 2
                median = (ordered[mid] if len(ordered) % 2
                          else 0.5 * (ordered[mid - 1] + ordered[mid]))
                (out.setdefault(bench, {}).setdefault(key, {})[arm]) = {
                    "median_delta": float(median),
                    "p_value": p_value,
                    "num_pairs": len(shared),
                }
    return out


def _merge_pairs(runs):
    """Flatten the per-pair errors of every fold into one ``{pair: value}`` map."""
    merged = {}
    for run in runs.values():
        merged.update(run.get("pair_med_px") or {})
    return merged


def print_paired_tests(paired, reference=REFERENCE_VARIANT):
    """Print the paired comparison against the reference arm.

    Args:
        paired: Output of :func:`paired_tests`.
        reference: The arm every other one was compared against.
    """
    if not paired:
        return
    print(f"\n===== paired med_px improvement over '{reference}' (positive = the arm "
          "is more accurate) =====")
    print("  Wilcoxon signed-rank over the pairs both arms scored.")
    for bench in sorted(paired):
        for key in sorted(paired[bench]):
            train, test = key
            print(f"\n  {bench}  {train} -> {test}")
            for variant, entry in paired[bench][key].items():
                print(f"    {variant:<24} {entry['median_delta']:>10.4f} px   "
                      f"p = {entry['p_value']:<10.4g} n = {entry['num_pairs']}")


def save_long_csv(path, table, arms):
    """Write the long-format CSV: one row per (benchmark, arm, test set, metric).

    Args:
        path: Destination CSV path.
        table: Output of :func:`build_table`.
        arms: Arm display order; rows are emitted in it.
    """
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["benchmark", "variant", "train_set", "test_set",
                         "metric", "mean", "std", "std_source", "num_runs",
                         "lower_is_better"])
        for bench in sorted(table):
            for key in sorted(table[bench]):
                train, test = key
                for metric in sorted(table[bench][key]):
                    scores = table[bench][key][metric]
                    for arm in arms:
                        entry = scores.get(arm)
                        if entry is None:
                            continue
                        writer.writerow([
                            bench, arm, train, test, metric,
                            f"{entry['mean']:.6g}", f"{entry['std']:.6g}",
                            entry["std_source"], entry["count"],
                            int(metric in LOWER_BETTER),
                        ])


def save_headline_csv(path, table, arms):
    """Write the wide reading table: arms across, headline metrics down.

    Args:
        path: Destination CSV path.
        table: Output of :func:`build_table`.
        arms: Arm display order; one column per arm.
    """
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["benchmark", "train_set", "test_set", "metric",
                         "lower_is_better"] + list(arms))
        for bench in sorted(table):
            groups = table[bench]
            available = {metric for metrics in groups.values() for metric in metrics}
            for metric in pick_metrics(bench, available):
                for key in sorted(groups):
                    scores = groups[key].get(metric)
                    if not scores:
                        continue
                    train, test = key
                    row = [bench, train, test, metric, int(metric in LOWER_BETTER)]
                    for arm in arms:
                        entry = scores.get(arm)
                        row.append("" if entry is None else f"{entry['mean']:.6g}")
                    writer.writerow(row)


def main():
    """Aggregate every arm's results and write the ablation summary files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eval_dir", default="workspace/eval/fewshot",
                        help="directory holding <benchmark>/<variant>/<tag>.json, i.e. "
                             "the sweep's ${OUT_DIR}/${GROUP}")
    parser.add_argument("--out_dir", default=None,
                        help="where to write the summary files (default: --eval_dir)")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="restrict to these benchmark subdirectories (Retina / "
                             "i.e. Retina). A sweep passes its own, so its tables "
                             "land beside its results and describe only them; omit it "
                             "to compare the arms across every benchmark at once")
    parser.add_argument("--reference_variant", dest="ref", default=REFERENCE_VARIANT,
                        help="the arm the paired tests are taken against")
    args = parser.parse_args()

    results, warnings = collect(args.eval_dir, args.benchmarks)
    for warning in warnings:
        print(f"!!! {warning}")
    if not results:
        scope = "" if args.benchmarks is None else f" for {', '.join(args.benchmarks)}"
        print(f"No ablation JSONs under {args.eval_dir}{scope}; nothing to do.")
        return

    arms = order_variants({arm for bench in results.values() for arm in bench})
    table = build_table(results)
    print_tables(table, arms)

    paired = paired_tests(results, arms, reference=args.ref)
    print_paired_tests(paired, reference=args.ref)


    out_dir = args.out_dir or args.eval_dir
    os.makedirs(out_dir, exist_ok=True)
    long_csv = os.path.join(out_dir, "ablation_summary.csv")
    headline_csv = os.path.join(out_dir, "ablation_headline.csv")
    summary_json = os.path.join(out_dir, "ablation_summary.json")
    save_long_csv(long_csv, table, arms)
    save_headline_csv(headline_csv, table, arms)
    # JSON keys have to be strings, so the (train, test) tuples are joined.
    save_json(summary_json, {
        "variants": arms,
        "reference_variant": args.ref,
        "metrics": {bench: {" / ".join(key): metrics
                            for key, metrics in by_key.items()}
                    for bench, by_key in table.items()},
        "paired": {bench: {" / ".join(key): entry for key, entry in by_key.items()}
                   for bench, by_key in paired.items()},
    })
    print(f"\nwrote {long_csv}\n      {headline_csv}\n      {summary_json}")


if __name__ == "__main__":
    main()
