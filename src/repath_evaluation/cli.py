from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .algorithms import (
    RepathMatcher,
    SftmMatcher,
    SimiloMatcher,
    VistaMatcher,
    WaterMatcher,
    WebEvoMatcher,
)
from .algorithms.repath.runtime import clear_model_caches
from .calibration import run_calibration
from .data import discover_sites, load_page_pair, load_site_split
from .doctor import doctor_command
from .evaluation import aggregate_summary, build_queries, metadata, run_site
from .statistical_analysis import configure_parser as configure_statistics_parser

_ALGORITHM_DISPLAY_NAMES = {
    "water": "WATER",
    "vista": "VISTA",
    "sftm": "SFTM",
    "webevo": "WebEvo",
    "similo": "Similo",
    "repath": "REPATH",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _select_sites(
    data_root: Path,
    names: list[str] | None,
    split_config: str | None,
) -> list[Path]:
    discovered = discover_sites(data_root)
    split = load_site_split(Path(split_config)) if split_config else None
    selectors = list(names) if names else [split.default_group if split else "all"]

    requested: list[str] = []
    for selector in selectors:
        normalized = selector.casefold()
        if normalized in {"development", "test", "all"}:
            if split is None:
                if normalized == "all":
                    requested.extend(path.name for path in discovered)
                    continue
                raise SystemExit(f"--sites {selector} requires --split-config")
            requested.extend(split.group(normalized))
        else:
            requested.append(selector)

    requested = list(dict.fromkeys(requested))
    exact = {path.name: path for path in discovered}
    folded: dict[str, list[Path]] = {}
    for path in discovered:
        folded.setdefault(path.name.casefold(), []).append(path)

    selected: list[Path] = []
    missing: list[str] = []
    for name in requested:
        path = exact.get(name)
        if path is None:
            matches = folded.get(name.casefold(), [])
            if len(matches) == 1:
                path = matches[0]
        if path is None:
            missing.append(name)
        else:
            selected.append(path)

    if missing:
        raise SystemExit(f"Sites not found: {', '.join(missing)}")
    return selected


def _factory(name: str, args, runtime_mode: str | None = None):
    normalized = name.casefold()
    if normalized == "water":
        return WaterMatcher
    if normalized == "vista":
        return VistaMatcher
    if normalized == "sftm":
        return SftmMatcher
    if normalized == "webevo":
        return WebEvoMatcher
    if normalized == "similo":
        return SimiloMatcher
    if normalized == "repath":
        if not args.repath_model:
            raise SystemExit("--repath-model is required when using repath")
        if not args.repath_config:
            raise SystemExit("--repath-config is required when using repath")
        return lambda: RepathMatcher(
            model_path=Path(args.repath_model),
            config_path=Path(args.repath_config),
            device=args.repath_device,
            cache_models=runtime_mode != "cold",
            local_files_only=runtime_mode == "cold",
        )
    raise SystemExit(f"Unknown algorithm: {name}")


def inspect_command(args) -> None:
    sites = _select_sites(Path(args.data_root), args.sites, args.split_config)
    total = 0
    rows = []
    for site in sites:
        pair = load_page_pair(site)
        queries = build_queries(pair)
        eligible = sum(query.eligible for query in queries)
        total += eligible
        rows.append(
            {
                "site": pair.site,
                "match_map": len(pair.match_map),
                "eligible_queries": eligible,
                "excluded_or_missing": len(pair.match_map) - eligible,
                "old_grouped_nodes": len(pair.old_page.nodes),
                "new_grouped_nodes": len(pair.new_page.nodes),
                "old_pre_nodes": len(pair.old_page.pre_nodes),
                "new_pre_nodes": len(pair.new_page.pre_nodes),
                "screenshots": bool(
                    pair.old_page.full_screen_path and pair.new_page.full_screen_path
                ),
            }
        )
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    print(f"\nTotal eligible queries: {total}")

    expected = args.expected_queries
    if expected is None and args.split_config:
        split = load_site_split(Path(args.split_config))
        selectors = args.sites or [split.default_group]
        if len(selectors) == 1 and selectors[0].casefold() in {
            "development",
            "test",
            "all",
        }:
            expected = split.expected_for(selectors[0])
    if expected is not None and total != expected:
        raise SystemExit(f"Expected {expected}, found {total}")


def benchmark_command(args) -> None:
    data_root = Path(args.data_root)
    sites = _select_sites(data_root, args.sites, args.split_config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime_mode = getattr(args, "runtime_mode", None)
    all_rows = []
    site_summaries = []

    for algorithm in args.algorithms:
        if runtime_mode is not None and algorithm.casefold() == "repath":
            preload_factory = _factory(algorithm, args, runtime_mode=None)
            preload_matcher = preload_factory()
            preload_matcher.close()
            del preload_matcher
            if runtime_mode == "cold":
                clear_model_caches()

        factory = _factory(algorithm, args, runtime_mode=runtime_mode)
        for repeat in range(1, args.repeats + 1):
            for site in sites:
                rows, summary = run_site(
                    site,
                    factory,
                    repeat,
                    runtime_mode=runtime_mode,
                    algorithm_name=_ALGORITHM_DISPLAY_NAMES[algorithm.casefold()],
                )
                all_rows.extend(rows)
                site_summaries.append(summary)
                message = (
                    f"{summary['algorithm']:>12} | {summary['site']:<24} | "
                    f"repeat {repeat} | "
                    f"{summary['correct_top1']}/{summary['framework_queries']} "
                    f"({summary['top1_accuracy']:.4f})"
                )
                if runtime_mode is not None:
                    message += (
                        f" | {runtime_mode} mean "
                        f"{summary['mean_runtime_ms']:.3f} ms/query"
                    )
                print(message)

    query_frame = pd.DataFrame(all_rows)
    site_frame = pd.DataFrame(site_summaries)
    summary_frame = aggregate_summary(site_frame)
    query_frame.to_csv(output / "query_results.csv", index=False)
    site_frame.to_csv(output / "site_results.csv", index=False)
    summary_frame.to_csv(output / "summary.csv", index=False)

    metadata_arguments = {
        key: value
        for key, value in vars(args).items()
        if key != "func"
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata(metadata_arguments), indent=2, default=str),
        encoding="utf-8",
    )
    print("\nSummary")
    print(summary_frame.to_string(index=False))


# Backward-compatible import for code written against the benchmark-only release.
run_command = benchmark_command


def calibrate_command(args) -> None:
    base_config = Path(args.base_config) if args.base_config else None
    config_output = Path(args.config_output) if args.config_output else None
    try:
        summary, curve, _ = run_calibration(
            manifest_path=Path(args.manifest),
            development_scores_path=Path(args.development_scores),
            test_scores_path=Path(args.test_scores),
            output_dir=Path(args.output),
            tau=args.tau,
            target_precision=args.target_precision,
            minimum_end_to_end_coverage=args.min_end_to_end_coverage,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            tau_step=args.tau_step,
            base_config_path=base_config,
            config_output_path=config_output,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc

    selected_tau = float(summary.iloc[0]["tau_margin"])
    mode = "manual" if args.tau is not None else "development-OOF selected"
    print(f"Selected tau margin ({mode}): {selected_tau:.6f}\n")

    display_columns = [
        "name",
        "tau_margin",
        "accepted",
        "correct",
        "precision",
        "conditional_coverage",
        "end_to_end_coverage",
        "conditional_recall",
        "end_to_end_recall",
        "conditional_f1",
        "end_to_end_f1",
    ]
    print(summary[display_columns].to_string(index=False))
    print(f"\nSaved calibration outputs to {Path(args.output)}")
    if curve is not None:
        print(f"Development threshold candidates evaluated: {len(curve)}")
    if config_output is not None:
        print(f"Saved calibrated runtime configuration to {config_output}")


def compare_command(args) -> None:
    python = pd.read_csv(args.python_results)
    reference = pd.read_csv(args.reference_results)
    keys = ["algorithm", "site", "old_element_id"]
    required = set(keys + ["predicted_new_element_id"])
    for name, frame in (("python", python), ("reference", reference)):
        missing = required - set(frame.columns)
        if missing:
            raise SystemExit(f"{name} results missing columns: {sorted(missing)}")
    python = python.sort_values("repeat").drop_duplicates(keys)
    reference = (
        reference.sort_values("repeat") if "repeat" in reference.columns else reference
    )
    reference = reference.drop_duplicates(keys)
    merged = python.merge(
        reference,
        on=keys,
        suffixes=("_python", "_reference"),
        how="outer",
        indicator=True,
    )
    merged["same_prediction"] = (
        merged["predicted_new_element_id_python"].fillna(-999999).astype(float)
        == merged["predicted_new_element_id_reference"].fillna(-999999).astype(float)
    )
    mismatches = merged[~merged["same_prediction"] | (merged["_merge"] != "both")]
    print(f"Compared rows: {len(merged)}")
    print(f"Prediction parity: {merged['same_prediction'].mean():.4%}")
    if len(mismatches):
        print("\nMismatches")
        print(mismatches.head(50).to_string(index=False))
        if args.output:
            mismatches.to_csv(args.output, index=False)
        raise SystemExit(1)


def _add_site_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default="data",
        help="Dataset directory containing UITestFix site folders (default: data)",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        help="Site names or groups: test (default), development, or all",
    )
    parser.add_argument("--split-config", default="configs/site_split.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repath-eval", allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser(
        "inspect",
        allow_abbrev=False,
        help="Inspect UITestFix sites and query counts",
    )
    _add_site_selection_arguments(inspect)
    inspect.add_argument("--expected-queries", type=int)
    inspect.set_defaults(func=inspect_command)

    benchmark = sub.add_parser(
        "benchmark",
        aliases=["run"],
        allow_abbrev=False,
        help="Run baseline and REPATH Top-1 evaluation",
    )
    _add_site_selection_arguments(benchmark)
    benchmark.add_argument(
        "--algorithms",
        nargs="+",
        default=["water", "vista", "sftm", "webevo", "similo", "repath"],
        choices=["water", "vista", "sftm", "webevo", "similo", "repath"],
        help=(
            "Algorithms to evaluate "
            "(default: water vista sftm webevo similo repath)"
        ),
    )
    benchmark.add_argument(
        "--repeats",
        type=_positive_int,
        default=1,
        help="Number of complete benchmark repetitions (default: 1)",
    )
    benchmark.add_argument(
        "--runtime-mode",
        choices=["cold", "amortized"],
        default=None,
        help=(
            "Runtime protocol. Omit for accuracy-only execution; "
            "'cold' loads and prepares independently for every query; "
            "'amortized' times page loading, preparation, and all queries "
            "once per page pair, then divides the total by eligible queries."
        ),
    )
    benchmark.add_argument(
        "--output",
        default="results/benchmark",
        help=(
            "Output directory for benchmark files "
            "(default: results/benchmark; existing files are overwritten)"
        ),
    )
    benchmark.add_argument(
        "--repath-model",
        default="models/lgb_ranker.txt",
        help="Path to the trained REPATH LightGBM model",
    )
    benchmark.add_argument(
        "--repath-config",
        default="models/config_base.pkl",
        help="Path to the REPATH inference configuration",
    )
    benchmark.add_argument(
        "--repath-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device used by the REPATH SentenceTransformer",
    )
    benchmark.set_defaults(func=benchmark_command)

    calibrate = sub.add_parser(
        "calibrate",
        allow_abbrev=False,
        help="Evaluate or select a REPATH confidence-margin threshold",
    )
    calibrate.add_argument(
        "--manifest",
        default="artifacts/repath/repath_query_manifest.csv",
        help="Query manifest with qid, site, and split columns",
    )
    calibrate.add_argument(
        "--development-scores",
        default="artifacts/repath/development_oof_candidate_scores.csv",
        help="Pooled development OOF candidate scores",
    )
    calibrate.add_argument(
        "--test-scores",
        default="artifacts/repath/test_candidate_scores.csv",
        help="Held-out test candidate scores",
    )
    calibrate.add_argument(
        "--tau",
        type=float,
        help=(
            "Evaluate this fixed margin threshold. Omit to select tau using "
            "development OOF precision and coverage constraints."
        ),
    )
    calibrate.add_argument("--target-precision", type=float, default=0.99)
    calibrate.add_argument(
        "--min-end-to-end-coverage",
        type=float,
        default=0.80,
    )
    calibrate.add_argument("--tau-min", type=float, default=0.0)
    calibrate.add_argument("--tau-max", type=float, default=1.0)
    calibrate.add_argument("--tau-step", type=float, default=0.05)
    calibrate.add_argument("--output", default="results/calibration")
    calibrate.add_argument(
        "--base-config",
        help="Optional base config_base.pkl used to write a calibrated config",
    )
    calibrate.add_argument(
        "--config-output",
        help="Output path for calibrated config.pkl; requires --base-config",
    )
    calibrate.set_defaults(func=calibrate_command)

    doctor = sub.add_parser(
        "doctor",
        allow_abbrev=False,
        help="Validate benchmark dependencies and artifacts",
    )
    doctor.add_argument(
        "--data-root",
        default="data",
        help="Dataset directory containing UITestFix site folders (default: data)",
    )
    doctor.add_argument("--split-config", default="configs/site_split.yaml")
    doctor.add_argument("--repath-model", default="models/lgb_ranker.txt")
    doctor.add_argument("--repath-config", default="models/config_base.pkl")
    doctor.set_defaults(func=doctor_command)

    compare = sub.add_parser(
        "compare",
        allow_abbrev=False,
        help="Compare suite predictions with external reference predictions",
    )
    compare.add_argument("--python-results", required=True)
    compare.add_argument("--reference-results", required=True)
    compare.add_argument("--output")
    compare.set_defaults(func=compare_command)

    statistics = sub.add_parser(
        "statistics",
        allow_abbrev=False,
        help="Compute accuracy intervals and paired statistical comparisons",
    )
    configure_statistics_parser(statistics)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
