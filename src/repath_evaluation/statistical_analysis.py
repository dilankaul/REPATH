"""Query-level paired analysis; nominal intervals ignore within-website correlation."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ALGORITHMS = ("REPATH", "WATER", "VISTA", "SFTM", "WebEvo", "Similo")
SITES = {
    "book-2016-2019", "beijing-2017-2019", "instagram", "movie-2014-2019",
    "xfinity-2018-2020", "yahoo",
}
QUERY_COUNT = 318


def load_outcomes(path: str | Path) -> pd.DataFrame:
    """Validate repeat 1 completely, then align by (site, old_element_id)."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "algorithm", "site", "repeat", "old_element_id", "expected_new_element_id", "correct",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    repeats = pd.to_numeric(frame["repeat"], errors="coerce")
    if repeats.isna().any() or (repeats < 1).any() or (repeats % 1 != 0).any():
        raise ValueError("Repeat values must be positive integers")
    frame = frame.loc[repeats == 1].copy()
    if set(frame.algorithm) != set(ALGORITHMS):
        raise ValueError("Repeat 1 must contain exactly the six expected algorithms")
    keys = ["site", "old_element_id"]
    if frame[list(required)].eq("").any().any():
        raise ValueError("Empty required values in repeat 1")
    if not frame.correct.isin(["True", "False"]).all():
        raise ValueError("Correct must be True or False")
    if frame.duplicated(["algorithm", *keys]).any():
        raise ValueError("Duplicate query outcomes in repeat 1")
    reference = frame.loc[frame.algorithm == "REPATH"].set_index(keys).sort_index()
    if len(reference) != QUERY_COUNT or set(reference.index.get_level_values("site")) != SITES:
        raise ValueError("REPATH must contain 318 queries from the six held-out websites")
    outcomes = pd.DataFrame(index=reference.index)
    for algorithm in ALGORITHMS:
        rows = frame.loc[frame.algorithm == algorithm].set_index(keys).sort_index()
        if len(rows) != QUERY_COUNT or not rows.index.equals(reference.index):
            raise ValueError(f"Missing or mismatched query records for {algorithm}")
        if not rows.expected_new_element_id.equals(reference.expected_new_element_id):
            raise ValueError(f"Mismatched expected target mapping for {algorithm}")
        outcomes[algorithm] = rows.correct.eq("True")
    return outcomes


def contingency(repath: np.ndarray, baseline: np.ndarray) -> tuple[int, int, int, int]:
    """Return both correct, REPATH only, baseline only, both incorrect."""
    return tuple(int(mask.sum()) for mask in (
        repath & baseline, repath & ~baseline, ~repath & baseline, ~repath & ~baseline,
    ))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, restored to original comparison order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1))
    )
    return adjusted


def bootstrap_accuracies(outcomes: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Shared query indices for every algorithm; default_rng/PCG64, fixed batches."""
    if count < 1:
        raise ValueError("Bootstrap count must be positive")
    rng = np.random.default_rng(seed)
    samples = np.empty((count, outcomes.shape[1]))
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        indices = rng.integers(0, len(outcomes), size=(stop - start, len(outcomes)))
        samples[start:stop] = outcomes[indices].mean(axis=1) * 100
    return samples


def analyze(outcomes: pd.DataFrame, count: int = 10000, seed: int = 42):
    values = outcomes.loc[:, ALGORITHMS].to_numpy(dtype=bool)
    accuracy = values.mean(axis=0) * 100
    samples = bootstrap_accuracies(values, count, seed)
    bounds = np.quantile(samples, [0.025, 0.975], axis=0, method="linear")
    metadata = {"bootstrap_count": count, "seed": seed, "interval_type": "nominal 95% percentile"}
    individual = pd.DataFrame([
        {"algorithm": name, "queries": len(values), "correct": int(values[:, i].sum()),
         "accuracy_pct": accuracy[i], "ci_lower_pct": bounds[0, i],
         "ci_upper_pct": bounds[1, i], **metadata}
        for i, name in enumerate(ALGORITHMS)
    ])
    comparisons = []
    for i, name in enumerate(ALGORITHMS[1:], 1):
        a, b, c, d = contingency(values[:, 0], values[:, i])
        lower, upper = np.quantile(samples[:, 0] - samples[:, i], [0.025, 0.975])
        comparisons.append({
            "algorithm": "REPATH", "baseline": name, "queries": len(values),
            "repath_correct": a + b, "baseline_correct": a + c,
            "both_correct_a": a, "repath_only_correct_b": b,
            "baseline_only_correct_c": c, "both_incorrect_d": d,
            "accuracy_improvement_pp": accuracy[0] - accuracy[i],
            "mcnemar_p_value": binomtest(b, b + c, p=0.5).pvalue if b + c else 1.0,
            "ci_lower_pp": lower, "ci_upper_pp": upper, **metadata,
        })
    paired = pd.DataFrame(comparisons)
    paired["holm_adjusted_p_value"] = holm_adjust(paired.mcnemar_p_value.to_numpy())
    return individual, paired


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Share options and execution between standalone and unified CLIs."""
    parser.add_argument("--input", type=Path, default=Path("results/benchmark/query_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/statistics"))
    parser.add_argument("--bootstrap-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=partial(statistics_command, parser=parser))


def statistics_command(args: argparse.Namespace, *, parser: argparse.ArgumentParser) -> None:
    if args.bootstrap_count < 1 or args.seed < 0:
        parser.error("Bootstrap count must be positive and seed must be nonnegative")
    # Guard against accidentally overwriting the input or retained artifacts.
    output_paths = [args.output_dir / name for name in (
        "accuracy_confidence_intervals.csv", "statistical_comparison.csv",
    )]
    artifact_root = Path(__file__).resolve().parents[2] / "artifacts"
    if any(path.resolve().is_relative_to(artifact_root) for path in output_paths):
        parser.error("Output under artifacts/ is reserved for manual retention")
    if args.input.resolve() in [path.resolve() for path in output_paths]:
        parser.error("Output must not overwrite the input CSV")
    try:
        tables = analyze(load_outcomes(args.input), args.bootstrap_count, args.seed)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for table, path in zip(tables, output_paths):
        table.to_csv(path, index=False)
        print(f"Wrote {len(table)} rows to {path}")
    print("Nominal 95% intervals; query-level analysis ignores within-website correlations.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
