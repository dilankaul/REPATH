from __future__ import annotations

import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from ..algorithms.repath.runtime import clear_model_caches
from ..evaluation import metadata
from .runner import run_profile_site
from .timing import COMPONENTS, ComponentTimer


def aggregate(site_frame):
    repetitions = site_frame.groupby("repeat").sum(numeric_only=True)
    total = site_frame.runtime_total_ms.sum()
    rows = []
    for name in COMPONENTS:
        per_query = repetitions[f"{name}_ms"] / repetitions.framework_queries
        rows.append({
            "component": name,
            "runtime_mode": site_frame.runtime_mode.iloc[0],
            "repeats": len(repetitions),
            "evaluated_queries": int(site_frame.framework_queries.sum()),
            "total_measured_runtime_ms": total,
            "component_runtime_ms": site_frame[f"{name}_ms"].sum(),
            "component_share_pct": 100 * site_frame[f"{name}_ms"].sum() / total if total else 0,
            "mean_component_ms_per_query": per_query.mean(),
            "std_component_ms_per_query": per_query.std(ddof=1),
        })
    return pd.DataFrame(rows)


def profile_command(args):
    from ..cli import _factory, _select_sites

    sites = _select_sites(Path(args.data_root), args.sites, args.split_config)
    if not sites:
        raise SystemExit("No sites selected")
    output = Path(args.output or f"results/profiling/{args.runtime_mode}")
    output.mkdir(parents=True, exist_ok=True)
    preload = _factory("repath", args)()
    preload.close()
    del preload
    if args.runtime_mode == "cold":
        clear_model_caches()
    timer = ComponentTimer()
    rows, summaries = [], []
    with timer.instrument():
        factory = _factory("repath", args, args.runtime_mode)
        for repeat in range(1, args.repeats + 1):
            for site in sites:
                query_rows, summary = run_profile_site(site, factory, repeat, args.runtime_mode, timer)
                rows.extend(query_rows)
                summaries.append(summary)
                print(f"{site.name} | repeat {repeat} | {summary['correct_top1']}/{summary['framework_queries']} | {summary['mean_runtime_ms']:.3f} ms/query")
    site_frame = pd.DataFrame(summaries)
    pd.DataFrame(rows).to_csv(output / "component_query_results.csv", index=False)
    site_frame.to_csv(output / "component_site_results.csv", index=False)
    aggregate(site_frame).to_csv(output / "component_summary.csv", index=False)
    arguments = {key: value for key, value in vars(args).items() if key != "func"}
    info = metadata(arguments)
    packages = {}
    for package in ("numpy", "pandas", "lightgbm", "torch", "sentence-transformers", "transformers", "joblib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    info.update(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runtime_protocol=args.runtime_mode, selected_sites=[site.name for site in sites],
        repeats=args.repeats, evaluated_queries=int(site_frame.framework_queries.sum()),
        model_path=str(Path(args.repath_model).resolve()),
        configuration_path=str(Path(args.repath_config).resolve()), device=args.repath_device,
        package_versions=packages, timing_method="time.perf_counter_ns; exclusive nested intervals",
        model_loading_included=args.runtime_mode == "cold", component_definitions=COMPONENTS,
        accounting="Sum all scopes; query rows contain actual execution; page_pair_shared contains preparation, cleanup and loop overhead once per site/repeat",
    )
    (output / "profiling_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if args.export:
        export_results(output, Path(args.export))


def export_results(output, destination):
    """Retain a compact summary only when requested; never overwrite artifacts."""
    info = json.loads((output / "profiling_metadata.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    summary = destination / f"runtime_{info['runtime_protocol']}_component_summary.csv"
    if summary.exists():
        raise FileExistsError("Export destination already contains profiling artifacts; use a new directory")
    summary.write_bytes((output / "component_summary.csv").read_bytes())


def configure_parser(parser):
    from ..cli import _positive_int

    parser.epilog = (
        "Component-Level Runtime Profiling: a separate REPATH-only experiment, without "
        "confidence thresholding or model tuning. Cold loads and prepares a fresh matcher "
        "for every eligible query; amortized loads models before timing and reuses page-pair "
        "state and embeddings. CPU only. Components: " + ", ".join(COMPONENTS) + ". "
        "The query CSV includes qid, site, repeat, protocol, scope, expected/predicted IDs, "
        "correctness, runtime_ms and <component>_ms. Sum query and page_pair_shared scopes "
        "to reconcile site totals; shared costs occur once. Site CSV includes query count, "
        "accuracy, total/mean runtime, component totals and percentages. Summary CSV uses "
        "runtime-weighted shares and mean/sample SD of component ms/query across full-set "
        "repetitions (SD is empty for one repeat). Instrumentation affects elapsed time; "
        "use the original uninstrumented benchmark for publication runtime comparisons. "
        "Reproduce with: repath-eval profile --sites test --repeats 10 --runtime-mode cold "
        "(or amortized). Add --export artifacts/profiling to retain a compact summary. "
        "Generate the environment later with repath-eval environment "
        "--output results/profiling/runtime_environment.txt; this captures the environment "
        "at invocation, so run it on the same host with the same thread settings."
    )

    parser.add_argument("--runtime-mode", choices=["cold", "amortized"], default="cold")
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--output", help="Default: results/profiling/<runtime-mode>")
    parser.add_argument("--repath-model", default="models/lgb_ranker.txt")
    parser.add_argument("--repath-config", default="models/config_base.pkl")
    parser.add_argument("--repath-device", choices=["cpu"], default="cpu", help="CPU profiling only")
    parser.add_argument("--export", metavar="DIRECTORY", help="Optionally retain summary, e.g. artifacts/profiling")
    parser.set_defaults(func=profile_command, sites=["test"])
