from argparse import Namespace
from pathlib import Path

import pandas as pd

from repath_evaluation import cli


def test_cli_defaults_to_test_group_and_model_artifacts() -> None:
    args = cli.build_parser().parse_args(["benchmark"])

    assert args.data_root == "data"
    assert args.sites is None
    assert args.split_config == "configs/site_split.yaml"
    assert args.runtime_mode is None
    assert args.algorithms == ["water", "vista", "sftm", "webevo", "similo", "repath"]
    assert args.repeats == 1
    assert args.output == "results/benchmark"
    assert args.repath_model == "models/lgb_ranker.txt"
    assert args.repath_config == "models/config_base.pkl"


def test_run_remains_a_backward_compatible_alias() -> None:
    args = cli.build_parser().parse_args(
        ["run", "--algorithms", "water", "--output", "out"]
    )

    assert args.func is cli.benchmark_command


def test_calibration_defaults_and_manual_tau() -> None:
    args = cli.build_parser().parse_args(["calibrate", "--tau", "0.4"])

    assert args.tau == 0.4
    assert args.manifest == "artifacts/repath/repath_query_manifest.csv"
    assert args.development_scores.endswith("development_oof_candidate_scores.csv")
    assert args.test_scores.endswith("test_candidate_scores.csv")
    assert args.output == "results/calibration"


def test_runtime_mode_supports_cold_and_amortized_protocols() -> None:
    parser = cli.build_parser()

    cold = parser.parse_args(["benchmark", "--runtime-mode", "cold"])
    amortized = parser.parse_args(["benchmark", "--runtime-mode", "amortized"])

    assert cold.runtime_mode == "cold"
    assert amortized.runtime_mode == "amortized"


def test_inspect_defaults_to_repository_data_directory() -> None:
    args = cli.build_parser().parse_args(["inspect"])

    assert args.data_root == "data"
    assert args.sites is None
    assert args.split_config == "configs/site_split.yaml"


def _make_discoverable_site(root: Path, name: str) -> Path:
    site = root / name
    for relative in ["old/domNodeInfo.json", "new/domNodeInfo.json", "result/result.json"]:
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    return site


def test_site_group_selectors_use_test_by_default_and_support_all(tmp_path) -> None:
    development = _make_discoverable_site(tmp_path, "DevSite")
    test = _make_discoverable_site(tmp_path, "TestSite")
    config = tmp_path / "split.yaml"
    config.write_text(
        "default_group: test\n"
        "development_sites:\n  - DevSite\n"
        "test_sites:\n  - TestSite\n",
        encoding="utf-8",
    )

    assert cli._select_sites(tmp_path, None, str(config)) == [test]
    assert cli._select_sites(tmp_path, ["development"], str(config)) == [development]
    assert cli._select_sites(tmp_path, ["all"], str(config)) == [development, test]
    assert cli._select_sites(tmp_path, ["testsite"], str(config)) == [test]


def test_run_writes_accuracy_only_outputs_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "_select_sites", lambda *_: [tmp_path / "site"])
    monkeypatch.setattr(cli, "_factory", lambda *_args, **_kwargs: object)
    monkeypatch.setattr(
        cli,
        "run_site",
        lambda *_args, **_kwargs: (
            [
                {
                    "algorithm": "WATER",
                    "site": "site",
                    "repeat": 1,
                    "old_element_id": 1,
                    "expected_new_element_id": 2,
                    "predicted_new_element_id": None,
                    "correct": False,
                    "status": "no_prediction",
                    "details": "",
                }
            ],
            {
                "algorithm": "WATER",
                "site": "site",
                "repeat": 1,
                "framework_queries": 1,
                "correct_top1": 0,
                "top1_accuracy": 0.0,
                "no_prediction": 1,
            },
        ),
    )
    output = tmp_path / "output"
    args = Namespace(
        data_root=str(tmp_path),
        sites=None,
        split_config="configs/site_split.yaml",
        output=str(output),
        algorithms=["water"],
        repeats=1,
        runtime_mode=None,
    )

    cli.run_command(args)

    assert {path.name for path in output.iterdir()} == {
        "query_results.csv",
        "site_results.csv",
        "summary.csv",
        "run_metadata.json",
    }
    query = pd.read_csv(output / "query_results.csv")
    site = pd.read_csv(output / "site_results.csv")
    summary = pd.read_csv(output / "summary.csv")
    assert "predicted_new_element_id" in query.columns
    assert "runtime_ms" not in query.columns
    assert "mean_runtime_ms" not in site.columns
    assert "mean_runtime_ms" not in summary.columns
    assert bool(query.loc[0, "correct"]) is False


def test_run_writes_runtime_outputs_when_runtime_mode_is_selected(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli, "_select_sites", lambda *_: [tmp_path / "site"])
    monkeypatch.setattr(cli, "_factory", lambda *_args, **_kwargs: object)
    monkeypatch.setattr(
        cli,
        "run_site",
        lambda *_args, **_kwargs: (
            [
                {
                    "algorithm": "WATER",
                    "site": "site",
                    "repeat": 1,
                    "old_element_id": 1,
                    "expected_new_element_id": 2,
                    "predicted_new_element_id": 2,
                    "correct": True,
                    "status": "ok",
                    "details": "",
                    "runtime_mode": "cold",
                    "runtime_ms": 12.0,
                }
            ],
            {
                "algorithm": "WATER",
                "site": "site",
                "repeat": 1,
                "framework_queries": 1,
                "correct_top1": 1,
                "top1_accuracy": 1.0,
                "no_prediction": 0,
                "runtime_mode": "cold",
                "runtime_samples": 1,
                "runtime_total_ms": 12.0,
                "mean_runtime_ms": 12.0,
            },
        ),
    )
    output = tmp_path / "runtime-output"
    args = Namespace(
        data_root=str(tmp_path),
        sites=["test"],
        split_config="configs/site_split.yaml",
        output=str(output),
        algorithms=["water"],
        repeats=1,
        runtime_mode="cold",
    )

    cli.run_command(args)

    query = pd.read_csv(output / "query_results.csv")
    site = pd.read_csv(output / "site_results.csv")
    summary = pd.read_csv(output / "summary.csv")
    assert query.loc[0, "runtime_mode"] == "cold"
    assert query.loc[0, "runtime_ms"] == 12.0
    assert site.loc[0, "runtime_mode"] == "cold"
    assert site.loc[0, "mean_runtime_ms"] == 12.0
    assert summary.loc[0, "runtime_mode"] == "cold"
    assert summary.loc[0, "mean_runtime_ms"] == 12.0


def test_statistics_cli_options_and_execution(monkeypatch, tmp_path) -> None:
    from repath_evaluation import statistical_analysis

    parser = cli.build_parser()
    defaults = parser.parse_args(["statistics"])
    assert defaults.input == Path("results/benchmark/query_results.csv")
    assert defaults.output_dir == Path("results/statistics")
    assert defaults.bootstrap_count == 10000
    assert defaults.seed == 42

    source = tmp_path / "input.csv"
    output = tmp_path / "statistics"
    monkeypatch.setattr(statistical_analysis, "load_outcomes", lambda path: path)
    calls = []

    def fake_analyze(outcomes, count, seed):
        calls.append((outcomes, count, seed))
        return pd.DataFrame({"accuracy": [95]}), pd.DataFrame({"p_value": [0.01]})

    monkeypatch.setattr(statistical_analysis, "analyze", fake_analyze)
    args = parser.parse_args([
        "statistics", "--input", str(source), "--output-dir", str(output),
        "--bootstrap-count", "20", "--seed", "7",
    ])
    args.func(args)
    assert calls == [(source, 20, 7)]
    assert (output / "accuracy_confidence_intervals.csv").exists()
    assert (output / "statistical_comparison.csv").exists()
