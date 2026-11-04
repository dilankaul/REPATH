import json

import pandas as pd
import pytest
from test_timing import _pair, _TimedMatcher

from repath_evaluation.cli import build_parser
from repath_evaluation.profiling import aggregate, profile_command, runner
from repath_evaluation.profiling.timing import COMPONENTS, ComponentTimer


@pytest.mark.parametrize("mode,constructed", [("cold", 2), ("amortized", 1)])
def test_protocol_and_prediction_parity(monkeypatch, tmp_path, mode, constructed):
    import repath_evaluation.evaluation.runner as benchmark

    pair = _pair(tmp_path, query_count=2)
    monkeypatch.setattr(runner, "load_page_pair", lambda _: pair)
    monkeypatch.setattr(benchmark, "load_page_pair", lambda _: pair)
    _TimedMatcher.constructed = _TimedMatcher.prepared = 0
    timer = ComponentTimer()
    load_locations = []

    def factory():
        load_locations.append(bool(timer.values["sbert_load"]))
        return timer.wrap("sbert_load", _TimedMatcher)()

    rows, site = runner.run_profile_site(tmp_path, factory, 1, mode, timer)
    assert _TimedMatcher.constructed == _TimedMatcher.prepared == constructed
    assert site["page_pair_load_ms"] > 0
    assert site["page_pair_preparation_ms"] > 0
    if mode == "amortized":
        shared = next(row for row in rows if row["scope"] == "page_pair_shared")
        for name in ("page_pair_load", "page_pair_preparation"):
            assert shared[f"{name}_ms"] == site[f"{name}_ms"]
            assert all(row[f"{name}_ms"] == 0 for row in rows if row["scope"] == "query")
    plain, _ = benchmark.run_site(tmp_path, _TimedMatcher, 1, runtime_mode=mode, algorithm_name="REPATH")
    query_rows = [row for row in rows if row["scope"] == "query"]
    assert [row["predicted_new_element_id"] for row in query_rows] == [row["predicted_new_element_id"] for row in plain]
    assert site["sbert_load_ms"] > 0 if mode == "cold" else site["sbert_load_ms"] == 0
    assert sum(row["runtime_ms"] for row in rows) == pytest.approx(site["runtime_total_ms"])
    assert sum(site[f"{key}_ms"] for key in COMPONENTS) == pytest.approx(site["runtime_total_ms"])
    assert sum(site[f"{key}_pct"] for key in COMPONENTS) == pytest.approx(100)
    if mode == "amortized":
        assert sum(row["scope"] == "page_pair_shared" for row in rows) == 1


def test_nested_timers_exclude_children(monkeypatch):
    ticks = iter([0, 10, 30, 50])
    monkeypatch.setattr("repath_evaluation.profiling.timing.time.perf_counter_ns", lambda: next(ticks))
    timer = ComponentTimer()
    child = timer.wrap("sbert_encode", lambda: None)
    timer.wrap("semantic_similarity", child)()
    assert timer.values["sbert_encode"] == 20
    assert timer.values["semantic_similarity"] == 30
    assert timer.reconcile(60)["other_overhead"] == 10
    with pytest.raises(ValueError, match="exceed"):
        timer.reconcile(49)
    timer.stack.append([0, 0])
    with pytest.raises(ValueError, match="Unclosed"):
        timer.reconcile(60)


def test_runtime_weighted_summary_and_independent_repeats():
    rows = []
    for repeat, scale in [(1, 1), (2, 2)]:
        for total, cost in [(10, 1), (90, 45)]:
            rows.append({
                "repeat": repeat, "runtime_mode": "cold", "framework_queries": 2,
                "runtime_total_ms": total * scale,
                **{f"{key}_ms": cost * scale if key == "sbert_encode" else 0 for key in COMPONENTS},
            })
    summary = aggregate(pd.DataFrame(rows)).set_index("component").loc["sbert_encode"]
    assert summary.component_share_pct == pytest.approx(46)
    assert summary.mean_component_ms_per_query == pytest.approx(17.25)
    assert summary.std_component_ms_per_query == pytest.approx(pd.Series([11.5, 23]).std())


@pytest.mark.parametrize("mode", ["cold", "amortized"])
def test_command_outputs_and_default_paths(monkeypatch, tmp_path, mode):
    from repath_evaluation import cli, profiling

    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["profile", "--runtime-mode", mode])
    assert args.sites == ["test"]
    assert args.repeats == 1
    monkeypatch.setattr(cli, "_select_sites", lambda *args: [tmp_path])
    monkeypatch.setattr(cli, "_factory", lambda *args, **kwargs: _TimedMatcher)
    monkeypatch.setattr(runner, "load_page_pair", lambda _: _pair(tmp_path, query_count=2))
    monkeypatch.setattr(profiling.subprocess, "check_output", lambda *args, **kwargs: "commit\n")
    profile_command(args)
    output = tmp_path / "results" / "profiling" / mode
    assert {file.name for file in output.iterdir()} == {
        "component_query_results.csv", "component_site_results.csv",
        "component_summary.csv", "profiling_metadata.json",
    }
    info = json.loads((output / "profiling_metadata.json").read_text())
    assert info["evaluated_queries"] == 2
    assert info["model_loading_included"] == (mode == "cold")
    for name in ("page_pair_load", "configuration_load", "page_pair_preparation", "lightgbm_load"):
        assert name in info["component_definitions"]
    assert "lightgbm_load_ms" in pd.read_csv(output / "component_query_results.csv").columns
    site_frame = pd.read_csv(output / "component_site_results.csv")
    assert {"lightgbm_load_ms", "lightgbm_load_pct"} <= set(site_frame.columns)
    summary = pd.read_csv(output / "component_summary.csv")
    assert "lightgbm_load" in set(summary.component)
    assert summary.component_share_pct.sum() == pytest.approx(100)
    assert summary.std_component_ms_per_query.isna().all()


def test_instrumentation_restored_and_cached_encode_measured(monkeypatch):
    from repath_evaluation.algorithms.repath import features, runtime

    class Model:
        def encode(self, texts):
            return texts

    model = Model()
    monkeypatch.setattr(runtime, "_load_semantic_model", lambda *args: model)
    original = features.RepathFeaturePipeline.build_query
    original_booster_init = runtime.lgb.Booster.__init__
    timer = ComponentTimer()
    with timer.instrument():
        loaded = runtime._load_semantic_model("model", "cpu")
        timer.reset()
        assert loaded.encode(["a"]) == ["a"]
        assert timer.values["sbert_encode"] > 0
        assert timer.values["sbert_load"] == 0
    assert features.RepathFeaturePipeline.build_query is original
    assert runtime.lgb.Booster.__init__ is original_booster_init
    assert "encode" not in model.__dict__


@pytest.mark.parametrize("mode", ["cold", "amortized"])
def test_instrumented_repath_rankings_match_real_booster(monkeypatch, tmp_path, mode):
    from pathlib import Path

    from test_repath import FakeSemanticModel, dom_node, raw_node

    from repath_evaluation.algorithms.repath import runtime
    from repath_evaluation.algorithms.repath.matcher import RepathMatcher
    from repath_evaluation.domain.models import PagePair, PageSnapshot
    from repath_evaluation.evaluation import runner as benchmark

    old = [dom_node(0, raw_node(index=0, tag="button", text="Sign in", xpath="/button[1]", parent=None))]
    new = [
        dom_node(i, raw_node(index=i, tag="button", text=text, xpath=f"/button[{i + 1}]", parent=None))
        for i, text in enumerate(["Log in", "Sign in"])
    ]
    pair = PagePair(
        "synthetic", tmp_path,
        PageSnapshot("synthetic", "old", tmp_path, old, [], None, None),
        PageSnapshot("synthetic", "new", tmp_path, new, [], None, None),
        {0: 1}, [], [],
    )
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(runtime, "_new_semantic_model", lambda *args, **kwargs: FakeSemanticModel())
    monkeypatch.setattr(runner, "load_page_pair", lambda _: pair)
    monkeypatch.setattr(benchmark, "load_page_pair", lambda _: pair)

    def factory():
        return RepathMatcher(root / "models/lgb_ranker.txt", root / "models/config_base.pkl", cache_models=False)

    plain, _ = benchmark.run_site(tmp_path, factory, 1, runtime_mode=mode, algorithm_name="REPATH")
    timer = ComponentTimer()
    with timer.instrument():
        rows, summary = runner.run_profile_site(tmp_path, factory, 1, mode, timer)
    assert rows[0]["status"] == plain[0]["status"] == "ok"
    assert rows[0]["predicted_new_element_id"] == plain[0]["predicted_new_element_id"]
    for name in ("sbert_encode", "candidate_retrieval", "nonsemantic_features", "semantic_similarity", "feature_matrix", "lambdarank_inference"):
        assert summary[f"{name}_ms"] > 0
    assert summary["sbert_load_ms"] > 0 if mode == "cold" else summary["sbert_load_ms"] == 0

    for name in ("configuration_load", "lightgbm_load"):
        assert summary[f"{name}_ms"] > 0 if mode == "cold" else summary[f"{name}_ms"] == 0
    assert summary["page_pair_load_ms"] > 0
    assert summary["page_pair_preparation_ms"] > 0
    assert sum(summary[f"{key}_ms"] for key in COMPONENTS) == pytest.approx(summary["runtime_total_ms"])
    assert sum(summary[f"{key}_pct"] for key in COMPONENTS) == pytest.approx(100)


def test_preparation_excludes_nested_components(monkeypatch):
    ticks = iter([0, 10, 20, 40, 50, 60, 100, 120])
    monkeypatch.setattr("repath_evaluation.profiling.timing.time.perf_counter_ns", lambda: next(ticks))
    timer = ComponentTimer()
    encode = timer.wrap("sbert_encode", lambda: None)
    cache = timer.wrap("semantic_similarity", encode)
    retrieval = timer.wrap("candidate_retrieval", lambda: None)

    def prepare():
        cache()
        retrieval()

    timer.wrap("page_pair_preparation", prepare)()
    assert timer.values["sbert_encode"] == 20
    assert timer.values["semantic_similarity"] == 20
    assert timer.values["candidate_retrieval"] == 40
    assert timer.values["page_pair_preparation"] == 40
    assert timer.reconcile(130)["other_overhead"] == 10
