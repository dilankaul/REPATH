from __future__ import annotations

import gc
import time

from ..data.loader import load_page_pair
from ..evaluation.protocol import build_queries
from ..evaluation.runner import _row, _runtime_summary
from .timing import COMPONENTS, MEASURED


def _times(total, values):
    return {"runtime_ms": total / 1e6, **{f"{key}_ms": value / 1e6 for key, value in values.items()}}


def run_profile_site(site_path, factory, repeat, mode, timer):
    reference = load_page_pair(site_path)
    queries = [query for query in build_queries(reference) if query.eligible]
    rows = []
    totals = dict.fromkeys(COMPONENTS, 0)
    total_runtime = 0

    def query_row(query, result, error, elapsed, values):
        row = _row(
            algorithm="REPATH", site=reference.site, repeat=repeat,
            old_element_id=query.old_element_id,
            expected_new_element_id=query.expected_new_element_id,
            result=result, error=error, runtime_mode=mode,
        )
        row.update(scope="query", qid=f"{reference.site}::{query.old_element_id}")
        row.update(_times(elapsed, values))
        return row

    if mode == "cold":
        for query in queries:
            timer.reset()
            matcher = result = error = None
            start = time.perf_counter_ns()
            try:
                pair = load_page_pair(site_path)
                matcher = factory()
                matcher.prepare(pair)
                result = matcher.match(query.old_element_id)
            except Exception as exc:  # noqa: BLE001
                error = exc
            finally:
                if matcher is not None:
                    matcher.close()
                elapsed = time.perf_counter_ns() - start
            values = timer.reconcile(elapsed)
            rows.append(query_row(query, result, error, elapsed, values))
            total_runtime += elapsed
            for key in totals:
                totals[key] += values[key]
            del matcher, result
            gc.collect()
    elif mode == "amortized":
        matcher = factory()
        timer.reset()
        measurements = []
        start = time.perf_counter_ns()
        try:
            pair = load_page_pair(site_path)
            matcher.prepare(pair)
            for query in queries:
                before = dict(timer.values)
                result = error = None
                query_start = time.perf_counter_ns()
                try:
                    result = matcher.match(query.old_element_id)
                except Exception as exc:  # noqa: BLE001
                    error = exc
                elapsed = time.perf_counter_ns() - query_start
                delta = {key: timer.values[key] - before[key] for key in MEASURED}
                measurements.append((query, result, error, elapsed, delta))
        finally:
            matcher.close()
            total_runtime = time.perf_counter_ns() - start
        totals = timer.reconcile(total_runtime)
        query_totals = dict.fromkeys(COMPONENTS, 0)
        query_runtime = 0
        for query, result, error, elapsed, delta in measurements:
            values = timer.reconcile(elapsed, delta)
            query_runtime += elapsed
            for key in query_totals:
                query_totals[key] += values[key]
            rows.append(query_row(query, result, error, elapsed, values))
        shared = {key: totals[key] - query_totals[key] for key in totals}
        shared_runtime = total_runtime - query_runtime
        if shared_runtime < 0 or any(value < 0 for value in shared.values()):
            raise ValueError("Shared page-pair timings overlap query intervals")
        if sum(shared.values()) != shared_runtime:
            raise ValueError("Shared page-pair components do not reconcile")
        # Shared rows are additive, never copies of amortized query means.
        rows.append({
            "algorithm": "REPATH", "site": reference.site, "repeat": repeat,
            "runtime_mode": mode, "scope": "page_pair_shared",
            **_times(shared_runtime, shared),
        })
    else:
        raise ValueError(f"Unknown profiling mode: {mode}")

    query_rows = [row for row in rows if row["scope"] == "query"]
    summary = _runtime_summary("REPATH", reference.site, repeat, query_rows, mode, total_runtime)
    for key, value in totals.items():
        summary[f"{key}_ms"] = value / 1e6
        summary[f"{key}_pct"] = 100 * value / total_runtime if total_runtime else 0.0
    return rows, summary
