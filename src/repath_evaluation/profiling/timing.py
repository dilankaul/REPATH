from __future__ import annotations

import time
from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch

from ..algorithms.repath import features, runtime

COMPONENTS = {
    "sbert_load": "SBERT initialization within the timed interval",
    "sbert_encode": "SentenceTransformer.encode execution",
    "candidate_retrieval": "Query candidate selection, merging, scoring and shortlisting",
    "nonsemantic_features": "build_pair_rows lexical, structural, attribute and geometry features",
    "semantic_similarity": "Semantic preparation, embedding cache and similarity, excluding encoding",
    "feature_matrix": "Ordered 50-feature matrix construction and validation",
    "lambdarank_inference": "LightGBM prediction and final ranking",
    "other_overhead": "Residual page loading, initialization, preparation, cleanup and instrumentation",
}
MEASURED = tuple(name for name in COMPONENTS if name != "other_overhead")


class ComponentTimer:
    """Exclusive nested intervals for a single-threaded CPU profiling experiment."""

    def __init__(self):
        self.stack = []
        self.reset()

    def reset(self):
        if self.stack:
            raise ValueError("Cannot reset an active timing interval")
        self.values = dict.fromkeys(MEASURED, 0)

    def wrap(self, name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            start = time.perf_counter_ns()
            frame = [start, 0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - start
                if self.stack.pop() is not frame or elapsed < frame[1]:
                    raise ValueError("Inconsistent or overlapping profiling intervals")
                self.values[name] += elapsed - frame[1]
                if self.stack:
                    self.stack[-1][1] += elapsed
        return measured

    def reconcile(self, total_ns, values=None):
        if self.stack:
            raise ValueError("Unclosed profiling intervals")
        values = dict(self.values if values is None else values)
        residual = total_ns - sum(values.values())
        if total_ns < 0 or residual < 0 or any(value < 0 for value in values.values()):
            raise ValueError(f"Component times exceed authoritative runtime: residual={residual} ns")
        return {**values, "other_overhead": residual}

    @contextmanager
    def instrument(self):
        # Patches exist only during this separate, single-threaded experiment.
        with ExitStack() as patches:
            timer = self

            class SemanticModel:
                def __init__(self, model):
                    self.model = model
                    self.encode = timer.wrap("sbert_encode", model.encode)

                def __getattr__(self, name):
                    return getattr(self.model, name)

            def model_wrapper(loader):
                @wraps(loader)
                def load(*args, **kwargs):
                    model = loader(*args, **kwargs)
                    return model if isinstance(model, SemanticModel) else SemanticModel(model)
                return load

            patches.enter_context(patch.object(
                runtime, "_new_semantic_model",
                self.wrap("sbert_load", model_wrapper(runtime._new_semantic_model)),
            ))
            patches.enter_context(patch.object(
                runtime, "_load_semantic_model", model_wrapper(runtime._load_semantic_model),
            ))
            targets = [
                (features, "build_pair_rows", "nonsemantic_features"),
                (features, "extract_base_candidate_indices", "candidate_retrieval"),
                (runtime.RepathRuntime, "_validated_matrix", "feature_matrix"),
                (features.RepathFeaturePipeline, "build_query", "candidate_retrieval"),
                (features.RepathFeaturePipeline, "_cache_texts", "semantic_similarity"),
                (features.RepathFeaturePipeline, "_add_semantic_features", "semantic_similarity"),
                (features.RepathFeaturePipeline, "_feature_matrix", "feature_matrix"),
                (runtime.RepathRuntime, "rank", "lambdarank_inference"),
            ]
            for owner, name, component in targets:
                patches.enter_context(patch.object(owner, name, self.wrap(component, getattr(owner, name))))
            yield
