from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from ...domain.models import PagePair
from .features import ALL_MODEL_FEATURES, RepathFeaturePipeline


@lru_cache(maxsize=4)
def _load_booster(model_path: str) -> lgb.Booster:
    return lgb.Booster(model_file=model_path)


def _new_semantic_model(
    model_name: str,
    device: str,
    *,
    local_files_only: bool,
) -> Any:
    from huggingface_hub import logging as hf_logging
    from huggingface_hub.utils import disable_progress_bars
    from transformers.utils import logging as transformers_logging

    disable_progress_bars()
    hf_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        device=device,
        local_files_only=local_files_only,
    )


@lru_cache(maxsize=4)
def _load_semantic_model(model_name: str, device: str) -> Any:
    return _new_semantic_model(
        model_name,
        device,
        local_files_only=False,
    )


def clear_model_caches() -> None:
    """Release process-wide REPATH model caches between timing phases."""
    _load_booster.cache_clear()
    _load_semantic_model.cache_clear()


class RepathRuntime:
    """Direct in-process REPATH ranking runtime."""

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        device: str = "cpu",
        cache_models: bool = True,
        local_files_only: bool = False,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"REPATH model not found: {self.model_path}"
            )
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"REPATH base config not found: {self.config_path}"
            )

        loaded_config = joblib.load(self.config_path)
        if not isinstance(loaded_config, dict):
            raise TypeError(
                "REPATH config_base.pkl must contain a dictionary"
            )
        self.config: dict[str, Any] = loaded_config

        configured_features = self.config.get(
            "ALL_FEATURES",
            self.config.get("FEATURES", []),
        )
        self.feature_names = [str(name) for name in configured_features]

        if self.feature_names != ALL_MODEL_FEATURES:
            raise TypeError(
                "config_base.pkl does not contain the exact ordered "
                "50-feature schema used by REPATH Notebook 01"
            )

        semantic_model_name = str(
            self.config.get("sbert_model", "all-MiniLM-L6-v2")
        )

        if local_files_only and cache_models:
            raise ValueError(
                "local_files_only timing mode requires cache_models=False"
            )

        self.model = (
            _load_booster(str(self.model_path))
            if cache_models
            else lgb.Booster(model_file=str(self.model_path))
        )
        if self.model.num_feature() != len(self.feature_names):
            raise TypeError(
                "LightGBM/config feature mismatch: "
                f"model expects {self.model.num_feature()}, "
                f"config lists {len(self.feature_names)}"
            )

        self.semantic_model = (
            _load_semantic_model(semantic_model_name, device)
            if cache_models
            else _new_semantic_model(
                semantic_model_name,
                device,
                local_files_only=local_files_only,
            )
        )
        self.feature_pipeline = RepathFeaturePipeline(
            semantic_model=self.semantic_model,
            feature_names=self.feature_names,
            config=self.config,
        )
        self.page_pair: PagePair | None = None

    def prepare(self, page_pair: PagePair) -> None:
        self.page_pair = page_pair
        self.feature_pipeline.prepare(page_pair)

    def rank(self, old_node_id: int) -> list[tuple[int, float]]:
        if self.page_pair is None:
            raise RuntimeError("REPATH runtime has not been prepared")

        candidate_ids, feature_matrix = (
            self.feature_pipeline.build_query(old_node_id)
        )
        if not candidate_ids:
            return []

        matrix = self._validated_matrix(feature_matrix, candidate_ids)

        scores = np.asarray(
            self.model.predict(matrix),
            dtype=np.float64,
        )
        if scores.shape != (len(candidate_ids),):
            raise TypeError(
                "LightGBM returned an unexpected score shape: "
                f"{scores.shape}"
            )

        # Notebook 02 sorts with stable mergesort. Therefore equal scores
        # retain the original candidate-pool order.
        order = np.argsort(-scores, kind="mergesort")
        return [
            (int(candidate_ids[index]), float(scores[index]))
            for index in order
        ]

    def _validated_matrix(self, feature_matrix, candidate_ids) -> np.ndarray:
        matrix = np.asarray(feature_matrix, dtype=np.float32)
        expected_shape = (
            len(candidate_ids),
            len(self.feature_names),
        )
        if matrix.shape != expected_shape:
            raise TypeError(
                "REPATH feature shape mismatch: "
                f"expected {expected_shape}, got {matrix.shape}"
            )

        return matrix
