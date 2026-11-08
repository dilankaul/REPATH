"""Python reproduction of the WATER locator-repair method.

Original project: https://github.com/shauvik/water
Implemented for the REPATH evaluation framework.
"""

from __future__ import annotations

import time
from typing import Any

from ...domain.models import MatchResult, PreDomNode, ScoredCandidate
from ...utilities.similarity import levenshtein_distance
from ..base import ElementMatcher

UPSTREAM_REPOSITORY = "https://github.com/shauvik/water"
UPSTREAM_COMMIT = "c57311e56afb8e52ccd315e460e1a6f5084a7b95"


class WaterMatcher(ElementMatcher):
    """WATER matching baseline adapted to deterministic query-level ranking."""

    @property
    def name(self) -> str:
        return "WATER"

    @staticmethod
    def _raw_first(node: PreDomNode, *names: str, default: Any = "") -> Any:
        for name in names:
            if name in node.raw and node.raw[name] is not None:
                return node.raw[name]
        return default

    @classmethod
    def _raw_bool(cls, node: PreDomNode, *names: str, default: bool = False) -> bool:
        value = cls._raw_first(node, *names, default=default)
        if isinstance(value, str):
            return value.casefold() not in {"", "0", "false", "no", "none"}
        return bool(value)

    @classmethod
    def _raw_int(cls, node: PreDomNode, *names: str, default: int = 0) -> int:
        value = cls._raw_first(node, *names, default=default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _link_text(cls, node: PreDomNode) -> str:
        return str(cls._raw_first(node, "linkText", "link_text", default=""))

    @staticmethod
    def _classes(node: PreDomNode) -> str:
        return node.attr("class")

    @staticmethod
    def _nonempty_equal(left: str, right: str) -> bool:
        return bool(left) and bool(right) and left == right

    @classmethod
    def _similarity(cls, source: PreDomNode, candidate: PreDomNode) -> float:
        if source.tag != candidate.tag:
            return 0.0

        maximum = max(len(source.xpath), len(candidate.xpath))
        if maximum == 0:
            return float("nan")

        rho1 = 1.0 - levenshtein_distance(source.xpath, candidate.xpath) / maximum

        a = source.bbox
        b = candidate.bbox
        rho2 = 0.0
        if (
            abs(a.x - b.x) <= 5
            and abs((a.x + a.width) - (b.x + b.width)) <= 5
            and abs(a.y - b.y) <= 5
            and abs((a.y + a.height) - (b.y + b.height)) <= 5
        ):
            rho2 += 1.0

        if cls._raw_bool(source, "clickable", "isClickable") == cls._raw_bool(
            candidate, "clickable", "isClickable"
        ):
            rho2 += 1.0
        if source.displayed == candidate.displayed:
            rho2 += 1.0
        if cls._raw_int(source, "zIndex", "z_index") == cls._raw_int(
            candidate, "zIndex", "z_index"
        ):
            rho2 += 1.0

        rho2 /= 5.0
        return rho1 * 0.9 + rho2 * 0.1

    @classmethod
    def _has_exact_property_match(cls, source: PreDomNode, candidate: PreDomNode) -> bool:
        return any(
            (
                cls._nonempty_equal(source.explicit_id, candidate.explicit_id),
                cls._nonempty_equal(source.xpath, candidate.xpath),
                cls._nonempty_equal(cls._classes(source), cls._classes(candidate)),
                cls._nonempty_equal(cls._link_text(source), cls._link_text(candidate)),
                cls._nonempty_equal(source.name, candidate.name),
            )
        )

    def _rank_pre_matches(self, source: PreDomNode) -> list[tuple[PreDomNode, float]]:
        exact = [
            (candidate, 1.0)
            for candidate in self.pair.new_page.pre_nodes
            if self._has_exact_property_match(source, candidate)
        ]
        if exact:
            return exact

        fallback: list[tuple[PreDomNode, float]] = []
        for candidate in self.pair.new_page.pre_nodes:
            score = self._similarity(source, candidate)
            if score > 0.5:
                fallback.append((candidate, score))
        fallback.sort(key=lambda item: item[1], reverse=True)
        return fallback

    def match(self, old_node_id: int) -> MatchResult:
        start = time.perf_counter_ns()
        old_group = self.pair.old_page.resolve_node(old_node_id)
        if old_group is None:
            return MatchResult((), time.perf_counter_ns() - start, "missing_old_node")
        if not self.pair.old_page.pre_nodes or not self.pair.new_page.pre_nodes:
            return MatchResult((), time.perf_counter_ns() - start, "missing_pre_dom")

        source_pre = self.pair.old_page.resolve_pre(old_group.element_id)
        if source_pre is None:
            return MatchResult((), time.perf_counter_ns() - start, "missing_old_pre_node")

        ranking: list[ScoredCandidate] = []
        seen_grouped_ids: set[int] = set()
        for matched_pre, score in self._rank_pre_matches(source_pre):
            if matched_pre.core_element_id < 0:
                continue
            core_pre = self.pair.new_page.resolve_pre(matched_pre.core_element_id)
            if core_pre is None:
                continue
            grouped = self.pair.new_page.grouped_node_by_xpath(core_pre.xpath)
            if grouped is None or grouped.node_id in seen_grouped_ids:
                continue
            seen_grouped_ids.add(grouped.node_id)
            ranking.append(ScoredCandidate(grouped.node_id, score))

        elapsed = time.perf_counter_ns() - start
        if not ranking:
            return MatchResult((), elapsed, "no_prediction")
        return MatchResult(tuple(ranking), elapsed)


__all__ = ["UPSTREAM_COMMIT", "UPSTREAM_REPOSITORY", "WaterMatcher"]
