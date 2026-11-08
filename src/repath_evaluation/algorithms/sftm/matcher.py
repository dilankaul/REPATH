# SPDX-License-Identifier: GPL-3.0-or-later
#
# Original SFTM implementation:
# Copyright (C) 2021 Mantu, Sacha Brisset.
#
# Python port and REPATH integration modifications, 2026.
# Original source: https://github.com/amaris/sftm_tree_matching

"""SFTM tree-matching baseline for REPATH."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Tag

from ...domain.models import MatchResult, PagePair, PageSnapshot, ScoredCandidate
from ..base import ElementMatcher

UPSTREAM_REPOSITORY = "https://github.com/amaris/sftm_tree_matching"
UPSTREAM_COMMIT = "4f8683b0e5a21050d1679f1c350f558a5a4171e5"
UPSTREAM_LICENSE = "GPL-3.0-or-later"

_TOKEN_SPLIT_RE = re.compile(r"\W+", flags=re.ASCII)
_STANDARD_SEGMENT_RE = re.compile(r"^(?P<tag>[A-Za-z0-9_:-]+)(?:\[(?P<index>\d+)\])?$")
_SVG_SEGMENT_RE = re.compile(
    r"^\*\[name\(\)\s*=\s*['\"]svg['\"]\](?:\[(?P<index>\d+)\])?$",
    flags=re.IGNORECASE,
)


def _round_to_int_positive(value: float) -> int:
    """Match Kotlin ``roundToInt`` for the positive values used here."""

    return int(math.floor(value + 0.5))


@dataclass(frozen=True)
class SimilarityPropagationParameters:
    parent: float = 0.25
    parent_inv: float = 0.7
    envelop: tuple[float, ...] = (0.8, 0.1, 0.01)


@dataclass(frozen=True)
class MetropolisParameters:
    gamma: float = 1.0
    lambda_: float = 0.7
    iterations: int = 1


@dataclass(frozen=True)
class SftmParameters:
    """Defaults from the authors' ``TreeMatcher.Parameters`` implementation."""

    propagation: SimilarityPropagationParameters = field(
        default_factory=SimilarityPropagationParameters
    )
    metropolis: MetropolisParameters = field(default_factory=MetropolisParameters)
    no_match_cost: float = 1.2
    limit_neighbors: int = 100
    max_tokens_per_value: int = 8
    signature_attribute: str = "signature"
    preserve_upstream_quirks: bool = True

    def __post_init__(self) -> None:
        if self.no_match_cost <= 0:
            raise ValueError("no_match_cost must be positive")
        if self.limit_neighbors <= 0:
            raise ValueError("limit_neighbors must be positive")
        if self.max_tokens_per_value <= 0:
            raise ValueError("max_tokens_per_value must be positive")
        if not 0 <= self.metropolis.gamma <= 1:
            raise ValueError("metropolis gamma must be in [0, 1]")
        if self.metropolis.iterations <= 0:
            raise ValueError("metropolis iterations must be positive")


@dataclass(eq=False)
class _Node:
    order: int
    tokens: list[str]
    xpath: str
    signature: str = ""
    parent: _Node | None = None
    children: list[_Node] = field(default_factory=list)


@dataclass(eq=False, frozen=True)
class _Edge:
    source: _Node | None
    target: _Node | None
    score: float
    order: int


class _InMemoryIndex:
    """Port of the authors' ``InMemoryIndex`` including its truncation order."""

    def __init__(self, max_neighbors: int, max_token_appearance: int) -> None:
        self.max_neighbors = max_neighbors
        self.max_token_appearance = max_token_appearance
        self.index: dict[str, set[_Node]] = {}
        self._removed_tokens: set[str] = set()
        self._nodes: set[_Node] = set()
        self._idf_precomputation = 0.0

    @classmethod
    def build(
        cls,
        nodes: list[_Node],
        *,
        max_neighbors: int,
        max_token_appearance: int,
    ) -> _InMemoryIndex:
        result = cls(max_neighbors, max_token_appearance)
        for node in nodes:
            for token in node.tokens:
                result._add(token, node)
        result._idf_precomputation = math.log(len(result._nodes)) if result._nodes else 0.0
        return result

    def _add(self, token: str, node: _Node) -> None:
        self._nodes.add(node)
        current = self.index.get(token)
        appearance = (len(current) if current is not None else 0) + 1
        if token in self._removed_tokens:
            return
        if appearance > self.max_token_appearance:
            self._removed_tokens.add(token)
            self.index.pop(token, None)
            return
        if current is None:
            self.index[token] = {node}
        else:
            current.add(node)

    def find_neighbors(self, targets: list[_Node]) -> dict[_Node, dict[_Node, float]]:
        neighbors: dict[_Node, dict[_Node, float]] = {}
        for target in targets:
            hits: dict[_Node, float] = {}
            for token in target.tokens:
                sources = self.index.get(token)
                if not sources:
                    continue
                idf = self._idf_precomputation - math.log(len(sources))
                for source in sorted(sources, key=lambda node: node.order):
                    hits[source] = hits.get(source, 0.0) + idf
            if hits:
                # The authors' code sorts ascending and takes the first N.  It is
                # counter-intuitive, but retained as an implementation quirk.
                ordered = sorted(hits.items(), key=lambda item: (item[1], item[0].order))
                neighbors[target] = dict(ordered[: self.max_neighbors])
        return neighbors


def _attribute_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


def _tokenize_value(value: str, max_tokens: int) -> list[str]:
    tokens = [token for token in _TOKEN_SPLIT_RE.sub(" ", value).split(" ") if token]
    if not tokens:
        return []
    if len(tokens) == 1:
        return tokens
    if len(tokens) <= max_tokens:
        return [*tokens, value]
    return [value]


def _tokenize_attributes(
    attributes: Iterable[tuple[object, object]],
    tag_name: str,
    parameters: SftmParameters,
) -> list[str]:
    result: list[str] = []
    for raw_key, raw_value in attributes:
        key = str(raw_key)
        if key == parameters.signature_attribute or key.startswith("data"):
            continue
        value = _attribute_value(raw_value)
        if not value.strip():
            result.append(key)
            continue
        result.extend(
            f"{key}:{token}"
            for token in _tokenize_value(value, parameters.max_tokens_per_value)
        )
        result.append(key)
    result.append(tag_name)
    return result


def _tokenize_tag(tag: Tag, parameters: SftmParameters) -> list[str]:
    return _tokenize_attributes(tag.attrs.items(), tag.name or "", parameters)


def _parse_html(html: str, parameters: SftmParameters) -> list[_Node]:
    """Parse a page using the same tree/token conventions as official SFTM.

    BeautifulSoup is used instead of Jsoup, so malformed-HTML repair may differ.
    XPath, tokenization, body-root selection, and element-only traversal follow
    the authors' implementation.
    """

    soup = BeautifulSoup(html, "html.parser")
    body = soup.body
    if body is None:
        body = soup.new_tag("body")
        for child in list(soup.contents):
            body.append(child.extract())

    nodes: list[_Node] = []

    def copy(tag: Tag, parent_node: _Node | None, partial_xpath: str) -> _Node:
        if parent_node is None:
            xpath = f"/{tag.name}"
        else:
            siblings = [
                child
                for child in tag.parent.children
                if isinstance(child, Tag) and child.name == tag.name
            ]
            index = next((i for i, child in enumerate(siblings) if child is tag), 0)
            xpath = f"{partial_xpath}/{tag.name}[{index}]"
        tokens = [*_tokenize_tag(tag, parameters), xpath]
        node = _Node(
            order=len(nodes),
            tokens=tokens,
            xpath=xpath,
            signature=_attribute_value(tag.attrs.get(parameters.signature_attribute, "")),
            parent=parent_node,
        )
        nodes.append(node)
        for child in tag.children:
            if isinstance(child, Tag):
                child_node = copy(child, node, xpath)
                node.children.append(child_node)
        return node

    copy(body, None, "")
    return nodes




def _author_xpath_from_dataset_key(key: tuple[tuple[str, int], ...]) -> str:
    parts: list[str] = []
    for position, (tag, one_based_index) in enumerate(key):
        if position == 0 and tag == "body":
            parts.append("body")
        else:
            parts.append(f"{tag}[{max(one_based_index - 1, 0)}]")
    return "/" + "/".join(parts)


def _parse_snapshot(page: PageSnapshot, parameters: SftmParameters) -> list[_Node]:
    """Build the SFTM tree from captured DOM metadata when HTML is unavailable.

    Prefer ``preDomNodeInfo`` because it normally contains a broader DOM view;
    fall back to ``domNodeInfo``.  Attribute tokenization and zero-based SFTM
    XPath generation remain the same as the authors' parser.  Malformed-HTML
    repair cannot be reproduced without the original HTML, so the selected
    input mode is exposed on ``SftmMatcher.input_mode``.
    """

    records = page.pre_nodes if page.pre_nodes else page.nodes
    keyed_records: list[tuple[tuple[tuple[str, int], ...], object]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for record in sorted(records, key=lambda item: item.list_index):
        key = _dataset_xpath_key(record.xpath)
        if not key or key[0][0] != "body" or key in seen:
            continue
        seen.add(key)
        keyed_records.append((key, record))

    nodes: list[_Node] = []
    by_key: dict[tuple[tuple[str, int], ...], _Node] = {}
    for key, record in keyed_records:
        xpath = _author_xpath_from_dataset_key(key)
        attributes = record.attributes.items()
        tokens = [*_tokenize_attributes(attributes, record.tag, parameters), xpath]
        node = _Node(
            order=len(nodes),
            tokens=tokens,
            xpath=xpath,
            signature=record.attr(parameters.signature_attribute),
        )
        nodes.append(node)
        by_key[key] = node

    for key, _record in keyed_records:
        node = by_key[key]
        parent = by_key.get(key[:-1]) if len(key) > 1 else None
        node.parent = parent
        if parent is not None:
            parent.children.append(node)

    return nodes


def _add_parent_tokens(nodes: list[_Node], index: dict[str, set[_Node]]) -> None:
    rarest: dict[_Node, str | None] = {}
    for node in nodes:
        candidates = [token for token in node.tokens if not token.startswith("/BODY")]
        rarest[node] = min(
            candidates,
            key=lambda token: len(index.get(token, ())) if token in index else len(nodes),
            default=None,
        )
    for node in nodes:
        if node.parent is None:
            continue
        parent_token = rarest.get(node.parent)
        if node.parent in rarest:
            # Kotlin string interpolation renders a null String? as "null".
            suffix = "null" if parent_token is None else parent_token
            node.tokens.append(f"#C_{suffix}")


def _increase(
    neighbors: dict[_Node, dict[_Node, float]],
    source: _Node,
    target: _Node,
    increment: float,
) -> None:
    if abs(increment) < 0.001:
        return
    hits = neighbors.setdefault(target, {})
    new_score = hits.get(source, 0.0) + increment
    if abs(new_score) < 0.001:
        return
    hits[source] = new_score


def _propagate(
    neighbors: dict[_Node, dict[_Node, float]],
    parameters: SimilarityPropagationParameters,
) -> dict[_Node, dict[_Node, float]]:
    current = neighbors
    for envelop in parameters.envelop:
        updated: dict[_Node, dict[_Node, float]] = {}
        for target in sorted(current, key=lambda node: node.order):
            for source, score in sorted(current[target].items(), key=lambda item: item[0].order):
                _increase(updated, source, target, score)
                if source.parent is None or target.parent is None:
                    continue
                parent_score = current.get(target.parent, {}).get(source.parent, 0.0)
                _increase(
                    updated,
                    source,
                    target,
                    envelop * parent_score * parameters.parent,
                )
                _increase(
                    updated,
                    source.parent,
                    target.parent,
                    envelop * score * parameters.parent_inv,
                )
        current = updated
    return current


def _greedy_matching(edges: list[_Edge]) -> tuple[list[_Edge], float]:
    # Official defaults are gamma=1 and one iteration, reducing Metropolis to
    # a score-descending greedy maximal matching.
    ordered = sorted(edges, key=lambda edge: (-edge.score, edge.order))
    selected: list[_Edge] = []
    used_sources: set[_Node] = set()
    used_targets: set[_Node] = set()
    for edge in ordered:
        if edge.source is not None and edge.source in used_sources:
            continue
        if edge.target is not None and edge.target in used_targets:
            continue
        selected.append(edge)
        if edge.source is not None:
            used_sources.add(edge.source)
        if edge.target is not None:
            used_targets.add(edge.target)
    objective = sum(edge.score for edge in selected) / len(selected) if selected else 0.0
    return selected, objective


def _dataset_xpath_key(xpath: str) -> tuple[tuple[str, int], ...]:
    segments: list[tuple[str, int]] = []
    for raw_segment in xpath.strip().strip("/").split("/"):
        segment = raw_segment.strip()
        if not segment:
            continue
        svg = _SVG_SEGMENT_RE.match(segment)
        if svg:
            segments.append(("svg", int(svg.group("index") or 1)))
            continue
        standard = _STANDARD_SEGMENT_RE.match(segment)
        if not standard:
            continue
        tag = standard.group("tag").casefold()
        if tag == "html":
            continue
        segments.append((tag, int(standard.group("index") or 1)))
    return tuple(segments)


def _author_xpath_key(xpath: str) -> tuple[tuple[str, int], ...]:
    segments: list[tuple[str, int]] = []
    for raw_segment in xpath.strip().strip("/").split("/"):
        standard = _STANDARD_SEGMENT_RE.match(raw_segment.strip())
        if not standard:
            continue
        tag = standard.group("tag").casefold()
        if tag == "html":
            continue
        raw_index = standard.group("index")
        if tag == "body" and raw_index is None:
            index = 1
        else:
            index = int(raw_index or 0) + 1
        segments.append((tag, index))
    return tuple(segments)


def _page_xpath_map(page: PageSnapshot) -> dict[tuple[tuple[str, int], ...], int]:
    result: dict[tuple[tuple[str, int], ...], int] = {}
    for node in sorted(page.nodes, key=lambda item: item.list_index):
        key = _dataset_xpath_key(node.xpath)
        if key and key not in result:
            result[key] = node.node_id
    return result


class SftmMatcher(ElementMatcher):
    """Python port of SFTM pinned to the authors' implementation revision."""

    def __init__(self, parameters: SftmParameters | None = None) -> None:
        self.parameters = parameters or SftmParameters()
        self.mapping: dict[int, int] = {}
        self.matching_objective = 0.0
        self.unresolved_old_xpaths = 0
        self.unresolved_new_xpaths = 0
        self.input_mode = "unprepared"

    @property
    def name(self) -> str:
        return "SFTM"

    def prepare(self, page_pair: PagePair) -> None:
        super().prepare(page_pair)
        self.mapping = {}
        self.matching_objective = 0.0
        self.unresolved_old_xpaths = 0
        self.unresolved_new_xpaths = 0
        self.input_mode = "unprepared"

        old_path = page_pair.old_page.html_path
        new_path = page_pair.new_page.html_path
        if old_path is not None and new_path is not None:
            source_nodes = _parse_html(_read_html(old_path), self.parameters)
            target_nodes = _parse_html(_read_html(new_path), self.parameters)
            self.input_mode = "html"
        else:
            source_nodes = _parse_snapshot(page_pair.old_page, self.parameters)
            target_nodes = _parse_snapshot(page_pair.new_page, self.parameters)
            self.input_mode = "dom_metadata"
        if not source_nodes or not target_nodes:
            return

        max_appearance = _round_to_int_positive(math.sqrt(len(source_nodes)))
        source_index = _InMemoryIndex.build(
            source_nodes,
            max_neighbors=self.parameters.limit_neighbors,
            max_token_appearance=max_appearance,
        )
        # The pinned upstream TreeMatcher builds this index from sourceNodes,
        # despite naming it targetIndex.  Preserve that behaviour by default.
        parent_index_nodes = (
            source_nodes if self.parameters.preserve_upstream_quirks else target_nodes
        )
        target_index = _InMemoryIndex.build(
            parent_index_nodes,
            max_neighbors=self.parameters.limit_neighbors,
            max_token_appearance=max_appearance,
        )
        _add_parent_tokens(source_nodes, source_index.index)
        _add_parent_tokens(target_nodes, target_index.index)
        source_index = _InMemoryIndex.build(
            source_nodes,
            max_neighbors=self.parameters.limit_neighbors,
            max_token_appearance=max_appearance,
        )
        neighbors = source_index.find_neighbors(target_nodes)
        neighbors = _propagate(neighbors, self.parameters.propagation)

        edges: list[_Edge] = []
        order = 0
        for target in sorted(neighbors, key=lambda node: node.order):
            for source, score in sorted(neighbors[target].items(), key=lambda item: item[0].order):
                edges.append(_Edge(source, target, score, order))
                order += 1
        no_match_score = 1.0 / self.parameters.no_match_cost
        for source in source_nodes:
            edges.append(_Edge(source, None, no_match_score, order))
            order += 1
        for target in target_nodes:
            edges.append(_Edge(None, target, no_match_score, order))
            order += 1

        matching, self.matching_objective = _greedy_matching(edges)
        old_xpath_map = _page_xpath_map(page_pair.old_page)
        new_xpath_map = _page_xpath_map(page_pair.new_page)
        for edge in matching:
            if edge.source is None or edge.target is None:
                continue
            old_id = old_xpath_map.get(_author_xpath_key(edge.source.xpath))
            new_id = new_xpath_map.get(_author_xpath_key(edge.target.xpath))
            if old_id is None:
                self.unresolved_old_xpaths += 1
                continue
            if new_id is None:
                self.unresolved_new_xpaths += 1
                continue
            self.mapping[old_id] = new_id

    def match(self, old_node_id: int) -> MatchResult:
        started = time.perf_counter_ns()
        source = self.pair.old_page.resolve_node(old_node_id)
        if source is None:
            return MatchResult((), time.perf_counter_ns() - started, "missing_old_node")
        predicted = self.mapping.get(source.node_id)
        elapsed = time.perf_counter_ns() - started
        if predicted is None:
            return MatchResult((), elapsed, "no_prediction")
        return MatchResult((ScoredCandidate(predicted, 1.0),), elapsed)

    def close(self) -> None:
        self.mapping = {}


def _read_html(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


__all__ = [
    "MetropolisParameters",
    "SftmMatcher",
    "SftmParameters",
    "SimilarityPropagationParameters",
    "UPSTREAM_COMMIT",
    "UPSTREAM_LICENSE",
    "UPSTREAM_REPOSITORY",
]
