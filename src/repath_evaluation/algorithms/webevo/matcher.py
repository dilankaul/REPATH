"""Python reproduction of the WebEvo matching method.

Original project: https://github.com/webevoexp/webevo
Implemented for the REPATH evaluation framework.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ...domain.models import DomNode, MatchResult, PagePair, PageSnapshot, PreDomNode, ScoredCandidate
from ..base import ElementMatcher

UPSTREAM_REPOSITORY = "https://github.com/webevoexp/webevo"
UPSTREAM_COMMIT = "d195630b8dd5d85790d5aebb5c81c29bd03f01ca"
UPSTREAM_MODULE = "graphic-image-analysis"

WEBEVO_TAG_ORDER = (
    "div",
    "li",
    "a",
    "img",
    "iframe",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "span",
    "p",
)


@dataclass(frozen=True)
class WebEvoParameters:
    image_similarity_threshold: float = 0.60
    hash_size: int = 16
    text_candidate_limit: int = 5

    def __post_init__(self) -> None:
        if not 0 <= self.image_similarity_threshold <= 1:
            raise ValueError("image_similarity_threshold must be in [0, 1]")
        if self.hash_size <= 0:
            raise ValueError("hash_size must be positive")
        if self.text_candidate_limit <= 0:
            raise ValueError("text_candidate_limit must be positive")


@dataclass(frozen=True)
class _Candidate:
    order: int
    node_id: int
    tag: str
    text: str
    image_path: Path | None


def _levenshtein(left: str, right: str) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _string_similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return math.nan
    return 1.0 - _levenshtein(left, right) / denominator


def _java_gray(image: Image.Image) -> Image.Image:
    """Reproduce ``converToGrey``: white alpha background and truncated weights."""

    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(white, rgba).convert("RGB")
    array = np.asarray(composited, dtype=np.uint16)
    gray = (
        (array[..., 0] * 299 // 1000)
        + (array[..., 1] * 587 // 1000)
        + (array[..., 2] * 114 // 1000)
    ).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _inverse(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return Image.fromarray(255 - array, mode="RGB")


def _fingerprint(image: Image.Image, size: int) -> np.ndarray:
    # Java Image.SCALE_SMOOTH is implementation-dependent; Lanczos is the
    # closest deterministic Pillow high-quality downsampling operation.
    resized = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    # The authors' FingerPrint then applies ColorConvertOp(CS_GRAY).
    gray = np.asarray(resized.convert("L"), dtype=np.uint8).reshape(-1)
    # Java Math.round(float) for a positive mean is floor(x + 0.5).
    mean = int(math.floor(float(gray.sum()) / len(gray) + 0.5))
    return gray >= mean


def _fingerprint_similarity(left: Image.Image, right: Image.Image, size: int) -> float:
    left_hash = _fingerprint(left, size)
    right_hash = _fingerprint(right, size)
    return float(np.count_nonzero(left_hash == right_hash) / left_hash.size)


def _image_stages(
    target: Image.Image,
    candidate: Image.Image,
    parameters: WebEvoParameters,
) -> tuple[bool, float, str]:
    original = _fingerprint_similarity(target, candidate, parameters.hash_size)
    if original >= parameters.image_similarity_threshold:
        return True, original, "original"

    target_gray = _java_gray(target)
    candidate_gray = _java_gray(candidate)
    gray = _fingerprint_similarity(target_gray, candidate_gray, parameters.hash_size)
    if gray >= parameters.image_similarity_threshold:
        return True, gray, "grayscale"

    inverted_candidate = _inverse(candidate_gray)
    inverted = _fingerprint_similarity(
        target_gray,
        inverted_candidate,
        parameters.hash_size,
    )
    if inverted >= parameters.image_similarity_threshold:
        return True, inverted, "candidate_inverse"
    return False, max(original, gray, inverted), "none"


def _dataset_xpath_key(xpath: str) -> str:
    return xpath.strip().casefold().replace('"', "'")


def _grouped_node_for_pre(page: PageSnapshot, pre: PreDomNode) -> DomNode | None:
    core = page.pre_by_index.get(pre.core_element_id) or page.pre_by_element_id.get(
        pre.core_element_id
    )
    candidate_xpaths = [
        core.xpath if core is not None else "",
        pre.xpath,
    ]
    keyed = {
        _dataset_xpath_key(node.xpath): node
        for node in sorted(page.nodes, key=lambda item: item.list_index)
    }
    for xpath in candidate_xpaths:
        node = keyed.get(_dataset_xpath_key(xpath))
        if node is not None:
            return node
    return page.resolve_node(pre.element_id)


def _candidate_image_paths(page: PageSnapshot, node: DomNode) -> tuple[Path, ...]:
    paths: list[Path] = []
    for identifier in (node.node_id, node.list_index, node.element_id):
        for suffix in (".png", ".jpg", ".jpeg"):
            path = page.root / f"{identifier}{suffix}"
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def _resolve_image_path(page: PageSnapshot, node: DomNode) -> Path | None:
    return next((path for path in _candidate_image_paths(page, node) if path.is_file()), None)


def _load_image(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    try:
        with Image.open(path) as image:
            return image.convert("RGBA").copy()
    except (OSError, ValueError):
        return None


class WebEvoMatcher(ElementMatcher):
    """Port of the authors' semantics-based visual-search module.

    The fixed benchmark already supplies changed-element queries, so WebEvo's
    separate DOM-change and history filters are outside this matcher. Candidate
    tag order, text split, top-five shortlist, image stages, and 0.60 threshold
    follow the pinned authors' implementation.
    """

    def __init__(self, parameters: WebEvoParameters | None = None) -> None:
        self.parameters = parameters or WebEvoParameters()
        self._candidates: list[_Candidate] = []
        self._old_image_paths: dict[int, Path] = {}

    @property
    def name(self) -> str:
        return "WebEvo"

    def prepare(self, page_pair: PagePair) -> None:
        super().prepare(page_pair)
        self._candidates = []
        self._old_image_paths = {}

        for node in page_pair.old_page.nodes:
            path = _resolve_image_path(page_pair.old_page, node)
            if path is not None:
                self._old_image_paths[node.node_id] = path

        raw_candidates: list[tuple[int, int, str, str, Path | None]] = []
        if page_pair.new_page.pre_nodes:
            for pre in sorted(page_pair.new_page.pre_nodes, key=lambda item: item.list_index):
                grouped = _grouped_node_for_pre(page_pair.new_page, pre)
                if grouped is None:
                    continue
                raw_candidates.append(
                    (
                        pre.list_index,
                        grouped.node_id,
                        pre.tag.casefold(),
                        pre.text,
                        _resolve_image_path(page_pair.new_page, grouped),
                    )
                )
        else:
            for node in sorted(page_pair.new_page.nodes, key=lambda item: item.list_index):
                raw_candidates.append(
                    (
                        node.list_index,
                        node.node_id,
                        node.tag.casefold(),
                        node.text,
                        _resolve_image_path(page_pair.new_page, node),
                    )
                )

        order = 0
        for tag in WEBEVO_TAG_ORDER:
            for _source_order, node_id, candidate_tag, text, image_path in raw_candidates:
                if candidate_tag != tag:
                    continue
                self._candidates.append(_Candidate(order, node_id, candidate_tag, text, image_path))
                order += 1

    def _shortlist(self, target_text: str) -> list[tuple[_Candidate, float]]:
        if target_text == "":
            return [(candidate, math.nan) for candidate in self._candidates if candidate.text == ""]
        scored = [
            (candidate, _string_similarity(candidate.text, target_text))
            for candidate in self._candidates
            if candidate.text != ""
        ]
        # Collections.sort is stable; deterministic candidate order resolves ties.
        scored.sort(key=lambda item: (-item[1], item[0].order))
        return scored[: self.parameters.text_candidate_limit]

    def match(self, old_node_id: int) -> MatchResult:
        started = time.perf_counter_ns()
        source = self.pair.old_page.resolve_node(old_node_id)
        if source is None:
            return MatchResult((), time.perf_counter_ns() - started, "missing_old_node")
        target = _load_image(self._old_image_paths.get(source.node_id))
        if target is None:
            return MatchResult(
                (),
                time.perf_counter_ns() - started,
                "missing_element_screenshot",
            )

        comparisons = 0
        for candidate, text_similarity in self._shortlist(source.text):
            candidate_image = _load_image(candidate.image_path)
            if candidate_image is None:
                continue
            comparisons += 1
            accepted, image_similarity, stage = _image_stages(
                target,
                candidate_image,
                self.parameters,
            )
            if accepted:
                elapsed = time.perf_counter_ns() - started
                details = (
                    f"text_similarity={text_similarity};"
                    f"image_similarity={image_similarity:.6f};"
                    f"image_stage={stage};comparisons={comparisons}"
                )
                return MatchResult(
                    (ScoredCandidate(candidate.node_id, image_similarity),),
                    elapsed,
                    details=details,
                )

        return MatchResult(
            (),
            time.perf_counter_ns() - started,
            "no_prediction",
            f"comparisons={comparisons}",
        )

    def close(self) -> None:
        self._candidates = []
        self._old_image_paths = {}


__all__ = [
    "UPSTREAM_COMMIT",
    "UPSTREAM_MODULE",
    "UPSTREAM_REPOSITORY",
    "WEBEVO_TAG_ORDER",
    "WebEvoMatcher",
    "WebEvoParameters",
]
