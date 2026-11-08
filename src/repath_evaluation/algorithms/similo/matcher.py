"""Python reproduction of the Similo matching method.

Original project: https://github.com/michelnass/Similo2
Implemented for the REPATH evaluation framework.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import Any

from ...domain.models import DomNode, MatchResult, PageSnapshot, ScoredCandidate
from ...utilities.similarity import levenshtein_distance
from ..base import ElementMatcher

UPSTREAM_REPOSITORY = "https://github.com/michelnass/Similo2"
UPSTREAM_COMMIT = "743ddede7df92814ddc3fb7f96e09bdb508e285a"
UPSTREAM_FILE = "WidgetLocator/widgetlocator/WidgetLocator.java"


class SimiloMatcher(ElementMatcher):
    TAG_WEIGHT = 1.5
    CLASS_WEIGHT = 0.5
    NAME_WEIGHT = 1.5
    ID_WEIGHT = 1.5
    HREF_WEIGHT = 0.5
    ALT_WEIGHT = 0.5
    XPATH_WEIGHT = 0.5
    ID_XPATH_WEIGHT = 0.5
    BUTTON_WEIGHT = 0.5
    LOCATION_WEIGHT = 0.5
    AREA_WEIGHT = 0.5
    SHAPE_WEIGHT = 0.5
    VISIBLE_TEXT_WEIGHT = 1.5
    NEIGHBOUR_TEXT_WEIGHT = 1.5

    LOCATION_DISTANCE_LIMIT = 100
    NEIGHBOUR_MARGIN = 50
    MAX_NEIGHBOUR_ELEMENT_HEIGHT = 100
    MAX_TARGET_WIDTH_FOR_NEIGHBOURS = 600
    MAX_TARGET_HEIGHT_FOR_NEIGHBOURS = 100

    # Exact querySelectorAll list from the author implementation.
    SUPPORTED_TAGS = frozenset(
        {
            "input",
            "textarea",
            "button",
            "select",
            "a",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "li",
            "span",
            "div",
            "p",
            "th",
            "tr",
            "td",
            "label",
            "svg",
        }
    )

    @property
    def name(self) -> str:
        return "Similo"

    def prepare(self, page_pair) -> None:
        super().prepare(page_pair)
        self.old_candidates = [node for node in page_pair.old_page.nodes if self._eligible(node)]
        self.new_candidates = [node for node in page_pair.new_page.nodes if self._eligible(node)]
        self.old_id_xpaths = self._build_id_xpath_map(page_pair.old_page)
        self.new_id_xpaths = self._build_id_xpath_map(page_pair.new_page)
        self.old_neighbour_text = self._build_neighbour_map(self.old_candidates)
        self.new_neighbour_text = self._build_neighbour_map(self.new_candidates)

    @staticmethod
    def _normalize(value: Any) -> str:
        """Mirror ``object2String`` for the string-valued benchmark fields."""

        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _metadata_string(cls, value: Any) -> str:
        """Mirror ``addMetadata``: trim first, omit empty, then lower-case."""

        normalized = cls._normalize(value)
        return normalized.lower() if normalized else ""

    @classmethod
    def _first_nonblank(cls, *values: Any) -> str:
        for value in values:
            normalized = cls._normalize(value)
            if normalized:
                return normalized
        return ""

    @classmethod
    def _exact_similarity(cls, left: Any, right: Any) -> float:
        left_n = cls._metadata_string(left)
        right_n = cls._metadata_string(right)
        if not left_n or not right_n:
            return 0.0
        return 1.0 if left_n == right_n else 0.0

    @staticmethod
    def _strip_string(value: str) -> str:
        # Java Character.isAlphabetic/isDigit behaviour is most closely matched
        # by Python's Unicode-aware isalpha/isdigit predicates.
        return "".join(
            character for character in value if character.isalpha() or character.isdigit()
        )

    @classmethod
    def _string_similarity(cls, left: Any, right: Any) -> float:
        """Port ``stringSimilarity`` including its integer-percent rounding.

        The original lower-cases metadata before scoring, strips punctuation
        only for the edit-distance calculation, but retains the unstripped
        longer-string length as the denominator.
        """

        left_n = cls._metadata_string(left)
        right_n = cls._metadata_string(right)
        if not left_n or not right_n:
            return 0.0
        if left_n == right_n:
            return 1.0
        if len(left_n) < len(right_n):
            left_n, right_n = right_n, left_n
        distance = levenshtein_distance(cls._strip_string(left_n), cls._strip_string(right_n))
        percentage = (len(left_n) - distance) * 100 // len(left_n)
        return percentage / 100.0

    # Retain the earlier internal name for callers/tests that used it.
    _edit_similarity = _string_similarity

    @staticmethod
    def _integer_similarity(left: int, right: int) -> float:
        maximum = max(left, right)
        if maximum <= 0:
            return 0.0
        distance = abs(left - right)
        percentage = (maximum - distance) * 100 // maximum
        return percentage / 100.0

    @classmethod
    def _location_similarity(cls, left: DomNode, right: DomNode) -> float:
        dx = left.bbox.x - right.bbox.x
        dy = left.bbox.y - right.bbox.y
        # The Java source casts sqrt to int before applying the 100-pixel ramp.
        pixel_distance = int(math.sqrt(dx * dx + dy * dy))
        return max(cls.LOCATION_DISTANCE_LIMIT - pixel_distance, 0) / cls.LOCATION_DISTANCE_LIMIT

    @staticmethod
    def _shape(node: DomNode) -> int:
        if node.bbox.height <= 0:
            return 0
        return node.bbox.width * 100 // node.bbox.height

    @classmethod
    def _is_button(cls, node: DomNode) -> bool:
        tag = cls._normalize(node.tag)
        type_value = cls._normalize(node.attr("type"))
        class_name = cls._normalize(node.attr("class"))
        if tag.casefold() == "a" and "btn" in class_name:
            return True
        if tag.casefold() == "button":
            return True
        return tag.casefold() == "input" and type_value.casefold() in {
            "button",
            "submit",
            "reset",
        }

    @classmethod
    def _is_valid_text(cls, text: Any) -> bool:
        if text is None:
            return False
        trimmed = cls._normalize(text)
        if len(trimmed) < 3 or len(trimmed) > 50:
            return False
        return "\n" not in trimmed and "\t" not in trimmed

    @classmethod
    def _visible_text(cls, node: DomNode) -> str:
        # Locator.getVisibleText: valid textContent, then value, then placeholder.
        if cls._is_valid_text(node.text):
            return cls._metadata_string(node.text)
        value = cls._metadata_string(node.attr("value"))
        if value:
            return value
        return cls._metadata_string(node.attr("placeholder"))

    @classmethod
    def _raw_id_xpath(cls, node: DomNode) -> str:
        for key in ("idxpath", "idXPath", "id_xpath"):
            value = node.raw.get(key)
            if value is not None and cls._normalize(value):
                return cls._metadata_string(value)
        return ""

    @classmethod
    def _node_identifier(cls, node: DomNode) -> str:
        return cls._first_nonblank(node.explicit_id, node.attr("id"))

    @classmethod
    def _build_id_xpath_map(cls, page: PageSnapshot) -> dict[int, str]:
        """Approximate the upstream DOM-based ``getIdXPath`` deterministically.

        A raw IdXPath is preferred when supplied by the dataset. Otherwise the
        nearest ancestor selected by ``document.getElementById`` replaces the
        absolute XPath prefix.
        """

        first_node_for_id: dict[str, int] = {}
        for candidate in page.nodes:
            identifier = cls._node_identifier(candidate)
            if identifier and identifier not in first_node_for_id:
                # document.getElementById returns the first matching DOM element.
                first_node_for_id[identifier] = candidate.list_index
        result: dict[int, str] = {}

        for node in page.nodes:
            raw_id_xpath = cls._raw_id_xpath(node)
            if raw_id_xpath:
                result[node.list_index] = raw_id_xpath
                continue

            target_xpath = cls._normalize(node.xpath)
            current: DomNode | None = node
            visited: set[int] = set()
            id_xpath = ""
            while current is not None and current.node_id not in visited:
                visited.add(current.node_id)
                identifier = cls._node_identifier(current)
                if identifier and first_node_for_id.get(identifier) == current.list_index:
                    prefix = f"//*[@id='{identifier}']"
                    ancestor_xpath = cls._normalize(current.xpath)
                    if target_xpath == ancestor_xpath:
                        id_xpath = prefix
                    elif ancestor_xpath and target_xpath.startswith(
                        ancestor_xpath.rstrip("/") + "/"
                    ):
                        id_xpath = prefix + target_xpath[len(ancestor_xpath) :]
                    else:
                        id_xpath = prefix if current is node else target_xpath
                    break
                if current.parent_id is None:
                    current = None
                else:
                    current = page.nodes_by_id.get(current.parent_id) or page.nodes_by_index.get(
                        current.parent_id
                    )

            result[node.list_index] = cls._metadata_string(id_xpath or target_xpath)
        return result

    @staticmethod
    def _intersects(a, b) -> bool:
        return (
            a.x < b.x + b.width
            and a.x + a.width > b.x
            and a.y < b.y + b.height
            and a.y + a.height > b.y
        )

    @classmethod
    def _build_neighbour_text(cls, target: DomNode, candidates: list[DomNode]) -> str:
        rectangle = target.bbox
        if (
            rectangle.height > cls.MAX_TARGET_HEIGHT_FOR_NEIGHBOURS
            or rectangle.width > cls.MAX_TARGET_WIDTH_FOR_NEIGHBOURS
        ):
            return ""

        class Rect:
            def __init__(self, x: int, y: int, width: int, height: int):
                self.x = x
                self.y = y
                self.width = width
                self.height = height

        expanded = Rect(
            rectangle.x - cls.NEIGHBOUR_MARGIN,
            rectangle.y - cls.NEIGHBOUR_MARGIN,
            rectangle.width + 2 * cls.NEIGHBOUR_MARGIN,
            rectangle.height + 2 * cls.NEIGHBOUR_MARGIN,
        )
        unique_words: OrderedDict[str, None] = OrderedDict()
        for neighbour in candidates:
            if neighbour is target:
                continue
            neighbour_box = neighbour.bbox
            if neighbour_box.height > cls.MAX_NEIGHBOUR_ELEMENT_HEIGHT:
                continue
            neighbour_rect = Rect(
                neighbour_box.x,
                neighbour_box.y,
                neighbour_box.width,
                neighbour_box.height,
            )
            if not cls._intersects(expanded, neighbour_rect):
                continue
            visible_text = cls._visible_text(neighbour)
            if not visible_text:
                continue
            for word in visible_text.split():
                lowered = word.lower()
                if lowered not in unique_words:
                    unique_words[lowered] = None
        return " ".join(unique_words)

    @classmethod
    def _build_neighbour_map(cls, candidates: list[DomNode]) -> dict[int, str]:
        return {
            node.list_index: cls._build_neighbour_text(node, candidates) for node in candidates
        }

    @staticmethod
    def _compatible_word(target: str, available: list[str]) -> bool:
        for word in available:
            if len(target) < len(word) and (word.startswith(target) or word.endswith(target)):
                return True
            if len(word) < len(target) and (target.startswith(word) or target.endswith(word)):
                return True
            if target == word:
                return True
        return False

    @classmethod
    def _neighbour_similarity(cls, left: Any, right: Any) -> float:
        left_n = cls._metadata_string(left)
        right_n = cls._metadata_string(right)
        if not left_n or not right_n:
            return 0.0
        left_words = left_n.split()
        right_words = right_n.split()
        exists_count = sum(
            len(word) for word in left_words if cls._compatible_word(word, right_words)
        )
        word_count = max(
            len(left_n) - len(left_words) + 1,
            len(right_n) - len(right_words) + 1,
        )
        if word_count <= 0:
            return 0.0
        percentage = min(exists_count * 100 // word_count, 100)
        return percentage / 100.0

    @classmethod
    def _eligible(cls, node: DomNode) -> bool:
        return (
            node.element_type != 1
            and cls._normalize(node.tag).casefold() in cls.SUPPORTED_TAGS
            and node.displayed
            and node.bbox.width > 0
            and node.bbox.height > 0
        )

    def score(self, source: DomNode, candidate: DomNode) -> float:
        source_neighbour = self.old_neighbour_text.get(source.list_index, "")
        candidate_neighbour = self.new_neighbour_text.get(candidate.list_index, "")
        source_id_xpath = self.old_id_xpaths.get(
            source.list_index, self._metadata_string(source.xpath)
        )
        candidate_id_xpath = self.new_id_xpaths.get(
            candidate.list_index, self._metadata_string(candidate.xpath)
        )

        return (
            self.TAG_WEIGHT * self._exact_similarity(source.tag, candidate.tag)
            + self.CLASS_WEIGHT
            * self._string_similarity(source.attr("class"), candidate.attr("class"))
            + self.NAME_WEIGHT
            * self._exact_similarity(
                self._first_nonblank(source.name, source.attr("name")),
                self._first_nonblank(candidate.name, candidate.attr("name")),
            )
            + self.ID_WEIGHT
            * self._exact_similarity(
                self._first_nonblank(source.explicit_id, source.attr("id")),
                self._first_nonblank(candidate.explicit_id, candidate.attr("id")),
            )
            + self.HREF_WEIGHT
            * self._string_similarity(source.attr("href"), candidate.attr("href"))
            + self.ALT_WEIGHT * self._string_similarity(source.attr("alt"), candidate.attr("alt"))
            + self.XPATH_WEIGHT * self._string_similarity(source.xpath, candidate.xpath)
            + self.ID_XPATH_WEIGHT * self._string_similarity(source_id_xpath, candidate_id_xpath)
            + self.BUTTON_WEIGHT
            * (1.0 if self._is_button(source) == self._is_button(candidate) else 0.0)
            + self.LOCATION_WEIGHT * self._location_similarity(source, candidate)
            + self.AREA_WEIGHT * self._integer_similarity(source.bbox.area, candidate.bbox.area)
            + self.SHAPE_WEIGHT
            * self._integer_similarity(self._shape(source), self._shape(candidate))
            + self.VISIBLE_TEXT_WEIGHT
            * self._string_similarity(self._visible_text(source), self._visible_text(candidate))
            + self.NEIGHBOUR_TEXT_WEIGHT
            * self._neighbour_similarity(source_neighbour, candidate_neighbour)
        )

    def match(self, old_node_id: int) -> MatchResult:
        start = time.perf_counter_ns()
        source = self.pair.old_page.resolve_node(old_node_id)
        if source is None:
            return MatchResult((), time.perf_counter_ns() - start, "missing_old_node")
        if not self._eligible(source):
            return MatchResult((), time.perf_counter_ns() - start, "ineligible_old_node")

        ranking = [
            ScoredCandidate(candidate.node_id, self.score(source, candidate))
            for candidate in self.new_candidates
        ]
        # Collections.sort is stable; candidate extraction order resolves ties.
        ranking.sort(key=lambda item: -item.score)
        status = "ok" if ranking else "no_prediction"
        return MatchResult(tuple(ranking), time.perf_counter_ns() - start, status)


__all__ = [
    "SimiloMatcher",
    "UPSTREAM_COMMIT",
    "UPSTREAM_FILE",
    "UPSTREAM_REPOSITORY",
]
