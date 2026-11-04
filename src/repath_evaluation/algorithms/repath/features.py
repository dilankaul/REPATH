from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

import numpy as np

from ...domain.models import PagePair

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# Frozen inference settings from
# REPATH_01_Generate_Complete_50_Feature_CSVs_Clean.ipynb.
CONTAINER = 1
SBERT_BATCH_SIZE = 128
MAX_CANDIDATES = 1000
MIN_ACCESSIBLE_TEXT_LENGTH = 2
MAX_ACCESSIBLE_TEXT_LENGTH = 150
INTERACTIVE_ANCESTOR_MAX_DEPTH = 1
MAX_INTERACTIVE_DESCENDANT_CHILDREN = 0
FALLBACK_TOP_K = 25
FALLBACK_MIN_SCORE = 0.15
USE_GENERIC_ACCESSIBLE_TEXT = True
USE_ATTRIBUTE_INTERACTION = True
USE_INTERACTIVE_DESCENDANTS = True
USE_QUERY_SPECIFIC_FALLBACK = True
QUERY_SHORTLIST_TOP_K = 25
PROTECT_EXACT_ID = True
PROTECT_EXACT_DATA_TEST_ID = True
PROTECT_EXACT_NAME = True
PROTECT_EXACT_ACCESSIBLE_TEXT = True
FORCE_INCLUDE_POSITIVE = False


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def normalise_text(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def normalise_tokens(value: Any) -> str:
    text = normalise_text(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def token_set(value: Any) -> set:
    return {token for token in normalise_tokens(value).split() if token}


def token_jaccard(left: Any, right: Any) -> float:
    left_tokens, right_tokens = token_set(left), token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sequence_similarity(left: Any, right: Any) -> float:
    left_text, right_text = normalise_text(left), normalise_text(right)
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    return float(SequenceMatcher(None, left_text, right_text).ratio())


def node_attributes(node: dict[str, Any]) -> dict[str, str]:
    raw = node.get("attributes", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        str(key).strip().lower(): str(value)
        for key, value in raw.items() if value is not None
    }


def node_attribute(node: dict[str, Any], *names: str) -> str:
    attributes = node_attributes(node)
    for name in names:
        value = attributes.get(name.lower())
        if value is not None and str(value) != "":
            return str(value)

    aliases = {
        "aria-label": ["ariaLabel", "arialabel"],
        "data-testid": ["dataTestId", "datatestid"],
        "data-test": ["dataTest", "datatest"],
        "data-qa": ["dataQa", "dataqa"],
        "class": ["className", "classname"],
        "tabindex": ["tabIndex"],
    }
    for name in names:
        keys = [name, name.lower(), name.replace("-", "")] + aliases.get(name.lower(), [])
        for key in keys:
            value = node.get(key)
            if value is not None and str(value) != "":
                return str(value)
    return ""


def first_number(node: dict[str, Any], names: Iterable[str], default: float = 0.0) -> float:
    for name in names:
        value = node.get(name)
        try:
            if value is not None and str(value).strip() != "":
                return float(value)
        except (TypeError, ValueError):
            pass
    return float(default)


def first_bool(node: dict[str, Any], names: Iterable[str], default: bool) -> bool:
    for name in names:
        if name not in node or node.get(name) is None:
            continue
        value = node.get(name)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return default


def node_tag(node: dict[str, Any]) -> str:
    return normalise_text(
        node.get("tagName", node.get("tag", node.get("name", "")))
    )


def raw_node_text(node: dict[str, Any]) -> str:
    for key in ["text", "innerText", "textContent", "value"]:
        value = normalise_text(node.get(key, ""))
        if value:
            return value
    return ""


def accessible_text(node: dict[str, Any]) -> str:
    """First meaningful user-facing text source, independent of tag name."""
    values = [
        raw_node_text(node),
        node_attribute(node, "aria-label", "arialabel"),
        node_attribute(node, "title"),
        node_attribute(node, "alt"),
        node_attribute(node, "value"),
        node_attribute(node, "placeholder"),
        node_attribute(node, "labeltext", "labelText"),
    ]
    for value in values:
        value = normalise_text(value)
        if value:
            return value
    return ""


def node_text(node: dict[str, Any]) -> str:
    # Rich combined text used by the pair features and fallback scoring.
    values = [
        raw_node_text(node),
        node_attribute(node, "labeltext", "labelText"),
        node_attribute(node, "placeholder"),
        node_attribute(node, "aria-label", "arialabel"),
        node_attribute(node, "title"),
        node_attribute(node, "alt"),
        node_attribute(node, "name"),
        node_attribute(node, "id"),
        node_attribute(node, "data-testid", "data-test", "data-qa"),
    ]
    return normalise_text(" ".join(value for value in values if value))


def node_xpath(node: dict[str, Any]) -> str:
    for key in ["xpath", "xPath", "XPath", "absoluteXPath", "fullXPath"]:
        value = normalise_text(node.get(key, ""))
        if value:
            return value
    return normalise_text(node_attribute(node, "xpath"))


def dataset_visible(node: dict[str, Any]) -> bool:
    # Offline approximation of runtime display/visibility/opacity and bbox >= 2x2.
    displayed = first_bool(
        node,
        ["isDisplayed", "displayed", "isVisible", "visible"],
        default=True,
    )
    width = first_number(node, ["width", "bboxW", "w"], 0.0)
    height = first_number(node, ["height", "bboxH", "h"], 0.0)
    return displayed and width >= 2.0 and height >= 2.0


def element_child_count(node: dict[str, Any]) -> int | None:
    for key in ["children", "childNodes", "childElementIds"]:
        value = node.get(key)
        if isinstance(value, list):
            return len(value)
    for key in ["childrenCount", "childCount", "numberOfChildren"]:
        value = safe_int(node.get(key))
        if value is not None:
            return value
    if "isLeaf" in node:
        return 0 if first_bool(node, ["isLeaf"], default=False) else None
    return None


def has_no_element_children(node: dict[str, Any]) -> bool:
    child_count = element_child_count(node)
    if child_count is not None:
        return child_count == 0
    # elementType is a useful dataset fallback, but not exact live DOM state.
    return safe_int(node.get("elementType"), CONTAINER) != CONTAINER


INTERACTIVE_TAGS = {"input", "textarea", "select", "button"}
INTERACTIVE_ROLES = {
    "button", "link", "menuitem", "menuitemcheckbox", "menuitemradio",
    "tab", "checkbox", "radio", "textbox", "searchbox", "combobox",
    "switch", "option", "slider", "spinbutton", "treeitem",
}


def valid_href(value: Any) -> bool:
    href = normalise_text(value)
    return bool(href) and href != "#" and not href.startswith("javascript")


def parse_tabindex(node: dict[str, Any]) -> int | None:
    value = node_attribute(node, "tabindex")
    return safe_int(value, None)


def is_interactive_candidate(node: dict[str, Any]) -> bool:
    if not dataset_visible(node):
        return False

    tag = node_tag(node)
    role = normalise_text(node_attribute(node, "role"))
    if tag in INTERACTIVE_TAGS:
        return True

    if not USE_ATTRIBUTE_INTERACTION:
        return tag == "a" and valid_href(node_attribute(node, "href"))

    tabindex = parse_tabindex(node)
    return (
        valid_href(node_attribute(node, "href"))
        or role in INTERACTIVE_ROLES
        or bool(node_attribute(node, "onclick"))
        or normalise_text(node_attribute(node, "contenteditable")) == "true"
        or (tabindex is not None and tabindex >= 0)
    )


def is_generic_text_leaf_candidate(node: dict[str, Any]) -> bool:
    if not USE_GENERIC_ACCESSIBLE_TEXT or not dataset_visible(node):
        return False
    if is_interactive_candidate(node):
        return False
    if not has_no_element_children(node):
        return False
    text = accessible_text(node)
    return MIN_ACCESSIBLE_TEXT_LENGTH <= len(text) <= MAX_ACCESSIBLE_TEXT_LENGTH


def _node_reference_map(nodes: list[dict[str, Any]]) -> dict[str, int]:
    mapping = {}
    for index, node in enumerate(nodes):
        for key in ["nodeId", "nodeID", "backendNodeId", "backendNodeID", "uid"]:
            value = node.get(key)
            if value is not None and str(value).strip() != "":
                mapping.setdefault(str(value).strip(), index)
    return mapping


def _parent_xpath(xpath: str) -> str:
    xpath = normalise_text(xpath).rstrip("/")
    if not xpath or "/" not in xpath:
        return ""
    return xpath.rsplit("/", 1)[0]


def build_parent_index_map(nodes: list[dict[str, Any]]) -> dict[int, int | None]:
    """Resolve parent links from explicit indices/IDs, then fall back to XPath."""
    parent_map: dict[int, int | None] = {index: None for index in range(len(nodes))}
    reference_map = _node_reference_map(nodes)

    xpath_to_index = {}
    for index, node in enumerate(nodes):
        xpath = node_xpath(node)
        if xpath:
            xpath_to_index.setdefault(xpath, index)

    index_keys = [
        "parentIndex", "parentIdx", "parent_index", "parentNodeIndex",
        "parentElementIndex",
    ]
    reference_keys = [
        "parentNodeId", "parentNodeID", "parentBackendNodeId",
        "parentBackendNodeID", "parentId",
    ]

    for index, node in enumerate(nodes):
        resolved = None

        for key in index_keys:
            candidate = safe_int(node.get(key), None)
            if candidate is not None and 0 <= candidate < len(nodes) and candidate != index:
                resolved = candidate
                break

        if resolved is None:
            for key in reference_keys:
                value = node.get(key)
                if value is None:
                    continue
                candidate = reference_map.get(str(value).strip())
                if candidate is not None and candidate != index:
                    resolved = candidate
                    break

        if resolved is None:
            generic_parent = node.get("parent")
            if isinstance(generic_parent, dict):
                for key in ["index", "idx", "nodeIndex", "nodeId", "backendNodeId"]:
                    if key not in generic_parent:
                        continue
                    value = generic_parent.get(key)
                    candidate = safe_int(value, None)
                    if (
                        (key.lower().endswith("index") or key in {"index", "idx"})
                        and candidate is not None
                        and 0 <= candidate < len(nodes)
                        and candidate != index
                    ):
                        resolved = candidate
                        break
                    candidate = reference_map.get(str(value).strip())
                    if candidate is not None and candidate != index:
                        resolved = candidate
                        break
            elif generic_parent is not None:
                candidate = safe_int(generic_parent, None)
                if candidate is not None and 0 <= candidate < len(nodes) and candidate != index:
                    resolved = candidate

        if resolved is None:
            parent_xpath = _parent_xpath(node_xpath(node))
            candidate = xpath_to_index.get(parent_xpath)
            if candidate is not None and candidate != index:
                resolved = candidate

        parent_map[index] = resolved

    return parent_map


def nearest_interactive_ancestor(
    node_index: int,
    nodes: list[dict[str, Any]],
    parent_map: dict[int, int | None],
    max_depth: int = INTERACTIVE_ANCESTOR_MAX_DEPTH,
):
    current = parent_map.get(node_index)
    for depth in range(1, max_depth + 1):
        if current is None or current < 0 or current >= len(nodes):
            break
        if is_interactive_candidate(nodes[current]):
            return current, depth
        current = parent_map.get(current)
    return None, None


def is_interactive_descendant_candidate(
    node_index: int,
    nodes: list[dict[str, Any]],
    parent_map: dict[int, int | None],
):
    if not USE_INTERACTIVE_DESCENDANTS:
        return False, None, None
    node = nodes[node_index]
    if not dataset_visible(node) or is_interactive_candidate(node):
        return False, None, None

    ancestor_index, ancestor_depth = nearest_interactive_ancestor(
        node_index, nodes, parent_map
    )
    if ancestor_index is None:
        return False, None, None

    child_count = element_child_count(node)

    # Balanced preset: only direct leaf descendants are retained.  When a
    # nonzero descendant allowance is configured, known child counts
    # may be used, but unknown structure is not accepted automatically.
    if MAX_INTERACTIVE_DESCENDANT_CHILDREN <= 0 or child_count is None:
        structurally_small = has_no_element_children(node)
    else:
        structurally_small = child_count <= MAX_INTERACTIVE_DESCENDANT_CHILDREN

    return structurally_small, ancestor_index, ancestor_depth


def extract_base_candidate_indices(
    new_nodes: list[dict[str, Any]],
    parent_map: dict[int, int | None],
):
    """Page-wide generic pool before the query-specific fallback."""
    interactive = [
        index for index, node in enumerate(new_nodes)
        if is_interactive_candidate(node)
    ]
    text_leaf = [
        index for index, node in enumerate(new_nodes)
        if is_generic_text_leaf_candidate(node)
    ]

    interactive_descendants = []
    descendant_ancestor = {}
    descendant_depth = {}
    for index in range(len(new_nodes)):
        ok, ancestor_index, depth = is_interactive_descendant_candidate(
            index, new_nodes, parent_map
        )
        if ok:
            interactive_descendants.append(index)
            descendant_ancestor[index] = ancestor_index
            descendant_depth[index] = depth

    merged, kinds, seen = [], {}, set()
    ordered_sources = (
        [(index, "interactive") for index in interactive]
        + [(index, "text_leaf") for index in text_leaf]
        + [(index, "interactive_descendant") for index in interactive_descendants]
    )
    for index, kind in ordered_sources:
        if index in seen:
            continue
        seen.add(index)
        merged.append(index)
        kinds[index] = kind

    stats = {
        "interactive_count": len(interactive),
        "text_leaf_count": len(text_leaf),
        "interactive_descendant_count": len(interactive_descendants),
        "descendant_ancestor": descendant_ancestor,
        "descendant_depth": descendant_depth,
    }
    return merged, kinds, stats


def node_tag_family(tag: str) -> str:
    tag = normalise_text(tag)
    if tag in {"input", "textarea", "select", "button", "option"}:
        return "form_control"
    if tag == "a":
        return "link"
    if tag in {"img", "svg", "picture", "canvas", "video", "audio", "i"}:
        return "media_or_icon"
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "span", "label", "strong", "em", "small", "b", "bdi"}:
        return "text"
    if tag in {"li", "ul", "ol", "menu"}:
        return "list"
    if tag in {"td", "th", "tr", "table", "thead", "tbody"}:
        return "table"
    if tag in {"div", "section", "article", "main", "aside", "header", "footer", "nav", "form"}:
        return "container"
    if "-" in tag:
        return "custom_element"
    return "other"


def tag_compatibility(old_node: dict[str, Any], new_node: dict[str, Any]) -> float:
    old_tag, new_tag = node_tag(old_node), node_tag(new_node)
    if old_tag and old_tag == new_tag:
        return 1.0
    if old_tag and new_tag and node_tag_family(old_tag) == node_tag_family(new_tag):
        return 0.65
    return 0.0


def role_compatibility(old_node: dict[str, Any], new_node: dict[str, Any]) -> float:
    old_role = normalise_text(node_attribute(old_node, "role"))
    new_role = normalise_text(node_attribute(new_node, "role"))
    if old_role and old_role == new_role:
        return 1.0
    if old_role in INTERACTIVE_ROLES and new_role in INTERACTIVE_ROLES:
        return 0.5
    return 0.0


def attribute_similarity(old_node: dict[str, Any], new_node: dict[str, Any]) -> float:
    similarities = []
    for names in [
        ("id",),
        ("data-testid", "data-test", "data-qa"),
        ("name",),
        ("class",),
        ("aria-label",),
        ("title",),
        ("placeholder",),
    ]:
        old_value = node_attribute(old_node, *names)
        new_value = node_attribute(new_node, *names)
        if not old_value or not new_value:
            continue
        similarities.append(max(
            sequence_similarity(old_value, new_value),
            token_jaccard(old_value, new_value),
        ))
    return max(similarities, default=0.0)


def geometry_similarity(old_node: dict[str, Any], new_node: dict[str, Any]) -> float:
    scores = []
    for names in [(["width", "bboxW", "w"]), (["height", "bboxH", "h"])]:
        old_value = first_number(old_node, names, 0.0)
        new_value = first_number(new_node, names, 0.0)
        if old_value > 0 and new_value > 0:
            scores.append(min(old_value, new_value) / max(old_value, new_value))
    return float(np.mean(scores)) if scores else 0.0


def fallback_similarity(
    old_node: dict[str, Any],
    new_node: dict[str, Any],
    old_index: int,
    new_index: int,
) -> float:
    text_score = max(
        sequence_similarity(node_text(old_node), node_text(new_node)),
        token_jaccard(node_text(old_node), node_text(new_node)),
    )
    attr_score = attribute_similarity(old_node, new_node)
    tag_score = tag_compatibility(old_node, new_node)
    role_score = role_compatibility(old_node, new_node)
    xpath_score = token_jaccard(node_xpath(old_node), node_xpath(new_node))
    geometry_score = geometry_similarity(old_node, new_node)
    index_score = 1.0 / (1.0 + float(np.log1p(abs(new_index - old_index))))

    return float(
        0.35 * text_score
        + 0.20 * attr_score
        + 0.10 * tag_score
        + 0.10 * role_score
        + 0.10 * xpath_score
        + 0.10 * geometry_score
        + 0.05 * index_score
    )


def select_query_fallback_candidates(
    old_node: dict[str, Any],
    old_index: int,
    new_nodes: list[dict[str, Any]],
    excluded_indices: Iterable[int],
):
    """Rank rejected visible nodes without consulting the true_new_idx label."""
    if not USE_QUERY_SPECIFIC_FALLBACK or FALLBACK_TOP_K <= 0:
        return [], {}, {}, 0

    excluded = set(excluded_indices)
    ranked = []
    for new_index, new_node in enumerate(new_nodes):
        if new_index in excluded or not dataset_visible(new_node):
            continue
        score = fallback_similarity(old_node, new_node, old_index, new_index)
        ranked.append((score, new_index))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    score_by_index = {index: float(score) for score, index in ranked}
    rank_by_index = {index: rank for rank, (_, index) in enumerate(ranked, start=1)}
    selected = [
        index for score, index in ranked
        if score >= FALLBACK_MIN_SCORE
    ][:FALLBACK_TOP_K]
    return selected, score_by_index, rank_by_index, len(ranked)


def merge_query_candidates(
    base_indices: list[int],
    base_kinds: dict[int, str],
    fallback_indices: list[int],
    cap: int | None,
):
    merged, kinds, seen = [], {}, set()
    for index, kind in (
        [(index, base_kinds.get(index, "base")) for index in base_indices]
        + [(index, "fallback") for index in fallback_indices]
    ):
        if index in seen:
            continue
        seen.add(index)
        merged.append(index)
        kinds[index] = kind

    uncapped = list(merged)
    if cap is not None:
        merged = merged[:cap]
        kinds = {index: kinds[index] for index in merged}
    return merged, kinds, uncapped



def exact_shortlist_protection_reason(
    old_node: dict[str, Any],
    new_node: dict[str, Any],
) -> str:
    """Return label-independent reasons for preserving a candidate."""
    reasons = []

    if PROTECT_EXACT_ID:
        old_value = normalise_text(node_attribute(old_node, "id"))
        new_value = normalise_text(node_attribute(new_node, "id"))
        if old_value and old_value == new_value:
            reasons.append("exact_id")

    if PROTECT_EXACT_DATA_TEST_ID:
        old_value = normalise_text(
            node_attribute(old_node, "data-testid", "data-test", "data-qa")
        )
        new_value = normalise_text(
            node_attribute(new_node, "data-testid", "data-test", "data-qa")
        )
        if old_value and old_value == new_value:
            reasons.append("exact_data_test_id")

    if PROTECT_EXACT_NAME:
        old_value = normalise_text(node_attribute(old_node, "name"))
        new_value = normalise_text(node_attribute(new_node, "name"))
        if old_value and old_value == new_value:
            reasons.append("exact_name")

    if PROTECT_EXACT_ACCESSIBLE_TEXT:
        old_value = accessible_text(old_node)
        new_value = accessible_text(new_node)
        if (
            old_value
            and len(old_value) >= MIN_ACCESSIBLE_TEXT_LENGTH
            and old_value == new_value
        ):
            reasons.append("exact_accessible_text")

    return "|".join(reasons)


def rank_query_candidate_pool(
    old_node: dict[str, Any],
    old_index: int,
    new_nodes: list[dict[str, Any]],
    candidate_indices: list[int],
):
    """Rank the full retrieved pool without using the ground-truth label."""
    scored = []
    protection_by_index = {}

    for new_index in candidate_indices:
        new_node = new_nodes[new_index]
        score = fallback_similarity(
            old_node=old_node,
            new_node=new_node,
            old_index=old_index,
            new_index=new_index,
        )
        scored.append((float(score), int(new_index)))

        reason = exact_shortlist_protection_reason(old_node, new_node)
        if reason:
            protection_by_index[new_index] = reason

    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked_indices = [index for _, index in scored]
    score_by_index = {index: score for score, index in scored}
    rank_by_index = {
        index: rank
        for rank, index in enumerate(ranked_indices, start=1)
    }
    return (
        ranked_indices,
        score_by_index,
        rank_by_index,
        protection_by_index,
    )


def apply_query_shortlist(
    full_candidate_indices: list[int],
    ranked_candidate_indices: list[int],
    protection_by_index: dict[int, str],
    top_k: int | None,
) -> list[int]:
    """
    Keep the top-K cheap retrieval results plus exact protected matches.

    The output follows the original balanced-pool order so candidate order is
    stable and does not directly encode the cheap retrieval rank.
    """
    if top_k is None:
        return list(full_candidate_indices)

    top_k = max(0, int(top_k))
    selected = set(ranked_candidate_indices[:top_k])
    selected.update(protection_by_index.keys())

    return [
        index for index in full_candidate_indices
        if index in selected
    ]


def shortlist_label(top_k: int | None) -> str:
    return "full" if top_k is None else str(int(top_k))


def diagnose_positive_candidate(
    true_new_index: int,
    old_node: dict[str, Any],
    old_index: int,
    new_nodes: list[dict[str, Any]],
    parent_map: dict[int, int | None],
    base_indices: list[int],
    base_kinds: dict[int, str],
    fallback_indices: list[int],
    fallback_score_by_index: dict[int, float],
    fallback_rank_by_index: dict[int, int],
    uncapped_final_indices: list[int],
    final_indices: list[int],
    final_kinds: dict[int, str],
) -> dict[str, Any]:
    """Explain why the official positive was retained or rejected."""
    if true_new_index < 0 or true_new_index >= len(new_nodes):
        return {
            "reject_reason": "true_new_idx_out_of_range",
            "eligible_before_cap": False,
            "uncapped_candidate_rank": None,
            "positive_retained": False,
            "positive_candidate_source": "",
            "positive_tag": "",
            "positive_role": "",
            "positive_text": "",
            "positive_accessible_text": "",
            "positive_accessible_text_length": 0,
            "positive_displayed": False,
            "positive_width": np.nan,
            "positive_height": np.nan,
            "interactive_ok": False,
            "text_leaf_ok": False,
            "interactive_descendant_ok": False,
            "interactive_ancestor_index": None,
            "interactive_ancestor_depth": None,
            "leaf_ok": False,
            "base_candidate_ok": False,
            "fallback_selected": False,
            "fallback_rank": None,
            "fallback_score": np.nan,
            "positive_href": "",
            "positive_aria_label": "",
            "positive_onclick": "",
            "positive_tabindex": "",
            "positive_contenteditable": "",
        }

    node = new_nodes[true_new_index]
    displayed = first_bool(
        node,
        ["isDisplayed", "displayed", "isVisible", "visible"],
        default=True,
    )
    width = first_number(node, ["width", "bboxW", "w"], 0.0)
    height = first_number(node, ["height", "bboxH", "h"], 0.0)
    accessible = accessible_text(node)

    interactive_ok = is_interactive_candidate(node)
    text_leaf_ok = is_generic_text_leaf_candidate(node)
    descendant_ok, ancestor_index, ancestor_depth = is_interactive_descendant_candidate(
        true_new_index, new_nodes, parent_map
    )
    base_candidate_ok = true_new_index in base_indices
    fallback_selected = true_new_index in fallback_indices
    eligible_before_cap = true_new_index in uncapped_final_indices
    retained = true_new_index in final_indices

    uncapped_rank = (
        uncapped_final_indices.index(true_new_index) + 1
        if eligible_before_cap else None
    )
    fallback_rank = fallback_rank_by_index.get(true_new_index)
    fallback_score = fallback_score_by_index.get(true_new_index, np.nan)

    if retained:
        reason = "retained"
    elif not displayed:
        reason = "not_displayed"
    elif width < 2.0 or height < 2.0:
        reason = "bbox_too_small_or_missing"
    elif eligible_before_cap:
        reason = "exceeds_candidate_cap"
    elif fallback_rank is None:
        reason = "not_in_visible_fallback_pool"
    elif fallback_score < FALLBACK_MIN_SCORE:
        reason = "fallback_score_below_threshold"
    elif fallback_rank > FALLBACK_TOP_K:
        reason = "outside_fallback_top_k"
    else:
        reason = "not_selected_by_generalized_retrieval"

    return {
        "reject_reason": reason,
        "eligible_before_cap": eligible_before_cap,
        "uncapped_candidate_rank": uncapped_rank,
        "positive_retained": retained,
        "positive_candidate_source": final_kinds.get(true_new_index, ""),
        "positive_tag": node_tag(node),
        "positive_role": normalise_text(node_attribute(node, "role")),
        "positive_text": raw_node_text(node),
        "positive_accessible_text": accessible,
        "positive_accessible_text_length": len(accessible),
        "positive_displayed": displayed,
        "positive_width": width,
        "positive_height": height,
        "interactive_ok": interactive_ok,
        "text_leaf_ok": text_leaf_ok,
        "interactive_descendant_ok": bool(descendant_ok),
        "interactive_ancestor_index": ancestor_index,
        "interactive_ancestor_depth": ancestor_depth,
        "leaf_ok": has_no_element_children(node),
        "base_candidate_ok": base_candidate_ok,
        "fallback_selected": fallback_selected,
        "fallback_rank": fallback_rank,
        "fallback_score": fallback_score,
        "positive_href": node_attribute(node, "href"),
        "positive_aria_label": node_attribute(node, "aria-label"),
        "positive_onclick": node_attribute(node, "onclick"),
        "positive_tabindex": node_attribute(node, "tabindex"),
        "positive_contenteditable": node_attribute(node, "contenteditable"),
    }



# ------------------------------------------------------------------
# Runtime-aligned deterministic feature helpers
# ------------------------------------------------------------------
# These functions mirror the deterministic feature definitions in app.py.
# They use only information available in old/new domNodeInfo.json files.

RUNTIME_XPATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


def runtime_norm(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split()).strip()


def runtime_token_set(value: Any) -> set:
    return {
        token for token in runtime_norm(value).split()
        if len(token) >= 2
    }


def runtime_seq_sim(left: Any, right: Any) -> float:
    left_text, right_text = runtime_norm(left), runtime_norm(right)
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    return float(SequenceMatcher(None, left_text, right_text).ratio())


def runtime_jaccard(left: Any, right: Any) -> float:
    left_tokens = runtime_token_set(left)
    right_tokens = runtime_token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def runtime_class_jaccard(left: Any, right: Any) -> float:
    left_tokens = {
        token for token in runtime_norm(left).split() if token
    }
    right_tokens = {
        token for token in runtime_norm(right).split() if token
    }
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def runtime_xpath_token_sim(left: Any, right: Any) -> float:
    def tokens(value: Any) -> set:
        if not value:
            return set()
        return {
            runtime_norm(token)
            for token in RUNTIME_XPATH_TOKEN_RE.findall(str(value))
            if len(runtime_norm(token)) >= 2
        }

    left_tokens, right_tokens = tokens(left), tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def runtime_node_attr(node: dict[str, Any], key: str) -> str:
    """Prefer the actual top-level DOM field, then the attributes map."""
    aliases = {
        "id": ["id"],
        "name": ["name"],
        "class": ["className", "classname", "class"],
        "placeholder": ["placeholder"],
        "type": ["type"],
        "value": ["value"],
        "aria-label": ["ariaLabel", "arialabel", "aria-label"],
        "data-testid": ["dataTestId", "datatestid", "data-testid", "data-test", "data-qa"],
        "role": ["role"],
        "title": ["title"],
        "for": ["htmlFor", "for"],
        "label": ["labelText", "labeltext", "label"],
    }
    for alias in aliases.get(key, [key]):
        value = node.get(alias)
        if value is not None and str(value).strip() != "":
            return str(value)

    if key == "data-testid":
        return node_attribute(node, "data-testid", "data-test", "data-qa")
    if key == "aria-label":
        return node_attribute(node, "aria-label", "arialabel")
    if key == "class":
        return node_attribute(node, "class", "classname")
    return node_attribute(node, key)


def node_depth_value(node: dict[str, Any]) -> float:
    for key in ["depth", "domDepth", "level"]:
        value = first_number(node, [key], np.nan)
        if np.isfinite(value):
            return float(value)
    dom_path = node.get("domPath")
    if isinstance(dom_path, list):
        return float(len(dom_path))
    return 0.0


def node_visible_value(node: dict[str, Any]) -> bool:
    return first_bool(
        node,
        ["isDisplayed", "isVisible", "displayed", "visible"],
        default=True,
    )


def node_enabled_value(node: dict[str, Any]) -> bool:
    return first_bool(
        node,
        ["isEnable", "isEnabled", "enabled"],
        default=True,
    )


def node_bbox(node: dict[str, Any]):
    x = first_number(node, ["x", "bboxX", "left"], np.nan)
    y = first_number(node, ["y", "bboxY", "top"], np.nan)
    width = first_number(node, ["width", "bboxW", "w"], np.nan)
    height = first_number(node, ["height", "bboxH", "h"], np.nan)

    if not all(np.isfinite(value) for value in [x, y, width, height]):
        return None
    if width <= 0 or height <= 0:
        return None
    return float(x), float(y), float(width), float(height)


def runtime_geometry_features(
    old_node: dict[str, Any],
    candidate_node: dict[str, Any],
) -> dict[str, Any]:
    old_box = node_bbox(old_node)
    candidate_box = node_bbox(candidate_node)

    if old_box is None or candidate_box is None:
        return {
            "geo_iou": 0.0,
            "geo_center_dist": 0.0,
            "geo_size_ratio": 0.0,
            "bbox_dx": 0.0,
            "bbox_dy": 0.0,
            "bbox_area_ratio": 0.0,
            "geometry_available": 0,
            "old_bbox_x": np.nan,
            "old_bbox_y": np.nan,
            "old_bbox_w": np.nan,
            "old_bbox_h": np.nan,
            "new_bbox_x": np.nan,
            "new_bbox_y": np.nan,
            "new_bbox_w": np.nan,
            "new_bbox_h": np.nan,
        }

    ox, oy, ow, oh = old_box
    nx, ny, nw, nh = candidate_box

    old_center_x = ox + ow / 2.0
    old_center_y = oy + oh / 2.0
    new_center_x = nx + nw / 2.0
    new_center_y = ny + nh / 2.0

    center_distance = float(np.sqrt(
        (old_center_x - new_center_x) ** 2
        + (old_center_y - new_center_y) ** 2
    ))

    intersection_width = max(
        0.0,
        min(ox + ow, nx + nw) - max(ox, nx),
    )
    intersection_height = max(
        0.0,
        min(oy + oh, ny + nh) - max(oy, ny),
    )
    intersection = intersection_width * intersection_height

    old_area = ow * oh
    new_area = nw * nh
    union = old_area + new_area - intersection + 1e-6
    area_ratio = old_area / (new_area + 1e-6)

    return {
        "geo_iou": float(np.clip(intersection / union, 0.0, 1.0)),
        "geo_center_dist": center_distance,
        "geo_size_ratio": float(np.log1p(area_ratio)),
        "bbox_dx": float(abs(ox - nx)),
        "bbox_dy": float(abs(oy - ny)),
        "bbox_area_ratio": float(np.clip(area_ratio, 0.0, 100.0)),
        "geometry_available": 1,
        "old_bbox_x": ox,
        "old_bbox_y": oy,
        "old_bbox_w": ow,
        "old_bbox_h": oh,
        "new_bbox_x": nx,
        "new_bbox_y": ny,
        "new_bbox_w": nw,
        "new_bbox_h": nh,
    }


def build_runtime_context_maps(
    nodes: list[dict[str, Any]],
    parent_map: dict[int, int | None],
) -> dict[str, dict[int, Any]]:
    """
    Build deterministic text and structural context from the grouped UITestFix
    DOM. All indices refer to domNodeInfo.json entries.
    """
    children_map: dict[int | None, list[int]] = defaultdict(list)
    for index, parent_index in parent_map.items():
        if parent_index is not None and 0 <= parent_index < len(nodes):
            children_map[parent_index].append(index)

    for children in children_map.values():
        children.sort()

    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}

    label_for_text: dict[str, str] = {}
    for node in nodes:
        if node_tag(node) != "label":
            continue
        target = runtime_norm(runtime_node_attr(node, "for"))
        text = raw_node_text(node)
        if target and text:
            label_for_text.setdefault(target, text)

    def sibling_texts(index: int, direction: str, limit: int = 2) -> str:
        parent_index = parent_map.get(index)
        siblings = children_map.get(parent_index, [])
        if index not in siblings:
            return ""
        position = siblings.index(index)
        if direction == "before":
            selected = list(reversed(siblings[max(0, position - limit):position]))
        else:
            selected = siblings[position + 1:position + 1 + limit]
        texts = [raw_node_text(nodes[item]) for item in selected]
        return " | ".join(text for text in texts if text)

    def first_heading_descendant(root_index: int) -> str:
        queue = list(children_map.get(root_index, []))
        visited = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if node_tag(nodes[current]) in heading_tags:
                text = raw_node_text(nodes[current])
                if text:
                    return text
            queue.extend(children_map.get(current, []))
        return ""

    contexts = {
        "parent_text": {},
        "parent_tag": {},
        "sibling_text": {},
        "sibling_before": {},
        "sibling_after": {},
        "child_text": {},
        "context_text": {},
        "nearby_text": {},
        "heading_context": {},
        "ancestor_chain": {},
        "label_text": {},
        "dom_depth": {},
        # Structural context used by the eight structural model features.
        "ancestor_tags_nearest": {},
        "ancestor_tags_root": {},
        "sibling_tags": {},
        "child_tags": {},
        "sibling_position": {},
        "sibling_count": {},
        "parent_map": parent_map,
        "children_map": dict(children_map),
    }

    for index, node in enumerate(nodes):
        parent_index = parent_map.get(index)
        parent_node = (
            nodes[parent_index]
            if parent_index is not None and 0 <= parent_index < len(nodes)
            else None
        )

        parent_text = raw_node_text(parent_node)[:200] if parent_node else ""
        parent_tag = node_tag(parent_node) if parent_node else ""

        sibling_before = sibling_texts(index, "before", limit=2)
        sibling_after = sibling_texts(index, "after", limit=2)
        sibling_text = " | ".join(
            value for value in [sibling_before, sibling_after] if value
        )

        direct_children = children_map.get(index, [])
        child_values = [
            raw_node_text(nodes[child])
            for child in direct_children[:3]
        ]
        child_text = " | ".join(value for value in child_values if value)

        nearby_text = parent_text[:120]

        context_parts = [
            raw_node_text(node),
            parent_text,
            sibling_text,
            child_text,
            runtime_node_attr(node, "aria-label"),
            runtime_node_attr(node, "placeholder"),
            runtime_node_attr(node, "title"),
            runtime_node_attr(node, "value"),
        ]
        context_text = normalise_text(
            " ".join(value for value in context_parts if value)
        )

        heading_context = ""
        current = parent_index
        depth = 0
        while current is not None and 0 <= current < len(nodes) and depth < 8:
            current_node = nodes[current]
            if node_tag(current_node) in heading_tags:
                heading_context = raw_node_text(current_node)
                if heading_context:
                    break
            heading_context = first_heading_descendant(current)
            if heading_context:
                break
            current = parent_map.get(current)
            depth += 1

        ancestor_parts = []
        ancestor_tags_nearest = []
        current = parent_index
        depth = 0
        while current is not None and 0 <= current < len(nodes) and depth < 12:
            current_node = nodes[current]
            tag = node_tag(current_node)
            class_tokens = runtime_norm(runtime_node_attr(current_node, "class")).split()
            first_class = class_tokens[0] if class_tokens else ""
            if tag:
                ancestor_tags_nearest.append(tag)
                if depth < 5:
                    ancestor_parts.append(
                        f"{tag}.{first_class}" if first_class else tag
                    )
            current = parent_map.get(current)
            depth += 1

        label_text = ""
        current = index
        depth = 0
        while current is not None and 0 <= current < len(nodes) and depth < 8:
            current_node = nodes[current]
            if node_tag(current_node) == "label":
                label_text = raw_node_text(current_node)
                if label_text:
                    break
            current = parent_map.get(current)
            depth += 1

        if not label_text:
            node_id = runtime_norm(runtime_node_attr(node, "id"))
            label_text = label_for_text.get(node_id, "")

        if not label_text:
            label_text = runtime_node_attr(node, "label")

        siblings = children_map.get(parent_index, [])
        sibling_position = 0.0
        if index in siblings and len(siblings) > 1:
            sibling_position = siblings.index(index) / (len(siblings) - 1)

        sibling_indices = []
        if index in siblings:
            position = siblings.index(index)
            sibling_indices = (
                siblings[max(0, position - 2):position]
                + siblings[position + 1:position + 3]
            )

        contexts["parent_text"][index] = parent_text
        contexts["parent_tag"][index] = parent_tag
        contexts["sibling_text"][index] = sibling_text
        contexts["sibling_before"][index] = sibling_before
        contexts["sibling_after"][index] = sibling_after
        contexts["child_text"][index] = child_text
        contexts["context_text"][index] = context_text
        contexts["nearby_text"][index] = nearby_text
        contexts["heading_context"][index] = heading_context
        contexts["ancestor_chain"][index] = " > ".join(ancestor_parts)
        contexts["label_text"][index] = label_text
        contexts["dom_depth"][index] = node_depth_value(node)
        contexts["ancestor_tags_nearest"][index] = ancestor_tags_nearest
        contexts["ancestor_tags_root"][index] = list(reversed(ancestor_tags_nearest))
        contexts["sibling_tags"][index] = [
            node_tag(nodes[sibling])
            for sibling in sibling_indices
            if node_tag(nodes[sibling])
        ]
        contexts["child_tags"][index] = [
            node_tag(nodes[child])
            for child in direct_children
            if node_tag(nodes[child])
        ]
        contexts["sibling_position"][index] = float(sibling_position)
        contexts["sibling_count"][index] = len(siblings)

    return contexts


def xpath_tag_sequence(xpath: Any) -> list[str]:
    """Return an index-free, root-to-leaf tag sequence from an XPath."""
    xpath = normalise_text(xpath)
    if not xpath:
        return []
    sequence = []
    for segment in xpath.split("/"):
        segment = re.sub(r"\[.*?\]", "", segment).strip()
        segment = segment.split("::")[-1].strip()
        match = re.match(r"[a-zA-Z][a-zA-Z0-9_-]*", segment)
        if match:
            sequence.append(match.group(0).lower())
    return sequence


def ordered_sequence_similarity(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_values = list(left or [])
    right_values = list(right or [])
    if not left_values and not right_values:
        return 1.0
    if not left_values or not right_values:
        return 0.0
    return float(SequenceMatcher(None, left_values, right_values).ratio())


def lcs_similarity(left: Iterable[Any], right: Iterable[Any]) -> float:
    """Order-aware similarity normalised by the longer sequence."""
    a, b = list(left or []), list(right or [])
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for left_value in a:
        current = [0]
        for column, right_value in enumerate(b, start=1):
            if left_value == right_value:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    return float(previous[-1] / max(len(a), len(b)))


def sequence_jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set = {str(value) for value in (left or []) if str(value)}
    right_set = {str(value) for value in (right or []) if str(value)}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return float(len(left_set & right_set) / len(left_set | right_set))


def runtime_structural_features(
    old_node: dict[str, Any],
    candidate_node: dict[str, Any],
    old_index: int,
    candidate_index: int,
    old_context: dict[str, dict[int, Any]],
    new_context: dict[str, dict[int, Any]],
) -> dict[str, float]:
    """
    Eight genuinely structural features. Attribute values such as id, class,
    and aria-label are intentionally excluded from this group.
    """
    old_depth = float(old_context["dom_depth"].get(old_index, 0.0))
    new_depth = float(new_context["dom_depth"].get(candidate_index, 0.0))
    depth_denominator = max(old_depth, new_depth, 1.0)

    old_ancestor_nearest = old_context["ancestor_tags_nearest"].get(
        old_index, []
    )
    new_ancestor_nearest = new_context["ancestor_tags_nearest"].get(
        candidate_index, []
    )
    old_ancestor_root = old_context["ancestor_tags_root"].get(old_index, [])
    new_ancestor_root = new_context["ancestor_tags_root"].get(
        candidate_index, []
    )

    old_xpath_tags = xpath_tag_sequence(node_xpath(old_node))
    new_xpath_tags = xpath_tag_sequence(node_xpath(candidate_node))

    old_sibling_position = float(
        old_context["sibling_position"].get(old_index, 0.0)
    )
    new_sibling_position = float(
        new_context["sibling_position"].get(candidate_index, 0.0)
    )

    return {
        "tag_match": float(
            bool(node_tag(old_node))
            and node_tag(old_node) == node_tag(candidate_node)
        ),
        "dom_depth_norm_diff": float(
            np.clip(abs(old_depth - new_depth) / depth_denominator, 0.0, 1.0)
        ),
        "parent_chain_similarity": ordered_sequence_similarity(
            old_ancestor_nearest[:4],
            new_ancestor_nearest[:4],
        ),
        "ancestor_path_similarity": lcs_similarity(
            old_ancestor_root,
            new_ancestor_root,
        ),
        "xpath_token_sim": sequence_jaccard(
            old_xpath_tags,
            new_xpath_tags,
        ),
        "sibling_position_diff": float(
            np.clip(
                abs(old_sibling_position - new_sibling_position),
                0.0,
                1.0,
            )
        ),
        "sibling_context_jaccard": sequence_jaccard(
            old_context["sibling_tags"].get(old_index, []),
            new_context["sibling_tags"].get(candidate_index, []),
        ),
        "child_tag_jaccard": sequence_jaccard(
            old_context["child_tags"].get(old_index, []),
            new_context["child_tags"].get(candidate_index, []),
        ),
    }



def runtime_attribute_features(
    old_node: dict[str, Any],
    candidate_node: dict[str, Any],
    old_index: int,
    candidate_index: int,
    old_context: dict[str, dict[int, Any]],
    new_context: dict[str, dict[int, Any]],
) -> dict[str, Any]:
    old_id = runtime_node_attr(old_node, "id")
    new_id = runtime_node_attr(candidate_node, "id")
    old_data_test_id = runtime_node_attr(old_node, "data-testid")
    new_data_test_id = runtime_node_attr(candidate_node, "data-testid")
    old_name = runtime_node_attr(old_node, "name")
    new_name = runtime_node_attr(candidate_node, "name")
    old_class = runtime_node_attr(old_node, "class")
    new_class = runtime_node_attr(candidate_node, "class")

    id_subset_value = token_subset(old_id, new_id)
    data_subset_value = token_subset(old_data_test_id, new_data_test_id)
    name_subset_value = token_subset(old_name, new_name)

    old_visible = node_visible_value(old_node)
    new_visible = node_visible_value(candidate_node)
    old_enabled = node_enabled_value(old_node)
    new_enabled = node_enabled_value(candidate_node)

    old_depth = float(old_context["dom_depth"].get(old_index, 0.0))
    new_depth = float(new_context["dom_depth"].get(candidate_index, 0.0))

    return {
        "label_seqsim": runtime_seq_sim(
            old_context["label_text"].get(old_index, ""),
            new_context["label_text"].get(candidate_index, ""),
        ),
        "heading_seqsim": runtime_seq_sim(
            old_context["heading_context"].get(old_index, ""),
            new_context["heading_context"].get(candidate_index, ""),
        ),
        "aria_seqsim": runtime_seq_sim(
            runtime_node_attr(old_node, "aria-label"),
            runtime_node_attr(candidate_node, "aria-label"),
        ),
        "placeholder_seqsim": runtime_seq_sim(
            runtime_node_attr(old_node, "placeholder"),
            runtime_node_attr(candidate_node, "placeholder"),
        ),
        "title_seqsim": runtime_seq_sim(
            runtime_node_attr(old_node, "title"),
            runtime_node_attr(candidate_node, "title"),
        ),
        "id_seqsim": runtime_seq_sim(old_id, new_id),
        "name_seqsim": runtime_seq_sim(old_name, new_name),
        "class_seqsim": runtime_seq_sim(old_class, new_class),
        "ancestor_jaccard": runtime_jaccard(
            old_context["ancestor_chain"].get(old_index, ""),
            new_context["ancestor_chain"].get(candidate_index, ""),
        ),
        "parent_seqsim": runtime_seq_sim(
            old_context["parent_text"].get(old_index, ""),
            new_context["parent_text"].get(candidate_index, ""),
        ),
        "sibling_before_jaccard": runtime_jaccard(
            old_context["sibling_text"].get(old_index, ""),
            new_context["sibling_before"].get(candidate_index, ""),
        ),
        "sibling_after_jaccard": runtime_jaccard(
            old_context["sibling_text"].get(old_index, ""),
            new_context["sibling_after"].get(candidate_index, ""),
        ),
        "nearby_jaccard": runtime_jaccard(
            old_context["nearby_text"].get(old_index, ""),
            new_context["nearby_text"].get(candidate_index, ""),
        ),
        "role_match": int(
            bool(runtime_norm(runtime_node_attr(old_node, "role")))
            and runtime_norm(runtime_node_attr(old_node, "role"))
            == runtime_norm(runtime_node_attr(candidate_node, "role"))
        ),
        "type_match": int(
            bool(runtime_norm(runtime_node_attr(old_node, "type")))
            and runtime_norm(runtime_node_attr(old_node, "type"))
            == runtime_norm(runtime_node_attr(candidate_node, "type"))
        ),
        "visible_match": int(old_visible == new_visible),
        "enabled_match": int(old_enabled == new_enabled),
        "id_subset": id_subset_value,
        "dataTestId_subset": data_subset_value,
        "attr_subset_any": int(
            bool(id_subset_value)
            or bool(data_subset_value)
            or bool(name_subset_value)
        ),
        "old_visible": int(old_visible),
        "new_visible": int(new_visible),
        "old_enabled": int(old_enabled),
        "new_enabled": int(new_enabled),
        "old_dom_depth": old_depth,
        "new_dom_depth": new_depth,
    }


ATTRIBUTE_FEATURE_COLUMNS = [
    "label_seqsim", "heading_seqsim", "aria_seqsim",
    "placeholder_seqsim", "title_seqsim", "id_seqsim", "name_seqsim",
    "class_seqsim",
    "ancestor_jaccard", "parent_seqsim",
    "sibling_before_jaccard", "sibling_after_jaccard", "nearby_jaccard",
    "role_match", "type_match", "visible_match", "enabled_match",
    "id_subset", "dataTestId_subset", "attr_subset_any",
]

GEOMETRY_FEATURE_COLUMNS = [
    "geo_iou", "geo_center_dist", "geo_size_ratio",
    "bbox_dx", "bbox_dy",
]

MODEL_LEXICAL_FEATURES = [
    "idx_distance_log",
    "old_text_len", "new_text_len", "len_diff", "text_len_ratio",
    "text_exact", "text_substr", "old_text_empty", "new_text_empty",
    "text_token_overlap", "text_seqsim", "n_candidates",
]

MODEL_SEMANTIC_FEATURES = [
    "semantic_sim",
    "parent_text_sim", "sibling_text_sim", "child_text_sim", "context_text_sim",
]

MODEL_STRUCTURAL_FEATURES = [
    "tag_match",
    "dom_depth_norm_diff",
    "parent_chain_similarity",
    "ancestor_path_similarity",
    "xpath_token_sim",
    "sibling_position_diff",
    "sibling_context_jaccard",
    "child_tag_jaccard",
]

MODEL_ATTRIBUTE_FEATURES = ATTRIBUTE_FEATURE_COLUMNS
MODEL_GEOMETRY_FEATURES = GEOMETRY_FEATURE_COLUMNS

ALL_MODEL_FEATURES = (
    MODEL_LEXICAL_FEATURES
    + MODEL_SEMANTIC_FEATURES
    + MODEL_STRUCTURAL_FEATURES
    + MODEL_ATTRIBUTE_FEATURES
    + MODEL_GEOMETRY_FEATURES
)

assert len(ALL_MODEL_FEATURES) == 50
assert len(ALL_MODEL_FEATURES) == len(set(ALL_MODEL_FEATURES))

# All 50 model features are exported by this notebook.
MODEL_FEATURE_COLUMNS = list(ALL_MODEL_FEATURES)

PAIR_COLUMNS = [
    "qid", "site", "split", "old_idx", "cand_new_idx",
    "candidate_extract_idx", "candidate_kind",
    "source_interactive", "source_text_leaf",
    "source_interactive_descendant", "source_fallback",
    "fallback_score", "fallback_rank",
    "shortlist_score", "shortlist_rank",
    "shortlist_protected", "shortlist_protection_reason",
    "label", "relevance", "idx_distance", "old_text_n", "new_text_n",
    "old_parent_text_n", "new_parent_text_n",
    "old_sibling_text_n", "new_sibling_text_n",
    "old_child_text_n", "new_child_text_n",
    "old_context_text_n", "new_context_text_n",
    "old_text_len", "new_text_len", "len_diff",
    "text_len_ratio", "text_exact", "text_substr", "old_text_empty",
    "new_text_empty", "text_token_overlap", "text_seqsim",
    "idx_distance_log", "n_candidates",
    *MODEL_SEMANTIC_FEATURES,
    *MODEL_STRUCTURAL_FEATURES,
    *ATTRIBUTE_FEATURE_COLUMNS,
    *GEOMETRY_FEATURE_COLUMNS,
    # Retained only as diagnostics; not part of the 50-feature model.
    "bbox_area_ratio",
    "geometry_available",
    "old_bbox_x", "old_bbox_y", "old_bbox_w", "old_bbox_h",
    "new_bbox_x", "new_bbox_y", "new_bbox_w", "new_bbox_h",
    "old_visible", "new_visible", "old_enabled", "new_enabled",
    "old_dom_depth", "new_dom_depth",
]


def token_subset(old_value: Any, new_value: Any) -> int:
    old_tokens, new_tokens = token_set(old_value), token_set(new_value)
    return int(bool(old_tokens) and old_tokens.issubset(new_tokens))


def build_pair_rows(
    query,
    old_node,
    new_nodes,
    old_context,
    new_context,
    candidate_indices,
    candidate_kinds,
    fallback_score_by_index,
    fallback_rank_by_index,
    shortlist_score_by_index,
    shortlist_rank_by_index,
    shortlist_protection_by_index,
):
    true_new_index = int(query["true_new_idx"])
    candidates = [(index, new_nodes[index]) for index in candidate_indices]
    old_index = int(query["old_idx"])
    old_text = node_text(old_node)
    candidate_texts = [node_text(candidate) for _, candidate in candidates]
    text_scores = [
        sequence_similarity(old_text, text)
        for text in candidate_texts
    ]

    rows = []
    n_candidates = len(candidates)

    for extract_index, (
        (candidate_index, candidate),
        candidate_text,
        text_score,
    ) in enumerate(zip(
        candidates,
        candidate_texts,
        text_scores,
    )):
        label = int(candidate_index == true_new_index)
        index_distance = abs(candidate_index - old_index)
        old_length, new_length = len(old_text), len(candidate_text)
        maximum_length = max(old_length, new_length) or 1
        candidate_kind = candidate_kinds.get(candidate_index, "unknown")

        text_exact = int(bool(old_text) and old_text == candidate_text)
        text_substr = int(
            bool(old_text)
            and bool(candidate_text)
            and (
                old_text in candidate_text
                or candidate_text in old_text
            )
        )
        text_token_overlap = int(
            bool(
                runtime_token_set(old_text)
                & runtime_token_set(candidate_text)
            )
        )

        structural_features = runtime_structural_features(
            old_node=old_node,
            candidate_node=candidate,
            old_index=old_index,
            candidate_index=candidate_index,
            old_context=old_context,
            new_context=new_context,
        )
        attr_features = runtime_attribute_features(
            old_node=old_node,
            candidate_node=candidate,
            old_index=old_index,
            candidate_index=candidate_index,
            old_context=old_context,
            new_context=new_context,
        )
        geometry_features = runtime_geometry_features(
            old_node,
            candidate,
        )

        shortlist_rank = int(
            shortlist_rank_by_index.get(candidate_index, 0)
        )

        rows.append({
            "qid": query["qid"],
            "site": query["site"],
            "split": query["split"],
            "old_idx": old_index,
            "cand_new_idx": candidate_index,
            "candidate_extract_idx": extract_index,
            "candidate_kind": candidate_kind,
            "source_interactive": int(candidate_kind == "interactive"),
            "source_text_leaf": int(candidate_kind == "text_leaf"),
            "source_interactive_descendant": int(
                candidate_kind == "interactive_descendant"
            ),
            "source_fallback": int(candidate_kind == "fallback"),
            "fallback_score": float(
                fallback_score_by_index.get(candidate_index, 0.0)
            ),
            "fallback_rank": int(
                fallback_rank_by_index.get(candidate_index, 0)
            ),
            "shortlist_score": float(
                shortlist_score_by_index.get(candidate_index, 0.0)
            ),
            "shortlist_rank": shortlist_rank,
            "shortlist_protected": int(
                candidate_index in shortlist_protection_by_index
            ),
            "shortlist_protection_reason": (
                shortlist_protection_by_index.get(candidate_index, "")
            ),
            "label": label,
            "relevance": 3 if label else 0,
            "idx_distance": index_distance,
            "old_text_n": old_text,
            "new_text_n": candidate_text,
            "old_parent_text_n": old_context["parent_text"].get(old_index, ""),
            "new_parent_text_n": new_context["parent_text"].get(candidate_index, ""),
            "old_sibling_text_n": old_context["sibling_text"].get(old_index, ""),
            "new_sibling_text_n": new_context["sibling_text"].get(candidate_index, ""),
            "old_child_text_n": old_context["child_text"].get(old_index, ""),
            "new_child_text_n": new_context["child_text"].get(candidate_index, ""),
            "old_context_text_n": old_context["context_text"].get(old_index, ""),
            "new_context_text_n": new_context["context_text"].get(candidate_index, ""),
            "old_text_len": old_length,
            "new_text_len": new_length,
            "len_diff": abs(old_length - new_length),
            "text_len_ratio": (
                min(old_length, new_length) / maximum_length
            ),
            "text_exact": text_exact,
            "text_substr": text_substr,
            "old_text_empty": int(not old_text),
            "new_text_empty": int(not candidate_text),
            # Match runtime: binary shared-token signal.
            "text_token_overlap": text_token_overlap,
            "text_seqsim": text_score,
            "idx_distance_log": float(np.log1p(index_distance)),
            "n_candidates": n_candidates,
            **structural_features,
            **attr_features,
            **geometry_features,
        })

    return rows




SEMANTIC_TEXT_PAIRS = {
    "semantic_sim": ("old_text_n", "new_text_n"),
    "parent_text_sim": ("old_parent_text_n", "new_parent_text_n"),
    "sibling_text_sim": ("old_sibling_text_n", "new_sibling_text_n"),
    "child_text_sim": ("old_child_text_n", "new_child_text_n"),
    "context_text_sim": ("old_context_text_n", "new_context_text_n"),
}


class RepathFeaturePipeline:
    """Direct candidate retrieval and exact ordered 50-feature generation."""

    def __init__(
        self,
        semantic_model: SentenceTransformer,
        feature_names: list[str],
        config: dict[str, Any],
    ) -> None:
        self.semantic_model = semantic_model
        self.feature_names = [str(name) for name in feature_names]
        self.config = config

        if self.feature_names != ALL_MODEL_FEATURES:
            raise ValueError(
                "REPATH feature order does not match the exact Notebook 01 "
                "50-feature schema."
            )

        self.page_pair: PagePair | None = None
        self.old_nodes: list[dict[str, Any]] = []
        self.new_nodes: list[dict[str, Any]] = []
        self.old_parent_map: dict[int, int | None] = {}
        self.new_parent_map: dict[int, int | None] = {}
        self.old_context: dict[str, dict[int, Any]] = {}
        self.new_context: dict[str, dict[int, Any]] = {}
        self.base_indices: list[int] = []
        self.base_kinds: dict[int, str] = {}
        self.text_to_embedding: dict[str, np.ndarray] = {}

    def prepare(self, page_pair: PagePair) -> None:
        self.page_pair = page_pair

        # The benchmark ground truth and Notebook 01 both use list positions
        # from domNodeInfo.json. Preserve that exact order.
        self.old_nodes = [
            node.raw for node in sorted(
                page_pair.old_page.nodes,
                key=lambda item: item.list_index,
            )
        ]
        self.new_nodes = [
            node.raw for node in sorted(
                page_pair.new_page.nodes,
                key=lambda item: item.list_index,
            )
        ]

        self.old_parent_map = build_parent_index_map(self.old_nodes)
        self.new_parent_map = build_parent_index_map(self.new_nodes)
        self.old_context = build_runtime_context_maps(
            self.old_nodes,
            self.old_parent_map,
        )
        self.new_context = build_runtime_context_maps(
            self.new_nodes,
            self.new_parent_map,
        )
        (
            self.base_indices,
            self.base_kinds,
            _,
        ) = extract_base_candidate_indices(
            self.new_nodes,
            self.new_parent_map,
        )

        # Cache is page-local. The empty embedding is generated exactly as in
        # Notebook 01 and acts as the zero-text representation.
        self.text_to_embedding = {}
        self._cache_texts([""])

    def _cache_texts(self, texts: Iterable[Any]) -> None:
        normalized = [normalise_text(value) for value in texts]
        missing = [
            value
            for value in dict.fromkeys(normalized)
            if value not in self.text_to_embedding
        ]
        for start in range(0, len(missing), SBERT_BATCH_SIZE):
            batch = missing[start:start + SBERT_BATCH_SIZE]
            if not batch:
                continue
            embeddings = self.semantic_model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for value, embedding in zip(batch, embeddings):
                self.text_to_embedding[value] = np.asarray(
                    embedding,
                    dtype=np.float32,
                )

    def _add_semantic_features(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        semantic_texts: list[Any] = []
        for old_column, new_column in SEMANTIC_TEXT_PAIRS.values():
            semantic_texts.extend(row.get(old_column, "") for row in rows)
            semantic_texts.extend(row.get(new_column, "") for row in rows)

        self._cache_texts(semantic_texts)
        empty = self.text_to_embedding[""]

        for row in rows:
            for feature, (
                old_column,
                new_column,
            ) in SEMANTIC_TEXT_PAIRS.items():
                old_value = normalise_text(row.get(old_column, ""))
                new_value = normalise_text(row.get(new_column, ""))
                old_embedding = self.text_to_embedding.get(old_value, empty)
                new_embedding = self.text_to_embedding.get(new_value, empty)
                row[feature] = float(
                    np.sum(old_embedding * new_embedding)
                )

    def build_query(
        self,
        old_node_id: int,
    ) -> tuple[list[int], np.ndarray]:
        if self.page_pair is None:
            raise RuntimeError(
                "REPATH feature pipeline has not been prepared"
            )

        # matchMap keys are domNodeInfo.json list positions, not node IDs.
        old_index = int(old_node_id)
        if old_index < 0 or old_index >= len(self.old_nodes):
            return [], np.empty(
                (0, len(self.feature_names)),
                dtype=np.float32,
            )

        old_node = self.old_nodes[old_index]

        (
            fallback_indices,
            fallback_score_by_index,
            fallback_rank_by_index,
            _,
        ) = select_query_fallback_candidates(
            old_node=old_node,
            old_index=old_index,
            new_nodes=self.new_nodes,
            excluded_indices=self.base_indices,
        )

        (
            full_candidate_indices,
            full_candidate_kinds,
            _,
        ) = merge_query_candidates(
            base_indices=self.base_indices,
            base_kinds=self.base_kinds,
            fallback_indices=fallback_indices,
            cap=MAX_CANDIDATES,
        )

        (
            ranked_candidate_indices,
            shortlist_score_by_index,
            shortlist_rank_by_index,
            shortlist_protection_by_index,
        ) = rank_query_candidate_pool(
            old_node=old_node,
            old_index=old_index,
            new_nodes=self.new_nodes,
            candidate_indices=full_candidate_indices,
        )

        candidate_indices = apply_query_shortlist(
            full_candidate_indices=full_candidate_indices,
            ranked_candidate_indices=ranked_candidate_indices,
            protection_by_index=shortlist_protection_by_index,
            top_k=QUERY_SHORTLIST_TOP_K,
        )

        if not candidate_indices:
            return [], np.empty(
                (0, len(self.feature_names)),
                dtype=np.float32,
            )

        candidate_kinds = {
            index: full_candidate_kinds[index]
            for index in candidate_indices
        }

        # true_new_idx is intentionally unavailable to inference. The value -1
        # only makes the notebook row builder assign label=0; labels are not
        # part of the model feature vector.
        query = {
            "qid": f"{self.page_pair.site}::{old_index}",
            "site": self.page_pair.site,
            "split": "benchmark",
            "old_idx": old_index,
            "true_new_idx": -1,
        }

        rows = build_pair_rows(
            query=query,
            old_node=old_node,
            new_nodes=self.new_nodes,
            old_context=self.old_context,
            new_context=self.new_context,
            candidate_indices=candidate_indices,
            candidate_kinds=candidate_kinds,
            fallback_score_by_index=fallback_score_by_index,
            fallback_rank_by_index=fallback_rank_by_index,
            shortlist_score_by_index=shortlist_score_by_index,
            shortlist_rank_by_index=shortlist_rank_by_index,
            shortlist_protection_by_index=shortlist_protection_by_index,
        )
        self._add_semantic_features(rows)

        return self._feature_matrix(rows, candidate_indices)

    def _feature_matrix(
        self, rows: list[dict[str, Any]], candidate_indices: list[int],
    ) -> tuple[list[int], np.ndarray]:
        matrix = np.asarray(
            [
                [float(row[feature]) for feature in self.feature_names]
                for row in rows
            ],
            dtype=np.float32,
        )
        if matrix.shape != (
            len(candidate_indices),
            len(self.feature_names),
        ):
            raise ValueError(
                "Unexpected REPATH feature matrix shape: "
                f"{matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(
                "REPATH generated NaN or infinite feature values"
            )

        return [int(index) for index in candidate_indices], matrix
