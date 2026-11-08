"""Python implementation of the VISTA visual-matching baseline.

Original project: https://github.com/saltlab/vista
Upstream license: Apache-2.0
Adapted for the REPATH evaluation framework, 2026.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ...domain.models import DomNode, MatchResult, PagePair, ScoredCandidate
from ..base import ElementMatcher

UPSTREAM_REPOSITORY = "https://github.com/saltlab/vista"
UPSTREAM_COMMIT = "7bf2a588897637e0e1c9a2cc33e4d372a62bcaf0"


class VistaMatcher(ElementMatcher):
    """VISTA visual matching baseline pinned to the authors' implementation."""

    NNDR_RATIO = 1.0

    def __init__(self) -> None:
        self._old_image: np.ndarray | None = None
        self._new_image: np.ndarray | None = None
        self._new_gray: np.ndarray | None = None
        self._scene_sift_descriptors: np.ndarray | None = None
        self._scene_fast_descriptors: np.ndarray | None = None
        self._prepare_status: str | None = None

    @property
    def name(self) -> str:
        return "VISTA"

    def prepare(self, page_pair: PagePair) -> None:
        """Load and preprocess the page-level visual data."""
        super().prepare(page_pair)
        self._clear_cache()

        old_path = page_pair.old_page.full_screen_path
        new_path = page_pair.new_page.full_screen_path
        if old_path is None or new_path is None:
            self._prepare_status = "missing_screenshot"
            return

        self._old_image = cv2.imread(str(old_path), cv2.IMREAD_COLOR)
        self._new_image = cv2.imread(str(new_path), cv2.IMREAD_COLOR)
        if self._old_image is None or self._new_image is None:
            self._prepare_status = "unreadable_screenshot"
            return

        try:
            self._new_gray = cv2.cvtColor(self._new_image, cv2.COLOR_BGR2GRAY)
            self._scene_sift_descriptors = self._sift_scene_descriptors(self._new_gray)
            self._scene_fast_descriptors = self._fast_scene_descriptors(self._new_gray)
        except cv2.error:
            self._prepare_status = "feature_gate_failed"

    def _clear_cache(self) -> None:
        self._old_image = None
        self._new_image = None
        self._new_gray = None
        self._scene_sift_descriptors = None
        self._scene_fast_descriptors = None
        self._prepare_status = None

    def close(self) -> None:
        self._clear_cache()

    @staticmethod
    def _crop_reference(image: np.ndarray, node: DomNode) -> np.ndarray | None:
        height, width = image.shape[:2]
        x, y, w, h = node.bbox.x, node.bbox.y, node.bbox.width, node.bbox.height
        min_offset_x = min(x, width - w - x)
        min_offset_y = min(y, height - h - y)
        offset = min(min_offset_x, min_offset_y) // 5
        x0, y0 = x - offset, y - offset
        crop_w, crop_h = 2 * offset + w, 2 * offset + h
        if crop_w <= 0 or crop_h <= 0 or x0 < 0 or y0 < 0:
            return None
        x1, y1 = x0 + crop_w, y0 + crop_h
        if x1 > width or y1 > height:
            return None
        crop = image[y0:y1, x0:x1]
        return crop if crop.size else None

    @staticmethod
    def _flann_good_matches(
        desc1: np.ndarray | None,
        desc2: np.ndarray | None,
    ) -> int:
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) < 2:
            return 0
        desc1 = np.asarray(desc1, dtype=np.float32)
        desc2 = np.asarray(desc2, dtype=np.float32)
        matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
        pairs = matcher.knnMatch(desc1, desc2, k=2)
        good = 0
        for pair in pairs:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance <= second.distance * VistaMatcher.NNDR_RATIO:
                good += 1
        return good

    @staticmethod
    def _sift_scene_descriptors(scene_gray: np.ndarray) -> np.ndarray | None:
        sift = cv2.SIFT_create()
        _keypoints, descriptors = sift.detectAndCompute(scene_gray, None)
        return descriptors

    @staticmethod
    def _fast_scene_descriptors(scene_gray: np.ndarray) -> np.ndarray | None:
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(scene_gray, None)
        sift = cv2.SIFT_create()
        _keypoints, descriptors = sift.compute(scene_gray, keypoints)
        return descriptors

    @classmethod
    def _sift_gate_cached(
        cls,
        template_gray: np.ndarray,
        scene_descriptors: np.ndarray | None,
    ) -> bool:
        sift = cv2.SIFT_create()
        keypoints, template_descriptors = sift.detectAndCompute(template_gray, None)
        if not keypoints:
            return False
        good = cls._flann_good_matches(template_descriptors, scene_descriptors)
        threshold = int(len(keypoints) * 0.3)
        return good > threshold

    @classmethod
    def _fast_gate_cached(
        cls,
        template_gray: np.ndarray,
        scene_descriptors: np.ndarray | None,
    ) -> bool:
        fast = cv2.FastFeatureDetector_create()
        keypoints = fast.detect(template_gray, None)
        if not keypoints:
            return False
        sift = cv2.SIFT_create()
        keypoints, template_descriptors = sift.compute(template_gray, keypoints)
        good = cls._flann_good_matches(template_descriptors, scene_descriptors)
        threshold = int(len(keypoints) * 0.3)
        return good > threshold

    @classmethod
    def _sift_gate(cls, template_gray: np.ndarray, scene_gray: np.ndarray) -> bool:
        return cls._sift_gate_cached(
            template_gray,
            cls._sift_scene_descriptors(scene_gray),
        )

    @classmethod
    def _fast_gate(cls, template_gray: np.ndarray, scene_gray: np.ndarray) -> bool:
        return cls._fast_gate_cached(
            template_gray,
            cls._fast_scene_descriptors(scene_gray),
        )

    @classmethod
    def _feature_gate(cls, template: np.ndarray, scene: np.ndarray) -> bool:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        try:
            # The authors' Java implementation executes both detectors before OR-ing
            # their results; do not short-circuit, because runtime is benchmarked.
            sift_present = cls._sift_gate(template_gray, scene_gray)
            fast_present = cls._fast_gate(template_gray, scene_gray)
            return sift_present or fast_present
        except cv2.error:
            return False

    def _feature_gate_from_cache(self, template: np.ndarray) -> bool:
        if self._new_gray is None:
            return False
        try:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            sift_present = self._sift_gate_cached(
                template_gray,
                self._scene_sift_descriptors,
            )
            fast_present = self._fast_gate_cached(
                template_gray,
                self._scene_fast_descriptors,
            )
            return sift_present or fast_present
        except cv2.error:
            return False

    @staticmethod
    def _template_center(template: np.ndarray, scene: np.ndarray) -> tuple[float, float] | None:
        th, tw = template.shape[:2]
        sh, sw = scene.shape[:2]
        if th <= 0 or tw <= 0 or th > sh or tw > sw:
            return None
        result = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
        normalized = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
        _, _, _, max_location = cv2.minMaxLoc(normalized)
        return max_location[0] + tw / 2.0, max_location[1] + th / 2.0

    @staticmethod
    def _smallest_containing_node(
        point: tuple[float, float],
        nodes: list[DomNode],
    ) -> DomNode | None:
        if not nodes:
            return None
        px, py = point
        # Preserve the original Java initialization: first node height squared.
        size = nodes[0].bbox.height * nodes[0].bbox.height
        selected: DomNode | None = None
        for node in nodes:
            area = node.bbox.height * node.bbox.width
            if node.bbox.strictly_contains(px, py) and size > area:
                selected = node
                size = area
        return selected

    def match(self, old_node_id: int) -> MatchResult:
        start = time.perf_counter_ns()
        source = self.pair.old_page.resolve_node(old_node_id)
        if source is None:
            return MatchResult((), time.perf_counter_ns() - start, "missing_old_node")
        if self._prepare_status is not None:
            return MatchResult((), time.perf_counter_ns() - start, self._prepare_status)
        if self._old_image is None or self._new_image is None:
            return MatchResult((), time.perf_counter_ns() - start, "unreadable_screenshot")
        template = self._crop_reference(self._old_image, source)
        if template is None:
            return MatchResult((), time.perf_counter_ns() - start, "invalid_crop")
        if not self._feature_gate_from_cache(template):
            return MatchResult((), time.perf_counter_ns() - start, "feature_gate_failed")
        center = self._template_center(template, self._new_image)
        if center is None:
            return MatchResult((), time.perf_counter_ns() - start, "template_match_failed")
        node = self._smallest_containing_node(center, self.pair.new_page.nodes)
        if node is None:
            return MatchResult((), time.perf_counter_ns() - start, "no_containing_node")
        elapsed = time.perf_counter_ns() - start
        return MatchResult((ScoredCandidate(node.node_id, 1.0),), elapsed)


__all__ = ["UPSTREAM_COMMIT", "UPSTREAM_REPOSITORY", "VistaMatcher"]
