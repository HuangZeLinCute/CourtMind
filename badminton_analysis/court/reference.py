import cv2
import numpy as np


BADMINTON_COURT_WIDTH = 6.1
BADMINTON_COURT_LENGTH = 13.4
BADMINTON_SINGLES_MARGIN = 0.46
BADMINTON_BACK_SERVICE_OFFSET = 0.76
BADMINTON_SERVICE_LINE_FROM_NET = 1.98


class BadmintonCourtReference:
    """Project a standard badminton court model into image space and score line support."""

    def __init__(self):
        width = BADMINTON_COURT_WIDTH
        length = BADMINTON_COURT_LENGTH
        singles_margin = BADMINTON_SINGLES_MARGIN
        net_y = length / 2.0
        service_top = net_y - BADMINTON_SERVICE_LINE_FROM_NET
        service_bottom = net_y + BADMINTON_SERVICE_LINE_FROM_NET
        back_service_top = BADMINTON_BACK_SERVICE_OFFSET
        back_service_bottom = length - BADMINTON_BACK_SERVICE_OFFSET
        center_x = width / 2.0

        self.court_corners = np.array(
            [[0, 0], [width, 0], [width, length], [0, length]],
            dtype=np.float32,
        )
        self.lines = [
            ((0, 0), (width, 0), 1.35),
            ((width, 0), (width, length), 1.35),
            ((width, length), (0, length), 1.35),
            ((0, length), (0, 0), 1.35),
            ((singles_margin, 0), (singles_margin, length), 0.95),
            ((width - singles_margin, 0), (width - singles_margin, length), 0.95),
            ((0, back_service_top), (width, back_service_top), 1.0),
            ((0, back_service_bottom), (width, back_service_bottom), 1.0),
            ((0, service_top), (width, service_top), 1.15),
            ((0, service_bottom), (width, service_bottom), 1.15),
            ((center_x, 0), (center_x, service_top), 0.8),
            ((center_x, service_bottom), (center_x, length), 0.8),
            ((0, net_y), (width, net_y), 0.35),
        ]

    def prepare_line_support(self, line_mask, image_shape):
        if line_mask is None:
            return None

        height, width = image_shape[:2]
        support_mask = (line_mask > 0).astype(np.uint8) * 255
        if support_mask.shape[:2] != (height, width):
            support_mask = cv2.resize(support_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        distance_map = cv2.distanceTransform(255 - support_mask, cv2.DIST_L2, 3)
        return {
            "distance_map": distance_map,
            "tolerance": max(5.0, min(width, height) * 0.013),
        }

    def score_line_support(self, image_corners, line_support, image_shape):
        if line_support is None:
            return 0.0, self._empty_details()

        height, width = image_shape[:2]
        corners = np.array(image_corners, dtype=np.float32)
        if corners.shape != (4, 2):
            return 0.0, self._empty_details()

        matrix = cv2.getPerspectiveTransform(self.court_corners, corners)
        distance_map = line_support["distance_map"]
        tolerance = line_support["tolerance"]

        weighted_score = 0.0
        total_weight = 0.0
        coverage_values = []
        supported_lines = 0

        for start, end, weight in self.lines:
            start_image, end_image = self._project_line(matrix, start, end)
            samples = self._sample_line(start_image, end_image, width, height)
            if samples is None:
                continue

            distances = distance_map[samples[:, 1], samples[:, 0]]
            line_scores = 1.0 - np.clip(distances / tolerance, 0.0, 1.0)
            line_score = float(np.mean(line_scores))
            coverage = float(np.mean(distances <= tolerance))
            weighted_score += weight * (0.68 * line_score + 0.32 * coverage)
            total_weight += weight
            coverage_values.append(coverage)
            if coverage >= 0.42:
                supported_lines += 1

        if total_weight <= 0:
            return 0.0, self._empty_details(tolerance)

        score = float(np.clip(weighted_score / total_weight, 0.0, 1.0))
        details = {
            "reference_score": round(score, 4),
            "reference_coverage": round(float(np.mean(coverage_values)) if coverage_values else 0.0, 4),
            "reference_supported_lines": int(supported_lines),
            "reference_tolerance_px": round(float(tolerance), 2),
        }
        return score, details

    def project_lines(self, image_corners):
        corners = np.array(image_corners, dtype=np.float32)
        if corners.shape != (4, 2):
            return []

        matrix = cv2.getPerspectiveTransform(self.court_corners, corners)
        projected_lines = []
        for start, end, weight in self.lines:
            start_image, end_image = self._project_line(matrix, start, end)
            projected_lines.append((start_image, end_image, weight))
        return projected_lines
    def _project_line(self, matrix, start, end):
        points = np.array([[start, end]], dtype=np.float32)
        projected = cv2.perspectiveTransform(points, matrix)[0]
        return projected[0], projected[1]

    def _sample_line(self, start, end, width, height):
        length = float(np.linalg.norm(end - start))
        if length < 1.0:
            return None

        sample_count = max(18, int(length / 5.0))
        xs = np.linspace(float(start[0]), float(end[0]), sample_count)
        ys = np.linspace(float(start[1]), float(end[1]), sample_count)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if np.mean(valid) < 0.35:
            return None

        xs = np.clip(np.rint(xs[valid]).astype(np.int32), 0, width - 1)
        ys = np.clip(np.rint(ys[valid]).astype(np.int32), 0, height - 1)
        return np.column_stack([xs, ys])

    def _empty_details(self, tolerance=0.0):
        return {
            "reference_score": 0.0,
            "reference_coverage": 0.0,
            "reference_supported_lines": 0,
            "reference_tolerance_px": round(float(tolerance), 2),
        }
