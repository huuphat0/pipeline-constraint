"""Hàm chi phí ReKep từ DSL đã kiểm tra; giá trị <= 0 nghĩa là thỏa."""

import numpy as np


def _point(point_id, end_effector, keypoints):
    value = end_effector if point_id == -1 else keypoints[point_id]
    if value is None:
        raise ValueError(f"Thiếu tọa độ cho keypoint {point_id}")
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"Tọa độ keypoint {point_id} không hợp lệ")
    return value


def measure(spec, end_effector, keypoints):
    ids, op = spec["ids"], spec["op"]
    points = [_point(i, end_effector, keypoints) for i in ids]
    if op in {"distance_le", "distance_ge"}:
        return float(np.linalg.norm(points[0] - points[1]))
    if op in {"orientation_change_le", "orientation_change_ge"}:
        current = points[1] - points[0]
        reference = np.asarray(spec["reference_vector_mm"], dtype=float)
        denom = np.linalg.norm(current) * np.linalg.norm(reference)
        if denom < 1e-8:
            raise ValueError("Cặp điểm trục bị trùng nhau")
        cosine = float(np.clip((current @ reference) / denom, -1, 1))
        return float(np.degrees(np.arccos(cosine)))
    if op in {"height_above_table_le", "height_above_table_ge"}:
        normal = np.asarray(spec["table_normal"], dtype=float)
        origin = np.asarray(spec["table_point_mm"], dtype=float)
        return float((points[0] - origin) @ normal / np.linalg.norm(normal))
    if op == "distance_to_reference_le":
        return float(np.linalg.norm(points[0] - np.asarray(spec["reference_position_mm"], dtype=float)))
    raise ValueError(f"Phép toán chưa hỗ trợ: {op}")


def evaluate(spec, end_effector, keypoints):
    """Trả cost có dấu; <=0 nghĩa là quan hệ thỏa."""
    value = measure(spec, end_effector, keypoints)
    if spec["op"].endswith("_ge"):
        return float(spec["threshold"] - value)
    return float(value - spec["threshold"])
