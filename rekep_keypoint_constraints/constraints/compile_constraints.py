"""Bước 8: lọc ứng viên, hiệu chuẩn ngưỡng và biên dịch Python an toàn."""

import argparse
import json
import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # gốc repo, để import rekep_keypoint_constraints.*
from rekep_keypoint_constraints.common.constraint_runtime import measure


OP_ARITY = {"distance_le": 2, "distance_ge": 2,
            "orientation_change_le": 2, "orientation_change_ge": 2,
            "height_above_table_le": 1, "height_above_table_ge": 1,
            "distance_to_reference_le": 1}

FALLBACK = {
    "grasp": [
        ("subgoal", "distance_le", ["end_effector", "bottle_center"], "Đưa đầu công tác tới vùng thân chai để nắm.")],
    "transport_to_target": [
        ("path", "orientation_change_le", ["axis_positive", "axis_negative"], "Giữ tư thế chai khi mang tới cốc."),
        ("subgoal", "distance_le", ["axis_positive", "cup_top"], "Đưa đầu chai tới vùng cốc trước khi rót.")],
    "tilt_to_pour": [
        ("subgoal", "orientation_change_ge", ["axis_positive", "axis_negative"], "Nghiêng chai ở cuối stage để rót."),
        ("subgoal", "distance_le", ["axis_positive", "cup_top"], "Đưa đầu chai tới gần miệng cốc khi rót.")],
    "restore_upright": [
        ("subgoal", "orientation_change_le", ["axis_positive", "axis_negative"], "Dựng chai gần hướng ban đầu để ngừng rót.")],
    "return_place": [
        ("path", "orientation_change_le", ["axis_positive", "axis_negative"], "Giữ chai gần thẳng trong lúc mang về."),
        ("subgoal", "distance_to_reference_le", ["axis_negative"], "Đặt chai gần vị trí ban đầu.")]
}


def fallback_ids(names, stage, tracks):
    bottle = stage["manipulated_object_id"]
    points = tracks["points"]
    def find(object_id, region, source=None):
        matches = [p["id"] for p in points if p["object_id"] == object_id
                   and p["region"] == region and (source is None or p.get("source") == source)]
        if len(matches) != 1:
            raise ValueError(f"Không xác định duy nhất điểm {region} cho vật {object_id}: {matches}")
        return matches[0]
    other_ids = sorted({p["object_id"] for p in points} - {bottle})
    if len(other_ids) != 1:
        raise ValueError("Cần đúng một vật đích khác vật thao tác trong ca thử này")
    aliases = {"end_effector": -1,
               "bottle_center": find(bottle, "center"),
               "axis_positive": find(bottle, "axis_positive", "pose3d_axis"),
               "axis_negative": find(bottle, "axis_negative", "pose3d_axis"),
               "cup_top": find(other_ids[0], "top")}
    return [aliases[name] for name in names]


def semantic_error(candidate, stage, tracks):
    op, ids, phase = candidate["op"], candidate["ids"], candidate["phase"]
    if op not in OP_ARITY or len(ids) != OP_ARITY[op] or len(set(ids)) != len(ids):
        return "sai phép toán hoặc số ID"
    if any(not isinstance(i, int) or i < -1 or i >= len(tracks["points"]) for i in ids):
        return "ID không tồn tại"
    if -1 in ids and not (stage["name"] == "grasp" and phase == "subgoal" and op == "distance_le"):
        return "end_effector chỉ áp dụng cho subgoal grasp"
    allowed = stage["path_eligible_ids"] if phase == "path" else stage["subgoal_eligible_ids"]
    if any(i >= 0 and i not in allowed for i in ids):
        return "keypoint thiếu dữ liệu trong phase"
    objects = [tracks["points"][i]["object_id"] for i in ids if i >= 0]
    bottle = stage["manipulated_object_id"]
    if op.startswith("orientation_change") and (len(set(objects)) != 1 or objects[0] != bottle):
        return "hướng phải dùng hai điểm cùng vật đang thao tác"
    if op in {"distance_le", "distance_ge"} and -1 not in ids and len(set(objects)) != 2:
        return "khoảng cách giữa hai điểm cùng vật cứng không biểu diễn mục tiêu task"
    if op.startswith("height_above_table") and objects[0] != bottle:
        return "độ cao phải đo trên vật đang thao tác"
    if op == "distance_to_reference_le" and objects[0] != bottle:
        return "vị trí tham chiếu phải thuộc vật đang thao tác"
    name = stage["name"]
    stage_ops = {
        "grasp": {("subgoal", "distance_le")},
        "transport_to_target": {("path", "orientation_change_le"),
                                ("path", "height_above_table_ge"), ("subgoal", "distance_le")},
        "tilt_to_pour": {("subgoal", "orientation_change_ge"), ("subgoal", "distance_le")},
        "restore_upright": {("subgoal", "orientation_change_le")},
        "return_place": {("path", "orientation_change_le"),
                         ("subgoal", "distance_to_reference_le"),
                         ("subgoal", "height_above_table_le")}
    }
    if (phase, op) not in stage_ops.get(name, set()):
        return "phép toán/phase không phù hợp mục tiêu stage"
    if name == "grasp" and (ids[0] != -1 or objects[0] != bottle):
        return "grasp phải nối end_effector với điểm trên chai"
    if op.startswith("orientation_change"):
        pair = [tracks["points"][i] for i in ids]
        if {p["region"] for p in pair} != {"axis_positive", "axis_negative"} or any(
                p.get("source") != "pose3d_axis" for p in pair):
            # Điểm bề mặt có thể trượt khi chai trong suốt/bị che; chỉ dùng
            # cặp trục có khoảng cách cứng đã kiểm tra trong ca thử này.
            return "cặp điểm bề mặt chưa đủ ổn định để đo góc trên ca thử này"
    return None


def setup_spec(candidate, stage, tracks, series, source):
    ids, op = candidate["ids"], candidate["op"]
    spec = {"stage": stage["stage"], "stage_name": stage["name"],
            "phase": candidate["phase"], "op": op, "ids": ids,
            "reason": candidate["reason"], "visual_evidence": candidate.get("visual_evidence", ""),
            "source": source, "necessity_status": "hypothesis",
            "instruction_review_pending": True,
            "evidence_quality": "pose3d_derived" if any(i >= 0 and tracks["points"][i].get("source") == "pose3d_axis"
                                                       for i in ids) else "observed_tracks"}
    if -1 in ids:
        spec["reason"] = "Đưa đầu công tác tới điểm nắm trên chai; chưa có phép đo đầu công tác trong video."
        spec["visual_evidence"] = "Không có tọa độ end_effector 3D trong video; mô tả tiếp xúc do VLM đưa ra chưa được kiểm chứng."
    if op.startswith("orientation_change"):
        first = tracks["frames"][0]["points"]
        spec["reference_vector_mm"] = np.subtract(first[ids[1]]["xyz_mm"],
                                                    first[ids[0]]["xyz_mm"]).tolist()
    if op.startswith("height_above_table"):
        plane = series["table_plane"]
        spec["table_normal"] = plane["normal"]
        spec["table_point_mm"] = plane["point"]
    if op == "distance_to_reference_le":
        spec["reference_position_mm"] = tracks["frames"][0]["points"][ids[0]]["xyz_mm"]
    spec["unit"] = "deg" if op.startswith("orientation_change") else "mm"
    return spec


def values_on_demo(spec, stage, tracks):
    start, end = stage["start_frame"], stage["end_frame"]
    frames = range(start, end + 1) if spec["phase"] == "path" else range(max(start, end - 4), end + 1)
    values = []
    for frame in frames:
        records = tracks["frames"][frame]["points"]
        if any(i == -1 or records[i]["xyz_mm"] is None for i in spec["ids"]):
            continue
        xyz = np.full((len(records), 3), np.nan)
        for i in spec["ids"]:
            xyz[i] = records[i]["xyz_mm"]
        values.append(measure(spec, None, xyz))
    return values, len(frames)


def calibrate(spec, stage, tracks):
    if -1 in spec["ids"]:
        spec.update({"threshold": 30.0, "threshold_source": "preset_grasp_30mm",
                     "measurement_coverage": 0.0, "demo_measured": False})
        return None
    values, count = values_on_demo(spec, stage, tracks)
    coverage = len(values) / count
    required = 0.9 if spec["phase"] == "path" else 0.6
    if coverage < required:
        return f"thiếu số đo: coverage={coverage:.2f} < {required:.2f}"
    if not np.isfinite(values).all():
        return "số đo không hữu hạn"
    op = spec["op"]
    margin = 5.0 if spec["unit"] == "deg" else 15.0
    if spec["phase"] == "path":
        estimate = float(np.percentile(values, 95 if op.endswith("_le") else 5))
    else:
        estimate = float(np.median(values))
    threshold = estimate + margin if op.endswith("_le") else estimate - margin
    if op.endswith("_ge"):
        threshold = max(0, threshold)
    if not math.isfinite(threshold):
        return "ngưỡng không hữu hạn"
    spec.update({"threshold": round(threshold, 2),
                 "threshold_source": "demo_quantile_plus_margin",
                 "measurement_coverage": round(coverage, 3),
                 "demo_measured": True,
                 "demo_value_min": round(float(min(values)), 2),
                 "demo_value_max": round(float(max(values)), 2),
                 "demo_value_median": round(float(np.median(values)), 2)})
    return None


def compile_all(raw, stages, tracks, series, out_dir):
    by_id = {x["stage"]: x for x in raw["stages"]}
    accepted, rejected = [], []
    for stage in stages["stages"]:
        if not stage["constraint_generation"]:
            continue
        selected = []
        for candidate in by_id[stage["stage"]]["candidates"]:
            error = semantic_error(candidate, stage, tracks)
            if error:
                rejected.append({"stage": stage["stage"], "candidate": candidate, "reason": error})
                continue
            selected.append((candidate, "vlm_candidate"))
        for phase, op, names, reason in FALLBACK.get(stage["name"], []):
            if any(c["phase"] == phase and c["op"] == op for c, _ in selected):
                continue
            ids = fallback_ids(names, stage, tracks)
            selected.append(({"phase": phase, "op": op, "ids": ids, "reason": reason,
                              "visual_evidence": "stage skill + keypoint/pose 3D của video mẫu"},
                             "geometry_policy_fallback"))
        for candidate, source in selected:
            error = semantic_error(candidate, stage, tracks)
            if error:
                rejected.append({"stage": stage["stage"], "candidate": candidate, "reason": error,
                                 "source": source})
                continue
            spec = setup_spec(candidate, stage, tracks, series, source)
            spec["instruction_review_pending"] = raw.get("instruction_review_pending", True)
            error = calibrate(spec, stage, tracks)
            if error:
                rejected.append({"stage": stage["stage"], "candidate": candidate, "reason": error,
                                 "source": source})
                continue
            spec["id"] = len(accepted)
            accepted.append(spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "constraints.json").write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Generated from validated DSL; do not edit by hand.",
             "from rekep_keypoint_constraints.common.constraint_runtime import evaluate", "",
             "SPECS = ["]
    lines += [f"    {spec!r}," for spec in accepted]
    lines += ["]", ""]
    counters = {}
    for spec in accepted:
        key = (spec["stage"], spec["phase"])
        counters[key] = counters.get(key, 0) + 1
        fn = f"stage{spec['stage']}_{spec['phase']}_constraint{counters[key]}"
        lines += [f"def {fn}(end_effector, keypoints):",
                  f"    return evaluate(SPECS[{spec['id']}], end_effector, keypoints)", ""]
    (out_dir / "constraints.py").write_text("\n".join(lines), encoding="utf-8")
    return accepted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parents[1]
    base = local / "artifacts/demo"
    parser.add_argument("--raw", default=str(base / "constraints/constraints_raw.json"))
    parser.add_argument("--stages", default=str(base / "stages/stages.json"))
    parser.add_argument("--tracks", default=str(base / "keypoints/keypoints_tracks.json"))
    parser.add_argument("--series", default=str(root / "out/json/demo_series.json"))
    parser.add_argument("--out", default=str(base / "constraints"))
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    accepted, rejected = compile_all(read(args.raw), read(args.stages), read(args.tracks), read(args.series), args.out)
    print(json.dumps({"accepted": [(x["stage"], x["phase"], x["op"], x["ids"], x["threshold"], x["source"])
                                   for x in accepted], "rejected_count": len(rejected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
