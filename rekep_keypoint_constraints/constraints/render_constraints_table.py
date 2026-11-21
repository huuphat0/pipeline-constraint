"""Tạo bảng Markdown dễ đọc từ constraint đã biên dịch và kết quả kiểm tra."""

import argparse
import json
from pathlib import Path


STAGES = {
    "grasp": "Nắm chai",
    "transport_to_target": "Đưa chai tới cốc",
    "tilt_to_pour": "Nghiêng để rót",
    "restore_upright": "Dựng chai",
    "return_place": "Đặt chai về bàn",
    "retract": "Rút tay",
}
REGIONS = {
    "top": "vùng trên mask",
    "center": "vùng gần giữa mask",
    "axis_positive": "đầu trục ước lượng",
    "axis_negative": "cuối trục ước lượng",
}


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def make_table(specs, stages, tracks, validation):
    results = {row["id"]: row for row in validation["rules"]}
    points = {point["id"]: point for point in tracks["points"]}
    object_names = {}
    for point in tracks["points"]:
        label = point["object_label"].lower()
        object_names[point["object_id"]] = "chai" if "bottle" in label else "cốc" if "cup" in label else point["object_label"]

    def point_name(point_id):
        if point_id == -1:
            return "đầu công tác (−1)"
        point = points[point_id]
        region = REGIONS.get(point["region"], point["region"])
        if region.startswith("dino_region_"):
            region = "điểm DINO " + region.removeprefix("dino_region_")
        return f"{object_names[point['object_id']]} {region} ({point_id})"

    def condition(spec):
        op, ids = spec["op"], spec["ids"]
        if op.startswith("distance_to_reference"):
            return f"{point_name(ids[0])} trở lại gần vị trí đầu video"
        if op.startswith("distance"):
            return f"{point_name(ids[0])} đến gần {point_name(ids[1])}"
        if op.startswith("orientation_change"):
            return f"Hướng chai (trục {ids[0]}→{ids[1]}) lệch so với đầu video"
        if op.startswith("height_above_table"):
            return f"{point_name(ids[0])} cao hơn mặt bàn"
        raise ValueError(op)

    def describe(spec):
        symbol = "≥" if spec["op"].endswith("_ge") else "≤"
        unit = "°" if spec["unit"] == "deg" else spec["unit"]
        status = results[spec["id"]]["status"]
        caution = " (chưa xác minh đầu công tác)" if status.startswith("unverifiable") else ""
        return f"#{spec['id']}: {condition(spec)} {symbol} {spec['threshold']:g} {unit}{caution}"

    lines = ["# Bảng constraint — video mẫu", "",
             "| Stage | Tên | Sub-goal | Path |",
             "|---|---|---|---|"]
    for stage in stages["stages"]:
        matching = [spec for spec in specs if spec["stage"] == stage["stage"]]
        subgoal = "<br>".join(escape(describe(spec)) for spec in matching if spec["phase"] == "subgoal") or "Không có"
        path = "<br>".join(escape(describe(spec)) for spec in matching if spec["phase"] == "path") or "Không có"
        lines.append(f"| {stage['stage']} | {escape(STAGES.get(stage['name'], stage['name']))} | {subgoal} | {path} |")
    axis_ids = sorted(p["id"] for p in tracks["points"] if p.get("source") == "pose3d_axis")
    axis_note = f"ID {', '.join(map(str, axis_ids))} là trục chai suy ra từ pose 3D. " if axis_ids else ""
    lines += ["", "**Ghi chú:** `Sub-goal` kiểm tra ở cuối stage; `Path` áp dụng suốt stage. "
              + axis_note + "Nhãn vùng trên/gần giữa mask là gợi ý vị trí ảnh, chưa xác nhận bộ phận chức năng. "
              "Ngưỡng lấy từ video mẫu và cần duyệt lại. "
              "Chi tiết nguồn luật trong [`constraints.json`](constraints.json), kết quả kiểm tra trong "
              "[`validation.md`](../validation/validation.md).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1] / "artifacts/demo"
    parser.add_argument("--constraints", default=str(base / "constraints/constraints.json"))
    parser.add_argument("--stages", default=str(base / "stages/stages.json"))
    parser.add_argument("--tracks", default=str(base / "keypoints/keypoints_tracks.json"))
    parser.add_argument("--validation", default=str(base / "validation/validation.json"))
    parser.add_argument("--out", default=str(base / "constraints/constraints_table.md"))
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    content = make_table(read(args.constraints), read(args.stages), read(args.tracks), read(args.validation))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
