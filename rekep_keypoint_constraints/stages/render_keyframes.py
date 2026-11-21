"""Bước 6: ảnh keyframe đánh số keypoint nhất quán và báo cáo QA."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # gốc repo, để import rekep_keypoint_constraints.*
from rekep_keypoint_constraints.common.rgbd import RGBDSequence


def draw_one(seq, frame, tracks, stage):
    bgr = cv2.cvtColor(seq.rgb(frame), cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (seq.width, 36), (20, 20, 20), -1)
    cv2.putText(bgr, f"stage {stage['stage']} {stage['name']} | frame {frame} t={seq.times[frame]:.2f}s",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    visible, missing, bad_mask = [], [], []
    for point in tracks["points"]:
        point_id = point["id"]
        if not point["usable_for_constraints"]:
            continue
        rec = tracks["frames"][frame]["points"][point_id]
        if rec["status"] not in {"visible", "inferred"}:
            missing.append(point_id)
            continue
        u, v = map(int, np.rint(rec["pixel_uv"]))
        if not (0 <= u < seq.width and 0 <= v < seq.height):
            missing.append(point_id)
            continue
        if rec["status"] == "visible" and not seq.object_mask(frame, point["object_id"])[v, u]:
            bad_mask.append(point_id)
            continue
        color = (30, 210, 40) if point["object_label"] == "white cup" else (0, 175, 255)
        cv2.circle(bgr, (u, v), 6, color, 2 if rec["status"] == "inferred" else -1)
        cv2.circle(bgr, (u, v), 7, (0, 0, 0), 1)
        pos = (min(u + 8, seq.width - 28), max(45, v - 7))
        cv2.putText(bgr, str(point_id), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 0, 0), 3)
        cv2.putText(bgr, str(point_id), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 1)
        visible.append(point_id)
    return bgr, {"frame": frame, "visible_ids": visible, "missing_ids": missing,
                 "outside_mask_ids": bad_mask}


def render(seq, tracks, stages, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    legend = [{k: p[k] for k in ("id", "object_id", "object_label", "region", "quality")}
              for p in tracks["points"]]
    (out_dir / "keypoint_legend.json").write_text(json.dumps(legend, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    for stage in stages["stages"]:
        panels, qa = [], []
        for frame in stage["keyframes"]:
            panel, frame_qa = draw_one(seq, frame, tracks, stage)
            panels.append(cv2.resize(panel, (480, 360), interpolation=cv2.INTER_AREA))
            qa.append(frame_qa)
        while len(panels) < 3:
            panels.append(panels[-1])
        sheet = cv2.hconcat(panels[:3])
        footer = np.full((105, sheet.shape[1], 3), 245, dtype=np.uint8)
        line1 = f"green: cup | orange: bottle | path IDs: {stage['path_eligible_ids']}"
        line2 = f"subgoal IDs: {stage['subgoal_eligible_ids']}"
        line3 = "Hollow circles = pose3d inferred; missing IDs not drawn."
        for y, line in zip((28, 59, 88), (line1, line2, line3)):
            cv2.putText(footer, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1)
        sheet = cv2.vconcat([sheet, footer])
        filename = f"stage{stage['stage']}_{stage['name']}.jpg"
        cv2.imwrite(str(out_dir / filename), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        report.append({"stage": stage["stage"], "image": filename, "frames": qa,
                       "missing_required_path": sorted(set(stage["path_eligible_ids"]) -
                                                       set.intersection(*(set(x["visible_ids"]) for x in qa))),
                       "outside_mask_count": sum(len(x["outside_mask_ids"]) for x in qa)})
    (out_dir / "render_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parents[1]
    parser.add_argument("--video", default=str(root / "out/mp4/demo_color.mp4"))
    parser.add_argument("--depth", default=str(root / "out/npz/demo_depth.npz"))
    parser.add_argument("--masks", default=str(root / "out/gsam2_demo/masks_gsam2_merged.json"))
    parser.add_argument("--tracks", default=str(local / "artifacts/demo/keypoints/keypoints_tracks.json"))
    parser.add_argument("--stages", default=str(local / "artifacts/demo/stages/stages.json"))
    parser.add_argument("--out", default=str(local / "artifacts/demo/keyframes"))
    args = parser.parse_args()
    seq = RGBDSequence(args.video, args.depth, args.masks)
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    stages = json.loads(Path(args.stages).read_text(encoding="utf-8"))
    report = render(seq, tracks, stages, args.out)
    print(json.dumps([{"stage": x["stage"], "missing_required_path": x["missing_required_path"],
                       "outside_mask_count": x["outside_mask_count"]} for x in report],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
