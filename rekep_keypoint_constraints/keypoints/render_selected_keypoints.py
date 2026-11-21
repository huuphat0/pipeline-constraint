"""Vẽ riêng những keypoint thật sự xuất hiện trong constraint đã biên dịch."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


def render(video, keypoints_path, tracks_path, constraints_path, out_dir):
    proposal = json.loads(Path(keypoints_path).read_text(encoding="utf-8"))
    tracks = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
    constraints = json.loads(Path(constraints_path).read_text(encoding="utf-8"))
    usage = defaultdict(list)
    for rule in constraints:
        for point_id in rule["ids"]:
            if point_id >= 0:
                usage[point_id].append({"stage": rule["stage"], "phase": rule["phase"],
                                        "constraint_id": rule["id"]})

    observed_ids = {point["id"] for point in proposal["points"]}
    observed = sorted(observed_ids & usage.keys())
    inferred = sorted(usage.keys() - observed_ids)
    cap = cv2.VideoCapture(str(video))
    frame_number = int(proposal["source_frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Không đọc được frame {frame_number} từ {video}")

    for point_id in observed:
        point = proposal["points"][point_id]
        u, v = map(int, point["pixel_uv"])
        color = (0, 220, 0) if point["object_id"] == 0 else (0, 160, 255)
        cv2.circle(frame, (u, v), 6, color, -1)
        pos = (u + 8, v - 8)
        cv2.putText(frame, str(point_id), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, str(point_id), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "keypoints_used_preview.png"), frame)
    points = tracks["points"]
    payload = {
        "source_frame": frame_number,
        "selection": "IDs xuất hiện trong constraint đã biên dịch; không phải toàn bộ ứng viên ban đầu",
        "observed_ids": observed,
        "inferred_ids": inferred,
        "points": [{"id": point_id, "object_label": points[point_id]["object_label"],
                    "region": points[point_id]["region"],
                    "source": points[point_id].get("source", "mask_dinov2"),
                    "used_by": usage[point_id]} for point_id in sorted(usage)],
        "note": "Điểm trục suy ra từ pose không được vẽ như điểm quan sát trên ảnh gốc.",
    }
    (out_dir / "keypoints_used.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    root = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(root / "out/mp4/demo_color.mp4"))
    parser.add_argument("--keypoints", default=str(local / "artifacts/demo/keypoints/keypoints.json"))
    parser.add_argument("--tracks", default=str(local / "artifacts/demo/keypoints/keypoints_tracks.json"))
    parser.add_argument("--constraints", default=str(local / "artifacts/demo/constraints/constraints.json"))
    parser.add_argument("--out", default=str(local / "artifacts/demo/keypoints"))
    args = parser.parse_args()
    result = render(args.video, args.keypoints, args.tracks, args.constraints, args.out)
    print(json.dumps({"observed_ids": result["observed_ids"],
                      "inferred_ids": result["inferred_ids"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
