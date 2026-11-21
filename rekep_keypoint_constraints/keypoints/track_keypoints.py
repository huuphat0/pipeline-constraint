"""Bước 4: theo dõi ID keypoint theo frame, ghi rõ khi mất dấu."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Dinov2Model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # gốc repo, để import rekep_keypoint_constraints.*
from rekep_keypoint_constraints.keypoints.propose_keypoints import dinov2_features, sample_features
from rekep_keypoint_constraints.common.rgbd import RGBDSequence


def _inside(mask, uv):
    u, v = np.rint(uv).astype(int)
    return 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1] and mask[v, u]


def _bbox_center(objects, obj_id):
    for obj in objects:
        if int(obj["id"]) == obj_id:
            x0, y0, x1, y1 = obj["bbox"]
            return np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=float)
    return None


def _record(seq, frame, point, uv, method, confidence):
    u, v = map(int, np.rint(uv))
    z = int(seq.depth[frame, v, u])
    if z <= 0:
        return {"id": point["id"], "status": "invalid", "reason": "depth_missing",
                "pixel_uv": None, "xyz_mm": None}
    xyz = seq.backproject(u, v, z)
    return {"id": point["id"], "status": "visible", "method": method,
            "confidence": round(float(confidence), 3), "pixel_uv": [u, v],
            "xyz_mm": [round(float(x), 2) for x in xyz]}


def _reacquire(seq, frame, point, ref_descriptor, predicted_uv, fmap, mask):
    ys, xs = np.nonzero(mask & (seq.depth[frame] > 0))
    if not len(xs):
        return None
    # Lấy mẫu đều để giới hạn chi phí. Điểm gần dự đoán được ưu tiên.
    stride = max(1, len(xs) // 1200)
    pixels = np.stack((xs[::stride], ys[::stride]), axis=1)
    features = sample_features(fmap, pixels, seq.width, seq.height)
    ref = ref_descriptor / max(np.linalg.norm(ref_descriptor), 1e-8)
    norm = np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    similarity = (features / norm) @ ref
    distance = np.linalg.norm(pixels - predicted_uv, axis=1)
    score = similarity - np.minimum(distance, 200) * 0.001
    idx = int(np.argmax(score))
    if similarity[idx] < 0.78 or distance[idx] > 100:
        return None
    return pixels[idx].astype(float), float(similarity[idx])


def track(seq, points, descriptors, reacquire_every=5):
    if len(points) != len(descriptors):
        raise ValueError("Số point và descriptor không khớp")
    cap = cv2.VideoCapture(str(seq.video_path))
    ok, prev_bgr = cap.read()
    if not ok:
        raise ValueError("Video rỗng")
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    records = []
    initial = [_record(seq, 0, p, p["pixel_uv"], "reference", 1.0) for p in points]
    records.append({"frame": 0, "t": float(seq.times[0]), "points": initial})
    last_uv = [np.asarray(p["pixel_uv"], dtype=float) for p in points]
    last_bbox = [_bbox_center(seq.frame_objects(0), p["object_id"]) for p in points]
    # Vật có bbox gần như bất động trong toàn video: giữ pixel tham chiếu.
    # Optical flow trên bề mặt ít texture có thể trôi dù vật hoàn toàn đứng yên.
    static_ids = set()
    for obj_id in {p["object_id"] for p in points}:
        boxes = np.asarray([next((o["bbox"] for o in f["objects"] if int(o["id"]) == obj_id),
                                 [np.nan] * 4) for f in seq.masks["frames"]], dtype=float)
        if np.isfinite(boxes).all() and np.max(np.ptp(boxes, axis=0)) < 12:
            static_ids.add(obj_id)
    model = None
    for frame in range(1, seq.n_frames):
        ok, bgr = cap.read()
        if not ok:
            raise ValueError(f"Video kết thúc sớm tại frame {frame}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        previous = records[-1]["points"]
        active = [i for i, r in enumerate(previous) if r["status"] == "visible"]
        flow = {}
        if active:
            seeds = np.asarray([previous[i]["pixel_uv"] for i in active], dtype=np.float32).reshape(-1, 1, 2)
            candidate, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, seeds, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            for n, i in enumerate(active):
                if status[n, 0] and np.isfinite(candidate[n, 0]).all():
                    flow[i] = (candidate[n, 0].astype(float), float(error[n, 0]))

        current_objects = seq.frame_objects(frame)
        masks = {p["object_id"]: seq.object_mask(frame, p["object_id"]) for p in points}
        current = []
        lost = []
        for i, p in enumerate(points):
            mask = masks[p["object_id"]]
            center = _bbox_center(current_objects, p["object_id"])
            reason = None
            if p["object_id"] in static_ids:
                uv = np.asarray(p["pixel_uv"], dtype=float)
                u, v = map(int, uv)
                reference_z = p["xyz_mm"][2]
                if _inside(mask, uv) and seq.depth[frame, v, u] > 0 and abs(int(seq.depth[frame, v, u]) - reference_z) <= 30:
                    current.append({"id": p["id"], "status": "visible", "method": "static_anchor",
                                    "confidence": 1.0, "pixel_uv": [u, v], "xyz_mm": p["xyz_mm"]})
                else:
                    current.append({"id": p["id"], "status": "occluded", "reason": "static_anchor_hidden",
                                    "pixel_uv": None, "xyz_mm": None})
                continue
            if center is None:
                reason = "object_mask_missing"
            elif i in flow:
                uv, err = flow[i]
                if np.linalg.norm(uv - last_uv[i]) > 35:
                    reason = "flow_jump"
                elif err > 35:
                    reason = "flow_error"
                elif not _inside(mask, uv):
                    reason = "outside_object_mask"
                else:
                    u, v = map(int, np.rint(uv))
                    if seq.depth[frame, v, u] <= 0:
                        reason = "depth_missing"
                    else:
                        xyz = seq.backproject(u, v, seq.depth[frame, v, u])
                        last_xyz = previous[i]["xyz_mm"]
                        if last_xyz is not None and np.linalg.norm(xyz - last_xyz) > 70:
                            reason = "depth_jump"
                        else:
                            rec = _record(seq, frame, p, uv, "optical_flow", max(0.05, 1 - err / 35))
                            current.append(rec)
                            last_uv[i] = uv
                            last_bbox[i] = center
                            continue
            else:
                reason = "flow_lost"
            current.append({"id": p["id"], "status": "occluded" if center is not None else "invalid",
                            "reason": reason, "pixel_uv": None, "xyz_mm": None})
            if center is not None and frame % reacquire_every == 0:
                lost.append(i)

        if lost:
            if model is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = Dinov2Model.from_pretrained("facebook/dinov2-base", local_files_only=True).eval().to(device)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            fmap = dinov2_features(rgb, model)
            for i in lost:
                p = points[i]
                center = _bbox_center(current_objects, p["object_id"])
                predicted = last_uv[i] + (center - last_bbox[i]) if last_bbox[i] is not None else last_uv[i]
                match = _reacquire(seq, frame, p, descriptors[i], predicted, fmap, masks[p["object_id"]])
                if match is None:
                    continue
                uv, similarity = match
                current[i] = _record(seq, frame, p, uv, "dinov2_reacquire", similarity)
                last_uv[i] = uv
                last_bbox[i] = center

        records.append({"frame": frame, "t": round(float(seq.times[frame]), 5), "points": current})
        prev_gray = gray
    cap.release()
    if model is not None and torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    summary = []
    for p in points:
        recs = [f["points"][p["id"]] for f in records]
        summary.append({"id": p["id"], "object_label": p["object_label"], "region": p["region"],
                        "usable_for_constraints": p["usable_for_constraints"],
                        "visible_fraction": round(sum(r["status"] == "visible" for r in recs) / len(recs), 3),
                        "reacquire_count": sum(r.get("method") == "dinov2_reacquire" for r in recs)})
    return {"n_frames": seq.n_frames, "coordinate_frame": "color_camera", "length_unit": "mm",
            "static_object_ids": sorted(static_ids),
            "points": points, "summary": summary, "frames": records}


def add_pose_axis_points(result, seq, pose_path, half_length_mm=60.0):
    """Thêm 2 điểm trục ảo cho vật dài từ pose 3D từng frame.

    Đây là ước lượng hình học phụ thuộc model pose, không phải keypoint bề mặt
    được nhìn thấy. Provenance và status=inferred được lưu rõ ràng.
    """
    pose = json.loads(Path(pose_path).read_text(encoding="utf-8"))
    by_label = {p["object_label"]: p["object_id"] for p in result["points"]}
    for label, series in pose["objects"].items():
        if label not in by_label or not any(x.get("axis_confident") for x in series):
            continue
        obj_id = by_label[label]
        point_ids = [len(result["points"]), len(result["points"]) + 1]
        axes = []
        last = None
        for sample in series:
            axis = np.asarray(sample.get("axis", [0, 0, 0]), dtype=float)
            norm = np.linalg.norm(axis)
            if not sample.get("axis_confident") or norm < 1e-6:
                axes.append(None)
                continue
            axis /= norm
            if last is not None and float(axis @ last) < 0:
                axis = -axis
            axes.append(axis)
            last = axis
        for point_id, sign, region in zip(point_ids, (1, -1), ("axis_positive", "axis_negative")):
            initial_sample = series[0]
            initial_axis = axes[0]
            if initial_axis is None:
                raise ValueError(f"Pose ban đầu không có trục tin cậy cho {label}")
            center0 = np.array([initial_sample["center_mm"][k] for k in ("x", "y", "z")], dtype=float)
            xyz0 = center0 + sign * half_length_mm * initial_axis
            uv0 = seq.project(xyz0)
            point = {"id": point_id, "object_id": obj_id, "object_label": label,
                     "region": region, "pixel_uv": [round(float(v), 1) for v in uv0],
                     "xyz_mm": [round(float(v), 2) for v in xyz0], "depth_valid": False,
                     "height_above_table_mm": None, "quality": "pose3d_derived",
                     "usable_for_constraints": True, "reference_frame": 0,
                     "source": "pose3d_axis", "half_axis_length_mm": half_length_mm}
            result["points"].append(point)
            visible_count = 0
            for frame, sample in enumerate(series):
                if axes[frame] is None:
                    rec = {"id": point_id, "status": "invalid", "reason": "axis_unconfident",
                           "pixel_uv": None, "xyz_mm": None}
                else:
                    center = np.array([sample["center_mm"][k] for k in ("x", "y", "z")], dtype=float)
                    xyz = center + sign * half_length_mm * axes[frame]
                    uv = seq.project(xyz)
                    rec = {"id": point_id, "status": "inferred", "method": "pose3d_axis",
                           "confidence": 0.7, "pixel_uv": [round(float(v), 1) for v in uv],
                           "xyz_mm": [round(float(v), 2) for v in xyz]}
                    visible_count += 1
                result["frames"][frame]["points"].append(rec)
            result["summary"].append({"id": point_id, "object_label": label, "region": region,
                                      "usable_for_constraints": True, "observed_fraction": 0.0,
                                      "inferred_fraction": round(visible_count / len(series), 3),
                                      "reacquire_count": 0})
    result["pose3d_source"] = str(Path(pose_path).resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parents[1]
    parser.add_argument("--video", default=str(root / "out/mp4/demo_color.mp4"))
    parser.add_argument("--depth", default=str(root / "out/npz/demo_depth.npz"))
    parser.add_argument("--masks", default=str(root / "out/gsam2_demo/masks_gsam2_merged.json"))
    parser.add_argument("--keypoints", default=str(local / "artifacts/demo/keypoints/keypoints.json"))
    parser.add_argument("--descriptors", default=str(local / "artifacts/demo/keypoints/descriptors.npz"))
    parser.add_argument("--pose3d", default=str(root / "out/json/demo_pose3d.json"),
                        help="pose 3D để thêm hai điểm trục ảo cho vật dài; bỏ bằng chuỗi rỗng")
    parser.add_argument("--out", default=str(local / "artifacts/demo/keypoints/keypoints_tracks.json"))
    args = parser.parse_args()
    seq = RGBDSequence(args.video, args.depth, args.masks)
    points = json.loads(Path(args.keypoints).read_text(encoding="utf-8"))["points"]
    with np.load(args.descriptors) as data:
        descriptors = data["descriptors"]
    result = track(seq, points, descriptors)
    if args.pose3d:
        result = add_pose_axis_points(result, seq, args.pose3d)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
