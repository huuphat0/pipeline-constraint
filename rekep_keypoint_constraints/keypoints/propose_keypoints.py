"""Đề xuất keypoint kiểu ReKep: mask + DINOv2/PCA + XYZ + gộp 3D."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from transformers import Dinov2Model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # gốc repo, để import rekep_keypoint_constraints.*
from rekep_keypoint_constraints.common.rgbd import RGBDSequence


def dinov2_features(rgb, model):
    """Feature map DINOv2 giữ gần đúng tỷ lệ khung hình, không tải mạng."""
    h, w = rgb.shape[:2]
    patch = int(model.config.patch_size)
    rh, rw = round(h / patch) * patch, round(w / patch) * patch
    resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    image = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = ((image - mean) / std).unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode():
        tokens = model(pixel_values=image).last_hidden_state[:, 1:]
    return tokens.transpose(1, 2).reshape(1, -1, rh // patch, rw // patch)


def sample_features(feature_map, pixels_uv, width, height):
    pixels = np.asarray(pixels_uv, dtype=np.float32)
    grid = torch.from_numpy(np.stack((2 * pixels[:, 0] / max(width - 1, 1) - 1,
                                      2 * pixels[:, 1] / max(height - 1, 1) - 1), axis=-1))
    grid = grid.to(feature_map.device).reshape(1, -1, 1, 2)
    with torch.inference_mode():
        features = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=True)
    return features.squeeze(0).squeeze(-1).T.cpu().numpy()


def normalize_columns(values):
    """Đưa mỗi chiều đặc trưng/XYZ về [0,1] như phần gom cụm ReKep."""
    lo = values.min(axis=0)
    span = values.max(axis=0) - lo
    return (values - lo) / np.where(span > 1e-8, span, 1.0)


def merge_close_3d(candidates, bandwidth_mm):
    """Mean Shift trên XYZ, giữ pixel ứng viên thật gần tâm cụm nhất."""
    if len(candidates) < 2:
        return candidates
    xyz = np.asarray([c["xyz_mm"] for c in candidates], dtype=np.float64)
    centers = MeanShift(bandwidth=bandwidth_mm, bin_seeding=False).fit(xyz).cluster_centers_
    selected = []
    for center in centers:
        index = int(np.argmin(np.linalg.norm(xyz - center, axis=1)))
        if index not in selected:
            selected.append(index)
    return [candidates[i] for i in selected]


def assign_region_hints(candidates, mask):
    """Gán top/center để giữ hợp đồng downstream; đây chỉ là gợi ý 2D."""
    if not candidates:
        return candidates
    ys, xs = np.nonzero(mask)
    target = np.array([np.median(xs), np.median(ys)], dtype=float)
    top = min(range(len(candidates)), key=lambda i: candidates[i]["pixel_uv"][1])
    alternatives = [i for i in range(len(candidates)) if i != top]
    center = min(alternatives, key=lambda i: np.linalg.norm(
        np.asarray(candidates[i]["pixel_uv"], dtype=float) - target)) if alternatives else top
    for i, candidate in enumerate(candidates):
        candidate["region"] = "top" if i == top else "center" if i == center else f"dino_region_{candidate['cluster']}"
        candidate["role_hint_method"] = "image_top_or_mask_center" if i in {top, center} else "none"
    return candidates


def propose(seq, frame, num_candidates_per_mask=5, seed=0, table_plane=None,
            min_dist_mm=35.0, max_sample_pixels=1600, workspace_bounds_mm=None):
    rgb = seq.rgb(frame)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Dinov2Model.from_pretrained("facebook/dinov2-base", local_files_only=True).eval().to(device)
    fmap = dinov2_features(rgb, model)
    rng = np.random.default_rng(seed)
    points, descriptors, diagnostics = [], [], []
    for obj in seq.frame_objects(frame):
        label = obj["label"]
        if "hand" in label.lower():
            continue  # ReKep dùng end_effector riêng; tay người không là điểm môi trường.
        obj_id = int(obj["id"])
        mask = seq.object_mask(frame, obj_id)
        valid = mask & (seq.depth[frame] > 0)
        if valid.sum() < 100:
            continue
        max_height = None
        if table_plane:
            ys_all, xs_all = np.nonzero(valid)
            sampled = np.arange(0, len(xs_all), max(1, len(xs_all) // 3000))
            xyz_all = seq.backproject(xs_all[sampled], ys_all[sampled],
                                       seq.depth[frame, ys_all[sampled], xs_all[sampled]])
            heights = (xyz_all - np.asarray(table_plane["point"])) @ np.asarray(table_plane["normal"])
            max_height = float(np.percentile(heights, 95))
        # Theo ReKep: gom đặc trưng thị giác PCA3 cùng XYZ 3D đã chuẩn hóa.
        # Lấy mẫu tối đa 1600 pixel để giữ chi phí xử lý video ở mức hợp lý.
        ys, xs = np.nonzero(valid)
        take = rng.choice(len(xs), size=min(max_sample_pixels, len(xs)), replace=False)
        pixels = np.stack((xs[take], ys[take]), axis=1)
        xyz_samples = seq.backproject(pixels[:, 0], pixels[:, 1],
                                      seq.depth[frame, pixels[:, 1], pixels[:, 0]]).astype(np.float64)
        feats = sample_features(fmap, pixels, seq.width, seq.height)
        reduced = PCA(n_components=3, random_state=seed).fit_transform(feats)
        visual = normalize_columns(reduced)
        spatial = normalize_columns(xyz_samples)
        vectors = np.concatenate((visual, spatial), axis=1)
        n_cluster = min(num_candidates_per_mask, len(pixels))
        km = KMeans(n_clusters=n_cluster, random_state=seed, n_init=3).fit(vectors)
        candidates = []
        for cluster in range(n_cluster):
            members = np.flatnonzero(km.labels_ == cluster)
            if not len(members):
                continue
            # ReKep chọn điểm thật gần trung tâm trong không gian đặc trưng.
            idx = int(members[np.argmin(np.linalg.norm(
                visual[members] - km.cluster_centers_[cluster, :3], axis=1))])
            u, v = map(int, pixels[idx])
            xyz = xyz_samples[idx].tolist()
            height = None
            reject_reason = None
            if workspace_bounds_mm is not None:
                low, high = (np.asarray(bound, dtype=float) for bound in workspace_bounds_mm)
                if not (np.asarray(xyz) >= low).all() or not (np.asarray(xyz) <= high).all():
                    reject_reason = "outside_workspace_bounds"
            if table_plane:
                height = float((np.asarray(xyz) - table_plane["point"]) @ table_plane["normal"])
                y0, y1 = obj["bbox"][1], obj["bbox"][3]
                if height < 25 or (v >= y0 + 0.75 * (y1 - y0) and height < 0.55 * max_height):
                    reject_reason = reject_reason or "support_plane_ambiguous"
            candidate = {"cluster": cluster, "pixel_uv": [u, v],
                         "xyz_mm": [round(float(x), 2) for x in xyz],
                         "height_above_table_mm": round(height, 1) if height is not None else None,
                         "descriptor": feats[idx]}
            if reject_reason:
                diagnostics.append({"object_id": obj_id, "object_label": label,
                                    "cluster": cluster, "pixel_uv": [u, v],
                                    "xyz_mm": candidate["xyz_mm"], "reason": reject_reason})
            else:
                candidates.append(candidate)
        merged = merge_close_3d(candidates, min_dist_mm)
        for candidate in candidates:
            if not any(candidate is kept for kept in merged):
                diagnostics.append({"object_id": obj_id, "object_label": label,
                                    "cluster": candidate["cluster"],
                                    "pixel_uv": candidate["pixel_uv"],
                                    "xyz_mm": candidate["xyz_mm"], "reason": "merged_with_nearby_3d_candidate"})
        merged = assign_region_hints(merged, mask)
        merged.sort(key=lambda p: (p["pixel_uv"][1], p["pixel_uv"][0]))
        for candidate in merged:
            candidate["id"] = len(points)
            candidate["object_id"] = obj_id
            candidate["object_label"] = label
            candidate["depth_valid"] = True
            candidate["quality"] = "usable"
            candidate["usable_for_constraints"] = True
            candidate["reference_frame"] = frame
            candidate["source"] = "mask_dinov2_xyz"
            descriptors.append(candidate.pop("descriptor"))
            points.append(candidate)
    if not points:
        raise ValueError("Không đề xuất được keypoint nào có depth hợp lệ")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return points, np.asarray(descriptors, dtype=np.float32), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--video", default=str(root / "out/mp4/demo_color.mp4"))
    parser.add_argument("--depth", default=str(root / "out/npz/demo_depth.npz"))
    parser.add_argument("--masks", default=str(root / "out/gsam2_demo/masks_gsam2_merged.json"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--num-candidates-per-mask", type=int, default=5,
                        help="số cụm DINOv2 ban đầu trên mỗi vật, như ReKep")
    parser.add_argument("--merge-distance-mm", type=float, default=35.0,
                        help="Mean Shift 3D; 35 mm cho chai/cốc demo (ReKep dùng 80 mm cho vật lớn hơn)")
    parser.add_argument("--workspace-bounds-mm", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                        help="giới hạn XYZ trong hệ camera màu; để trống khi chưa hiệu chuẩn vùng robot")
    parser.add_argument("--series", default=str(root / "out/json/demo_series.json"),
                        help="file mặt bàn để gắn cờ điểm sát bàn/ảnh phản chiếu; bỏ bằng chuỗi rỗng")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "artifacts/demo/keypoints"))
    args = parser.parse_args()
    seq = RGBDSequence(args.video, args.depth, args.masks)
    table_plane = None
    if args.series:
        table_plane = json.loads(Path(args.series).read_text(encoding="utf-8"))["table_plane"]
    if args.num_candidates_per_mask < 1 or args.merge_distance_mm <= 0:
        parser.error("num-candidates-per-mask phải >=1 và merge-distance-mm phải >0")
    bounds = (args.workspace_bounds_mm[:3], args.workspace_bounds_mm[3:]) if args.workspace_bounds_mm else None
    points, descriptors, diagnostics = propose(
        seq, args.frame, num_candidates_per_mask=args.num_candidates_per_mask,
        table_plane=table_plane, min_dist_mm=args.merge_distance_mm,
        workspace_bounds_mm=bounds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"source_frame": args.frame, "coordinate_frame": "color_camera", "length_unit": "mm",
               "proposal_model": "facebook/dinov2-base",
               "proposal_method": "ReKep-style PCA3 + normalized XYZ KMeans + 3D MeanShift",
               "num_candidates_per_mask": args.num_candidates_per_mask,
               "merge_distance_mm": args.merge_distance_mm,
               "workspace_bounds_mm": bounds,
               "points": points}
    (out / "keypoints.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "rejected_candidates.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "descriptors.npz", descriptors=descriptors)
    preview = cv2.cvtColor(seq.rgb(args.frame), cv2.COLOR_RGB2BGR)
    for point in points:
        u, v = point["pixel_uv"]
        color = ((0, 220, 0) if point["object_id"] == 0 else (0, 160, 255))
        if not point["usable_for_constraints"]:
            color = (0, 0, 255)
        cv2.circle(preview, (u, v), 5, color, -1)
        cv2.putText(preview, str(point["id"]), (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(preview, str(point["id"]), (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.imwrite(str(out / "keypoints_preview.png"), preview)
    print(json.dumps({"count": len(points), "points": points}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
