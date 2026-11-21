"""Bước 2: kiểm tra RGB-D và tạo ảnh đối chiếu depth với RGB."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # gốc repo, để import rekep_keypoint_constraints.*
from rekep_keypoint_constraints.common.rgbd import RGBDSequence


def audit(seq, out_dir, frame=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = seq.rgb(frame)
    depth = seq.depth[frame]
    valid = depth > 0
    cloud = seq.point_cloud(frame, stride=5)
    if cloud.size == 0:
        raise ValueError("Frame kiểm tra không có depth hợp lệ")
    u, v = np.nonzero(valid)[1][::5], np.nonzero(valid)[0][::5]
    reproj = seq.project(cloud)
    reproj_error = np.linalg.norm(reproj - np.stack((u, v), axis=1), axis=1)
    object_coverage = {}
    for obj_id, label in seq.object_labels.items():
        mask = seq.object_mask(frame, obj_id)
        if mask.any():
            object_coverage[f"{obj_id}:{label}"] = round(float(np.mean(valid[mask])), 4)
    scaled = np.zeros_like(depth, dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [5, 95])
    scaled[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    color_depth = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    color_depth[~valid] = 0
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(bgr, 0.55, color_depth, 0.45, 0)
    cv2.imwrite(str(out_dir / "rgb_depth_overlay.png"), overlay)
    report = {
        "video": str(seq.video_path), "depth": str(seq.depth_path),
        "masks": str(seq.masks_path), "frame_count": seq.n_frames,
        "size_hw": [seq.height, seq.width], "fps": seq.fps,
        "coordinate_frame": "color_camera", "length_unit": "mm",
        "intrinsics": seq.intrinsics,
        "depth_valid_fraction_all": round(float(np.mean(seq.depth > 0)), 4),
        "depth_valid_fraction_frame": round(float(np.mean(valid)), 4),
        "depth_valid_fraction_by_object": object_coverage,
        "max_nominal_timestamp_offset_ms": round(float(np.max(np.abs(
            seq.times - np.arange(seq.n_frames) / seq.fps))) * 1000, 2),
        "reprojection_error_px_max": float(reproj_error.max()),
        "sample_cloud_points": len(cloud),
        "warning": "Video ghi fps cố định; depth được ghép theo timestamp màu trong file NPZ. Chưa hiệu chuẩn hệ camera sang robot."
    }
    (out_dir / "rgbd_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--video", default=str(root / "out/mp4/demo_color.mp4"))
    parser.add_argument("--depth", default=str(root / "out/npz/demo_depth.npz"))
    parser.add_argument("--masks", default=str(root / "out/gsam2_demo/masks_gsam2_merged.json"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "artifacts/demo/rgbd"))
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    report = audit(RGBDSequence(args.video, args.depth, args.masks), args.out, args.frame)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
