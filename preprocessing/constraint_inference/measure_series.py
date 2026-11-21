"""
measure_series.py

Số đo TỪNG KHUNG của mỗi vật, quy về MẶT BÀN thay vì hệ camera
(mục 7 + 8 của đề xuất nâng cấp trong out/bao_cao/tom_tat_chat.md):

    height_mm               độ cao tâm vật dọc pháp tuyến mặt bàn (mm)
    lift_mm                 height_mm trừ độ cao lúc vật nằm yên -> 0 = đang trên bàn
    speed_mm_s              tốc độ tâm vật (mm/s, đã làm mịn)
    tilt_from_vertical_deg  góc giữa trục vật (có hướng, 0..180°) và phương thẳng đứng:
                            trục lúc nghỉ nếu vật đứng gần thẳng, không thì pháp tuyến bàn

VÌ SAO: `tilt_from_rest_deg` cũ (góc lệch so với trục lúc vật đứng yên) chỉ đúng
khi lúc nghỉ vật ĐỨNG THẲNG, và vì trục PCA không có hướng nên bị chặn ở 90°.
Ở đây pháp tuyến bàn cho phương thẳng đứng thật, và trục được giữ CÓ HƯỚNG liên
tục theo thời gian (mỗi khung chọn dấu gần trục khung trước nhất) nên đọc được
tới 180°. `height_mm` của skill_params.py là hiệu ĐỘ SÂU so với mốc (dọc trục
nhìn của camera, không phải phương thẳng đứng) và chỉ có {start, max, end} mỗi
đoạn — không kiểm chứng được "đặt chạm bàn" hay tốc độ.

MẶT BÀN: RANSAC trên đám mây điểm của vài khung đầu (trung vị depth), bỏ pixel
thuộc mask vật/tay. Pháp tuyến quay về phía camera. Nếu mặt phẳng lớn nhất
không nằm DƯỚI các vật (vd tường, người phía sau), bỏ nó và tìm mặt kế tiếp.

Đầu vào lấy từ các tầng trước: masks (GSAM2), depth .npz (có fx/fy/cx/cy),
pose3d (object_pose.py: center_mm, axis từng khung).

Cách dùng (từ thư mục preprocessing/):
    python3 constraint_inference/measure_series.py \\
        --masks ../out/gsam2_demo/masks_gsam2_merged.json \\
        --depth ../out/npz/demo_depth.npz --pose3d ../out/json/demo_pose3d.json \\
        --out ../out/json/demo_series.json
"""

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

# cho phép chạy trực tiếp `python <thư mục>/<file>.py`: đưa gốc preprocessing/ vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skill_inference import hoi_skill_inference as H  # noqa: E402

PLANE_FRAMES = 10            # số khung đầu lấy trung vị depth để dựng mặt bàn
PLANE_STRIDE = 4             # lấy 1/4 pixel mỗi chiều cho RANSAC
PLANE_INLIER_MM = 6.0
PLANE_ITERS = 400
PLANE_MAX_DEPTH_MM = 2500.0
PLANE_MAX_TRIES = 3          # số mặt phẳng thử lần lượt nếu mặt lớn nhất không nằm dưới vật
MASK_DILATE_PX = 9           # nới mask vật/tay khi loại khỏi mặt bàn (bóng, rìa mask)
# tâm vật lúc nghỉ phải cao hơn mặt bàn trong khoảng này (mm) thì mới nhận mặt phẳng
REST_HEIGHT_RANGE_MM = (5.0, 300.0)
REST_SEC = 0.5               # cửa sổ đầu video coi là lúc vật còn nằm yên
SMOOTH_FRAMES = 7            # làm mịn tâm vật trước khi lấy vận tốc (~0.23 s)
VEL_HALF_FRAMES = 4          # vận tốc = sai phân trung tâm qua 2*4 khung (~0.27 s)
UPRIGHT_COS = math.cos(math.radians(45))   # khung đầu: trục lệch < 45° -> coi là "lên"
# Vật lúc nghỉ lệch pháp tuyến bàn dưới ngưỡng này -> coi là ĐANG ĐỨNG THẲNG và
# lấy trục lúc nghỉ làm phương thẳng đứng. Lý do: PCA trên mặt nhìn thấy của
# chai (camera chỉ thấy nửa trước) lệch hệ thống ~6° (đo trên demo: chai đứng
# yên ra 5,6–7,5°, sau khi đặt lại 9–11°), làm LLM đặt `tilt <= 5deg` rồi chính
# demo vi phạm. Vật nằm nghiêng lúc nghỉ (lệch >= ngưỡng) thì dùng pháp tuyến bàn.
UPRIGHT_REST_DEG = 15.0


def _seeded_rng():
    return np.random.default_rng(0)        # RANSAC tất định: chạy lại cho cùng mặt bàn


def table_points(per_frame, depth, intr, n_frames=PLANE_FRAMES, stride=PLANE_STRIDE):
    """Đám mây điểm (Nx3 mm, hệ camera) của phần cảnh KHÔNG thuộc vật/tay."""
    n = min(n_frames, len(depth), len(per_frame))
    stack = depth[:n].astype(np.float32)
    stack[stack == 0] = np.nan
    with np.errstate(all="ignore"):
        d = np.nanmedian(stack, axis=0)
    dh, dw = d.shape
    occupied = np.zeros((dh, dw), bool)
    for i in range(n):
        for m in per_frame[i].values():
            mm = m
            if mm.shape != (dh, dw):
                mm = cv2.resize(mm.astype(np.uint8), (dw, dh),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            occupied |= mm
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_PX, MASK_DILATE_PX))
    occupied = cv2.dilate(occupied.astype(np.uint8), k) > 0
    vs, us = np.mgrid[0:dh:stride, 0:dw:stride]
    z = d[vs, us]
    keep = ~occupied[vs, us] & np.isfinite(z) & (z > 0) & (z < PLANE_MAX_DEPTH_MM)
    us, vs, z = us[keep].astype(np.float64), vs[keep].astype(np.float64), z[keep].astype(np.float64)
    x = (us - intr["cx"]) * z / intr["fx"]
    y = (vs - intr["cy"]) * z / intr["fy"]
    return np.stack([x, y, z], axis=1)


def ransac_plane(pts, rng, iters=PLANE_ITERS, thresh=PLANE_INLIER_MM):
    """Trả (normal, point, inlier_mask) của mặt phẳng nhiều điểm nhất; pháp
    tuyến đã tinh chỉnh bằng SVD trên inlier và quay về phía camera (gốc toạ độ)."""
    best = None
    n = len(pts)
    if n < 3:
        return None
    for _ in range(iters):
        a, b, c = pts[rng.choice(n, 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        if ln < 1e-6:
            continue
        nrm /= ln
        inl = np.abs((pts - a) @ nrm) < thresh
        if best is None or inl.sum() > best[2].sum():
            best = (nrm, a, inl)
    if best is None:
        return None
    inl = best[2]
    p0 = pts[inl].mean(axis=0)
    _, _, vt = np.linalg.svd(pts[inl] - p0, full_matrices=False)
    nrm = vt[-1]
    if np.dot(nrm, -p0) < 0:                 # quay pháp tuyến về phía camera
        nrm = -nrm
    inl = np.abs((pts - p0) @ nrm) < thresh
    return nrm, p0, inl


def rest_centers(pose_objects, rest_sec=REST_SEC):
    """Tâm trung vị của mỗi vật trong `rest_sec` giây đầu (vật còn nằm yên)."""
    out = {}
    for label, rows in pose_objects.items():
        c = [[r["center_mm"]["x"], r["center_mm"]["y"], r["center_mm"]["z"]]
             for r in rows if r["t"] < rest_sec]
        if c:
            out[label] = np.median(np.array(c), axis=0)
    return out


def fit_table(per_frame, depth, intr, pose_objects, rest_sec=REST_SEC):
    """Mặt bàn = mặt phẳng lớn nhất mà các vật lúc nghỉ đều nằm TRÊN nó.
    Trả dict {normal, point, inlier_ratio, n_points, rest_height_mm, tries}."""
    pts = table_points(per_frame, depth, intr)
    rng = _seeded_rng()
    rests = rest_centers(pose_objects, rest_sec)
    remaining = pts
    lo, hi = REST_HEIGHT_RANGE_MM
    for attempt in range(1, PLANE_MAX_TRIES + 1):
        res = ransac_plane(remaining, rng)
        if res is None:
            break
        nrm, p0, inl = res
        heights = {lb: float(np.dot(c - p0, nrm)) for lb, c in rests.items()}
        if heights and all(lo <= h <= hi for h in heights.values()):
            return {"normal": [round(float(v), 5) for v in nrm],
                    "point": [round(float(v), 1) for v in p0],
                    "inlier_ratio": round(float(inl.sum()) / len(pts), 3),
                    "n_points": int(len(pts)),
                    "rest_height_mm": {lb: round(h, 1) for lb, h in heights.items()},
                    "tries": attempt}
        remaining = remaining[~inl]
        if len(remaining) < 100:
            break
    raise RuntimeError("Không tìm được mặt bàn nằm dưới các vật (đã thử "
                       f"{PLANE_MAX_TRIES} mặt phẳng) — kiểm tra depth / khung đầu video")


def _smooth(arr, k):
    """Trung bình trượt từng cột, giữ nguyên độ dài (mép dùng cửa sổ hụt)."""
    out = np.empty_like(arr)
    h = k // 2
    for i in range(len(arr)):
        out[i] = arr[max(0, i - h):i + h + 1].mean(axis=0)
    return out


def oriented_axes(rows, normal):
    """Trục CÓ HƯỚNG cho từng bản ghi (None nếu trục không tin cậy): khung đầu
    gần thẳng đứng thì hướng lên, các khung sau chọn dấu gần khung trước nhất."""
    n = np.asarray(normal)
    out, prev = [], None
    for r in rows:
        if not (r.get("axis_confident") and r.get("axis")):
            out.append(None)
            continue
        ax = np.asarray(r["axis"], dtype=float)
        ax /= np.linalg.norm(ax)
        if prev is None:
            if abs(np.dot(ax, n)) >= UPRIGHT_COS and np.dot(ax, n) < 0:
                ax = -ax
        elif np.dot(ax, prev) < 0:
            ax = -ax
        prev = ax
        out.append(ax)
    return out


def tilt_reference(rows, axes, normal, rest_sec=REST_SEC):
    """Phương 'thẳng đứng' để đo góc nghiêng: trục lúc nghỉ nếu vật đứng gần
    thẳng (khử lệch hệ thống của PCA), không thì pháp tuyến bàn.
    Trả (vector, info) với info = {reference, rest_tilt_deg}."""
    n = np.asarray(normal)
    rest = [a for a, r in zip(axes, rows) if a is not None and r["t"] < rest_sec]
    if not rest:
        return n, {"reference": "table_normal", "rest_tilt_deg": None}
    m = np.mean(rest, axis=0)
    m /= np.linalg.norm(m)
    rest_tilt = math.degrees(math.acos(float(np.clip(np.dot(m, n), -1.0, 1.0))))
    if rest_tilt < UPRIGHT_REST_DEG:
        return m, {"reference": "rest_axis", "rest_tilt_deg": round(rest_tilt, 1)}
    return n, {"reference": "table_normal", "rest_tilt_deg": round(rest_tilt, 1)}


def object_series(rows, normal, point, fps, rest_sec=REST_SEC):
    """Chuỗi số đo của MỘT vật từ các bản ghi pose3d. Trả (chuỗi, info góc)."""
    n = np.asarray(normal)
    p0 = np.asarray(point)
    rows = sorted(rows, key=lambda r: r["frame"])
    c = np.array([[r["center_mm"]["x"], r["center_mm"]["y"], r["center_mm"]["z"]] for r in rows])
    frames = np.array([r["frame"] for r in rows])
    cs = _smooth(c, SMOOTH_FRAMES)
    height = (cs - p0) @ n
    rest = [h for h, r in zip(height, rows) if r["t"] < rest_sec]
    rest_h = float(np.median(rest)) if rest else float(np.min(height))
    axes = oriented_axes(rows, n)
    ref, info = tilt_reference(rows, axes, n, rest_sec)

    out = []
    for i, r in enumerate(rows):
        rec = {"frame": int(r["frame"]), "t": r["t"],
               "height_mm": round(float(height[i]), 1),
               "lift_mm": round(float(height[i] - rest_h), 1)}
        # sai phân trung tâm trên tâm đã làm mịn; bỏ qua nếu khoảng frame bị đứt
        a, b = max(0, i - VEL_HALF_FRAMES), min(len(rows) - 1, i + VEL_HALF_FRAMES)
        if b > a:
            dt = (frames[b] - frames[a]) / fps
            if 0 < dt <= 2.5 * VEL_HALF_FRAMES * 2 / fps:
                rec["speed_mm_s"] = round(float(np.linalg.norm(cs[b] - cs[a]) / dt), 1)
        if axes[i] is not None:
            rec["tilt_from_vertical_deg"] = round(
                math.degrees(math.acos(float(np.clip(np.dot(axes[i], ref), -1.0, 1.0)))), 1)
        out.append(rec)
    return out, info


def compute(masks_json, depth_path, pose3d_path, rest_sec=REST_SEC):
    _, per_frame = H.load_mask_series(masks_json)
    depth, _ = H.load_depth(depth_path)
    intr = H.load_intrinsics(depth_path)
    if intr is None:
        raise RuntimeError(f"{depth_path} không có fx/fy/cx/cy — chạy lại depth_from_mcap.py")
    with open(pose3d_path, encoding="utf-8") as f:
        pose = json.load(f)
    fps = float(pose["fps"])
    table = fit_table(per_frame, depth, intr, pose["objects"], rest_sec)
    objects, tilt_ref = {}, {}
    for label, rows in pose["objects"].items():
        objects[label], tilt_ref[label] = object_series(rows, table["normal"], table["point"],
                                                        fps, rest_sec)
    return {"source_masks": masks_json, "source_depth": depth_path, "source_pose3d": pose3d_path,
            "fps": fps, "table_plane": table, "tilt_reference": tilt_ref,
            "note": "height_mm/lift_mm dọc pháp tuyến mặt bàn (tâm vật, không phải đáy); "
                    "tilt_from_vertical_deg = góc trục vật CÓ HƯỚNG (0..180°) với phương thẳng "
                    "đứng: trục lúc nghỉ nếu vật đứng gần thẳng (xem tilt_reference), không thì "
                    "pháp tuyến bàn; speed_mm_s = tốc độ tâm vật đã làm mịn. Hệ camera, chưa "
                    "hiệu chuẩn sang robot.",
            "objects": objects}


def stats(values):
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    return {"min": round(float(v.min()), 1), "max": round(float(v.max()), 1),
            "mean": round(float(v.mean()), 1), "n_frames": int(v.size)}


def segment_block(series_rows, t0, t1):
    """Thống kê số đo của một vật trong [t0, t1) — ghép vào `measured` của scene."""
    rows = [r for r in series_rows if t0 <= r["t"] < t1]
    if not rows:
        return {}
    m = {}
    lift = [r["lift_mm"] for r in rows]
    m["lift_mm"] = {"min": round(min(lift), 1), "max": round(max(lift), 1),
                    "start": lift[0], "end": lift[-1]}
    sp = [r["speed_mm_s"] for r in rows if "speed_mm_s" in r]
    if sp:
        m["speed_mm_s"] = {"max": round(max(sp), 1), "mean": round(sum(sp) / len(sp), 1),
                           "p90": round(float(np.percentile(sp, 90)), 1)}
    tilt = stats([r["tilt_from_vertical_deg"] for r in rows if "tilt_from_vertical_deg" in r])
    if tilt:
        m["tilt_from_vertical_deg"] = tilt
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--pose3d", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rest-sec", type=float, default=REST_SEC,
                    help="số giây đầu video coi như vật còn nằm yên trên bàn")
    args = ap.parse_args()
    res = compute(args.masks, args.depth, args.pose3d, args.rest_sec)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    tp = res["table_plane"]
    print(f"Đã ghi {args.out}")
    print(f"  mặt bàn: normal={tp['normal']}, inlier {tp['inlier_ratio']:.0%} của {tp['n_points']} "
          f"điểm, thử {tp['tries']} mặt; tâm vật lúc nghỉ cao {tp['rest_height_mm']} mm")
    print(f"  phương thẳng đứng để đo góc: {res['tilt_reference']}")
    for label, rows in res["objects"].items():
        lift = [r["lift_mm"] for r in rows]
        sp = [r["speed_mm_s"] for r in rows if "speed_mm_s" in r]
        tl = [r["tilt_from_vertical_deg"] for r in rows if "tilt_from_vertical_deg" in r]
        print(f"  {label}: {len(rows)} khung, lift {min(lift):.0f}..{max(lift):.0f} mm, "
              f"tốc độ max {max(sp) if sp else float('nan'):.0f} mm/s, "
              f"tilt {min(tl) if tl else float('nan'):.0f}..{max(tl) if tl else float('nan'):.0f}°")
    return 0


if __name__ == "__main__":
    sys.exit(main())
