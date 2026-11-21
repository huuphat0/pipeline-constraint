"""Bước 9: kiểm tra cost theo stage và phản chứng hình học có kiểm soát."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # gốc repo, để import rekep_keypoint_constraints.*
from rekep_keypoint_constraints.common.constraint_runtime import evaluate


def frame_points(record):
    return [point["xyz_mm"] for point in record["points"]]


def counterfactual(spec, points):
    """Làm sai chính quan hệ đang xét; không giả vờ đây là video độc lập."""
    changed = [None if p is None else list(p) for p in points]
    ids, op = spec["ids"], spec["op"]
    if -1 in ids:
        anchor = np.asarray(changed[ids[1]], dtype=float)
        positive = evaluate(spec, anchor, changed)
        negative = evaluate(spec, anchor + np.array([100.0, 0, 0]), changed)
        return "EE tại điểm nắm → EE lệch 100 mm", positive, negative
    if op in {"distance_le", "distance_ge"}:
        a, b = ids
        base = np.asarray(changed[a], dtype=float)
        target = np.asarray(changed[b], dtype=float)
        if op == "distance_ge":
            changed[a] = target.tolist()
            label = f"đưa điểm {a} trùng điểm {b}"
        else:
            direction = base - target
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-8 else np.array([1.0, 0, 0])
            changed[a] = (target + direction * (spec["threshold"] + 200.0)).tolist()
            label = f"dịch điểm {a} ra khỏi điểm {b}"
    elif op.startswith("orientation_change"):
        a, b = ids
        anchor = np.asarray(changed[a], dtype=float)
        ref = np.asarray(spec["reference_vector_mm"], dtype=float)
        if op.endswith("_ge"):
            changed[b] = (anchor + ref).tolist()
            label = "thay trục chai bằng hướng ban đầu"
        else:
            vector = np.array([ref[1], -ref[0], 0.0])
            if np.linalg.norm(vector) < 1e-8:
                vector = np.array([0.0, ref[2], -ref[1]])
            vector = vector / np.linalg.norm(vector) * np.linalg.norm(ref)
            changed[b] = (anchor + vector).tolist()
            label = "xoay trục chai khoảng 90 độ"
    elif op.startswith("height_above_table"):
        a = ids[0]
        normal = np.asarray(spec["table_normal"], dtype=float)
        normal /= np.linalg.norm(normal)
        delta = -200.0 if op.endswith("_ge") else 200.0
        changed[a] = (np.asarray(changed[a]) + delta * normal).tolist()
        label = f"dịch điểm {a} {delta:+g} mm theo pháp tuyến bàn"
    elif op == "distance_to_reference_le":
        a = ids[0]
        ref = np.asarray(spec["reference_position_mm"], dtype=float)
        changed[a] = (ref + np.array([spec["threshold"] + 100.0, 0, 0])).tolist()
        label = f"dịch điểm {a} khỏi vị trí ban đầu"
    else:
        raise ValueError(op)
    return label, evaluate(spec, None, points), evaluate(spec, None, changed)


def validate(specs, stages, tracks, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_stage = {stage["stage"]: stage for stage in stages["stages"]}
    results = []
    fig, axes = plt.subplots(len(specs), 1, figsize=(11, max(9, 1.8 * len(specs))), sharex=True)
    for spec, ax in zip(specs, axes):
        stage = by_stage[spec["stage"]]
        start, end = stage["start_frame"], stage["end_frame"]
        sample_frames = list(range(start, end + 1)) if spec["phase"] == "path" else list(range(max(start, end - 4), end + 1))
        costs, missing = [], []
        for frame in sample_frames:
            points = frame_points(tracks["frames"][frame])
            if -1 in spec["ids"] or any(points[i] is None for i in spec["ids"]):
                missing.append(frame)
                continue
            costs.append((frame, evaluate(spec, None, points)))
        coverage = len(costs) / len(sample_frames)
        violation_rate = sum(cost > 0 for _, cost in costs) / len(costs) if costs else None
        median_cost = float(np.median([c for _, c in costs])) if costs else None
        if -1 in spec["ids"]:
            status = "unverifiable_no_end_effector_track"
            point = frame_points(tracks["frames"][end])
            if point[spec["ids"][1]] is None:
                test = {"status": "unavailable"}
            else:
                label, control, altered = counterfactual(spec, point)
                test = {"change": label, "control_cost": round(control, 3),
                        "altered_cost": round(altered, 3), "passes": control <= 0 < altered}
        else:
            ok = (coverage >= (0.9 if spec["phase"] == "path" else 0.6)
                  and (violation_rate <= 0.1 if spec["phase"] == "path" else median_cost <= 0))
            frame = costs[-1][0] if costs else None
            if frame is not None:
                label, control, altered = counterfactual(spec, frame_points(tracks["frames"][frame]))
                test = {"change": label, "control_cost": round(control, 3),
                        "altered_cost": round(altered, 3), "passes": control <= 0 < altered}
            else:
                test = {"status": "unavailable"}
            status = "pass_demo_and_counterfactual" if ok and test.get("passes") else "needs_review"
        result = {"id": spec["id"], "stage": spec["stage"], "stage_name": spec["stage_name"],
                  "phase": spec["phase"], "op": spec["op"], "ids": spec["ids"],
                  "evidence_quality": spec["evidence_quality"], "status": status,
                  "sampled_frames": len(sample_frames), "measured_frames": len(costs),
                  "coverage": round(coverage, 3), "missing_frames": missing,
                  "violation_rate": round(violation_rate, 3) if violation_rate is not None else None,
                  "median_cost": round(median_cost, 3) if median_cost is not None else None,
                  "counterfactual": test}
        results.append(result)
        if costs:
            ax.plot([f for f, _ in costs], [c for _, c in costs], ".-", markersize=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvspan(start, end, alpha=0.05)
        ax.set_ylabel(f"#{spec['id']} cost ({spec['unit']})", fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Video frame (cost <= 0: thỏa)")
    fig.suptitle("Constraint costs trên video mẫu; mỗi hàng chỉ hiện phase được kiểm tra")
    fig.tight_layout()
    fig.savefig(out_dir / "cost_by_stage.png", dpi=140)
    plt.close(fig)
    summary = {"passed": sum(x["status"] == "pass_demo_and_counterfactual" for x in results),
               "unverifiable": sum(x["status"].startswith("unverifiable") for x in results),
               "needs_review": sum(x["status"] == "needs_review" for x in results)}
    payload = {"summary": summary, "rules": results,
               "interpretation": "Phản chứng thay đổi tọa độ của đúng điểm đang xét; chứng minh dấu cost và độ nhạy hình học, không chứng minh luật cần thiết hoặc robot thực hiện thành công."}
    (out_dir / "validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Kiểm tra constraint trên video mẫu", "", "Cost ≤ 0 nghĩa là thỏa. Phản chứng là biến đổi tọa độ có chủ đích, không phải video độc lập.", "",
             "| ID | Stage | Phase | Phép | Độ phủ | Tỉ lệ vi phạm | Phản chứng | Kết quả |",
             "|---:|---|---|---|---:|---:|---|---|"]
    for r in results:
        cf = r["counterfactual"]
        lines.append(f"| {r['id']} | {r['stage_name']} | {r['phase']} | {r['op']} {r['ids']} | "
                     f"{r['coverage']:.1%} | {r['violation_rate']:.1%} | "
                     f"{'đạt' if cf.get('passes') else 'chưa rõ'} | {r['status']} |" if r["violation_rate"] is not None else
                     f"| {r['id']} | {r['stage_name']} | {r['phase']} | {r['op']} {r['ids']} | "
                     f"{r['coverage']:.1%} | — | {'đạt' if cf.get('passes') else 'chưa rõ'} | {r['status']} |")
    lines += ["", "Các luật dùng điểm `pose3d_axis` phụ thuộc cùng nguồn pose 3D với phép chia stage; cần kiểm tra độc lập trên dữ liệu khác.",
              "Luật grasp thiếu quỹ đạo 3D của đầu công tác nên chỉ kiểm tra được dấu hàm bằng phản chứng tổng hợp.",
              "Ngưỡng được hiệu chuẩn từ video mẫu; phép kiểm tra trên chính video này chỉ là kiểm tra tính nhất quán."]
    (out_dir / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1] / "artifacts/demo"
    parser.add_argument("--constraints", default=str(base / "constraints/constraints.json"))
    parser.add_argument("--stages", default=str(base / "stages/stages.json"))
    parser.add_argument("--tracks", default=str(base / "keypoints/keypoints_tracks.json"))
    parser.add_argument("--out", default=str(base / "validation"))
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    result = validate(read(args.constraints), read(args.stages), read(args.tracks), args.out)
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
