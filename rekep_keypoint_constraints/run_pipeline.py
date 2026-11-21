"""Một lệnh tạo keypoint, stage, constraint và báo cáo từ video RGB-D MCAP."""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEGACY = ROOT / "preprocessing"
DEMO_MCAP = ROOT / "video_depthcam/demo_20260912_172945_0.mcap"


def upstream_paths(mcap, name, out, prompt):
    if mcap == DEMO_MCAP.resolve() and name == "demo":
        base = ROOT / "out"
        paths = {"video": base / "mp4/demo_color.mp4", "depth": base / "npz/demo_depth.npz",
                 "masks": base / "gsam2_demo/masks_gsam2_merged.json",
                 "skills": base / "json/skills_lowlevel.json",
                 "pose3d": base / "json/demo_pose3d.json", "series": base / "json/demo_series.json"}
        return paths, "reused_demo_upstream"
    if not prompt:
        raise ValueError("Video mới cần --prompt để phân đoạn vật, ví dụ: 'hand . bottle . cup .'")
    base = out / "upstream"
    cmd = [sys.executable, str(LEGACY / "run_pipeline.py"), str(mcap), "--name", name,
           "--prompt", prompt, "--out-root", str(base), "--to", "series"]
    return ({"video": base / "mp4" / f"{name}_color.mp4",
             "depth": base / "npz" / f"{name}_depth.npz",
             "masks": base / f"gsam2_{name}/masks_gsam2_merged.json",
             "skills": base / "json" / f"{name}_skills_lowlevel.json",
             "pose3d": base / "json" / f"{name}_pose3d.json",
             "series": base / "json" / f"{name}_series.json"}, cmd)


def run(args):
    mcap = Path(args.mcap).resolve()
    if not mcap.is_file() or mcap.suffix != ".mcap":
        raise ValueError(f"Input phải là file .mcap tồn tại: {mcap}")
    from mcap.reader import make_reader
    with mcap.open("rb") as handle:
        summary = make_reader(handle).get_summary()
    topics = {channel.topic for channel in summary.channels.values()}
    required_topics = {"/camera/camera/color/image_raw",
                       "/camera/camera/aligned_depth_to_color/image_raw",
                       "/camera/camera/color/camera_info"}
    absent = sorted(required_topics - topics)
    if absent:
        raise ValueError(f"MCAP thiếu topic cần cho RGB-D 3D: {absent}")
    name = args.name or ("demo" if mcap == DEMO_MCAP.resolve() else re.sub(r"[^a-zA-Z0-9_]", "_", mcap.stem))
    out = Path(args.out).resolve() if args.out else HERE / "artifacts" / name
    out.mkdir(parents=True, exist_ok=True)
    paths, upstream = upstream_paths(mcap, name, out, args.prompt)
    manifest_path = out / "run_manifest.json"
    manifest = {"mcap": str(mcap), "name": name, "out": str(out),
                "upstream_mode": upstream if isinstance(upstream, str) else "legacy_pipeline_to_series",
                "upstream_files": {k: str(v) for k, v in paths.items()},
                "keypoint_proposal": {"method": "rekep_style_pca3_xyz_meanshift3d",
                                      "num_candidates_per_mask": args.num_candidates_per_mask,
                                      "merge_distance_mm": args.merge_distance_mm},
                "instruction_source": "user" if args.instruction else "vlm_video_frames",
                "steps": [], "status": "running"}

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def step(label, cmd):
        print(f"[{label}] chạy...", flush=True)
        start = time.monotonic()
        log = out / f"{label}.log"
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        record = {"step": label, "seconds": round(time.monotonic() - start, 2),
                  "exit_code": process.returncode, "log": str(log)}
        manifest["steps"].append(record)
        if process.returncode:
            manifest["status"] = "failed"
            manifest["failed_step"] = label
            save()
            raise RuntimeError(f"Bước {label} lỗi, xem {log}")
        save()
        print(f"[{label}] xong sau {record['seconds']:.1f}s", flush=True)

    save()
    if not isinstance(upstream, str):
        step("upstream", upstream)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        manifest["status"] = "failed"
        manifest["missing_upstream_files"] = missing
        save()
        raise ValueError(f"Thiếu đầu ra RGB-D/skill/pose: {missing}")
    common = ["--video", str(paths["video"]), "--depth", str(paths["depth"]),
              "--masks", str(paths["masks"])]
    keypoints = out / "keypoints"
    stages = out / "stages"
    constraints = out / "constraints"
    keyframes = out / "keyframes"
    validation = out / "validation"
    def script(file, *options):
        return [sys.executable, str(HERE / file), *map(str, options)]

    step("rgbd", script("checks/audit_rgbd.py", *common, "--out", out / "rgbd"))
    step("propose", script("keypoints/propose_keypoints.py", *common, "--series", paths["series"],
                           "--num-candidates-per-mask", args.num_candidates_per_mask,
                           "--merge-distance-mm", args.merge_distance_mm,
                           "--out", keypoints))
    step("track", script("keypoints/track_keypoints.py", *common,
                         "--keypoints", keypoints / "keypoints.json",
                         "--descriptors", keypoints / "descriptors.npz",
                         "--pose3d", paths["pose3d"], "--out", keypoints / "keypoints_tracks.json"))
    stages.mkdir(parents=True, exist_ok=True)
    if args.instruction:
        data = {"instruction": args.instruction, "source": "user", "needs_review": False,
                "review_status": "user_provided", "uncertainties": []}
        (stages / "instruction.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["steps"].append({"step": "instruction", "source": "user", "seconds": 0})
        save()
    else:
        step("instruction", script("stages/infer_instruction.py", *common, "--out", stages))
    step("stages", script("stages/build_stages.py", "--skills", paths["skills"],
                          "--tracks", keypoints / "keypoints_tracks.json",
                          "--instruction", stages / "instruction.json", "--series", paths["series"],
                          "--out", stages / "stages.json"))
    step("keyframes", script("stages/render_keyframes.py", *common,
                             "--tracks", keypoints / "keypoints_tracks.json",
                             "--stages", stages / "stages.json", "--out", keyframes))
    step("generator", script("constraints/generate_constraints.py", "--stages", stages / "stages.json",
                             "--tracks", keypoints / "keypoints_tracks.json",
                             "--instruction", stages / "instruction.json",
                             "--keyframes", keyframes, "--out", constraints))
    step("compile", script("constraints/compile_constraints.py", "--raw", constraints / "constraints_raw.json",
                           "--stages", stages / "stages.json", "--tracks", keypoints / "keypoints_tracks.json",
                           "--series", paths["series"], "--out", constraints))
    step("validation", script("constraints/validate_demo.py", "--constraints", constraints / "constraints.json",
                              "--stages", stages / "stages.json", "--tracks", keypoints / "keypoints_tracks.json",
                              "--out", validation))
    step("table", script("constraints/render_constraints_table.py", "--constraints", constraints / "constraints.json",
                         "--stages", stages / "stages.json",
                         "--tracks", keypoints / "keypoints_tracks.json",
                         "--validation", validation / "validation.json",
                         "--out", constraints / "constraints_table.md"))
    step("selected_keypoints", script("keypoints/render_selected_keypoints.py", "--video", paths["video"],
                                       "--keypoints", keypoints / "keypoints.json",
                                       "--tracks", keypoints / "keypoints_tracks.json",
                                       "--constraints", constraints / "constraints.json",
                                       "--out", keypoints))
    summary = json.loads((validation / "validation.json").read_text(encoding="utf-8"))["summary"]
    manifest["validation_summary"] = summary
    manifest["status"] = "completed_with_unverifiable" if summary["unverifiable"] else "completed"
    save()
    print(json.dumps({"status": manifest["status"], "validation": summary,
                      "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap", nargs="?", default=str(DEMO_MCAP))
    parser.add_argument("--name", help="tên ngắn cho kết quả; mặc định là tên file")
    parser.add_argument("--prompt", help="nhãn vật cho Grounded-SAM2 khi xử lý video mới")
    parser.add_argument("--instruction", help="instruction người dùng, bỏ qua bước suy instruction bằng VLM")
    parser.add_argument("--num-candidates-per-mask", type=int, default=5,
                        help="số cụm DINOv2 ban đầu trên mỗi vật (mặc định 5)")
    parser.add_argument("--merge-distance-mm", type=float, default=35.0,
                        help="khoảng gộp keypoint XYZ trong demo chai/cốc (mặc định 35 mm)")
    parser.add_argument("--out", help="thư mục kết quả riêng; mặc định artifacts/<name>")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, RuntimeError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
