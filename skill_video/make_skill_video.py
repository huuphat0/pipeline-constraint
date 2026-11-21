"""Một lệnh: video RGB-D -> video gắn nhãn SKILL + SUB-SKILL (như out/mp4/demo_final_h264.mp4).

Các bước:
  1. Tiền xử lý (preprocessing/run_pipeline.py --to skills):
       .mcap -> mp4 màu + depth.npz -> mask GSAM2 -> skills_lowlevel.json
     Bỏ qua nếu dùng --from-run (tái sử dụng kết quả rekep_keypoint_constraints).
  2. Sub-skill (preprocessing/skill_inference/subskill_inference.py):
       Grasp -> Approach / Align / Close_Gripper
       Pour  -> Orient_Tilt / Hold_Pour_Angle / Return_Upright
  3. Vẽ video (skill_video/render_hoi_video.py): mask tay/vật, tên skill + sub-skill,
     điểm chạm, trục vật + toạ độ mm, dải timeline sub-skill.
  4. Convert sang H.264 bằng ffmpeg để phát được trên trình duyệt.

Cách dùng (từ gốc repo):
    # Từ video .mcap (chạy cả tiền xử lý)
    python skill_video/make_skill_video.py video_depthcam/demo.mcap \\
        --prompt "hand . plastic water bottle . white cup ."

    # Tái sử dụng thư mục kết quả của rekep_keypoint_constraints/run_pipeline.py
    python skill_video/make_skill_video.py --from-run rekep_keypoint_constraints/artifacts/demo
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPROCESSING = ROOT / "preprocessing"


def run(cmd, log):
    """Chạy một bước, ghi log; dừng ngay nếu lỗi."""
    print(f"$ {' '.join(map(str, cmd))}", flush=True)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(map(str, cmd))}\n")
        handle.flush()
        code = subprocess.run(list(map(str, cmd)), stdout=handle, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f"Lệnh lỗi (mã {code}), xem {log}")


def inputs_from_run(run_dir):
    """Tìm mp4/mask/depth/skills trong <run_dir>/upstream do run_pipeline.py tạo."""
    up = Path(run_dir).resolve() / "upstream"
    manifest = Path(run_dir).resolve() / "run_manifest.json"
    if manifest.is_file():
        files = json.loads(manifest.read_text(encoding="utf-8")).get("upstream_files", {})
        if all(k in files for k in ("video", "masks", "depth", "skills")):
            return {k: Path(files[k]) for k in ("video", "masks", "depth", "skills")}
    videos = sorted((up / "mp4").glob("*_color.mp4"))
    if len(videos) != 1:
        raise ValueError(f"Không xác định được video trong {up / 'mp4'}: {videos}")
    name = videos[0].name[:-len("_color.mp4")]
    return {"video": videos[0], "masks": up / f"gsam2_{name}" / "masks_gsam2_merged.json",
            "depth": up / "npz" / f"{name}_depth.npz",
            "skills": up / "json" / f"{name}_skills_lowlevel.json"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mcap", nargs="?", help="video ROS2 bag .mcap có màu + depth căn màu + camera_info")
    ap.add_argument("--prompt", help='nhãn vật cho GSAM2, vd "hand . plastic water bottle . white cup ."')
    ap.add_argument("--from-run", help="thư mục kết quả của rekep_keypoint_constraints/run_pipeline.py")
    ap.add_argument("--name", help="tên ngắn; mặc định lấy từ tên file")
    ap.add_argument("--out", help="thư mục kết quả; mặc định skill_video/output/<name>")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="GSAM2: giây giữa 2 keyframe Grounding DINO (mặc định 1.0, như video demo_final)")
    ap.add_argument("--scale", type=float, default=2.0, help="phóng to khung hình (mặc định 2)")
    ap.add_argument("--no-mask", action="store_true", help="không tô màu mask")
    ap.add_argument("--no-pose-axis", action="store_true", help="không vẽ trục vật + toạ độ mm")
    args = ap.parse_args()
    if bool(args.mcap) == bool(args.from_run):
        ap.error("truyền đúng MỘT trong hai: file .mcap hoặc --from-run")
    if args.mcap and not args.prompt:
        ap.error("video .mcap cần --prompt để tách vật bằng GSAM2")

    source = Path(args.mcap or args.from_run).resolve()
    name = args.name or re.sub(r"[^a-zA-Z0-9_]", "_", source.stem if args.mcap else source.name)
    out = Path(args.out).resolve() if args.out else HERE / "output" / name
    out.mkdir(parents=True, exist_ok=True)
    log = out / "make_skill_video.log"
    log.write_text("", encoding="utf-8")
    py = sys.executable

    # 1. Tiền xử lý tới skills (bước nào đã có file thì preprocessing tự bỏ qua)
    if args.mcap:
        if not source.is_file():
            raise SystemExit(f"LỖI: không thấy file {source}")
        upstream = out / "upstream"
        run([py, PREPROCESSING / "run_pipeline.py", source, "--name", name, "--prompt", args.prompt,
             "--out-root", upstream, "--interval", args.interval, "--to", "skills"], log)
        files = {"video": upstream / "mp4" / f"{name}_color.mp4",
                 "masks": upstream / f"gsam2_{name}" / "masks_gsam2_merged.json",
                 "depth": upstream / "npz" / f"{name}_depth.npz",
                 "skills": upstream / "json" / f"{name}_skills_lowlevel.json"}
    else:
        files = inputs_from_run(source)
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"LỖI: thiếu file đầu vào: {missing}")

    # 2. Sub-skill
    subskills = out / f"{name}_subskills.json"
    run([py, PREPROCESSING / "skill_inference" / "subskill_inference.py", files["masks"],
         "--depth", files["depth"], "--out", subskills], log)

    # 3. Vẽ video (codec mp4v của OpenCV)
    raw = out / f"{name}_skill_video.mp4"
    contacts = out / f"{name}_contacts_per_frame.json"
    cmd = [py, HERE / "render_hoi_video.py", files["video"], "--masks", files["masks"],
           "--skills", files["skills"], "--depth", files["depth"], "--subskills", subskills,
           "--dump-contacts", contacts, "--out", raw, "--scale", args.scale]
    cmd += ["--no-mask"] if args.no_mask else []
    cmd += ["--no-pose-axis"] if args.no_pose_axis else []
    run(cmd, log)

    # 4. H.264 để xem trên trình duyệt / VS Code
    h264 = out / f"{name}_skill_video_h264.mp4"
    if shutil.which("ffmpeg"):
        run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-c:v", "libx264", "-preset", "slow",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", h264], log)
    else:
        h264 = None
        print("CẢNH BÁO: không có ffmpeg — chỉ có bản mp4v, có thể không phát được trên trình duyệt")

    print(json.dumps({"video": str(h264 or raw), "subskills": str(subskills),
                      "contacts_per_frame": str(contacts), "log": str(log)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        sys.exit(1)
