"""
run_pipeline.py

Tiền xử lý MỘT video ROS2 bag (.mcap) RGB-D cho rekep_keypoint_constraints:

    mp4      data_io/mcap_to_mp4.py           -> <out>/mp4/<name>_color.mp4
    depth    data_io/depth_from_mcap.py       -> <out>/npz/<name>_depth.npz
    gsam2    segmentation/gsam2_video.py + merge_object_ids.py
                                              -> <out>/gsam2_<name>/masks_gsam2_merged.json
    skills   skill_inference/hoi_skill_inference.py -> <out>/json/<name>_skills_lowlevel.json
    pose3d   perception/object_pose.py        -> <out>/json/<name>_pose3d.json
    params   skill_inference/skill_params.py  -> <out>/json/<name>_skill_params.json
    series   constraint_inference/measure_series.py -> <out>/json/<name>_series.json

Bước nào đã có file đầu ra thì bỏ qua (thêm --force để chạy lại). Dùng --from/--to
để chạy một khúc. Thư mục GSAM2 chỉ định bằng --sam-dino (mặc định ../segmentation).

Cách dùng (từ thư mục preprocessing/):
    python3 run_pipeline.py ../video_depthcam/demo.mcap --name demo \\
        --prompt "hand . plastic water bottle . white cup ." --out-root ../out
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["mp4", "depth", "gsam2", "skills", "pose3d", "params", "series"]


def paths(root, name):
    j = os.path.join(root, "json")
    return {
        "mp4": os.path.join(root, "mp4", f"{name}_color.mp4"),
        "depth": os.path.join(root, "npz", f"{name}_depth.npz"),
        "gsam2_dir": os.path.join(root, f"gsam2_{name}"),
        "masks": os.path.join(root, f"gsam2_{name}", "masks_gsam2_merged.json"),
        "skills": os.path.join(j, f"{name}_skills_lowlevel.json"),
        "pose3d": os.path.join(j, f"{name}_pose3d.json"),
        "params": os.path.join(j, f"{name}_skill_params.json"),
        "series": os.path.join(j, f"{name}_series.json"),
    }


def build_commands(args, p):
    py = sys.executable
    raw_masks = os.path.join(p["gsam2_dir"], "masks_gsam2.json")
    return {
        # (lệnh, file đầu ra dùng để biết bước đã xong)
        "mp4": ([[py, "data_io/mcap_to_mp4.py", args.mcap, "--out", p["mp4"]]], p["mp4"]),
        "depth": ([[py, "data_io/depth_from_mcap.py", args.mcap, "--out", p["depth"]]], p["depth"]),
        "gsam2": ([[py, os.path.join(args.sam_dino, "gsam2_video.py"), p["mp4"],
                    "--prompt", args.prompt, "--interval", str(args.interval),
                    "--outdir", p["gsam2_dir"]],
                   [py, os.path.join(args.sam_dino, "merge_object_ids.py"), raw_masks,
                    "--out", p["masks"]]], p["masks"]),
        "skills": ([[py, "skill_inference/hoi_skill_inference.py", p["masks"],
                     "--depth", p["depth"], "--out", p["skills"]]], p["skills"]),
        "pose3d": ([[py, "perception/object_pose.py", p["masks"], "--depth", p["depth"],
                     "--out", p["pose3d"]]], p["pose3d"]),
        "params": ([[py, "skill_inference/skill_params.py", p["masks"], "--skills", p["skills"],
                     "--depth", p["depth"], "--out", p["params"]]], p["params"]),
        "series": ([[py, "constraint_inference/measure_series.py", "--masks", p["masks"],
                     "--depth", p["depth"], "--pose3d", p["pose3d"], "--out", p["series"]]],
                   p["series"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mcap")
    ap.add_argument("--name", required=True, help="tên ngắn của video, dùng đặt tên file đầu ra")
    ap.add_argument("--prompt", help='prompt GSAM2, vd "hand . plastic water bottle . white cup ."'
                                     " (bắt buộc khi phải chạy bước gsam2)")
    ap.add_argument("--out-root", default=os.path.join(HERE, "..", "out"))
    ap.add_argument("--sam-dino", default=os.path.join(HERE, "..", "segmentation"),
                    help="thư mục chứa gsam2_video.py, merge_object_ids.py")
    ap.add_argument("--interval", type=float, default=1.5, help="GSAM2: giây giữa 2 keyframe")
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    ap.add_argument("--to", dest="end", choices=STEPS, default=STEPS[-1])
    ap.add_argument("--force", action="store_true", help="chạy lại cả bước đã có đầu ra")
    args = ap.parse_args()

    root = os.path.abspath(args.out_root)
    args.mcap = os.path.abspath(args.mcap)
    args.sam_dino = os.path.abspath(args.sam_dino)
    p = paths(root, args.name)
    for d in ("mp4", "npz", "json"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    cmds = build_commands(args, p)
    todo = STEPS[STEPS.index(args.start):STEPS.index(args.end) + 1]
    t_all = time.time()
    for step in todo:
        steps, output = cmds[step]
        if output and os.path.isfile(output) and not args.force:
            print(f"[{step}] đã có {os.path.relpath(output, root)} -> bỏ qua")
            continue
        if step == "gsam2" and not args.prompt:
            ap.error("bước gsam2 cần --prompt")
        print(f"[{step}] chạy...", flush=True)
        t0 = time.time()
        for cmd in steps:
            r = subprocess.run(cmd, cwd=HERE)
            if r.returncode != 0:
                print(f"[{step}] LỖI (mã {r.returncode}): {' '.join(cmd)}")
                return r.returncode
        print(f"[{step}] xong sau {time.time() - t0:.0f} s")
    print(f"\nXong {len(todo)} bước sau {time.time() - t_all:.0f} s. Kết quả: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
