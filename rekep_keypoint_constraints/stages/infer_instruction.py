"""Suy chỉ dẫn task từ nhiều frame RGB bằng VLM cục bộ; luôn đánh dấu cần duyệt."""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # gốc repo, để import rekep_keypoint_constraints.*
from rekep_keypoint_constraints.common.model_client import call_role
from rekep_keypoint_constraints.common.rgbd import RGBDSequence


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "instruction": {"type": "string"},
        "visible_actions": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["instruction", "visible_actions", "evidence", "uncertainties"]
}


def montage(seq, times, out_path):
    frames = []
    for t in times:
        frame = min(seq.n_frames - 1, round(t * seq.fps))
        bgr = cv2.cvtColor(seq.rgb(frame), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (480, 360))
        cv2.rectangle(bgr, (0, 0), (480, 34), (0, 0, 0), -1)
        cv2.putText(bgr, f"t={seq.times[frame]:.2f}s", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        frames.append(bgr)
    panel = cv2.vconcat([cv2.hconcat(frames[:2]), cv2.hconcat(frames[2:4])])
    cv2.imwrite(str(out_path), panel)


def infer(seq, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "instruction_frames.jpg"
    montage(seq, [0, 3.5, 5.5, 9.5], image_path)
    prompt = (
        "Bốn ảnh theo thời gian của CÙNG MỘT video thao tác, thời gian ghi trên ảnh. "
        "Hãy suy ra một instruction ngắn cho robot mô tả mục tiêu tổng thể. "
        "Chỉ mô tả hành động nhìn được; đừng khẳng định chất lỏng/chất liệu/nắp nếu không chắc. "
        "Liệt kê hành động nhìn thấy và bằng chứng theo mốc thời gian. "
        "Trong uncertainties nêu mọi điểm không thể biết chắc chỉ từ ảnh. "
        "Trả về JSON đúng schema."
    )
    (out_dir / "instruction_prompt.txt").write_text(prompt, encoding="utf-8")
    result = call_role("vision", prompt, images=[str(image_path)], schema=SCHEMA, think=False)
    data = result["data"]
    raw_instruction = data["instruction"]
    uncertainty_text = " ".join(data["uncertainties"]).lower()
    if "nước hay chất lỏng khác" in uncertainty_text and "nước" in raw_instruction.lower():
        data["instruction"] = raw_instruction.replace("nước", "chất lỏng")
        data["normalization_note"] = "VLM không chắc chất lỏng là nước; dùng từ trung tính."
    if "đặt chai" in " ".join(data["visible_actions"]).lower() and "đặt chai" not in data["instruction"].lower():
        data["instruction"] = data["instruction"].rstrip(".") + ", rồi đặt chai trở lại bàn."
        data["normalization_note"] = data.get("normalization_note", "") + " Bổ sung hành động đặt chai mà VLM đã quan sát."
    data.update({"source": "vlm_video_frames", "model": result["model"],
                 "needs_review": True, "review_status": "pending",
                 "frames_seconds": [0, 3.5, 5.5, 9.5]})
    (out_dir / "instruction.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "instruction_raw.txt").write_text(result["raw"], encoding="utf-8")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parents[1]
    parser.add_argument("--video", default=str(root / "out/mp4/demo_color.mp4"))
    parser.add_argument("--depth", default=str(root / "out/npz/demo_depth.npz"))
    parser.add_argument("--masks", default=str(root / "out/gsam2_demo/masks_gsam2_merged.json"))
    parser.add_argument("--out", default=str(local / "artifacts/demo/stages"))
    args = parser.parse_args()
    data = infer(RGBDSequence(args.video, args.depth, args.masks), args.out)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
