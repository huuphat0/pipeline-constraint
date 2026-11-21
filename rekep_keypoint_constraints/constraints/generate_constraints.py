"""Bước 7: VLM đề xuất quan hệ keypoint; LLM rà soát ở dạng cấu trúc."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # gốc repo, để import rekep_keypoint_constraints.*
from rekep_keypoint_constraints.common.model_client import call_role


OPERATIONS = ["distance_le", "distance_ge", "orientation_change_le",
              "orientation_change_ge", "height_above_table_le", "height_above_table_ge",
              "distance_to_reference_le"]

CANDIDATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "constraints": {
            "type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "phase": {"type": "string", "enum": ["path", "subgoal"]},
                    "op": {"type": "string", "enum": OPERATIONS},
                    "ids": {"type": "array", "items": {"type": "integer"}},
                    "reason": {"type": "string"},
                    "visual_evidence": {"type": "string"}
                },
                "required": ["phase", "op", "ids", "reason", "visual_evidence"]
            }
        }
    }, "required": ["constraints"]
}

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "decisions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"stage": {"type": "integer"}, "candidate": {"type": "integer"},
                           "accept": {"type": "boolean"}, "note": {"type": "string"}},
            "required": ["stage", "candidate", "accept", "note"]
        }}
    }, "required": ["decisions"]
}


def stage_prompt(stage, stages, tracks, instruction):
    ids = sorted(set(stage["path_eligible_ids"] + stage["subgoal_eligible_ids"]))
    points = [{"id": p["id"], "object": p["object_label"], "region": p["region"],
               "source": p.get("source", "mask_dinov2")}
              for p in tracks["points"] if p["id"] in ids]
    context = {
        "instruction_provisional": instruction["instruction"],
        "instruction_uncertainties": instruction["uncertainties"],
        "stage": stage["stage"], "stage_name": stage["name"], "skill_labels": stage["skill_labels"],
        "stage_order": [s["name"] for s in stages["stages"]],
        "path_eligible_ids": stage["path_eligible_ids"],
        "subgoal_eligible_ids": stage["subgoal_eligible_ids"], "points": points,
        "grasp_stage": stage["grasp_stage"], "release_at_end": stage["release_at_end"]
    }
    guidance = {
        "grasp": "Đây là stage grasp RIÊNG: cần một subgoal đưa end_effector (-1) tới điểm nắm trên chai. Không đưa thêm path nếu không cần.",
        "transport_to_target": "Chai đang được mang tới cốc. Xét subgoal quan hệ vị trí chai-cốc và path giữ tư thế chai nếu việc đổ chất lỏng là nguy cơ có căn cứ.",
        "tilt_to_pour": "Cuối stage chai ở tư thế nghiêng để rót; ưu tiên quan hệ hướng chai và vị trí với cốc, tránh khoảng cách giữa hai điểm trên cùng chai.",
        "restore_upright": "Cuối stage chai được dựng thẳng lại sau khi rót; xét subgoal hướng chai gần hướng ban đầu để ngừng rót.",
        "return_place": "Cuối stage chai được đặt lại bàn; xét subgoal vị trí chai gần vị trí ban đầu và giữ tư thế phù hợp trên đường về."
    }
    return (
        "Bạn tạo ứng viên constraint kiểu ReKep từ ảnh đầu/giữa/cuối của MỘT stage. "
        "Số trên ảnh là ID keypoint, xanh là cốc, cam là chai; vòng tròn rỗng là điểm trục chai SUY RA từ pose 3D, không trực tiếp nhìn thấy. "
        "Chỉ nêu quan hệ thực sự cần cho mục tiêu task, tối đa 3 constraint. "
        "path phải duy trì trong cả stage; subgoal phải đúng ở CUỐI stage. "
        "Chỉ dùng ID trong path_eligible_ids cho path, subgoal_eligible_ids cho subgoal. "
        "Ngoại lệ: ID -1 là end_effector, chỉ được dùng cho subgoal của grasp; "
        "điểm -1 không có số đo 3D trên video và sẽ được đánh dấu chưa kiểm chứng. "
        "Các phép hợp lệ: distance_le/ge giữa 2 ID; orientation_change_le/ge "
        "giữa 2 ID CÙNG vật so với hướng cặp điểm lúc đầu video; "
        "height_above_table_le/ge của 1 ID; distance_to_reference_le của 1 ID "
        "so với vị trí ban đầu. Không đoán ngưỡng số; bước sau sẽ lấy từ số đo thật. "
        "Nếu cần hướng chai, ưu tiên cặp axis_positive/axis_negative suy từ pose 3D khi chúng có trong danh sách; "
        "ghi rõ đây là bằng chứng ước lượng. Không dùng distance_le giữa hai điểm trên cùng một vật cứng. "
        "Với stage tilt_to_pour, mục tiêu là tư thế nghiêng ở cuối stage; "
        "với restore_upright, mục tiêu là dựng lại ở cuối stage. "
        "Không giả định chất lỏng chắc chắn là nước hay biết trạng thái nắp. "
        "Nếu không đủ căn cứ, trả constraints rỗng. Gợi ý riêng cho stage hiện tại: "
        + guidance.get(stage["name"], "") + " Trả JSON đúng schema.\n\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def generate(stages, tracks, instruction, keyframes_dir, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"instruction": instruction["instruction"],
              "instruction_review_pending": instruction.get("needs_review", True),
              "vision_model": "qwen3-vl:8b-instruct", "reasoning_model": "qwen3.5:9b",
              "stages": []}
    for stage in stages["stages"]:
        if not stage["constraint_generation"]:
            result["stages"].append({"stage": stage["stage"], "name": stage["name"],
                                     "candidates": [], "status": "observational_only"})
            continue
        image = Path(keyframes_dir) / f"stage{stage['stage']}_{stage['name']}.jpg"
        prompt = stage_prompt(stage, stages, tracks, instruction)
        (out_dir / f"stage{stage['stage']}_prompt.txt").write_text(prompt, encoding="utf-8")
        response = call_role("vision", prompt, images=[str(image)], schema=CANDIDATE_SCHEMA, think=False)
        (out_dir / f"stage{stage['stage']}_vlm_raw.txt").write_text(response["raw"], encoding="utf-8")
        candidates = response["data"]["constraints"][:3]
        result["stages"].append({"stage": stage["stage"], "name": stage["name"],
                                 "image": str(image), "candidates": candidates,
                                 "vlm_cached": response["cached"]})
        print(f"stage {stage['stage']} {stage['name']}: {len(candidates)} ứng viên VLM", flush=True)
        (out_dir / "constraints_raw.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    review_input = [{"stage": s["stage"], "name": s["name"], "candidates": s["candidates"]}
                    for s in result["stages"] if s["candidates"]]
    if review_input:
        prompt = (
            "Rà soát ứng viên constraint ReKep dưới đây theo task và stage. "
            "Đánh dấu accept=false nếu quan hệ không cần cho task, sai nghĩa path/subgoal, "
            "dùng ID sai, hoặc chỉ bắt chước chi tiết tùy ý của video. "
            "Mỗi ứng viên phải có đúng một quyết định, candidate là chỉ số 0-based trong stage. "
            "Không tự thêm constraint mới. Trả JSON đúng schema.\n"
            + json.dumps({"instruction": instruction["instruction"], "stages": review_input},
                         ensure_ascii=False, indent=2)
        )
        (out_dir / "review_prompt.txt").write_text(prompt, encoding="utf-8")
        review = call_role("reasoning", prompt, schema=REVIEW_SCHEMA, think=False)
        (out_dir / "review_llm_raw.txt").write_text(review["raw"], encoding="utf-8")
        decisions = {(x["stage"], x["candidate"]): x for x in review["data"]["decisions"]}
        for entry in result["stages"]:
            for index, candidate in enumerate(entry["candidates"]):
                decision = decisions.get((entry["stage"], index))
                candidate["llm_accept"] = bool(decision and decision["accept"])
                candidate["llm_review_note"] = decision["note"] if decision else "LLM không trả quyết định"
        result["llm_review_cached"] = review["cached"]
    (out_dir / "constraints_raw.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local = Path(__file__).resolve().parents[1]
    base = local / "artifacts/demo"
    parser.add_argument("--stages", default=str(base / "stages/stages.json"))
    parser.add_argument("--tracks", default=str(base / "keypoints/keypoints_tracks.json"))
    parser.add_argument("--instruction", default=str(base / "stages/instruction.json"))
    parser.add_argument("--keyframes", default=str(base / "keyframes"))
    parser.add_argument("--out", default=str(base / "constraints"))
    args = parser.parse_args()
    stages = json.loads(Path(args.stages).read_text(encoding="utf-8"))
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    instruction = json.loads(Path(args.instruction).read_text(encoding="utf-8"))
    result = generate(stages, tracks, instruction, args.keyframes, args.out)
    for stage in result["stages"]:
        print(stage["stage"], stage["name"],
              [(c["phase"], c["op"], c["ids"], c.get("llm_accept")) for c in stage["candidates"]])


if __name__ == "__main__":
    main()
