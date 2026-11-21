"""Bước 5: ánh xạ skill mức thấp thành stage ReKep và đo độ phủ keypoint."""

import argparse
import json
from pathlib import Path


def phase_for_skill(skill, seen_pour):
    if skill in {"Reach", "Contact", "Grasp"}:
        return "grasp"
    if skill in {"Lift", "MoveToTarget"} and not seen_pour:
        return "transport_to_target"
    if skill == "Pour":
        return "pour"
    if skill in {"MoveToTarget", "Place", "Release"} and seen_pour:
        return "return_place"
    if skill == "Retract":
        return "retract"
    return f"other_{skill.lower()}"


def build_stages(skills, tracks, instruction, series=None, instruction_review_pending=True):
    fps = float(skills["fps"])
    segments = [s for s in skills["segments"] if s["skill_en"] != "Idle"]
    grouped = []
    seen_pour = False
    for seg in segments:
        phase = phase_for_skill(seg["skill_en"], seen_pour)
        if phase == "pour":
            seen_pour = True
        if grouped and grouped[-1]["phase"] == phase:
            grouped[-1]["segments"].append(seg)
        else:
            grouped.append({"phase": phase, "segments": [seg]})
    if series is not None:
        split_groups = []
        for group in grouped:
            if group["phase"] != "pour" or len(group["segments"]) != 1:
                split_groups.append(group)
                continue
            seg = group["segments"][0]
            obj_series = series["objects"].get(seg.get("object_label"), [])
            start = max(0, round(float(seg["time_sec"]) * fps))
            end = min(tracks["n_frames"] - 1, round(float(seg["end_sec"]) * fps) - 1)
            samples = [(int(x["frame"]), float(x["tilt_from_vertical_deg"])) for x in obj_series
                       if start <= int(x["frame"]) <= end and x.get("tilt_from_vertical_deg") is not None]
            peak = max(samples, key=lambda x: x[1])[0] if samples else None
            if peak is not None and start + 10 < peak < end - 10:
                split_groups.append({**group, "phase": "tilt_to_pour",
                                     "start_frame": start, "end_frame": peak})
                split_groups.append({**group, "phase": "restore_upright",
                                     "start_frame": peak + 1, "end_frame": end})
            else:
                split_groups.append(group)
        grouped = split_groups
    stages = []
    for index, group in enumerate(grouped, start=1):
        segs = group["segments"]
        start_t, end_t = float(segs[0]["time_sec"]), float(segs[-1]["end_sec"])
        start = group.get("start_frame", max(0, min(tracks["n_frames"] - 1, round(start_t * fps))))
        end = group.get("end_frame", max(start, min(tracks["n_frames"] - 1, round(end_t * fps) - 1)))
        if "start_frame" in group:
            start_t, end_t = float(tracks["frames"][start]["t"]), float(tracks["frames"][end]["t"])
        frame_records = tracks["frames"][start:end + 1]
        availability = {}
        path_ids, subgoal_ids = [], []
        for point in tracks["points"]:
            if not point["usable_for_constraints"]:
                continue
            point_id = point["id"]
            statuses = [f["points"][point_id]["status"] for f in frame_records]
            available = [x in {"visible", "inferred"} for x in statuses]
            overall = sum(available) / len(available)
            final = sum(available[-5:]) / len(available[-5:])
            availability[str(point_id)] = {"available_fraction": round(overall, 3),
                                            "observed_fraction": round(sum(x == "visible" for x in statuses) / len(statuses), 3),
                                            "final_available_fraction": round(final, 3),
                                            "source": point.get("source", "mask_dinov2")}
            keyframes = sorted(set([start, (start + end) // 2, end]))
            visible_at_keyframes = all(tracks["frames"][f]["points"][point_id]["status"] in {"visible", "inferred"}
                                       for f in keyframes)
            if overall >= 0.9 and visible_at_keyframes:
                path_ids.append(point_id)
            if final >= 0.6:
                subgoal_ids.append(point_id)
        phase = group["phase"]
        stages.append({
            "stage": index, "name": phase, "start_frame": start, "end_frame": end,
            "start_sec": round(start_t, 3), "end_sec": round(end_t, 3),
            "skill_labels": [s["skill_en"] for s in segs],
            "manipulated_object_id": segs[0].get("object_id"),
            "movable_object_ids": [segs[0].get("object_id")] if phase in
                {"transport_to_target", "pour", "tilt_to_pour", "restore_upright", "return_place"} else [],
            "grasp_stage": phase == "grasp", "release_at_end": phase == "return_place",
            "constraint_generation": phase != "retract",
            "keyframes": sorted(set([start, (start + end) // 2, end])),
            "keypoint_availability": availability,
            "path_eligible_ids": path_ids, "subgoal_eligible_ids": subgoal_ids,
        })
    return {"instruction": instruction, "instruction_review_pending": instruction_review_pending,
            "source_skills": skills.get("source_masks", ""), "stage_count": len(stages),
            "stage_rule": "grasp independent; pre-Pour transport; Pour split at measured peak tilt; post-Pour return/place; Retract observational",
            "stages": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    local = Path(__file__).resolve().parents[1]
    parser.add_argument("--skills", default=str(root / "out/json/skills_lowlevel.json"))
    parser.add_argument("--tracks", default=str(local / "artifacts/demo/keypoints/keypoints_tracks.json"))
    parser.add_argument("--instruction", default=str(local / "artifacts/demo/stages/instruction.json"))
    parser.add_argument("--series", default=str(root / "out/json/demo_series.json"))
    parser.add_argument("--out", default=str(local / "artifacts/demo/stages/stages.json"))
    args = parser.parse_args()
    skills = json.loads(Path(args.skills).read_text(encoding="utf-8"))
    tracks = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    instruction_data = json.loads(Path(args.instruction).read_text(encoding="utf-8"))
    instruction = instruction_data["instruction"]
    series = json.loads(Path(args.series).read_text(encoding="utf-8")) if args.series else None
    result = build_stages(skills, tracks, instruction, series,
                          instruction_data.get("needs_review", True))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: s[k] for k in ("stage", "name", "skill_labels", "start_frame", "end_frame",
                                            "path_eligible_ids", "subgoal_eligible_ids")}
                      for s in result["stages"]], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
