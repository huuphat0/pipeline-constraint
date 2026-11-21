"""
constraint_schema.py

Bước 1 của phần suy luận constraint (xem out/bao_cao/suy_luan_constraint_skill.md):
định nghĩa KHUÔN DỮ LIỆU dùng chung cho các bước sau.

    SCENE_VLM_SCHEMA   phần VLM điền ở bước A (chỉ MÔ TẢ cảnh, không suy luận)
    MEASURED_FIELDS    khối số đo từ depth do code ghép vào scene (không qua model)
    DIMENSIONS         checklist 11 chiều bước B BẮT BUỘC xét hết cho mỗi skill
    CONSTRAINT_SCHEMA  đầu ra của bước B cho MỘT đoạn skill
    validate()         kiểm tra một dict theo schema (dùng ở bước 2 và bước 5)

Các schema là JSON Schema dạng dict Python thuần, chỉ dùng tập con mà Ollama
hỗ trợ cho tham số `format` (type, properties, required, enum, items,
additionalProperties) — truyền thẳng vào Ollama để ép model trả đúng khuôn,
và dùng lại chính nó để kiểm tra ở bước 5. Không cần cài jsonschema/pydantic.

Mọi trường có thể không nhìn thấy đều có giá trị "unknown": model được phép
nói "không biết" thay vì bị khuôn ép phải đoán.

QUY ƯỚC `evidence` (bước B trích dẫn, bước 5 tra ngược vào scene):
    obj<ID>.<trường>=<giá trị>      obj1.state.contents=liquid, obj1.symmetry=rotational_vertical
    env.<trường>=<giá trị>          env.clutter=low, env.obstacles_near_path=[]
    rel:<ID> <quan hệ> <ID>         rel:1 left_of 0
    measured.<trường>[=<giá trị>]   measured.tilt_from_rest_deg.max=7.8  (giá trị không bắt buộc;
                                    có thì bước 5 đối chiếu với scene)
    segment.<trường>=<giá trị>      segment.next=Pour
Dùng ID vật (khớp mask GSAM2), KHÔNG dùng chỉ số mảng, để không nhầm khi
model liệt kê vật theo thứ tự khác.

Chạy trực tiếp để in schema và tự kiểm tra ví dụ mẫu:
    python3 constraint_inference/constraint_schema.py
    python3 constraint_inference/constraint_schema.py --print constraint
"""

import argparse
import json
import re
import sys

UNKNOWN = "unknown"

# skill không có thao tác -> bỏ qua, không suy luận constraint
SKIP_SKILLS = {"Idle"}

# ---------------------------------------------------------------------------
# Bước A: mô tả cảnh do VLM điền
# ---------------------------------------------------------------------------

LID_STATES = ["open", "closed", "none", UNKNOWN]            # none = vật không có nắp
CONTENTS = ["liquid", "solid", "granular", "empty", UNKNOWN]
FILL_LEVELS = ["full", "half", "low", "empty", UNKNOWN]
MATERIALS = ["plastic", "glass", "ceramic", "metal", "paper", "wood",
             "fabric", "food", "other", UNKNOWN]
YES_NO = ["yes", "no", UNKNOWN]
DEFORMABLE = ["rigid", "slightly", "highly", UNKNOWN]
SHAPES = ["cylinder", "box", "sphere", "bowl", "flat", "irregular", UNKNOWN]
# rotational_vertical: xoay quanh trục dọc không đổi hình dạng (chai, cốc không quai)
SYMMETRY = ["rotational_vertical", "mirror", "none", UNKNOWN]
SUPPORT_SURFACES = ["table", "shelf", "counter", "floor", "other", UNKNOWN]
CLUTTER = ["low", "medium", "high"]
TEMPERATURES = ["hot", "cold", "normal", UNKNOWN]           # chỉ nhìn được khi có hơi / đá
OPENING_SIZES = ["narrow", "wide", "none", UNKNOWN]         # miệng chai hẹp, miệng cốc/bát rộng
# cơ cấu mở/đóng: hinge = cửa/nắp bản lề, slide = ngăn kéo, screw_cap = nắp vặn,
# snap_cap = nắp bật/ấn -> quyết định contact_type khi Open/Close
ARTICULATIONS = ["none", "hinge", "slide", "screw_cap", "snap_cap", UNKNOWN]
WEIGHT_CLASSES = ["light", "medium", "heavy", UNKNOWN]
SIDES = ["top", "front", "back", "left", "right"]
RELATIONS = ["left_of", "right_of", "in_front_of", "behind", "above", "below",
             "inside", "on_top_of", "next_to"]


def _enum(values):
    return {"type": "string", "enum": list(values)}


OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "label": {"type": "string"},
        "state": {
            "type": "object",
            "properties": {
                "lid": _enum(LID_STATES),
                "contents": _enum(CONTENTS),
                "fill_level": _enum(FILL_LEVELS),
                "temperature": _enum(TEMPERATURES),
            },
            "required": ["lid", "contents", "fill_level", "temperature"],
            "additionalProperties": False,
        },
        "material": _enum(MATERIALS),
        "fragile": _enum(YES_NO),
        "deformable": _enum(DEFORMABLE),
        "shape": _enum(SHAPES),
        "symmetry": _enum(SYMMETRY),
        # yes = miệng hở không có nắp (cốc, bát); khác chai đã mở nắp (lid=open)
        "open_top": _enum(YES_NO),
        "opening_size": _enum(OPENING_SIZES),
        "articulation": _enum(ARTICULATIONS),
        "sharp": _enum(YES_NO),
        "weight_class": _enum(WEIGHT_CLASSES),
        # vị trí trong cảnh (đổi được giữa các đoạn)
        "items_on_top": {"type": "array", "items": {"type": "integer"}},   # ID vật đặt trên nó
        "near_edge": _enum(YES_NO),                                      # sát mép bàn/kệ
        "accessible_sides": {"type": "array", "items": _enum(SIDES)},    # phía tay/gripper tới được
        # bộ phận cầm được / bộ phận chức năng, vd ["body", "handle"], ["opening", "spout"]
        "graspable_parts": {"type": "array", "items": {"type": "string"}},
        "functional_parts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id", "label", "state", "material", "fragile", "deformable",
                 "shape", "symmetry", "open_top", "opening_size", "articulation", "sharp",
                 "weight_class", "items_on_top", "near_edge", "accessible_sides",
                 "graspable_parts", "functional_parts"],
    "additionalProperties": False,
}

# Thuộc tính KHÔNG đổi theo thời gian -> bước 3 bỏ phiếu đa số trên cả video
STATIC_FIELDS = ("material", "fragile", "deformable", "shape", "symmetry", "open_top",
                 "opening_size", "articulation", "sharp", "weight_class")


def unknown_object(obj_id, label):
    """Vật với mọi thuộc tính 'unknown' — điền khi VLM bỏ sót một ID."""
    return {
        "id": obj_id, "label": label,
        "state": {"lid": UNKNOWN, "contents": UNKNOWN, "fill_level": UNKNOWN,
                  "temperature": UNKNOWN},
        **{f: UNKNOWN for f in STATIC_FIELDS},
        "items_on_top": [], "near_edge": UNKNOWN, "accessible_sides": [],
        "graspable_parts": [], "functional_parts": [],
    }


# Định nghĩa HẸP của humans_nearby, dùng chung cho prompt mô tả và prompt chuyển
# JSON. Bản cũ ("người/tay khác ở gần") ra `yes` cho cả video demo vì có người
# đứng sau bàn, kéo theo ~10 constraint tốc độ/khoảng cách không thực thi được
# (`clear_from_humans`, `speed <= safe_limit`) — xem time_line.md.
HUMANS_NEARBY_DEF = (
    "another person's hand or arm inside the work area, i.e. on or directly above the table "
    "within reach of the tagged objects (closer than about one arm's length). People standing "
    "in the background, behind the table or only partly visible at the image border do NOT count")


SCENE_VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": OBJECT_SCHEMA},
        "environment": {
            "type": "object",
            "properties": {
                "support_surface": _enum(SUPPORT_SURFACES),
                # ID các vật nằm giữa vật đang cầm và đích
                "obstacles_near_path": {"type": "array", "items": {"type": "integer"}},
                "clutter": _enum(CLUTTER),
                # người/tay KHÁC trong tầm với của vùng làm việc (trên/sát mặt bàn);
                # người đứng phía sau, ngoài tầm với KHÔNG tính (xem HUMANS_NEARBY_DEF)
                "humans_nearby": _enum(YES_NO),
            },
            "required": ["support_surface", "obstacles_near_path", "clutter", "humans_nearby"],
            "additionalProperties": False,
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "integer"},
                    "relation": _enum(RELATIONS),
                    "object": {"type": "integer"},
                },
                "required": ["subject", "relation", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["objects", "environment", "relations"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Khối số đo do CODE ghép vào scene ở bước A (nguồn: skill_params.json,
# demo_pose3d.json). Không đưa vào schema của model — model không được điền.
# ---------------------------------------------------------------------------

MEASURED_FIELDS = {
    "height_mm": "độ cao vật so với mặt bàn {start, max, end} (mm). Có chuỗi từng khung "
                 "(measure_series.py) thì là độ nâng của TÂM vật dọc pháp tuyến mặt bàn so "
                 "với lúc vật nằm yên; không có thì lấy từ skill_params.json (hiệu độ sâu)",
    "lift_mm": "giống height_mm nhưng lấy từ chuỗi từng khung {min, max, start, end}",
    "speed_mm_s": "tốc độ tâm vật (mm/s, đã làm mịn) {max, mean, p90}",
    "tilt_from_vertical_deg": "góc nghiêng so với phương thẳng đứng {min, max, mean, n_frames}: "
                              "trục lúc nghỉ nếu vật đứng gần thẳng, không thì pháp tuyến mặt "
                              "bàn; trục giữ CÓ HƯỚNG liên tục nên đọc được tới 180° (lật ngược)",
    "tilt_deg": "góc trục vật trong hệ CAMERA {min, max, mean, n_frames}; "
                "chỉ tính các frame axis_confident=true. Chai đứng yên đã ~31° vì camera "
                "chúc xuống -> không dùng làm ngưỡng",
    "tilt_from_rest_deg": "góc lệch của trục vật so với trục lúc vật nằm yên trên bàn "
                          "{min, max, mean, n_frames} — xấp xỉ độ nghiêng THẬT so với "
                          "phương thẳng đứng; dùng cái này để đặt và kiểm ngưỡng",
    "displacement_mm": "khoảng cách tâm vật từ đầu tới cuối đoạn (mm, 3D)",
    "end_distance_mm": "khoảng cách 3D từ tâm vật đang cầm tới tâm từng vật khác "
                       "ở cuối đoạn {<id>: mm}",
    "grasp_part": "phần thân bị cầm, vd 'thân', 'phần dưới'",
    "grasp_along": "vị trí cầm dọc trục vật, 0 = đáy, 1 = đỉnh",
    "approach": "hướng tay tiếp cận {deg, label}",
}

# ---------------------------------------------------------------------------
# Checklist 11 chiều cho bước B
# ---------------------------------------------------------------------------
# question: câu hỏi PHẢN CHỨNG đưa vào prompt (tiếng Anh — model hiểu ổn định hơn)
# measure : khoá trong `measured` dùng để đối chiếu ở bước 5 (None = không đo được)

DIMENSIONS = [
    {"name": "grasp_region",
     "desc_vi": "Cầm vào phần nào của vật",
     "question": "If the gripper grasped a different part of the object, would the task fail or cause harm?",
     "measure": "grasp_along"},
    {"name": "grasp_orientation",
     "desc_vi": "Hướng tiếp cận khi cầm",
     "question": "If the gripper approached the object from a different direction, would the task fail or cause harm?",
     "measure": "approach"},
    {"name": "object_orientation_during_motion",
     "desc_vi": "Tư thế vật khi di chuyển",
     "question": "If the held object were tilted arbitrarily while moving, would the task fail or cause harm?",
     "measure": "tilt_from_rest_deg"},
    {"name": "path_shape",
     "desc_vi": "Hình dạng quỹ đạo",
     "question": "If the path between start and goal took any shape, would the task fail or cause harm?",
     "measure": None},
    {"name": "path_clearance",
     "desc_vi": "Khoảng cách an toàn với vật khác",
     "question": "If the object passed arbitrarily close to other objects, would the task fail or cause harm?",
     "measure": "height_mm"},
    {"name": "speed",
     "desc_vi": "Tốc độ, gia tốc",
     "question": "If the motion were much faster or slower, would the task fail or cause harm?",
     "measure": "speed_mm_s"},
    {"name": "end_position",
     "desc_vi": "Vị trí đích",
     "question": "If the object ended at a different position, would the task fail?",
     "measure": "height_mm"},
    {"name": "end_orientation",
     "desc_vi": "Tư thế ở đích",
     "question": "If the object ended in a different orientation, would the task fail?",
     "measure": "tilt_from_rest_deg"},
    {"name": "contact",
     "desc_vi": "Kiểu tiếp xúc với môi trường (Li & Brock 2026)",
     "question": "Must the object keep a specific contact with the environment (slide on a plane, "
                 "move along a line, rotate about a hinge)? Would breaking that contact make the task fail?",
     "measure": None},
    {"name": "rotation_about_symmetry_axis",
     "desc_vi": "Xoay quanh trục đối xứng của vật",
     "question": "If the object were rotated about its own symmetry axis, would the task fail?",
     "measure": None},
    {"name": "ordering",
     "desc_vi": "Thứ tự với skill trước, sau",
     "question": "Must this skill happen strictly after the previous skill and before the next one?",
     "measure": None},
]
DIMENSION_NAMES = [d["name"] for d in DIMENSIONS]

# ---------------------------------------------------------------------------
# Chiều nào ÁP DỤNG cho skill nào
# ---------------------------------------------------------------------------
# Bước B chỉ hỏi model các chiều áp dụng; chiều còn lại code điền
# `not_applicable`. Lý do: Reach/Contact/Retract (tay chưa cầm hoặc đã buông
# vật) từng bị gán `tilt(obj1) <= 3deg`, `low_speed`… vô nghĩa, và mỗi chiều
# thừa tốn thêm token thinking.
# Nguyên tắc chia:
#   - tay KHÔNG cầm vật: không có chiều về tư thế/đích của vật đang cầm;
#   - đang cầm sẵn (Lift, MoveToTarget, Pour…): chỗ cầm và hướng tiếp cận đã
#     chốt ở Grasp -> không xét lại grasp_region / grasp_orientation;
#   - Grasp: vật còn trên bàn -> không có quỹ đạo / đích của vật.
_HAND_ONLY = ["grasp_orientation", "path_shape", "path_clearance", "speed", "end_position",
              "ordering"]
_HOLDING = ["object_orientation_during_motion", "path_shape", "path_clearance", "speed",
            "end_position", "end_orientation", "contact", "rotation_about_symmetry_axis",
            "ordering"]
APPLICABLE_DIMENSIONS = {
    "Reach": _HAND_ONLY,
    "Contact": ["grasp_region", "grasp_orientation", "speed", "ordering"],
    "Grasp": ["grasp_region", "grasp_orientation", "speed", "contact",
              "rotation_about_symmetry_axis", "ordering"],
    "Lift": _HOLDING,
    "MoveToTarget": _HOLDING,
    "Pour": _HOLDING,
    "Rotate": _HOLDING,
    "Insert": _HOLDING,
    "GoHome": _HOLDING,
    "Place": _HOLDING,
    "Release": ["end_position", "end_orientation", "contact", "ordering"],
    "Retract": ["path_shape", "path_clearance", "speed", "ordering"],
    # thao tác có tiếp xúc liên tục với vật (đẩy, kéo, mở, nhấn): xét gần đủ
    "Push": [d for d in DIMENSION_NAMES if d != "rotation_about_symmetry_axis"],
    "Pull": [d for d in DIMENSION_NAMES if d != "rotation_about_symmetry_axis"],
    "Open": list(DIMENSION_NAMES),
    "Close": list(DIMENSION_NAMES),
    "Remove": list(DIMENSION_NAMES),
    "Press": ["grasp_region", "grasp_orientation", "path_clearance", "speed", "end_position",
              "contact", "ordering"],
}


def applicable_dimensions(skill):
    """Danh sách chiều áp dụng cho skill, GIỮ thứ tự của DIMENSIONS. Skill lạ
    (chưa có trong bảng) -> mọi chiều, để không mất thông tin."""
    allowed = set(APPLICABLE_DIMENSIONS.get(skill, DIMENSION_NAMES))
    return [d for d in DIMENSION_NAMES if d in allowed]


NOT_APPLICABLE = "not_applicable"
# model chỉ được chọn 3 giá trị đầu; NOT_APPLICABLE do code điền
STATUSES = ["constraint", "flexible", UNKNOWN]
ALL_STATUSES = STATUSES + [NOT_APPLICABLE]
# path: giữ trong suốt skill; subgoal: phải đạt khi kết thúc skill (ReKep, Huang et al. 2024)
PHASES = ["path", "subgoal", "none"]
# 4 kiểu ràng buộc tiếp xúc (Li & Brock 2026); none = chiều không phải `contact`
CONTACT_TYPES = ["free_space", "plane", "prismatic", "revolute", "none", UNKNOWN]

# THỨ TỰ TRƯỜNG CÓ Ý NGHĨA: Ollama sinh JSON đúng theo thứ tự này, nên model
# viết lý do và bằng chứng TRƯỚC rồi mới kết luận `status`. Chạy thử với thứ tự
# ngược lại (status đứng đầu), qwen3.5:9b chốt `constraint` cho gần như mọi chiều
# rồi mới tìm lý do bào chữa, và nhét chuỗi evidence vào `rule`.
DIMENSION_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "status": _enum(STATUSES),
        "phase": _enum(PHASES),
        # luật dạng gần code, có ngưỡng số, vd "tilt(obj1) <= 15deg"; "" khi không phải constraint
        "rule": {"type": "string"},
        "contact_type": _enum(CONTACT_TYPES),
    },
    "required": ["reason", "evidence", "status", "phase", "rule", "contact_type"],
    "additionalProperties": False,
}

def constraint_schema(dims=DIMENSION_NAMES, allow_not_applicable=False):
    """Khuôn đầu ra bước B cho tập chiều `dims` (mọi chiều đều `required` ->
    Ollama ép model trả lời đủ). allow_not_applicable=True: khuôn của KẾT QUẢ
    đã ghép (constraints_raw.json), nơi code điền `not_applicable`."""
    dim_schema = DIMENSION_RESULT_SCHEMA
    if allow_not_applicable:
        dim_schema = json.loads(json.dumps(DIMENSION_RESULT_SCHEMA))
        dim_schema["properties"]["status"] = _enum(ALL_STATUSES)
    return {
        "type": "object",
        "properties": {
            # ID vật đích (vd cốc khi MoveToTarget/Pour); -1 nếu skill không có đích
            "target_id": {"type": "integer"},
            "dimensions": {
                "type": "object",
                "properties": {name: dim_schema for name in dims},
                "required": list(dims),
                "additionalProperties": False,
            },
        },
        "required": ["target_id", "dimensions"],
        "additionalProperties": False,
    }


# khuôn model trả lời đủ 11 chiều (dùng ở llm_client --selftest, ví dụ mẫu)
CONSTRAINT_SCHEMA = constraint_schema()
# khuôn kết quả đã ghép, kiểm tra ở bước 5
RESULT_SCHEMA = constraint_schema(allow_not_applicable=True)


def not_applicable_result():
    return {"reason": "dimension does not apply to this skill", "evidence": [],
            "status": NOT_APPLICABLE, "phase": "none", "rule": "", "contact_type": "none"}


def fill_not_applicable(data, skill):
    """Đầu ra model (chỉ có chiều áp dụng) -> đủ 11 chiều theo thứ tự chuẩn."""
    dims = data["dimensions"]
    return dict(data, dimensions={name: dims.get(name) or not_applicable_result()
                                  for name in DIMENSION_NAMES})

NO_TARGET = -1

# Định nghĩa từng trường cho model. Schema chỉ ép KIỂU dữ liệu, không truyền
# được Ý NGHĨA: chạy thử không có đoạn này, model chép câu hỏi vào `rule`, để
# `phase` toàn "none" và lấy vật đang cầm làm `target_id`. Bước 4 đưa nguyên
# văn đoạn này vào prompt.
FIELD_GUIDE = f"""\
Output fields:
- target_id: ID of the object the held object is brought to or acts on (e.g. the cup when
  moving a bottle to pour). NOT the held object. Use {NO_TARGET} if the skill has no target.
- For each dimension, fill the fields in this order:
  - reason: one short sentence answering the dimension's question for THIS scene.
  - evidence: facts from the scene that support the reason, each in one of these forms:
      obj<ID>.<field>=<value>     e.g. obj1.state.contents=liquid
      env.<field>=<value>         e.g. env.obstacles_near_path=[]
      rel:<ID> <relation> <ID>    e.g. rel:1 left_of 0
      measured.<field>            e.g. measured.tilt_from_rest_deg.max
      segment.<field>=<value>     e.g. segment.next=Pour
    Only cite facts that are actually present in the scene.
  - status: "constraint" if changing this dimension freely would make the task fail or cause
    harm; "flexible" if the robot may choose freely; "unknown" if the scene does not tell.
    Free-space motion (path shape, speed) is flexible unless the scene gives a concrete reason.
  - phase: for a constraint, "path" if it must hold DURING the whole skill, "subgoal" if it
    must hold at the END of the skill. "none" for flexible or unknown.
  - rule: for a constraint, a short checkable expression with a number and unit when possible,
    e.g. "tilt(obj1) <= 15deg", "dist(obj1, obj0) <= 200mm", "height(obj1) <= 5mm",
    "speed(obj1) <= 300mm/s", "above(obj1.opening, obj0.opening_top)", "after(Lift)".
    Never use undefined names such as safe_limit, margin, low_speed: if you cannot give a
    number, the status is "unknown". Empty string "" for flexible or unknown.
    Do not repeat the question here.
  - contact_type: only for the "contact" dimension: free_space, plane (slides on a surface),
    prismatic (moves along a line, e.g. drawer), revolute (rotates about a hinge, e.g. door)
    or unknown. Use "none" for every other dimension.
"""

# Gợi ý điều kiện -> chiều bị ảnh hưởng, đưa vào prompt bước 4. Là GỢI Ý để
# kết luận nhất quán ở ca quen thuộc, không phải luật cứng: LLM vẫn phải tự
# xét ca khác. Mỗi dòng là "điều kiện quan sát được -> kết luận".
CONDITION_HINTS = """\
Typical conditions (hints, not hard rules; always check them against THIS scene):
Held object
- container with liquid/granular contents AND (lid open OR open_top=yes) -> object_orientation_during_motion
  is a path constraint (keep upright); tighter when fill_level is full. lid closed or contents empty -> flexible.
- open_top with loose contents, full container, fragile=yes or weight_class=heavy -> speed may be a constraint
  (low acceleration). Otherwise speed is flexible.
- temperature=hot -> grasp_region constraint (handle only). sharp=yes -> grasp the handle/blunt part,
  keep the sharp end pointing away from people.
- deformable=highly -> grasp a rigid part, do not squeeze the body.
- symmetry=rotational_vertical -> rotation_about_symmetry_axis flexible. A handle or spout breaks the symmetry.
- items_on_top not empty -> keep the object level while moving (like a liquid).
Target / other objects
- target opening_size=narrow -> end_position must be aligned tightly over the opening (Pour, Insert);
  wide opening -> looser.
- spout or one-sided opening -> end_orientation constraint (spout toward the target) when pouring.
- other object fragile=yes or tall/light (easy to tip) near the path -> path_clearance constraint.
- target already full or occupied -> end_position must be a free spot.
- articulation=slide -> contact prismatic; hinge -> revolute; screw_cap -> rotation about the cap axis
  (Open/Close). The path is then fixed by the mechanism, not flexible.
Environment
- obstacles_near_path not empty -> path_shape and path_clearance are constraints (go over or around).
  Empty -> path_shape flexible.
- clutter=high -> path_clearance and speed constraints.
- humans_nearby=yes means another person's hand is inside the work area (people in the background do
  not count) -> path_clearance constraint and a speed limit, both with numbers from the measurements.
  humans_nearby=no -> humans give no reason for any constraint.
- target near_edge=yes -> end_position must stay away from the edge (Place).
- accessible_sides limited (e.g. only front) -> grasp_orientation constraint (approach from a free side).
Unobservable properties
- Force, friction and exact speed cannot be seen in video. Make them constraints only when one of the
  conditions above gives a concrete reason; if a deciding property is "unknown", choose the safer option
  and say in the reason that it was unknown.
"""



def hints_for(dims):
    """CONDITION_HINTS chỉ giữ các gợi ý nhắc tới ít nhất một chiều trong `dims`
    (gợi ý không nhắc chiều nào, vd phần 'Unobservable', luôn giữ). Rút ngắn
    prompt bước B cho skill ít chiều (Reach, Contact, Retract)."""
    keep = []
    bullet = []

    def flush():
        text = "\n".join(bullet)
        named = [d for d in DIMENSION_NAMES if d in text]
        if bullet and (not named or any(d in dims for d in named)):
            keep.extend(bullet)

    for line in CONDITION_HINTS.splitlines():
        if line.startswith("- "):
            flush()
            bullet = [line]
        elif line.startswith("  ") and bullet:
            bullet.append(line)
        else:                                   # tiêu đề nhóm
            flush()
            bullet = []
            keep.append(line)
    flush()
    # bỏ tiêu đề nhóm không còn gợi ý nào bên dưới
    out = [ln for i, ln in enumerate(keep)
           if ln.startswith((" ", "-")) or i == 0
           or (i + 1 < len(keep) and keep[i + 1].startswith("- "))]
    return "\n".join(out) + "\n"


EVIDENCE_PATTERN = re.compile(
    r"^(obj\d+\.[\w.]+=.+|env\.\w+=.+|rel:\d+ \w+ \d+|measured\.[\w.]+(=.+)?|segment\.\w+=.+)$")

# ---------------------------------------------------------------------------
# Kiểm tra theo schema
# ---------------------------------------------------------------------------

_PY_TYPES = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool}


def validate(value, schema, path="$"):
    """Kiểm tra `value` theo tập con JSON Schema ở trên.

    Trả về danh sách lỗi dạng "đường_dẫn: mô tả"; rỗng nghĩa là hợp lệ.
    Không dừng ở lỗi đầu tiên để bước 5 ghi đủ lý do vào rejected.json."""
    errors = []
    expected = schema.get("type")
    if expected:
        py_type = _PY_TYPES[expected]
        # bool là lớp con của int trong Python -> loại riêng
        if not isinstance(value, py_type) or (expected in ("integer", "number")
                                              and isinstance(value, bool)):
            return [f"{path}: cần kiểu {expected}, nhận {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: '{value}' không thuộc {schema['enum']}")
    if expected == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: thiếu trường '{key}'")
        for key, sub in value.items():
            if key in props:
                errors.extend(validate(sub, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: trường lạ '{key}'")
    if expected == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errors


# ---------------------------------------------------------------------------
# Ví dụ mẫu (đoạn MoveToTarget 3.39–4.09 s của video demo_20260912) để tự kiểm tra
# ---------------------------------------------------------------------------

def _dim(status, phase="none", rule="", contact_type="none", evidence=(), reason=""):
    return {"reason": reason, "evidence": list(evidence), "status": status,
            "phase": phase, "rule": rule, "contact_type": contact_type}


EXAMPLE_SCENE_VLM = {
    "objects": [
        {"id": 1, "label": "plastic water bottle",
         "state": {"lid": UNKNOWN, "contents": "liquid", "fill_level": "full",
                   "temperature": "normal"},
         "material": "plastic", "fragile": "no", "deformable": "slightly",
         "shape": "cylinder", "symmetry": "rotational_vertical",
         "open_top": "no", "opening_size": "narrow", "articulation": "screw_cap",
         "sharp": "no", "weight_class": "light",
         "items_on_top": [], "near_edge": "no", "accessible_sides": ["top", "front", "left", "right"],
         "graspable_parts": ["body"], "functional_parts": ["cap", "opening"]},
        {"id": 0, "label": "white cup",
         "state": {"lid": "none", "contents": "empty", "fill_level": "empty",
                   "temperature": "normal"},
         "material": "ceramic", "fragile": "yes", "deformable": "rigid",
         "shape": "cylinder", "symmetry": "mirror",
         "open_top": "yes", "opening_size": "wide", "articulation": "none",
         "sharp": "no", "weight_class": "light",
         "items_on_top": [], "near_edge": "no", "accessible_sides": ["top", "front", "left", "right"],
         "graspable_parts": ["handle", "body"], "functional_parts": ["opening_top", "handle"]},
    ],
    "environment": {"support_surface": "table", "obstacles_near_path": [], "clutter": "low",
                    "humans_nearby": "no"},
    "relations": [{"subject": 1, "relation": "left_of", "object": 0}],
}

EXAMPLE_CONSTRAINT = {
    "target_id": 0,
    "dimensions": {
        "grasp_region": _dim("flexible", evidence=["obj1.graspable_parts=body"],
                             reason="Thân chai trụ đều, cầm chỗ nào trên thân cũng được"),
        "grasp_orientation": _dim("flexible", reason="Đã cầm xong ở skill trước"),
        "object_orientation_during_motion": _dim(
            "constraint", "path", "tilt(obj1) <= 15deg",
            evidence=["obj1.state.contents=liquid", "segment.next=Pour", "measured.tilt_from_rest_deg.max"],
            reason="Chai có nước và sắp rót nên nhiều khả năng đã mở nắp"),
        "path_shape": _dim("flexible", evidence=["env.obstacles_near_path=[]"],
                           reason="Không có vật cản"),
        "path_clearance": _dim(UNKNOWN, reason="Chưa đủ thông tin khoảng cách"),
        "speed": _dim("flexible", reason="Chai đầy nhưng di chuyển ngắn"),
        "end_position": _dim("constraint", "subgoal", "above(obj1.opening, obj0.opening_top)",
                             evidence=["segment.next=Pour", "rel:1 left_of 0"],
                             reason="Skill sau là rót nên miệng chai phải tới trên miệng cốc"),
        "end_orientation": _dim("flexible", reason="Góc rót thuộc skill Pour"),
        "contact": _dim("flexible", contact_type="free_space",
                        reason="Chai đi trong không gian tự do"),
        "rotation_about_symmetry_axis": _dim("flexible",
                                             evidence=["obj1.symmetry=rotational_vertical"]),
        "ordering": _dim("constraint", "none", "after(Lift) and before(Pour)",
                         evidence=["segment.next=Pour"], reason="Phải nhấc chai lên trước khi mang đi"),
    },
}


def self_test():
    """Kiểm tra schema tự nhất quán và ví dụ mẫu hợp lệ. Trả về số lỗi."""
    n_err = 0
    for name, (value, schema) in {"scene": (EXAMPLE_SCENE_VLM, SCENE_VLM_SCHEMA),
                                  "constraint": (EXAMPLE_CONSTRAINT, CONSTRAINT_SCHEMA)}.items():
        errs = validate(value, schema)
        print(f"[{'OK' if not errs else 'LỖI'}] ví dụ {name}")
        for e in errs:
            print("    ", e)
        n_err += len(errs)
    bad_ev = [ev for d in EXAMPLE_CONSTRAINT["dimensions"].values()
              for ev in d["evidence"] if not EVIDENCE_PATTERN.match(ev)]
    print(f"[{'OK' if not bad_ev else 'LỖI'}] định dạng evidence", bad_ev or "")
    n_err += len(bad_ev)
    # schema phải phát hiện được lỗi thật, không chỉ luôn trả "hợp lệ"
    broken = json.loads(json.dumps(EXAMPLE_CONSTRAINT))
    del broken["dimensions"]["speed"]
    broken["dimensions"]["contact"]["status"] = "maybe"
    caught = validate(broken, CONSTRAINT_SCHEMA)
    ok = len(caught) == 2
    print(f"[{'OK' if ok else 'LỖI'}] bắt được lỗi cố ý: {caught}")
    n_err += 0 if ok else 1
    print(f"{len(DIMENSIONS)} chiều: {', '.join(DIMENSION_NAMES)}")
    return n_err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", choices=["scene", "constraint", "dimensions"],
                    help="in schema/checklist ra màn hình thay vì tự kiểm tra")
    args = ap.parse_args()
    if args.print:
        obj = {"scene": SCENE_VLM_SCHEMA, "constraint": CONSTRAINT_SCHEMA,
               "dimensions": DIMENSIONS}[args.print]
        print(json.dumps(obj, ensure_ascii=False, indent=2))
        return 0
    return 1 if self_test() else 0


if __name__ == "__main__":
    sys.exit(main())
