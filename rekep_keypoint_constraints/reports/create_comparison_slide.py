"""Tạo đúng một slide so sánh cách lấy constraint của vật với ReKep 2024."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = Path(__file__).resolve().parents[2] / "out/bao_cao/so_sanh_constraint_rekep_2024.pptx"

NAVY = RGBColor(19, 33, 56)
BLUE = RGBColor(63, 79, 196)
TEAL = RGBColor(0, 135, 129)
INK = RGBColor(31, 43, 61)
MUTED = RGBColor(82, 98, 117)
PALE = RGBColor(245, 248, 252)
LIGHT_BLUE = RGBColor(238, 240, 255)
LIGHT_TEAL = RGBColor(233, 247, 244)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(215, 224, 234)


def text_box(slide, x, y, w, h, text, size, color=INK, bold=False,
             align=PP_ALIGN.LEFT, margin=0, valign=MSO_ANCHOR.MIDDLE):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = valign
    paragraph = tf.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = text
    font = run.font
    font.name = "Noto Sans Display"
    font.size = Pt(size)
    font.bold = bold
    font.color.rgb = color
    return shape


def rounded(slide, x, y, w, h, fill, line=None, radius=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background() if line is None else None
    if line is not None:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.8)
    if radius is not None:
        shape.adjustments[0] = radius
    return shape


def create():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = PALE

    # Đầu trang
    rounded(slide, 0, 0, 13.333, 1.27, NAVY, radius=0)
    text_box(slide, 0.58, 0.17, 11.9, 0.52,
             "Lấy constraint của vật: pipeline của tôi vs ReKep (2024)", 25, WHITE, True)
    text_box(slide, 0.60, 0.73, 11.9, 0.30,
             "Cùng dùng quan hệ keypoint; khác nguồn quan sát và cách xác nhận luật", 12.7,
             RGBColor(207, 219, 235))

    # Hai cột so sánh
    left_x, right_x, col_w = 0.58, 6.77, 5.98
    rounded(slide, left_x, 1.49, col_w, 0.57, BLUE)
    rounded(slide, right_x, 1.49, col_w, 0.57, TEAL)
    text_box(slide, left_x + 0.22, 1.56, col_w - 0.44, 0.38, "ReKep 2024", 17.2, WHITE, True)
    text_box(slide, right_x + 0.22, 1.56, col_w - 0.44, 0.38, "Pipeline video của tôi", 17.2, WHITE, True)

    rows = [
        ("ĐẦU VÀO", "RGB-D của cảnh hiện tại + instruction do người dùng đưa.",
         "Video RGB-D người làm mẫu; instruction người nhập hoặc VLM suy tạm."),
        ("KEYPOINT VẬT", "DINOv2 đề xuất điểm từ một quan sát; ánh xạ thành điểm 3D.",
         "Grounding DINO/SAM2 tạo mask; hình học + DINOv2 chọn điểm; theo dõi ID qua video."),
        ("SINH CONSTRAINT", "GPT-4o nhận ảnh đánh ID + instruction → stage, hàm cost Python sub-goal/path.",
         "Skill + chuyển động chia stage; Qwen3-VL đề xuất; code lọc DSL, bổ sung fallback."),
        ("KIỂM TRA / DÙNG", "Solver dùng cost để lập kế hoạch và điều chỉnh robot khi keypoint thay đổi.",
         "Ngưỡng từ video mẫu; tính cost theo frame và phản chứng. Hiện kiểm tra ngoại tuyến."),
    ]
    row_y, row_h, gap = 2.22, 0.91, 0.10
    for i, (label, left, right) in enumerate(rows):
        y = row_y + i * (row_h + gap)
        rounded(slide, left_x, y, col_w, row_h, WHITE, LINE)
        rounded(slide, right_x, y, col_w, row_h, WHITE, LINE)
        text_box(slide, left_x + 0.20, y + 0.11, 1.18, 0.20, label, 9.4, BLUE, True)
        text_box(slide, right_x + 0.20, y + 0.11, 1.18, 0.20, label, 9.4, TEAL, True)
        text_box(slide, left_x + 1.36, y + 0.08, 4.34, 0.73, left, 13.1, INK)
        text_box(slide, right_x + 1.36, y + 0.08, 4.34, 0.73, right, 13.1, INK)

    rounded(slide, 0.58, 6.33, 12.17, 0.58, LIGHT_TEAL)
    text_box(slide, 0.77, 6.38, 11.8, 0.46,
             "Khác biệt cốt lõi: ReKep sinh luật để điều khiển robot từ cảnh hiện tại; "
             "pipeline này rút luật từ video trình diễn và kiểm tra trên chính quỹ đạo mẫu.",
             12.8, NAVY, True)
    text_box(slide, 0.60, 7.02, 12.15, 0.28,
             "Nguồn ReKep: rekep-robot.github.io · Huang et al., arXiv:2409.01652 (2024)   |   "
             "Ca RGB-only của dự án chỉ cho constraint 2D (px), chưa tương đương luật 3D.",
             8.6, MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(create())
