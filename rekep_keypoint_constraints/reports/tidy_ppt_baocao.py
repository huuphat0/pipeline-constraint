"""Dọn bố cục báo cáo PPTX, giữ nguyên chữ, ảnh và video nhúng gốc."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "video_depthcam/video_vat_can/ppt_baocao_original.pptx"
OUTPUT = ROOT / "video_depthcam/video_vat_can/ppt_baocao_tidy_preview.pptx"

NAVY = RGBColor(19, 33, 56)
PALE = RGBColor(246, 248, 252)
WHITE = RGBColor(255, 255, 255)
MUTED = RGBColor(91, 108, 129)
FONT = "Noto Sans Display"


def move(shape, x, y, w, h):
    shape.left = Inches(x)
    shape.top = Inches(y)
    shape.width = Inches(w)
    shape.height = Inches(h)


def to_front(shape):
    element = shape._element
    parent = element.getparent()
    parent.remove(element)
    parent.append(element)


def header(slide, title_shape=None, title=None, number=None):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = PALE
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.12))
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()
    # Đặt dải màu sau các nội dung hiện có để chữ không bị che.
    tree = slide.shapes._spTree
    tree.remove(band._element)
    tree.insert(2, band._element)
    if title_shape is None:
        title_shape = slide.shapes.add_textbox(Inches(0.58), Inches(0.24), Inches(11.7), Inches(0.58))
        title_shape.text = title
    move(title_shape, 0.58, 0.22, 11.7, 0.62)
    to_front(title_shape)
    tf = title_shape.text_frame
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for paragraph in tf.paragraphs:
        paragraph.alignment = PP_ALIGN.LEFT
        for run in paragraph.runs:
            run.font.name = FONT
            run.font.size = Pt(25)
            run.font.bold = True
            run.font.color.rgb = WHITE
    if number is not None:
        box = slide.shapes.add_textbox(Inches(12.02), Inches(7.08), Inches(0.72), Inches(0.22))
        box.text = f"{number:02d} / 04"
        box.text_frame.margin_left = box.text_frame.margin_right = 0
        for p in box.text_frame.paragraphs:
            for run in p.runs:
                run.font.name = FONT
                run.font.size = Pt(9)
                run.font.color.rgb = MUTED


def create():
    prs = Presentation(SOURCE)
    if len(prs.slides) != 4:
        raise ValueError("Bản gốc đã đổi số slide; cần xem lại trước khi dọn")
    slide2, slide3, slide4 = prs.slides[1], prs.slides[2], prs.slides[3]

    # Slide 2: giữ ảnh mô phỏng và đưa video gốc vào vùng hiển thị.
    original2 = list(slide2.shapes)
    header(slide2, original2[1], number=2)
    move(original2[0], 0.58, 1.37, 12.18, 3.40)
    media = original2[2]
    move(media, 2.86, 4.91, 7.62, 2.14)

    # Slide 3: hai ảnh keyframe nằm cùng hàng; bảng constraint 3D ở phía dưới.
    original3 = list(slide3.shapes)
    header(slide3, original3[3], number=3)
    move(original3[0], 0.58, 1.37, 5.95, 1.92)
    move(original3[1], 6.80, 1.37, 5.95, 1.92)
    move(original3[2], 0.58, 3.56, 12.17, 2.91)

    # Slide 4: đặt montage và bảng constraint cùng trang để đối chiếu.
    original4 = list(slide4.shapes)
    header(slide4, title="Video vật cản: keyframe và constraint", number=4)
    move(original4[1], 1.89, 1.36, 9.55, 2.69)
    move(original4[0], 1.89, 4.23, 9.55, 2.57)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(create())
