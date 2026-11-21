"""RGB-D, mask và phép chiếu 3D dùng chung cho pipeline keypoint."""

import json
from pathlib import Path

import cv2
import numpy as np


class RGBDSequence:
    """Depth đã được ghép theo chỉ số frame màu bằng timestamp ở pipeline cũ.

    Tọa độ 3D trả về là milimét trong hệ camera màu, không phải hệ robot.
    """

    def __init__(self, video_path, depth_path, masks_path):
        self.video_path = Path(video_path).resolve()
        self.depth_path = Path(depth_path).resolve()
        self.masks_path = Path(masks_path).resolve()
        with np.load(self.depth_path) as data:
            self.depth = data["depth"]
            self.times = data["t"]
            self.fps = float(data["fps"])
            self.intrinsics = {k: float(data[k]) for k in ("fx", "fy", "cx", "cy")}
        self.masks = json.loads(self.masks_path.read_text(encoding="utf-8"))
        self.object_labels = {int(o["id"]): o["label"] for o in self.masks["objects"]}
        self.n_frames, self.height, self.width = self.depth.shape
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Không mở được video: {self.video_path}")
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
        if not (n_video == self.n_frames == len(self.times) == len(self.masks["frames"])):
            raise ValueError("Số frame RGB, depth, timestamp và mask không khớp")
        if video_size != (self.height, self.width):
            raise ValueError("Kích thước RGB và depth không khớp")
        if list(self.masks["size"]) != [self.height, self.width]:
            raise ValueError("Kích thước mask và depth không khớp")
        if self.intrinsics["fx"] <= 0 or self.intrinsics["fy"] <= 0:
            raise ValueError("Nội tham số camera không hợp lệ")

    def rgb(self, frame):
        """Trả ảnh RGB uint8 ở một frame; mở/đóng video để tránh giữ handle."""
        cap = cv2.VideoCapture(str(self.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise IndexError(f"Không đọc được RGB frame {frame}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def frame_objects(self, frame):
        return self.masks["frames"][frame]["objects"]

    @staticmethod
    def decode_rle(rle):
        """RLE row-major từ segmentation/common.py; kiểm tra tổng độ dài."""
        shape = tuple(rle["size"])
        counts = np.asarray(rle["counts"], dtype=np.int64)
        if counts.size == 0 or np.any(counts < 0) or int(counts.sum()) != int(np.prod(shape)):
            raise ValueError("RLE mask không hợp lệ")
        return np.repeat(np.arange(len(counts)) % 2 == 1, counts).reshape(shape)

    def object_mask(self, frame, object_id):
        for obj in self.frame_objects(frame):
            if int(obj["id"]) == object_id:
                return self.decode_rle(obj["rle"])
        return np.zeros((self.height, self.width), dtype=bool)

    def backproject(self, u, v, depth_mm):
        """Pixel (u,v) và depth mm -> XYZ mm hệ camera."""
        u, v, z = np.broadcast_arrays(np.asarray(u), np.asarray(v), np.asarray(depth_mm))
        k = self.intrinsics
        return np.stack(((u - k["cx"]) * z / k["fx"],
                         (v - k["cy"]) * z / k["fy"], z), axis=-1)

    def project(self, xyz_mm):
        """XYZ mm hệ camera -> pixel (u,v)."""
        xyz = np.asarray(xyz_mm)
        if np.any(xyz[..., 2] <= 0):
            raise ValueError("Không thể chiếu điểm có Z <= 0")
        k = self.intrinsics
        return np.stack((xyz[..., 0] * k["fx"] / xyz[..., 2] + k["cx"],
                         xyz[..., 1] * k["fy"] / xyz[..., 2] + k["cy"]), axis=-1)

    def point_cloud(self, frame, mask=None, stride=1):
        """Trả (N,3) XYZ mm chỉ ở pixel có depth hợp lệ."""
        valid = self.depth[frame] > 0
        if mask is not None:
            valid &= mask
        v, u = np.nonzero(valid)
        u, v = u[::stride], v[::stride]
        return self.backproject(u, v, self.depth[frame, v, u])
