"""Optional OpenCV-DNN adapter for an exported YOLO text/bubble detector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class YoloTextDetector:
    """Read YOLO ONNX output with classes 0=text and 1=bubble."""

    def __init__(self, model_path: str | Path, *, input_size: int = 640, conf: float = 0.25, nms: float = 0.45):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.input_size = input_size
        self.conf = conf
        self.nms = nms

    @staticmethod
    def _letterbox(image: np.ndarray, size: int):
        h, w = image.shape[:2]
        scale = min(size / max(w, 1), size / max(h, 1))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, scale, dx, dy

    def detect(self, image: np.ndarray):
        canvas, scale, dx, dy = self._letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        output = np.asarray(outputs)
        if output.ndim == 3:
            output = output[0]
            if output.shape[0] < output.shape[1]:
                output = output.T
        elif output.ndim == 2:
            pass
        else:
            output = output.reshape(-1, output.shape[-1])
        # Raw YOLOv8/11 exports usually have thousands of rows (xywh + class
        # scores); exports with built-in NMS have only a few hundred rows of
        # x1,y1,x2,y2,score,class. The column count alone is ambiguous for a
        # two-class raw model, so use the row count as the discriminator.
        raw_yolo = output.shape[0] > 100

        boxes: list[list[int]] = []
        scores: list[float] = []
        classes: list[int] = []
        for row in output:
            if not raw_yolo and len(row) >= 6:
                # Export with NMS: x1,y1,x2,y2,score,class.
                x1, y1, x2, y2, score, cls = row[:6]
                score, cls = float(score), int(cls)
            else:
                if len(row) < 6:
                    continue
                x, y, w, h = map(float, row[:4])
                class_scores = np.asarray(row[4:], dtype=np.float32)
                cls = int(class_scores.argmax())
                score = float(class_scores[cls])
                x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            if score < self.conf:
                continue
            x1 = int(np.clip((x1 - dx) / scale, 0, image.shape[1] - 1))
            y1 = int(np.clip((y1 - dy) / scale, 0, image.shape[0] - 1))
            x2 = int(np.clip((x2 - dx) / scale, 0, image.shape[1] - 1))
            y2 = int(np.clip((y2 - dy) / scale, 0, image.shape[0] - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            classes.append(cls)

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.nms)
        result = []
        for index in np.asarray(keep).reshape(-1):
            x, y, w, h = boxes[int(index)]
            result.append((np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), classes[int(index)], scores[int(index)]))
        return result
