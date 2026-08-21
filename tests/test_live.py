import numpy as np

from ocr_rus_cyrillic.live import LiveOCRSession
from ocr_rus_cyrillic.pipeline import OCRResult


class FakeOCR:
    def __init__(self):
        self.calls = 0

    def recognize_page(self, frame):
        self.calls += 1
        return OCRResult(
            text=f"кадр {self.calls}",
            confidence=0.95,
            certain=True,
            passes=1,
        )


def test_identical_frame_is_reused():
    fake = FakeOCR()
    session = LiveOCRSession(fake)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    first = session.process(frame)
    second = session.process(frame.copy())
    assert first.analyzed is True
    assert second.reused is True
    assert second.analyzed is False
    assert fake.calls == 1
    assert second.text == first.text


def test_changed_frame_is_analyzed():
    fake = FakeOCR()
    session = LiveOCRSession(fake)
    first = np.zeros((120, 160, 3), dtype=np.uint8)
    second = first.copy()
    second[30:80, 40:100] = 255
    session.process(first)
    result = session.process(second)
    assert result.changed is True
    assert result.analyzed is True
    assert fake.calls == 2
