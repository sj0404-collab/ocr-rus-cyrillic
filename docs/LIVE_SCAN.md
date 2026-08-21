# Live OCR для Android и локального ПК

Статус: рабочий baseline, обновлён 21 августа 2026 года.

## Главное правило

```text
кадр изменился       -> один OCR-анализ
кадр не изменился    -> вернуть предыдущий текст без повторного OCR
```

Для этого в проекте есть:

- `src/ocr_rus_cyrillic/live.py` — Python `LiveOCRSession`;
- `android/LiveOcrController.kt` — Android-контроллер для TFLite;
- `android/CyrillicRecognition.kt` — TFLite text-line recognizer.

## События live-контроллера

Каждый результат содержит:

```json
{
  "text": "распознанный текст",
  "confidence": 0.94,
  "certain": true,
  "frameId": 42,
  "changed": true,
  "analyzed": true,
  "reused": false,
  "latencyMs": 37
}
```

Если crop не изменился, `analyzed=false`, `reused=true`, а `text` остаётся прежним.

## Рекомендуемый поток CameraX

```text
CameraX ImageAnalysis
        |
        +-- STRATEGY_KEEP_ONLY_LATEST
        |
        +-- crop / perspective correction
        |
        +-- signature comparison
        |       |
        |       +-- same -> cached text
        |       +-- changed -> TFLite/YOLO OCR
        |
        +-- PP-OCRv3 primary
        +-- PP-OCRv5 verifier only when confidence is low
        +-- Russian orthography correction
        +-- UI: text + confidence + certain
```

Не ставьте в очередь все кадры камеры: нужен `ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST`, иначе приложение будет обрабатывать устаревшие изображения.

## Что считать кадром

Сигнатура должна считаться не от всего экрана, а от text-line или bubble crop. Рекомендуемый размер сигнатуры — `48 x 32` grayscale. Это позволяет игнорировать небольшое JPEG-шумовое отличие, но реагировать на изменение букв.

Если камера движется, а текстовая область остаётся той же, сравнивайте отдельные tracked crops, а не общий frame.

## Модели

```text
models/tflite/pp-ocrv4_mobile_det_float32.tflite
models/tflite/cyrillic_pp-ocrv3_mobile_rec_float32.tflite
models/tflite/cyrillic_pp-ocrv5_mobile_rec_float32.tflite
models/onnx/yolo_cyrillic_text_bubble_best.onnx
models/onnx/yolo_cyrillic_opencorpora_best.onnx
```

TFLite recognizer принимает `[1, 48, 320, 3]` NHWC. Для live-режима лучше передавать уже вырезанную строку, а не целую страницу.

## Ограничения

- точное время зависит от телефона, CPU/GPU delegate, размера crop и detector;
- полный page OCR существенно медленнее text-line OCR;
- `certain=false` нельзя заменять выдуманным текстом;
- для окончательной калибровки latency нужны замеры на целевом APK и реальном устройстве.
