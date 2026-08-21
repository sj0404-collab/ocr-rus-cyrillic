# EasyOCR / Manga OCR / Tesseract comparison

`compare_engines.py` запускает на одном и том же наборе страниц:

- наш Cyrillic PP-OCRv3 + PP-OCRv5 + орфографический verifier;
- EasyOCR с `ru`;
- Tesseract с `rus`;
- Manga OCR PC, если его японская модель и зависимости установились.

Manga OCR предназначен прежде всего для японской манги, поэтому русская оценка для него будет диагностической, а не полностью равноправным baseline. Сравнение считает CER на `manifest.json`, latency и сохраняет полный текст каждого движка.

Для честного вывода нужны одинаковые изображения, один и тот же preprocessing и ground truth. Synthetic benchmark — только smoke test; пользовательские скриншоты должны оставаться приватными или иметь разрешение на публикацию.
