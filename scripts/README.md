# Scripts

- `windows_smoke.py` — запускается на GitHub Actions `windows-latest`, проверяет загрузку ONNX-моделей, детектор, кириллический recognizer, post-processing и confidence threshold на маленьком fixture.

Для обучения/дообучения нужен отдельный размеченный набор. Windows workflow намеренно не запускает фиктивное переобучение без ground truth.
