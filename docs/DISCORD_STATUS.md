# OCR Rus Cyrillic — статус для Discord

**Дата:** 21.08.2026

## Текущий статус

✅ Кроссплатформенный Russian/Cyrillic OCR baseline собран.

✅ Live-cache реализован:

- новый кадр анализируется один раз;
- неизменившийся кадр возвращает тот же текст без повторного OCR;
- Android-код: `android/LiveOcrController.kt`;
- Python-код: `src/ocr_rus_cyrillic/live.py`.

✅ TFLite-модели:

- PP-OCRv3 Cyrillic — основной recognizer;
- PP-OCRv5 Cyrillic — дополнительный verifier;
- PP-OCRv4 mobile — detector.

✅ YOLO detector обучен на вариативной русской синтетике и отдельно на синтетике с предложениями OpenCorpora.

✅ Добавлены русская орфография, исправление confusable-символов, digits-lookalikes и консервативное разделение слитных слов.

## Тесты

Windows OCR benchmark: успешно

https://github.com/sj0404-collab/ocr-rus-cyrillic/actions/runs/32506236954

Windows engine comparison: успешно

https://github.com/sj0404-collab/ocr-rus-cyrillic/actions/runs/32506237007

Windows YOLO + OpenCorpora training: успешно

https://github.com/sj0404-collab/ocr-rus-cyrillic/actions/runs/32498505048

## Сравнение на synthetic manga-style страницах

```text
Наш pipeline:  CER 19.71%
EasyOCR ru:   CER 23.42%
Tesseract rus: CER 105.99%
```

Manga OCR показал CER 100% на русском synthetic наборе, поскольку его основная модель предназначена для японского текста.

## Модели в GitHub

https://github.com/sj0404-collab/ocr-rus-cyrillic

## Следующий этап

1. Замерить live latency на целевом Android-устройстве.
2. Проверить YOLO на разрешённых реальных страницах.
3. Добавить финальные ground-truth транскрипции для тестовых страниц.
4. Настроить переключение fast/quality режима.
5. Снизить среднее время обработки, не включая второй verifier на каждом кадре.

> Метрики synthetic benchmark не являются гарантией качества на любых камерах и шрифтах. Для финального APK нужен отдельный test set из реальных разрешённых изображений.
