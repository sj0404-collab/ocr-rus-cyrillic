# OCR Rus Cyrillic

Офлайн OCR для русскоязычного текста и кириллицы: мобильные ONNX-модели, ограничение алфавита, детектор строк и bounded-consensus слой для повторного анализа сомнительных результатов.

> Это инженерный baseline для APK, а не обещание абсолютной точности. Модель возвращает измеренный confidence и `certain=false`, если несколько визуальных проходов не согласны. Она не может честно гарантировать 100% на произвольных фото.

## Что уже собрано

- мобильная модель распознавания `cyrillic_pp-ocrv3_mobile_rec.onnx`;
- второй независимый verifier `cyrillic_pp-ocrv5_mobile_rec.onnx`, который запускается только на сомнительных crops;
- лёгкий детектор текста `pp-ocrv4_mobile_det.onnx`;
- русский/Cyrillic whitelist на этапе CTC-декодирования;
- 3 варианта предобработки: raw, grayscale, Otsu;
- до 4 ограниченных повторных проходов для сомнительных строк;
- консервативное исправление латинско-кириллических look-alike символов и опечаток в русском словаре;
- JSON-вывод для интеграции в Android.

## Быстрый запуск на ПК

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=src python -m ocr_rus_cyrillic.cli tests/fixtures/sample_word.png --json
```

Для страницы или фотографии:

```bash
PYTHONPATH=src python -m ocr_rus_cyrillic.cli path/to/page.jpg --json --target 0.90 --passes 4
```

После обучения YOLO можно подключить его ONNX-детектор:

```bash
PYTHONPATH=src python -m ocr_rus_cyrillic.cli path/to/page.jpg --json --yolo-model outputs/yolo_training/cyrillic_text_yolo/weights/best.onnx
```

YOLO-классы: `0=text`, `1=bubble`. Если YOLO-модель не найдена или не дала text boxes, pipeline автоматически возвращается к PP-OCR detector.

Пример результата:

```json
{
  "text": "проверка",
  "confidence": 0.96,
  "certain": true,
  "passes": 1
}
```

`confidence` — оценка модели и согласованности проходов, не математическая вероятность безошибочности. При `certain=false` приложение должно показать пользователю результат как требующий проверки, а не подменять его выдуманным текстом.

## Android / APK

Для APK уже подготовлены два варианта.

**ONNX Runtime Mobile:**

```text
models/onnx/pp-ocrv4_mobile_det.onnx
models/onnx/cyrillic_pp-ocrv3_mobile_rec.onnx
models/onnx/cyrillic_pp-ocrv5_mobile_rec.onnx   # optional verifier
models/dicts/cyrillic_dict.txt
models/dicts/ppocrv5_cyrillic_dict.txt
```

`ppocrv5` используется как второй ONNX-проход только при низкой уверенности первого движка. Если движки расходятся, результат остаётся `certain=false`.

**TFLite, фиксированный CPU-friendly float32 input:**

```text
models/tflite/pp-ocrv4_mobile_det_float32.tflite       # [1, 736, 736, 3]
models/tflite/cyrillic_pp-ocrv3_mobile_rec_float32.tflite # [1, 48, 320, 3]
models/tflite/cyrillic_pp-ocrv5_mobile_rec_float32.tflite # optional verifier, [1, 48, 320, 3]
models/dicts/cyrillic_dict.txt
models/dicts/ppocrv5_cyrillic_dict.txt
```

Для TFLite input — NHWC, для recognizer output — `[1, 40, 165]`. Порядок работы:

1. уменьшить фотографию с сохранением читаемой высоты букв;
2. запустить детектор;
3. выровнять каждый четырёхугольник текста;
4. подать crop в recognizer с размером `[1, 48, 320, 3]` и нормализацией `(pixel / 255 - 0.5) / 0.5`;
5. выполнить CTC decode, начиная с индекса blank `0`;
6. запретить неразрешённые Latin-классы;
7. повторить только сомнительные crops и вернуть `text`, `confidence`, `certain`.

Float32 TFLite проверен на ПК: logits совпали с ONNX с максимальным абсолютным отклонением около `7e-6` на тестовом crop, argmax-последовательность совпала. Это проверка эквивалентности конверсии, а не benchmark на реальных фото.

## Ограничения текущего результата

- Размеченный пользовательский набор изображений не был предоставлен; поэтому новая supervised-модель не обучалась.
- Синтетика помогает покрыть шрифты, но не заменяет фото, мангу, пузыри, наклон, шум и реальные камеры.
- Заявленные upstream-метрики относятся к их тестовому набору и не переносятся автоматически на ваш APK.
- «Анализировать до 100%» реализовано как конечное число повторных проходов и честный отказ от высокой уверенности; бесконечный цикл невозможен и не гарантирует правильный текст.

## Структура

```text
.
├── models/                         # ONNX/TFLite-модели и словарь
├── android/                        # Kotlin пример для TFLite text-line recognizer
├── training/                       # synthetic layouts and YOLO fine-tune scripts
├── src/ocr_rus_cyrillic/           # recognizer, corrector, page pipeline, CLI, YOLO adapter
├── tests/                          # тесты постобработки и маленький fixture
├── benchmark/README.md             # формат будущего CER/WER benchmark
├── MODEL_CARD.md                   # происхождение и SHA-256 моделей
└── requirements.txt
```

## Предыдущий каталог движков

Сравнение Tesseract, EasyOCR, PaddleOCR, Kraken и OCRmyPDF осталось в истории проекта и расширено мобильным baseline.
