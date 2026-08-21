# Android / APK integration

В репозитории лежат TFLite-ассеты для мобильной интеграции:

```text
models/tflite/pp-ocrv4_mobile_det_float32.tflite
models/tflite/cyrillic_pp-ocrv3_mobile_rec_float32.tflite
models/tflite/cyrillic_pp-ocrv5_mobile_rec_float32.tflite  # optional verifier
models/dicts/cyrillic_dict.txt
models/dicts/ppocrv5_cyrillic_dict.txt
```

Скопируйте их в `app/src/main/assets/` и добавьте TensorFlow Lite runtime в APK. Версия библиотеки должна соответствовать вашему Android toolchain; не добавляйте одновременно несколько несовместимых TFLite runtimes.

`CyrillicRecognition.kt` показывает полный цикл для **уже вырезанной горизонтальной строки**:

- фиксированный input `[1, 48, 320, 3]`, NHWC;
- BGR-порядок каналов;
- нормализация `(pixel / 255 - 0.5) / 0.5`;
- CTC blank index `0`;
- кириллический whitelist;
- `confidence` и `certain` без притворной гарантии 100%.

Для страницы ещё нужны детектор, perspective crop и сортировка областей по чтению. Детектор TFLite имеет фиксированный input `[1, 736, 736, 3]`; его DB post-processing должен восстановить полигоны текста. Готовый Python baseline находится в `src/ocr_rus_cyrillic/pipeline.py`.

**Colibri:** точная ссылка или API Colibri в задаче не указаны, поэтому Kotlin-файл не притворяется кодом конкретного Colibri SDK. Пришлите ссылку на Colibri/его шаблон APK — адаптирую загрузку assets и вызов `recognize` под него.
