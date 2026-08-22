# Дообучение русской версии

В этой рабочей сессии размеченный набор пользователя отсутствовал, поэтому модель не переобучалась на случайно скачанных страницах. Вместо этого выбран готовый мобильный Cyrillic checkpoint и добавлен Russian-only decode mask. Это безопаснее, чем объявить синтетическое дообучение «90%+» без независимого теста.

## Полный OCR-набор с разметкой

```powershell
python training/build_full_dataset.py --samples 400 --corpus outputs/ru_corpus.txt --art-dir benchmark/pollination_pages --output outputs/russian_manga_ocr_dataset
```

Для каждой страницы создаются:

- изображение;
- YOLO-разметка классов `text` и `bubble`;
- `manifest.json` с точным transcript, направлением, шрифтом и углом;
- `recognition/train.tsv` и `recognition/val.tsv` с парами `crop<TAB>text`.

Фоны берутся из оригинальных AI-generated страниц или процедурных текстур, а русские строки отрисовываются локально и имеют точный ground truth. Это полностью размеченный generated dataset, но он всё равно синтетический по изображению: Pollinations/image-generation не создаёт реальные фотографии.

## Русский текстовый корпус

На Windows workflow сначала извлекается до 100 000 предложений из OpenCorpora 2025 (CC BY-SA), затем они используются как русскоязычный корпус для генерации изображений:

```powershell
python training/extract_opencorpora.py --archive opencorpora-2025.tar.gz --output outputs/ru_corpus.txt --max-sentences 100000
python training/build_full_dataset.py --samples 400 --corpus outputs/ru_corpus.txt --output outputs/russian_manga_ocr_dataset
```

Генератор покрывает:

- разные TTF/OTF-шрифты, начертания и размеры;
- горизонтальные строки, абзацы, колонки LTR/RTL;
- вертикальный текст сверху вниз;
- дуговой текст;
- пузыри, фоновые текстуры, шум, blur, erosion/dilation и JPEG-артефакты;
- наклоны, цифры, кавычки, тире, проценты, `ё`, `ъ`, `ы`, `щ` и прочую пунктуацию.

## YOLO detector

На Windows runner запускается CPU fine-tune на полном размеченном наборе:

```powershell
python -m pip install -r training/requirements.txt
python training/train_yolo.py --data outputs/russian_manga_ocr_dataset/dataset.yaml --output outputs/yolo_training --epochs 15 --batch 4
```

Результаты: `best.pt` и экспортированный `best.onnx`. Адаптер `src/ocr_rus_cyrillic/yolo_detector.py` принимает этот ONNX: класс `0` передаётся в OCR, а класс `1` может использоваться для bubble-aware crop.

YOLO/Ultralytics имеет собственные условия лицензирования. Перед коммерческой публикацией APK проверьте лицензию Ultralytics и выбранного checkpoint; полученный detector не объявляется частью Apache-2.0 OCR-моделей.

Пятнадцать эпох на 200 синтетических страницах — это улучшенный smoke baseline, а не финальное качество. Для 90%+ нужны реальные размеченные кадры из целевого APK и отдельный test split по документам.

## Что нужно для настоящего Russian-only fine-tune

```text
train_data/
├── train_list.txt   # путь-картинка<TAB>транскрипция
├── val_list.txt
└── images/
```

Тексты и изображения должны быть использованы только при наличии лицензии. Не скрейпьте произвольные книги, сайты, мангу или документы для публичного датасета без проверки прав.

1. Оставить русский алфавит, цифры и нужную пунктуацию в словаре.
2. Зафиксировать `max_text_length`, размер строки `[3, 48, 320]`, версии Paddle/Python и seed.
3. Разделить данные по документам, а не случайно по соседним crop.
4. Оценивать CER, WER и exact-line accuracy на отложенном наборе.
5. Только после этого экспортировать Paddle → ONNX/TFLite и повторить сравнение logits и CER.

Цель `90%+` может быть достижима на конкретном домене и корпусе, но не является гарантией для любых камер, шрифтов, манги и рукописи.
