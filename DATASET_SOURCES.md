# Russian dataset sources used by the project

## OpenCorpora frequency list

The file `models/dicts/opencorpora_freqrnc2011.tsv` comes from the OpenCorpora 2025 data package published by the MAWO project:

- Source package: https://github.com/mawo-ru/mawo-nlp-data/releases/download/v1.0.0/opencorpora-2025.tar.gz
- Original project: https://opencorpora.org/
- License: CC BY-SA (OpenCorpora data)
- Use in this project: Russian word-frequency prior for conservative spelling correction and fused-word segmentation. The Windows YOLO training workflow additionally streams up to 100,000 annotated-corpus sentence strings into the synthetic image generator; it does not publish the full corpus in this repository.

This is not a visual OCR dataset. It supplies Russian lexical statistics; images are still generated or supplied under a compatible license.

## Suitable visual datasets for future fine-tuning

These are recorded here for a separate, license-aware training job:

- **Cyrillic Handwriting Dataset** — the Kaggle card states CC0/Public Domain, 73,830 Russian handwritten crops: https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset
- **HWR200** — Apache-2.0, Russian handwritten text photographed/scanned: https://huggingface.co/datasets/AntiplagiatCompany/HWR200
- **school_notebooks_RU** — MIT, Russian handwritten notebook detection/OCR annotations: https://huggingface.co/datasets/ai-forever/school_notebooks_RU
- **russian-old-orthography-ocr** — MIT, public 19th-century Russian images and text: https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr
- **synthetic_cyrillic** — MIT, generated Cyrillic handwriting dataset: https://huggingface.co/datasets/nastyboget/synthetic_cyrillic

Datasets with an unknown or unclear license, including some large synthetic OCR repositories, are not automatically downloaded into the public project. A commercial APK must separately review every dataset and model license.
