# YOLO detector training report

Windows runner: [run 32474627916](https://github.com/sj0404-collab/ocr-rus-cyrillic/actions/runs/32474627916)

Configuration:

- Windows runner, CPU;
- 200 synthetic pages;
- 15 epochs;
- YOLO nano checkpoint;
- classes: `0=text`, `1=bubble`;
- varied fonts, sizes, layouts, directions, textures and degradations.

Held-out synthetic metrics:

```text
mAP50:     0.88867
mAP50-95:  0.62624
```

The exported synthetic-only model is `models/onnx/yolo_cyrillic_text_bubble_best.onnx`.

A second run first extracted up to 100,000 Russian sentences from OpenCorpora 2025 and used them in the synthetic renderer. Its final held-out metrics were:

```text
mAP50:     0.75632
mAP50-95:  0.52320
```

That model is `models/onnx/yolo_cyrillic_opencorpora_best.onnx`. The synthetic-only model scores higher on its easier distribution, while the OpenCorpora model covers a broader Russian vocabulary. Neither metric establishes 90%+ accuracy on real manga pages or camera photos. The next validation step is to run both detectors on licensed/user-provided pages and measure text-box recall, reading order, CER and WER separately.
