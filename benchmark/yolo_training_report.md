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

The exported model is `models/onnx/yolo_cyrillic_text_bubble_best.onnx`.

These metrics only describe the generated synthetic validation split. They do not establish 90%+ accuracy on real manga pages or camera photos. The next validation step is to run the detector on licensed/user-provided pages and measure text-box recall, reading order, CER and WER separately.
