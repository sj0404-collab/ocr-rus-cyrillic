# Mobile Cyrillic OCR model

## What is shipped

- `onnx/cyrillic_pp-ocrv3_mobile_rec.onnx` — PP-OCRv3 mobile recognition model, converted to ONNX and distributed by the OAR OCR project.
- `onnx/pp-ocrv4_mobile_det.onnx` — lightweight PP-OCR text detector.
- `dicts/cyrillic_dict.txt` — PaddleOCR Cyrillic character dictionary.

The recognition model is a pretrained model, not a new model trained from scratch in this workspace. No labeled user dataset was supplied, so claiming a new 90%+ benchmark would be misleading. The wrapper masks non-Russian output classes and adds bounded visual consensus plus conservative Russian post-processing.

## Upstream and license

- Original recognition model: [PaddlePaddle/cyrillic_PP-OCRv3_mobile_rec](https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv3_mobile_rec), Apache-2.0.
- ONNX conversion/release: [GreatV/oar-ocr](https://github.com/GreatV/oar-ocr), check its repository and release license notices.
- Detector: [GreatV/oar-ocr release](https://github.com/GreatV/oar-ocr/releases/tag/v0.3.0).

SHA-256:

```text
6ab2b46cee27755f82cacd86a73706f00146f1938aa5c74549a4fb2d1f94ae9c  onnx/cyrillic_pp-ocrv3_mobile_rec.onnx
ab2a50dcd2c340852f2d0fbfa547d5eec79a0d04a774eb0b622d96d0d9d2ceeb  onnx/pp-ocrv4_mobile_det.onnx
369a82c6c8c479784a5d726448b83b1eafb5fef0a4129a5eaa3929625ddcd132  dicts/cyrillic_dict.txt
```

## Mobile formats

The repository includes both ONNX and fixed-shape TFLite artifacts:

```text
models/tflite/pp-ocrv4_mobile_det_float32.tflite
models/tflite/cyrillic_pp-ocrv3_mobile_rec_float32.tflite
```

TFLite input tensors are NHWC. The recognizer input is `[1, 48, 320, 3]`; the output is `[1, 40, 165]`. The detector input is `[1, 736, 736, 3]`. The float32 recognizer conversion was compared with ONNX Runtime on the same normalized crop: maximum absolute output difference was approximately `7e-6` and the argmax sequence matched. The detector TFLite artifact was loaded and invoked on the PC, but its full post-processing benchmark still belongs in the Android integration test.

TFLite SHA-256:

```text
3f5fd05d9c6fc1c5b11b832963f486a7ad483eb8e58fd0f9671028119b7160a1  tflite/cyrillic_pp-ocrv3_mobile_rec_float32.tflite
a2803d3c540e9077e561540285005b77dd2d47f7f5e470e8f2da2c993d9ad9f0  tflite/pp-ocrv4_mobile_det_float32.tflite
```
