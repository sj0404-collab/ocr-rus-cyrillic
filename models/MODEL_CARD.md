# Mobile Cyrillic OCR model

## What is shipped

- `onnx/cyrillic_pp-ocrv3_mobile_rec.onnx` — PP-OCRv3 mobile recognition model, converted to ONNX and distributed by the OAR OCR project.
- `onnx/pp-ocrv4_mobile_det.onnx` — lightweight PP-OCR text detector.
- `onnx/cyrillic_pp-ocrv5_mobile_rec.onnx` — optional second Cyrillic verifier, loaded only for uncertain crops.
- `dicts/cyrillic_dict.txt` — PP-OCRv3 Cyrillic character dictionary.
- `dicts/ppocrv5_cyrillic_dict.txt` — PP-OCRv5 Cyrillic character dictionary.

The recognition model is a pretrained model, not a new model trained from scratch in this workspace. No labeled user dataset was supplied, so claiming a new 90%+ benchmark would be misleading. The wrapper masks non-Russian output classes and adds bounded visual consensus plus conservative Russian post-processing. On uncertain crops, the optional PP-OCRv5 ONNX verifier runs as a second engine; agreement increases confidence, disagreement forces `certain=false`.

A first YOLO text/bubble detector was trained on 200 varied synthetic pages on the Windows runner for 15 CPU epochs. Its held-out synthetic metrics were `mAP50=0.88867` and `mAP50-95=0.62624`.

A second detector was trained with up to 100,000 Russian sentences extracted from OpenCorpora 2025 before synthetic rendering. Its held-out metrics were `mAP50=0.75632` and `mAP50-95=0.52320`; it is stored as `onnx/yolo_cyrillic_opencorpora_best.onnx`. The lower synthetic score reflects a harder Russian-corpus distribution, so this model should be benchmarked on real Russian pages before replacing the first detector. Neither score is a real-camera accuracy guarantee.

## Upstream and license

- Primary recognition model: [PaddlePaddle/cyrillic_PP-OCRv3_mobile_rec](https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv3_mobile_rec), Apache-2.0.
- Optional verifier: [PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec](https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec), Apache-2.0.
- ONNX conversion/release: [GreatV/oar-ocr](https://github.com/GreatV/oar-ocr), check its repository and release license notices.
- Detector: [GreatV/oar-ocr release](https://github.com/GreatV/oar-ocr/releases/tag/v0.3.0).

SHA-256:

```text
6ab2b46cee27755f82cacd86a73706f00146f1938aa5c74549a4fb2d1f94ae9c  onnx/cyrillic_pp-ocrv3_mobile_rec.onnx
a18d96d7c8d73d90f2ed056549caa1de3a8e6cb744cccba16cd593ea8cd2d569  onnx/cyrillic_pp-ocrv5_mobile_rec.onnx
ab2a50dcd2c340852f2d0fbfa547d5eec79a0d04a774eb0b622d96d0d9d2ceeb  onnx/pp-ocrv4_mobile_det.onnx
369a82c6c8c479784a5d726448b83b1eafb5fef0a4129a5eaa3929625ddcd132  dicts/cyrillic_dict.txt
db40aa52ceb112055be80c694afdf655d5d2c4f7873704524cc16a447ca913ba  dicts/ppocrv5_cyrillic_dict.txt
```

## Mobile formats

The repository includes both ONNX and fixed-shape TFLite artifacts:

```text
models/tflite/pp-ocrv4_mobile_det_float32.tflite
models/tflite/cyrillic_pp-ocrv3_mobile_rec_float32.tflite
models/tflite/cyrillic_pp-ocrv5_mobile_rec_float32.tflite
```

TFLite input tensors are NHWC. The recognizer input is `[1, 48, 320, 3]`; the PP-OCRv3 output is `[1, 40, 165]`, and the PP-OCRv5 verifier output is `[1, 40, 852]`. The detector input is `[1, 736, 736, 3]`. Both float32 recognizer conversions were compared with ONNX Runtime on the same normalized crop: the PP-OCRv3 maximum absolute output difference was approximately `7e-6`; the PP-OCRv5 difference was approximately `7e-7`; both argmax sequences matched. The detector TFLite artifact was loaded and invoked on the PC, but its full post-processing benchmark still belongs in the Android integration test.

TFLite SHA-256:

```text
3f5fd05d9c6fc1c5b11b832963f486a7ad483eb8e58fd0f9671028119b7160a1  tflite/cyrillic_pp-ocrv3_mobile_rec_float32.tflite
c51eb8df3eb94cce31f906c23ceb572151d27b0692a8e8bea62a38f6e54f7808  tflite/cyrillic_pp-ocrv5_mobile_rec_float32.tflite
a2803d3c540e9077e561540285005b77dd2d47f7f5e470e8f2da2c993d9ad9f0  tflite/pp-ocrv4_mobile_det_float32.tflite
5140cb61bb27e2c737e28f6a3ad58e65c321d2ae39b599c16a829b8f2c5d04db  onnx/yolo_cyrillic_text_bubble_best.onnx
af125a1f879a0b21f4d051ef0f1a1aae10933dd9f4929d21d1cfd3b44f2449ed  onnx/yolo_cyrillic_opencorpora_best.onnx
```
