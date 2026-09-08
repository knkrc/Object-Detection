Measured on arm64 Darwin 25.6.0, Python 3.13.9. Timings cover `Detector.detect()` end to end,
including drawing the boxes — the wait a user actually sees.

| Model | Device | ms / image | FPS |
|---|---|---|---|
| yolov8n.pt | mps | 9.3 | 107.2 |
| yolov8n.pt | cpu | 25.3 | 39.4 |
| yolov8s.pt | mps | 15.5 | 64.5 |
| yolov8s.pt | cpu | 49.8 | 20.1 |
| yolov8m.pt | mps | 28.9 | 34.6 |
| yolov8m.pt | cpu | 99.9 | 10.0 |

README badge for these numbers:

```markdown
[![Speed](https://img.shields.io/badge/yolov8n-~40_FPS_on_CPU-success)](#how-fast-is-it)
```
