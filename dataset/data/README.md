---
license: mit
task_categories:
  - object-detection
language:
  - en
tags:
  - yolo
  - ppe-detection
  - safety
  - computer-vision
  - construction
size_categories:
  - 10K<n<100K
---

# PPE Detection Dataset

딥러닝 기반 건설현장 안전 장비(PPE) 착용 모니터링을 위한 데이터셋

## Dataset Description

개인보호구(Personal Protective Equipment) 착용 상태를 감지하기 위한 YOLO 형식의 객체 탐지 데이터셋입니다.

### Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | helmet | 안전 헬멧 |
| 1 | vest | 반사 안전 조끼 |

### Dataset Statistics

| Split | Images | Labels | Helmet Objects | Vest Objects |
|-------|--------|--------|----------------|--------------|
| Train | 10,556 | 10,556 | 27,240 | 11,334 |
| Val | 2,262 | 2,262 | 5,973 | 2,279 |
| Test | 2,263 | 2,263 | 5,944 | 2,436 |
| **Total** | **15,081** | **15,081** | **39,157** | **16,049** |

### Data Format

YOLO 형식 (normalized coordinates):
```
class_id x_center y_center width height
```

Example:
```
0 0.456789 0.345678 0.123456 0.234567
1 0.567890 0.456789 0.145678 0.256789
```

## Dataset Structure

```
ppe-dataset/
├── train/
│   ├── images/     # 10,556 images
│   └── labels/     # 10,556 label files
├── val/
│   ├── images/     # 2,262 images
│   └── labels/     # 2,262 label files
└── test/
    ├── images/     # 2,263 images
    └── labels/     # 2,263 label files
```

## Usage

### Download with Hugging Face CLI

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download dataset
huggingface-cli download jhboyo/ppe-dataset --repo-type dataset --local-dir ./dataset
```

### Using with uv

```bash
uv tool install huggingface-hub
uv tool run hf download jhboyo/ppe-dataset --repo-type dataset --local-dir ./dataset/data
```

### YOLO Training Configuration

Create a YAML configuration file:

```yaml
# ppe_dataset.yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: helmet
  1: vest
```

### Training with YOLOv8

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
model.train(
    data='ppe_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## Data Sources

This dataset is merged from two Kaggle datasets:

1. **Hard Hat Detection** (5,000 images)
   - Original classes: helmet, head, person
   - Used: helmet only

2. **Safety Helmet and Reflective Jacket** (10,500 images)
   - Classes: Safety-Helmet, Reflective-Jacket
   - Used: both classes

### Preprocessing

- VOC to YOLO format conversion for Dataset 1
- Class ID unification (helmet: 0, vest: 1)
- File naming with prefix (ds1_, ds2_) to avoid conflicts
- Random split: 70% train, 15% val, 15% test (seed=42)

## License

MIT License

## Citation

```bibtex
@dataset{ppe_detection_2024,
  title={PPE Detection Dataset for Construction Safety},
  author={SafetyVisionAI Team},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/jhboyo/ppe-dataset}
}
```

## Project

This dataset is part of the **Safety Vision AI** project - a deep learning-based construction site safety equipment monitoring platform.

- GitHub: [SafetyVisionAI](https://github.com/jhboyo/SafetyVisionAI)
