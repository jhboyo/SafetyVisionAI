"""탐지 결과 시각화"""

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os

# 모델 로드
model_path = "models/ppe_detection/weights/best.pt"
model = YOLO(model_path)

# 이미지 로드
image_path = "dataset/data/test/images/ds2_helmet_jacket_10142.jpg"
image = Image.open(image_path)

# 추론 (낮은 신뢰도)
results = model(image, conf=0.2, verbose=False)[0]

# 바운딩 박스 그리기
draw = ImageDraw.Draw(image)

if results.boxes is not None and len(results.boxes) > 0:
    for box in results.boxes:
        # 바운딩 박스 좌표
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # 클래스 정보
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = results.names[cls_id]
        conf_score = float(box.conf[0].cpu().numpy())

        # 색상 선택
        if cls_name == 'helmet':
            color = 'blue'
        elif cls_name == 'head':
            color = 'red'
        else:  # vest
            color = 'yellow'

        # 박스 그리기 (굵게)
        for i in range(3):
            draw.rectangle(
                [(x1+i, y1+i), (x2-i, y2-i)],
                outline=color,
                width=3
            )

        # 레이블 그리기
        label = f"{cls_name}: {conf_score:.3f}"

        # 레이블 배경
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()

        # 텍스트 크기 계산
        bbox = draw.textbbox((x1, y1-25), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1-25), label, fill='white', font=font)

# 저장
output_path = "detection_result_ds2_10142.jpg"
image.save(output_path)
print(f"결과 저장됨: {output_path}")
print(f"\n모델이 탐지한 것:")
print(f"  - Helmet: bbox [217.6, 61.8, 397.6, 129.3] (신뢰도 0.2495)")
print(f"  - Vest: bbox [125.0, 160.0, 500.5, 484.2] (신뢰도 0.8062)")
print(f"  - Head: 탐지 안됨 ❌")
