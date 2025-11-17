# TFGuard - 산업안전 재해 방지 개체탐지 시스템

## 프로젝트 개요
딥러닝 기반의 작업자 개인보호구(PPE) 착용 감지 및 산업안전 재해 방지를 위한 머신러닝 모델 개발 프로젝트

## 주요 기능
- 개인보호구(헬멧, 안전조끼, 안전화 등) 착용 상태 감지
- 작업자 얼굴 신원 확인
- 산업현장 위험 요소 탐지
- 실시간 안전 모니터링

## 프로젝트 구조
```
tfguard/
├── materials/          # 프로젝트 관련 문서 및 자료
│   ├── papers/        # 연구논문
│   ├── patents/       # 특허 자료
│   └── company/       # 회사 자료
├── data/              # 데이터셋
│   ├── train_images/  # 훈련용 이미지
│   ├── val_images/    # 검증용 이미지
│   └── test_images/   # 테스트용 이미지
├── models/            # 훈련된 모델 파일
├── src/               # 소스 코드
├── notebooks/         # Jupyter 노트북 (실험, 데이터 분석)
├── configs/           # 설정 파일
├── pyproject.toml     # Python 의존성 및 프로젝트 설정 (uv 사용)
├── uv.lock           # 의존성 락파일
├── main.py           # 메인 실행 파일
├── .gitignore        # Git 제외 파일 목록
├── CLAUDE.md         # 프로젝트 지침서
└── README.md         # 프로젝트 설명
```

## 개발 환경
- Python 3.8+
- uv (패키지 관리자)
- TensorFlow/PyTorch
- OpenCV
- YOLO/Fast R-CNN 등 객체 탐지 모델

## 환경 설정
```bash
# uv로 의존성 설치
uv sync

# 가상환경 활성화 (자동으로 관리됨)
uv run python main.py
```

## 명령어
### 모델 훈련
```bash
uv run python src/train.py --config configs/train_config.yaml
```

### 추론/테스트
```bash
uv run python src/inference.py --model models/best_model.pth --input data/test_images/
```

### 데이터 전처리
```bash
uv run python src/preprocess.py --input data/raw/ --output data/processed/
```

### 메인 애플리케이션 실행
```bash
uv run python main.py
```

## 참고 자료
- materials/papers/딥 러닝 기반 작업자 개인보호구 착용 및 얼굴 신원 확인 시스템에 관한 연구.pdf
- materials/3조_팀소개_팀플주제_선정.pdf
- materials/New AI_BM_canvas_3조.pdf
- materials/patents/ (특허 관련 자료들)
- materials/company/ (미스릴 회사 자료)