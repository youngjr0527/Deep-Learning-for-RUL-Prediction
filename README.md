# Deep Learning 기반 CMAPSS 데이터셋 RUL 예측

이 프로젝트는 NASA의 CMAPSS(Commercial Modular Aero-Propulsion System Simulation) 데이터셋을 사용하여 항공 엔진의 잔여 유효 수명(RUL, Remaining Useful Life)을 예측하는 딥러닝 모델을 구현한 것입니다. 현재 LSTM, BiLSTM, TCN(Temporal Convolutional Network)과 Transformer 모델이 구현되어 있습니다.

## 목차
- [개요](#개요)
- [데이터셋](#데이터셋)
- [모델 설명](#모델-설명)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [결과 시각화](#결과-시각화)
- [프로젝트 구조](#프로젝트-구조)
- [기여하기](#기여하기)

## 개요

예측 유지보수(Predictive Maintenance)는 산업 자산의 상태를 모니터링하고 고장이 발생하기 전에 유지보수가 필요한 시점을 예측하는 기술입니다. 이 프로젝트에서는 시계열 데이터를 활용하여 항공 엔진의 잔여 유효 수명(RUL)을 예측하는 다양한 딥러닝 모델을 구현하고 비교합니다.

주요 특징:
- 다양한 딥러닝 모델(LSTM, BiLSTM, TCN, Transformer) 비교 가능
- 명령줄 인자를 통한 간편한 모델 및 하이퍼파라미터 선택
- Early stopping 기능으로 효율적인 학습
- 결과 시각화 및 CSV 형식 저장

## 데이터셋

CMAPSS 데이터셋은 NASA에서 제공하는 항공 엔진 데이터셋으로, 시뮬레이션된 엔진의 다양한 센서 측정값을 포함하고 있습니다. 데이터셋은 정상 상태에서 시작해 성능이 저하되어 최종적으로 고장나는 엔진의 시계열 데이터를 담고 있습니다.

이 프로젝트에서는 다음과 같은 센서 데이터를 사용합니다:
- T30 (HPC 출구 온도)
- T50 (LPT 출구 온도)
- P30 (HPC 출구 압력)
- PS30 (정적 압력 - HPC 출구)
- phi (연료-공기 비율)

## 모델 설명

### LSTM (Long Short-Term Memory)
LSTM은 기존 RNN의 장기 의존성 문제를 해결하기 위해 설계된 특수한 RNN 아키텍처입니다. 입력, 망각, 출력 게이트를 사용하여 시계열 데이터에서 중요한 패턴을 학습합니다.

### BiLSTM (Bidirectional LSTM)
BiLSTM은 LSTM을 양방향으로 확장한 모델로, 시퀀스를 정방향과 역방향으로 동시에 처리하여 과거와 미래의 문맥을 모두 고려할 수 있습니다.

### TCN (Temporal Convolutional Network)
TCN은 긴 시퀀스 데이터를 처리하기 위한 컨볼루션 기반 아키텍처입니다. 주요 특징으로 인과성(causality), 확장된 컨볼루션(dilated convolution), 잔차 연결(residual connections)이 있어 시계열 예측 작업에 효과적입니다.

### Transformer
Transformer는 "Attention is All You Need" 논문에서 소개된 아키텍처로, 인코더-디코더 구조와 셀프 어텐션 메커니즘을 기반으로 합니다. 이 프로젝트에서는 RUL 예측에 맞게 수정된 Transformer 모델을 구현하였습니다.

## 설치 방법

### 필수 요구사항
- Python 3.8 이상
- PyTorch 1.9 이상
- NumPy, Pandas, Matplotlib, scikit-learn

### 설치

```bash
# 저장소 클론
git clone https://github.com/youngjr0527/PyTorch-Transformer-for-RUL-Prediction.git
cd PyTorch-Transformer-for-RUL-Prediction

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 기본 사용법

모델 훈련 및 평가:

```bash
# LSTM 모델 사용
python main.py --model lstm

# BiLSTM 모델 사용
python main.py --model bilstm

# TCN 모델 사용
python main.py --model tcn

# Transformer 모델 사용
python main.py --model transformer
```

### 주요 매개변수

```bash
# 모델 선택
--model [lstm/bilstm/tcn/transformer]  # 사용할 모델

# 데이터셋 관련
--dataset [FD001/FD002/FD003/FD004]  # CMAPSS 데이터셋 선택 (기본값: FD001)

# 훈련 관련
--epochs [값]          # 훈련 에포크 수 (기본값: 100)
--batch_size [값]      # 배치 크기 (기본값: 32)
--lr [값]              # 학습률 (기본값: 0.001)
--dropout [값]         # 드롭아웃 비율 (기본값: 0.1)
--seed [값]            # 랜덤 시드 (기본값: 42)

# Early stopping 관련
--patience [값]        # 조기 중단 인내 값 (기본값: 10)
--min_delta [값]       # 개선으로 간주할 최소 변화량 (기본값: 0.0001)

# Transformer 모델 관련
--d_model [값]         # Transformer 임베딩 차원 (기본값: 128)
--heads [값]           # Transformer 어텐션 헤드 수 (기본값: 4)
--n_layers [값]        # Transformer 인코더 레이어 수 (기본값: 2)
```

### 예제

```bash
# 더 작은 Transformer 모델로 50 에포크 훈련
python main.py --model transformer --d_model 64 --heads 2 --n_layers 1 --epochs 50

# LSTM 모델로 학습률과 드롭아웃 조정
python main.py --model lstm --lr 0.0005 --dropout 0.2

# BiLSTM 모델로 학습
python main.py --model bilstm --batch_size 64 --epochs 150

# TCN 모델로 훈련
python main.py --model tcn --lr 0.001 --dropout 0.15

# Early stopping 매개변수 조정
python main.py --model transformer --d_model 256 --heads 8 --n_layers 4 --seq_len 30 --lr 0.0005 --patience 15 --lr_scheduler
```

## 결과 시각화

모델 훈련이 완료되면 다음과 같은 결과물이 생성됩니다:

1. 최적 모델 가중치: `best_[모델타입]_model_[날짜시간].pth`
2. 예측 결과 CSV 파일: `predictions_[모델타입]_[날짜시간].csv`
3. RUL 예측 시각화 이미지: `RUL_Prediction_[모델타입]_[날짜시간].png`

RUL 예측 그래프는 실제값과 예측값을 비교하여 모델의 성능을 시각적으로 평가할 수 있게 해줍니다.

## 프로젝트 구조

리팩토링 후 더 모듈화된 구조로 코드가 정리되었습니다:

```
PyTorch-Transformer-for-RUL-Prediction/
├── models/                  # 모델 구현 디렉토리
│   ├── __init__.py         # 모델 패키지 초기화
│   ├── lstm_model.py       # LSTM 모델 구현
│   ├── bilstm_model.py     # BiLSTM 모델 구현
│   ├── tcn_model.py        # TCN 모델 구현
│   └── transformer_model.py # Transformer 모델 구현
├── data/                    # 데이터 관련 디렉토리
│   ├── __init__.py         # 데이터 패키지 초기화
│   ├── data_utils.py       # 데이터셋 및 데이터 로더 유틸리티
│   └── data_loader.py      # 데이터 로드 및 전처리 함수
├── utils/                   # 유틸리티 디렉토리
│   ├── __init__.py         # 유틸리티 패키지 초기화
│   └── visualization.py    # 결과 시각화 함수
├── datasets/                # 데이터셋 저장 디렉토리
├── main.py                  # 메인 실행 파일
├── train.py                 # 훈련 관련 코드
└── requirements.txt         # 필요한 패키지 목록
```
