
# 🧠 EEG KPI Analysis Pipeline

좋은 음악 vs 싫은 음악 청취 시 뇌파 변화를 자동으로 비교 분석하는 머신러닝용 KPI 추출 파이프라인입니다.

원시 EEG 데이터(텍스트 파일)를 자동으로 로드하여 전처리, 분할(Epoching), 정제(Artifact Rejection), 그리고 뇌파 특징(KPI) 추출 과정을 거쳐, 최종적으로 **CSV 형식의 분석 테이블**을 생성합니다.

---

## 🎯 Project Overview

**목표:** Condition(G: Good/B: Bad)별 EEG 신호의 뇌파 지표(KPI)를 자동 추출하여 SPSS 분석용 CSV 파일 생성.

**사용 기술:**
- **MNE-Python:** EEG 신호 처리 (필터링, Epoching)
- **Welch's Method:** 주파수 대역별 전력(Band Power) 계산
- **Cross-Channel Analysis:** 비대칭도(Asymmetry), 코히런스(Coherence)
- **OmegaConf:** 설정 관리

---

## 📦 Installation

### 1. 가상환경 생성 (권장)
```bash
# Python 3.9+ 필수
python -m venv .venv

# 가상환경 활성화
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

**주요 라이브러리:**
- `mne>=1.8.0` - EEG 신호 처리
- `pandas>=2.2.0` - 데이터 테이블 관리
- `numpy>=1.26.0` - 수치 계산
- `scipy>=1.13.0` - 신호 처리 (FFT, Coherence)
- `omegaconf>=2.3.0` - YAML 설정 관리

---

## 🚀 Quick Start

### 1. 데이터 준비
`raw_data/` 폴더에 EEG 데이터 파일을 배치합니다.

**파일 형식:**
- 파일명: `[Subject_Name]_[G|B]_[NNN].txt`
  - 예: `나현_G_001.txt`, `철수_B_002.txt`
- G = Condition 1 (Good), B = Condition 2 (Bad)
- NNN = 세자리 시행 번호 (001~999)

**데이터 구조 (CSV):**
```
Timestamp(HH:mm:ss.SSS),Ch1(uV),Ch2(uV)
00:00:00.149,-6.64,-15.82
00:00:00.150,0.34,-5.63
...
```

### 2. 파이프라인 실행
```bash
# 기본 설정으로 실행
python main.py

# 커스텀 설정 파일 지정
python main.py -c ./configs/analysis_config.yaml

# 런타임 설정 오버라이드 (예: 필터 대역 변경)
python main.py PREPROCESSING.filter_band.low=1.0 PREPROCESSING.filter_band.high=45.0
```

### 3. 결과 확인
```bash
# 생성된 파일 확인
ls -la output/
```

**생성 파일:**
- `output/eeg_kpi_analysis_results.csv` - 분석 결과 테이블
- `output/analysis_summary.txt` - 처리 요약 보고서
- `output/analysis.log` - 상세 로그

---

## ⚙️ Configuration

### `configs/analysis_config.yaml`

**주요 설정:**

```yaml
PATHS:
  data_dir: ./raw_data              # 입력 데이터 폴더
  output_dir: ./output              # 출력 폴더
  log_file: ./output/analysis.log   # 로그 파일

PREPROCESSING:
  sampling_rate: 250                # 샘플링 레이트 (Hz)
  filter_band:
    low: 0.5                        # 대역통과 필터 하한 (Hz)
    high: 50.0                      # 대역통과 필터 상한 (Hz)
  notch_freq: 60.0                  # 노치 필터 (전원 잡음, Hz)
  artifact_threshold_uv: 150.0      # 아티팩트 임계값 (µV)

EPOCH:
  window_sec: 4.0                   # Epoch 길이 (초)
  overlap_sec: 2.0                  # Epoch 오버랩 (초)

BANDS:
  Delta: [0.5, 4.0]                 # 주파수 대역
  Theta: [4.0, 8.0]
  Alpha: [8.0, 13.0]
  Beta: [13.0, 30.0]
  Gamma: [30.0, 50.0]

KPI_SELECT:
  core:
    - band_powers                   # 필수 KPI
    - basic_stats
  optional:
    - asymmetry                     # 선택 KPI
    - coherence
    - sef90
    - center_freq
    - spectral_entropy
    - ratios
```

---

## 📊 Outputs

### 1. `eeg_kpi_analysis_results.csv`

**구조:** 메타데이터 + Cross-Channel KPI + Ch1 KPI + Ch2 KPI

| 컬럼 | 설명 |
|------|------|
| `Subject_ID` | 피험자 이름 |
| `Condition` | 조건 (1=G, 2=B) |
| `Trial_No` | 시행 번호 |
| `FileName` | 원본 파일명 |
| `Ch1_Band_Delta` | Ch1 Delta 대역 전력 |
| `Ch2_Band_Alpha` | Ch2 Alpha 대역 전력 |
| `Asym_Band_Alpha` | Alpha 비대칭도: ln(Ch2) - ln(Ch1) |
| `Conn_Coh_Alpha` | Ch1-Ch2 Coherence (Alpha 대역) |
| `Ch1_Stat_Mean` | Ch1 평균 진폭 |
| `Ch1_Ratio_TBR` | Ch1 Theta/Beta 비율 |
| ... | 총 28개 KPI |

**인코딩:** UTF-8-BOM (Excel 한글 호환)

### 2. `analysis_summary.txt`

```
======================================================================
EEG KPI 분석 보고서
======================================================================

실행 일시: 2026-01-21 19:39:33
완료 일시: 2026-01-21 19:39:33
소요 시간: 0.35초

총 파일 수: 1
성공: 1
실패: 0

[Failed Files List]
없음

======================================================================
```

---

## 🔄 Pipeline Stages

### Stage 1: Data Loading (Phase 3)
- 텍스트 파일 읽기 (pandas CSV 파서)
- 컬럼명 정규화 (`Timestamp(HH:mm:ss.SSS)` → `Timestamp`)
- 단위 변환 (µV → V)
- 샘플링 레이트 검증

### Stage 2: Preprocessing (Phase 3)
1. Notch Filter: 60Hz 제거
2. Bandpass Filter: 0.5~50Hz 추출

### Stage 3: Epoching (Phase 4)
- 고정 길이 Epoch 분할
  - 길이: 4초
  - 오버랩: 2초

### Stage 4: Artifact Rejection (Phase 4)
- 진폭 기준 검사 (150µV 초과 제거)
- Q16 규칙: 3개 미만 Epoch는 분석 불가

### Stage 5: Feature Extraction (Phase 5)
**Band Powers (Welch's Method):**
- Ch1/Ch2 각각: Delta, Theta, Alpha, Beta, Gamma

**Basic Statistics:**
- Mean, Std, Skewness, Kurtosis

**Cross-Channel:**
- Asymmetry (Alpha): ln(Ch2_Power) - ln(Ch1_Power)
- Coherence: 5개 대역별 Ch1-Ch2 간 coherence

**Ratios:**
- TBR (Theta/Beta)
- Engagement (Beta / (Alpha+Theta))

### Stage 6: Integration & Reporting (Phase 6)
- DataFrame 생성
- 컬럼 정렬 (Metadata → Cross-Channel → Ch1 → Ch2)
- CSV 저장 (UTF-8-BOM)
- 요약 보고서 생성

---

## 🧪 Testing

각 단계별 테스트 스크립트 제공:

```bash
# Phase 3: 데이터 로드 & 전처리 테스트
python tests/test_phase3.py

# Phase 4: Epoching & Artifact Rejection 테스트
python tests/test_phase4.py

# Phase 5: Feature Extraction 테스트
python tests/test_phase5.py
```

---

## 📝 KPI 명명 규칙 (Snake Case)

모든 KPI는 다음 패턴을 따릅니다:

```
[Channel]_[Category]_[Subcategory]

예:
- Ch1_Band_Alpha         # Ch1 채널의 Alpha 대역 전력
- Ch2_Stat_Mean          # Ch2 채널의 평균 통계
- Asym_Band_Alpha        # Alpha 비대칭도
- Conn_Coh_Beta          # Beta 대역 Coherence
- Ch1_Ratio_TBR          # Ch1 TBR 비율
```

---

## ⚠️ Error Handling

**파일 처리 실패:**
- 개별 파일 실패 시 다음 파일로 진행
- 실패 원인을 `analysis.log`에 기록
- `analysis_summary.txt`에 실패 파일 목록 별도 보고

**KPI 계산 실패:**
- 개별 KPI 실패 시 해당 값만 `NaN`
- 전체 파이프라인 중단 안 함
- Warning 로그 출력

**3개 미만 Epoch (Q16 규칙):**
- 분석 불가능한 파일로 판정
- 해당 파일 건너뜀
- 로그에 경고 메시지 기록

---

## 📂 Project Structure

```
eeg_kpi_project/
├── main.py                         # 메인 실행 파일
├── configs/
│   └── analysis_config.yaml        # 분석 설정
├── core/
│   ├── __init__.py
│   ├── data_scanner.py             # 파일 스캔
│   ├── loader.py                   # 데이터 로드
│   ├── preprocessor.py             # 필터링
│   ├── epocher.py                  # Epoch 생성
│   ├── cleaner.py                  # Artifact 제거
│   └── feature_extractor.py        # KPI 추출
├── utils/
│   ├── __init__.py
│   └── config_loader.py            # 설정 로더
├── tests/
│   ├── test_phase3.py              # 로드 & 전처리 테스트
│   ├── test_phase4.py              # Epoching 테스트
│   └── test_phase5.py              # KPI 추출 테스트
├── raw_data/                       # 입력 데이터 폴더 (사용자 배치)
├── output/                         # 결과 폴더 (자동 생성)
├── requirements.txt                # 의존성 패키지
└── README.md                       # 이 파일
```

---

## 🤝 Contributing

버그 리포트 및 개선 사항은 GitHub Issues를 통해 제출해주세요.

---

## 📄 License

This project is provided as-is for educational and research purposes.

      * **중요:** CSV 파일에는 `configs/base_config.yaml`에 정의된 `CHANNELS` (예: 'Fp1', 'Fp2')와 `STIM_CHANNEL` (예: 'stim') 열이 반드시 포함되어야 합니다.
      * `stim` 채널에는 `EVENT_IDS`에 정의된 숫자(예: `1`=교회 시작, `2`=시장 시작)가 마킹되어 있어야 합니다.

2.  **설정:** `configs/base_config.yaml` 파일을 열어 자신의 데이터 스펙(채널, 샘플링 레이트, 이벤트 ID)에 맞게 수정합니다.

3.  **실행:** 터미널에서 `main.py` 파일을 실행합니다.

    ```bash
    # 기본 설정(base_config.yaml)으로 실행
    python main.py

    # (선택) 설정을 덮어쓰며 실행 (YAML 파일을 직접 수정할 필요 없음)
    # 예: ICA를 켜고, Epoch 길이를 2초로 변경하여 테스트
    python main.py --USE_ICA=True --EPOCH_DURATION_SEC=2.0
    ```

4.  **결과 확인:** 파이프라인이 완료되면 `results/` 폴더에 `final_kpi_table.csv` 파일이 생성됩니다.

-----

## 💡 5. 최종 분석 (Analyze.py) 가이드

`final_kpi_table.csv` 파일은 "1. 특징 공학"의 산출물입니다. 이 데이터를 사용하여 "2. 모델링" (예: `analyze.py` 생성)을 수행할 때, 연구의 타당성을 위해 다음 사항을 강력히 권장합니다.

  * **결측치 처리:** `NaN`/`Inf` 값이 포함된 행(Epoch)이나 열(KPI)은 `0`이나 평균으로 대체하지 말고, 통계적 왜곡을 막기 위해 \*\*제거(삭제)\*\*하는 것을 원칙으로 합니다.
  * **데이터 누수 방지:** `StandardScaler` (표준화) 등은 `sklearn.pipeline.Pipeline` 내에서 사용해야 합니다.
  * **피험자 독립성:** 모델 검증 시 `KFold` 대신 `GroupKFold(groups=df['source_file'])`를 사용하여, '처음 보는' 피험자의 데이터로 검증해야 합니다.
  * **KPI 선별:** `sklearn.linear_model.LassoCV` (LASSO)를 사용하여 50개+의 KPI 중 환경 분류에 가장 유의미한 지표를 자동으로 선별(Feature Selection)할 수 있습니다.