# 📜 config.py
# ⚙️ 이 파일에서 프로젝트의 모든 주요 설정을 관리합니다.
# "교회 vs 시장" 환경 분류 목표에 맞게 수정되었습니다.

# --- 1. 경로 설정 (Paths) ---
# (변경 없음)
DATA_PATH = "./data_raw/"
RESULTS_PATH = "./results/"
RESULT_FILENAME = "final_kpi_table.csv"


# --- 2. EEG 신호 기본 설정 (Signal Specs) ---
# 🧠 분석에 사용할 EEG 채널 (예: Fp1, Fp2)
CHANNELS = ['Fp1', 'Fp2']
# 📡 (🔥 신규) CSV에 포함된 '스티뮬러스(이벤트)' 채널 이름
# M1 로드 및 M4 분할에 필수적입니다.
STIM_CHANNEL = 'stim'
# Hz 단위의 샘플링 레이트
SAMPLE_RATE = 250


# --- 3. 모듈 2 & 3: 전처리 및 ICA 설정 (Preprocessing & ICA) ---
# 📉 저주파수 컷오프 (High-pass)
FILTER_L_FREQ = 0.5
# 📈 고주파수 컷오프 (Low-pass)
FILTER_H_FREQ = 40.0
# ⚡️ 전원 노이즈 필터 (Notch)
NOTCH_FREQ = 60.0
# 👁️ (🔥 신규) M3 ICA에서 EOG(눈 깜빡임) 탐지용으로 사용할 채널
# (Fp1이 보통 가장 감지가 잘 됩니다)
EOG_CHANNEL_NAME = 'Fp1'  # cfg.CHANNELS[0] 대신 명시적으로 지정
# 🔢 (🔥 신규) M3 ICA 재현성을 위한 Random State
ICA_RANDOM_STATE = 97


# --- 4. 모듈 4: 데이터 분할 & 정제 설정 (Epoching) ---
# (🔥 중요: "교회 vs 시장" 목표에 맞게 섹션 전체가 재설계됨)

# 🗺️ "교회", "시장" 상태(Block)를 나타내는 stim 채널의 이벤트 ID
EVENT_IDS = {
    'church': 1,
    'market': 2
}
# ⏱️ 10분짜리 긴 블록을 잘게 쪼갤 '미니 Epoch'의 길이 (초)
EPOCH_DURATION_SEC = 5.0
# ♻️ Epoch 간의 중첩(Overlap) 시간 (초) (0.0 = 중첩 없음)
EPOCH_OVERLAP_SEC = 0.0
# 🗑️ Epoch 정제(Rejection) 기준 (진폭 기준, 100µV)
REJECT_THRESHOLD_UV = 100.0


# --- 5. 모듈 5: 핵심 변수 추출 설정 (Features) ---
# B. 주파수축 변수 (Bands)
BANDS = {
    'Delta': (1.0, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0),
    'Gamma': (30.0, 40.0)  # M2의 FILTER_H_FREQ와 일치
}

# (🔥 신규) B. 주파수축: Welch PSD 계산 시 윈도우 길이 (초)
# (해상도 1/N초. 예: 2초 -> 0.5Hz 해상도)
WELCH_WINDOW_SEC = 2.0
# (🔥 신규) B. 주파수축: 1/f 지수(기울기) 계산 시 사용할 주파수 범위 (Hz)
# (FOOOF 라이브러리 사용을 권장)
APERIODIC_FIT_RANGE_HZ = [1.0, 30.0]

# C. 비선형 변수 (Non-linear)
SAMPEN_M = 2  # 샘플 엔트로피(SampEn) 패턴 길이
SAMPEN_R_RATIO = 0.2  # 허용 반경 (표준편차의 20%)

# (🔥 신규) C. 동역학: 알파 버스트 임계값 (평균 + N * 표준편차)
ALPHA_BURST_THRESHOLD_SD = 1.0
# (🔥 신규) C. 동역학: 파워 변동성(Spectrogram) 계산 시 윈도우 길이 (초)
POWER_VAR_WINDOW_SEC = 1.0
# (🔥 신규) C. 동역학: 파워 변동성 계산 시 윈도우 중첩 비율 (0.0 ~ 1.0)
POWER_VAR_OVERLAP_RATIO = 0.5  # 50% 중첩

# (🔥 신규) C. 연결성: Coherence, PLV, wPLI 계산 시 윈도우 길이 (초)
CONN_WINDOW_SEC = 2.0