# 📜 config.py
# ⚙️ 이 파일에서 프로젝트의 모든 주요 설정을 관리합니다.
# 연구자는 이 파일만 수정하여 파이프라인 전체를 제어할 수 있습니다.

# --- 1. 경로 설정 (Paths) ---
# 📥 원본 데이터가 있는 폴더 경로
DATA_PATH = "./data_raw/"
# 📤 최종 KPI 테이블이 저장될 폴더 경로
RESULTS_PATH = "./results/"
# 💾 최종 CSV 파일 이름
RESULT_FILENAME = "final_kpi_table.csv"


# --- 2. EEG 신호 기본 설정 (Signal Specs) ---
# 🧠 분석에 사용할 채널 이름 목록 (순서 중요)
# 예: 2채널 장비가 Fp1, Fp2 순서로 CSV에 저장된 경우
CHANNELS = ['Fp1', 'Fp2']
# Hz 단위의 샘플링 레이트
SAMPLE_RATE = 250


# --- 3. 모듈 2: 전처리 & 필터링 설정 (Preprocessing) ---
# 📉 저주파수 컷오프 (High-pass)
FILTER_L_FREQ = 0.5  # 0.5Hz 미만은 제거
# 📈 고주파수 컷오프 (Low-pass)
FILTER_H_FREQ = 40.0  # 40Hz 초과는 제거
# ⚡️ 전원 노이즈 필터 (Notch)
NOTCH_FREQ = 60.0  # 60Hz (미국/한국) 또는 50.0 (유럽)


# --- 4. 모듈 4: 데이터 분할 & 정제 설정 (Epoching) ---
# 🅰️ '첫 대면' (A. 형태학적/시간축 분석용) Epoch 설정
# 자극 제시 시점(0)을 기준으로 시작 시간(초)
EPOCH_A_TMIN = -1.0  # 자극 제시 1초 전부터
# 자극 제시 시점(0)을 기준으로 종료 시간(초)
EPOCH_A_TMAX = 3.0  # 자극 제시 3초 후까지

# 🅱️ '연속 거닐기' (B, C. 주파수/비선형 분석용) Epoch 설정
# 판단 마커(0)를 기준으로 시작 시간(초)
EPOCH_BC_TMIN = -10.0  # 판단 10초 전부터
# 판단 마커(0)를 기준으로 종료 시간(초)
EPOCH_BC_TMAX = 0.0  # 판단 시점까지

# 🗑️ Epoch 정제(Rejection) 기준 (진폭 기준)
# 이 값을 초과하는 Epoch는 노이즈로 간주하고 폐기
REJECT_THRESHOLD_UV = 100.0  # 100µV


# --- 5. 모듈 5: 핵심 변수 추출 설정 (Features) ---
# B. 주파수축 변수 (Bands)
# 각 주파수 대역의 (시작, 끝) Hz 정의
BANDS = {
    'Delta': (1.0, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 13.0),
    'Beta': (13.0, 30.0),
    'Gamma': (30.0, 40.0)  # 필터링(40Hz)과 일치시킴
}

# C. 비선형 변수 (Non-linear)
# 샘플 엔트로피(SampEn) 계산 시 사용할 파라미터
SAMPEN_M = 2  # 패턴 길이
SAMPEN_R_RATIO = 0.2  # 허용 반경 (표준편차의 20%)