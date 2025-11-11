# 📜 features/features_A.py
# 🧮 [모듈 5-A] 형태학적 & 시간축 KPI 계산 함수들
# (🔥 "교회 vs 시장" 목표에 맞게 ERP 로직이 제거됨)

import numpy as np
import config  # 설정값(sfreq 등)을 사용하기 위해 임포트
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

def get_A_features(epoch_data: np.ndarray, cfg: config, kpi_row: dict):
    """
    Epoch 데이터(단일 Epoch)에서 A 카테고리의 모든 KPI를 추출합니다.
    (🔥 수정됨: ERP 관련 로직 삭제. 5초 Epoch의 일반 통계만 계산)

    Args:
        epoch_data (np.ndarray): (n_channels, n_samples) 형태의 2D 배열.
        cfg (config): config.py 모듈 객체
        kpi_row (dict): KPI 결과를 누적할 딕셔너리 (수정됨)
    """
    
    sfreq = cfg.SAMPLE_RATE
    ch_names = cfg.CHANNELS

    # --- (🔥 삭제됨) "1. ERP-like 특징..." 섹션 삭제 ---
    # (5초 상태 Epoch에서는 ERP를 계산하지 않습니다.)

    # --- 2. 각 채널(Fp1, Fp2)을 순회하며 KPI 계산 ---
    for i, ch_name in enumerate(ch_names):
        x = epoch_data[i, :]  # (n_samples,) 1D 배열
        
        # 0으로 나누기 오류 방지용 상수
        epsilon = 1e-10 
        
        # --- A-1. 진폭/크기 특징 (Amplitude/Magnitude) ---
        kpi_row[f'{ch_name}_A_amp_max'] = np.max(x)
        kpi_row[f'{ch_name}_A_amp_min'] = np.min(x)
        kpi_row[f'{ch_name}_A_amp_p2p'] = np.max(x) - np.min(x)
        kpi_row[f'{ch_name}_A_amp_mean'] = np.mean(x)
        kpi_row[f'{ch_name}_A_amp_rms'] = np.sqrt(np.mean(np.square(x)))

        # --- A-2. 시간/지연 특징 (Temporal/Latency) ---
        # ⚡️ 영점 교차율 (ZCR)
        kpi_row[f'{ch_name}_A_zcr'] = ((x[:-1] * x[1:]) < 0).sum() / (len(x) - 1)
        
        # 📐 파형 슬로프 (Mean Absolute Slope)
        dx = np.diff(x)
        kpi_row[f'{ch_name}_A_slope_mean'] = np.mean(np.abs(dx))
        
        # 🌀 Hjorth Mobility (이동성)
        # (dx가 이미 계산됨)
        var_x = np.var(x)
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / (var_x + epsilon))
        kpi_row[f'{ch_name}_A_hjorth_mobility'] = mobility
        
        # ⏱️ (추가) 주요 피크 개수 (Num Peaks)
        # (노이즈로 인한 자잘한 피크를 제외하기 위해, 표준편차의 절반 이상 높이만 카운트)
        peaks, _ = find_peaks(x, height=np.std(x) * 0.5)
        kpi_row[f'{ch_name}_A_num_peaks'] = len(peaks)

        # --- A-3. 적분 특징 (Integral) ---
        # 🗺️ AUC (Area Under the Curve)
        kpi_row[f'{ch_name}_A_auc'] = np.trapz(np.abs(x), dx=1/sfreq)

        # --- A-4. 통계적/분포적 특징 (Statistical/Distributional) ---
        # M2️⃣ Hjorth Activity (활동성) / 분산
        kpi_row[f'{ch_name}_A_stat_variance'] = var_x
        
        # M3️⃣ 3차 모멘트 (Skewness, 왜도)
        kpi_row[f'{ch_name}_A_stat_skewness'] = skew(x)
        
        # M4️⃣ 4차 모멘트 (Kurtosis, 첨도)
        kpi_row[f'{ch_name}_A_stat_kurtosis'] = kurtosis(x)
        
        # 🌀 Hjorth Complexity (복잡성)
        ddx = np.diff(dx)
        var_ddx = np.var(ddx)
        mobility_dx = np.sqrt(var_ddx / (var_dx + epsilon))
        complexity = mobility_dx / (mobility + epsilon)
        kpi_row[f'{ch_name}_A_hjorth_complexity'] = complexity
        
    # --- (🔥 삭제됨) "A-5. ERP-like 특징..." 섹션 삭제 ---

    # (kpi_row 딕셔너리가 수정되었으므로, 별도 반환값 없음)