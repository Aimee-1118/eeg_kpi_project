# 📜 features/features_A.py
# 🧮 [모듈 5-A] 형태학적 & 시간축 KPI 계산 함수들

import numpy as np
import config  # 설정값(sfreq 등)을 사용하기 위해 임포트
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

def get_A_features(epoch_data: np.ndarray, cfg: config, kpi_row: dict):
    """
    'A' 유형 Epoch 데이터(단일 Epoch)에서 A 카테고리의 모든 KPI를 추출합니다.
    추출된 KPI는 'kpi_row' 딕셔너리에 직접 추가됩니다.

    Args:
        epoch_data (np.ndarray): (n_channels, n_samples) 형태의 2D 배열.
                                 (예: (2, 1000))
        cfg (config): config.py 모듈 객체
        kpi_row (dict): KPI 결과를 누적할 딕셔너리 (수정됨)
    """
    
    sfreq = cfg.SAMPLE_RATE
    ch_names = cfg.CHANNELS

    # --- 1. ERP-like 특징을 위한 시간 -> 샘플 변환 ---
    # 베이스라인(-1초 ~ 0초) 이후가 실제 자극 구간입니다.
    # EPOCH_A_TMIN이 -1.0이면, 0초는 'sfreq'번째 샘플이 됩니다.
    try:
        zero_sample_idx = int(abs(cfg.EPOCH_A_TMIN) * sfreq)
        
        # P300-like window (예: 250ms ~ 400ms)
        p3_start_idx = zero_sample_idx + int(0.250 * sfreq)
        p3_end_idx = zero_sample_idx + int(0.400 * sfreq)
        
        # LPP-like window (예: 400ms ~ 1000ms)
        lpp_start_idx = zero_sample_idx + int(0.400 * sfreq)
        lpp_end_idx = zero_sample_idx + int(1.000 * sfreq)
        
    except Exception as e:
        print(f"[ERROR M5-A] ERP 시간 인덱스 계산 실패: {e}. config.py의 EPOCH_A_TMIN 값을 확인하세요.")
        return # 이 Epoch 계산 중단

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
        dx = np.diff(x)
        var_x = np.var(x)
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / (var_x + epsilon))
        kpi_row[f'{ch_name}_A_hjorth_mobility'] = mobility
        
        # ⏱️ (추가) 주요 피크 개수 (Num Peaks)
        # (노이즈로 인한 자잘한 피크를 제외하기 위해, 표준편차의 절반 이상 높이만 카운트)
        peaks, _ = find_peaks(x, height=np.std(x) * 0.5)
        kpi_row[f'{ch_name}_A_num_peaks'] = len(peaks)

        # --- A-3. 적분 특징 (Integral) ---
        # 🗺️ AUC (Area Under the Curve) - (전체 Epoch의 총 활동량)
        # (dx=1/sfreq 를 통해 실제 시간 단위의 면적을 계산)
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
        
        
        # --- A-5. ERP-like 특징 (사건 관련) ---
        # 위에서 계산한 시간 인덱스를 사용
        
        # P300-like (250~400ms)
        x_p3 = x[p3_start_idx:p3_end_idx]
        if len(x_p3) > 0:
            # 🧠 P300 진폭 (Peak)
            kpi_row[f'{ch_name}_A_erp_p3_peak'] = np.max(x_p3)
            # 🧠 P300 잠복기 (Latency)
            # (Epoch 시작(-1초) 기준이 아닌, 자극(0초) 기준 Latency (ms))
            latency_samples = np.argmax(x_p3) + p3_start_idx - zero_sample_idx
            kpi_row[f'{ch_name}_A_erp_p3_latency_ms'] = (latency_samples / sfreq) * 1000.0

        # LPP-like (400~1000ms)
        x_lpp = x[lpp_start_idx:lpp_end_idx]
        if len(x_lpp) > 0:
            # ❤️ LPP (Mean Amplitude or AUC)
            kpi_row[f'{ch_name}_A_erp_lpp_mean'] = np.mean(x_lpp)
            kpi_row[f'{ch_name}_A_erp_lpp_auc'] = np.trapz(np.abs(x_lpp), dx=1/sfreq)

    # (참고: 이 함수는 kpi_row 딕셔너리를 직접 수정했으므로,
    #  별도로 값을 반환(return)할 필요가 없습니다.)