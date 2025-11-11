# 📜 features/features_B.py
# 🧮 [모듈 5-B] 주파수축 KPI 계산 함수들
# (🔥 config.py 설정값 연동 및 1/f 지수 계산 업그레이드)

import numpy as np
import config
from scipy.signal import welch
import antropy as ant
# (🔥 신규) 1/f 지수(기울기)의 정교한 계산을 위해 fooof 라이브러리 임포트
from fooof import FOOOF
# (🔥 신규) 안전한 로그 계산을 위해 utils.py에서 safe_log 임포트
from .utils import safe_log

def get_B_features(epoch_data: np.ndarray, cfg: config, kpi_row: dict):
    """
    'B/C' 유형 Epoch 데이터(단일 Epoch)에서 B 카테고리의 모든 KPI를 추출합니다.
    추출된 KPI는 'kpi_row' 딕셔너리에 직접 추가됩니다.
    (🔥 수정됨: config 설정값 연동, fooof 라이브러리 적용, safe_log 적용)

    Args:
        epoch_data (np.ndarray): (n_channels, n_samples) 형태의 2D 배열.
        cfg (config): config.py 모듈 객체
        kpi_row (dict): KPI 결과를 누적할 딕셔너리 (수정됨)
    """
    
    sfreq = cfg.SAMPLE_RATE
    ch_names = cfg.CHANNELS
    bands = cfg.BANDS
    
    # 0으로 나누기 오류 방지용 상수
    epsilon = 1e-10 
    
    band_powers_per_channel = {ch_name: {} for ch_name in ch_names}
    
    # (🔥 신규) FOOOF 모델 객체 초기화 (매 채널마다 재사용)
    # Aperiodic(배경) 모드만 'fixed'로 설정하여 기울기만 피팅
    fm = FOOOF(peak_width_limits=[0.5, 12.0], 
               max_n_peaks=8, 
               min_peak_height=0.0,
               peak_threshold=2.0,
               aperiodic_mode='fixed', # 'fixed' 또는 'knee'
               verbose=False) # FOOOF의 로그 메시지 끄기

    # --- 1. 각 채널(Fp1, Fp2)을 순회하며 PSD 기반 KPI 계산 ---
    for i, ch_name in enumerate(ch_names):
        x = epoch_data[i, :]
        
        # --- 1a. PSD 계산 (Welch's Method) ---
        # (🔥 수정됨) config.py의 WELCH_WINDOW_SEC 설정을 사용
        nperseg = int(sfreq * cfg.WELCH_WINDOW_SEC)
        if len(x) < nperseg:
            nperseg = len(x)
            
        freqs, psd = welch(x, sfreq=sfreq, nperseg=nperseg, nfft=nperseg)
        freq_res = freqs[1] - freqs[0]

        # --- B-1. 스펙트럼 파워/크기 특징 ---
        abs_powers = {}
        for band_name, (f_low, f_high) in bands.items():
            band_mask = (freqs >= f_low) & (freqs < f_high)
            if np.sum(band_mask) == 0:
                abs_powers[band_name] = 0.0
            else:
                abs_powers[band_name] = np.trapz(psd[band_mask], dx=freq_res)
        
        # (🔥 수정됨) total_power 계산 범위를 config의 BANDS에 정의된 범위로 한정
        # (델타 ~ 감마 밴드의 합)
        total_power_bands = np.sum(list(abs_powers.values()))

        # KPI 딕셔너리에 저장
        kpi_row[f'{ch_name}_B_pow_total'] = total_power_bands
        for band_name, abs_p in abs_powers.items():
            rel_p = (abs_p / (total_power_bands + epsilon)) * 100.0
            kpi_row[f'{ch_name}_B_pow_abs_{band_name.lower()}'] = abs_p
            kpi_row[f'{ch_name}_B_pow_rel_{band_name.lower()}'] = rel_p
            band_powers_per_channel[ch_name][band_name] = abs_p

        # --- B-2. 스펙트럼 주파수/위치 특징 ---
        alpha_mask = (freqs >= bands['Alpha'][0]) & (freqs < bands['Alpha'][1])
        if np.sum(alpha_mask) > 0:
            kpi_row[f'{ch_name}_B_loc_peak_alpha_hz'] = freqs[alpha_mask][np.argmax(psd[alpha_mask])]
        
        psd_cumsum = np.cumsum(psd) * freq_res
        total_power_psd = psd_cumsum[-1] # Welch로 계산된 전체 파워 (0 ~ sfreq/2)
        try:
            sef90_idx = np.searchsorted(psd_cumsum, 0.90 * total_power_psd)
            kpi_row[f'{ch_name}_B_loc_sef90_hz'] = freqs[sef90_idx]
        except IndexError:
            kpi_row[f'{ch_name}_B_loc_sef90_hz'] = np.nan
        
        kpi_row[f'{ch_name}_B_loc_centroid_hz'] = np.sum(freqs * psd) / (np.sum(psd) + epsilon)

        # --- B-3. 스펙트럼 형태/분포 특징 ---
        kpi_row[f'{ch_name}_B_shape_spec_ent'] = ant.spectral_entropy(psd, sfreq=sfreq, method='welch', normalize=True)
        
        # (🔥 수정됨) 1/f 지수 (FOOOF 라이브러리 사용)
        try:
            # config.py의 피팅 범위(예: 1-30Hz) 설정
            fit_range = cfg.APERIODIC_FIT_RANGE_HZ 
            fm.add_data(freqs, psd, freq_range=fit_range)
            fm.fit()
            
            # 1/f의 지수(기울기)와 절편(Offset) 추출
            ap_params = fm.get_params('aperiodic_params')
            kpi_row[f'{ch_name}_B_shape_1f_exponent'] = ap_params[1] # Exponent (χ)
            kpi_row[f'{ch_name}_B_shape_1f_offset'] = ap_params[0]   # Offset
        except Exception as e:
            # (피팅 실패 시)
            print(f"[ERROR M5-B] FOOOF 피팅 실패: {e}")
            kpi_row[f'{ch_name}_B_shape_1f_exponent'] = np.nan
            kpi_row[f'{ch_name}_B_shape_1f_offset'] = np.nan
            
        # --- B-4. 밴드 간 비율 특징 (채널 내부) ---
        p_theta = band_powers_per_channel[ch_name].get('Theta', 0)
        p_alpha = band_powers_per_channel[ch_name].get('Alpha', 0)
        p_beta = band_powers_per_channel[ch_name].get('Beta', 0)
        p_delta = band_powers_per_channel[ch_name].get('Delta', 0)

        kpi_row[f'{ch_name}_B_ratio_tbr'] = p_theta / (p_beta + epsilon)
        kpi_row[f'{ch_name}_B_ratio_engagement'] = p_beta / (p_alpha + p_theta + epsilon)
        kpi_row[f'{ch_name}_B_ratio_dar'] = p_delta / (p_alpha + epsilon)


    # --- 2. 채널 간 비대칭성 계산 (B-4의 일부) ---
    if len(ch_names) == 2:
        ch1_name = ch_names[0] # Left (Fp1)
        ch2_name = ch_names[1] # Right (Fp2)
        
        # (🔥 수정됨) 'np.log' 대신 'safe_log' 사용
        alpha_L = band_powers_per_channel[ch1_name].get('Alpha', 0)
        alpha_R = band_powers_per_channel[ch2_name].get('Alpha', 0)
        kpi_row['B_asym_alpha_ln_R-L'] = safe_log(alpha_R) - safe_log(alpha_L)
        
        beta_L = band_powers_per_channel[ch1_name].get('Beta', 0)
        beta_R = band_powers_per_channel[ch2_name].get('Beta', 0)
        kpi_row['B_asym_beta_ln_R-L'] = safe_log(beta_R) - safe_log(beta_L)
    
    # (kpi_row 딕셔너리가 수정되었으므로, 별도 반환값 없음)