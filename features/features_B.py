# 📜 features/features_B.py
# 🧮 [모듈 5-B] 주파수축 KPI 계산 함수들
# (🔥 수동 Spectral Entropy 계산으로 변경하여 오류 해결 및 최적화)

import numpy as np
from scipy.signal import welch
import antropy as ant
from fooof import FOOOF
from .utils import safe_log
from omegaconf import DictConfig

def get_B_features(epoch_data: np.ndarray, cfg: DictConfig, kpi_row: dict):
    """
    'B/C' 유형 Epoch 데이터(단일 Epoch)에서 B 카테고리의 모든 KPI를 추출합니다.
    """
    
    sfreq = cfg.SAMPLE_RATE
    ch_names = cfg.CHANNELS
    bands = cfg.BANDS
    
    epsilon = 1e-10 
    
    band_powers_per_channel = {ch_name: {} for ch_name in ch_names}
    
    # FOOOF 모델 초기화
    fm = FOOOF(peak_width_limits=[0.5, 12.0], 
               max_n_peaks=8, 
               min_peak_height=0.0,
               peak_threshold=2.0,
               aperiodic_mode='fixed', 
               verbose=False)

    # --- 1. 각 채널(Fp1, Fp2)을 순회하며 PSD 기반 KPI 계산 ---
    for i, ch_name in enumerate(ch_names):
        x = epoch_data[i, :]
        
        # --- 1a. PSD 계산 (Welch's Method) ---
        nperseg = int(sfreq * cfg.WELCH_WINDOW_SEC)
        if len(x) < nperseg:
            nperseg = len(x)
            
        # (🔥 수정됨) fs=sfreq 인자명 정확히 사용
        freqs, psd = welch(x, fs=sfreq, nperseg=nperseg, nfft=nperseg)
        freq_res = freqs[1] - freqs[0]

        # --- B-1. 스펙트럼 파워/크기 특징 ---
        abs_powers = {}
        for band_name, (f_low, f_high) in bands.items():
            band_mask = (freqs >= f_low) & (freqs < f_high)
            if np.sum(band_mask) == 0:
                abs_powers[band_name] = 0.0
            else:
                abs_powers[band_name] = np.trapz(psd[band_mask], dx=freq_res)
        
        total_power_bands = np.sum(list(abs_powers.values()))

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
        else:
            kpi_row[f'{ch_name}_B_loc_peak_alpha_hz'] = np.nan
        
        psd_cumsum = np.cumsum(psd) * freq_res
        total_power_psd = psd_cumsum[-1]
        try:
            sef90_idx = np.searchsorted(psd_cumsum, 0.90 * total_power_psd)
            kpi_row[f'{ch_name}_B_loc_sef90_hz'] = freqs[sef90_idx]
        except IndexError:
            kpi_row[f'{ch_name}_B_loc_sef90_hz'] = np.nan
        
        kpi_row[f'{ch_name}_B_loc_centroid_hz'] = np.sum(freqs * psd) / (np.sum(psd) + epsilon)

        # --- B-3. 스펙트럼 형태/분포 특징 ---
        # (🔥 수정됨) ant.spectral_entropy 함수 대신 수동 계산 (속도 최적화 & 오류 방지)
        # Spectral Entropy = Shannon entropy of normalized PSD
        try:
            psd_norm = psd / (np.sum(psd) + epsilon)
            # log2(0) 방지 위해 epsilon 추가 또는 조건부 처리 필요하나, psd는 보통 양수임
            se = -np.sum(psd_norm * np.log2(psd_norm + epsilon))
            # Normalize (0~1 사이 값으로)
            se /= np.log2(len(psd_norm))
            kpi_row[f'{ch_name}_B_shape_spec_ent'] = se
        except Exception:
            kpi_row[f'{ch_name}_B_shape_spec_ent'] = 0.0
        
        # 1/f 지수 (FOOOF)
        try:
            fit_range = cfg.APERIODIC_FIT_RANGE_HZ 
            fm.add_data(freqs, psd, freq_range=fit_range)
            fm.fit()
            
            ap_params = fm.get_params('aperiodic_params')
            kpi_row[f'{ch_name}_B_shape_1f_exponent'] = ap_params[1]
            kpi_row[f'{ch_name}_B_shape_1f_offset'] = ap_params[0]
        except Exception as e:
            kpi_row[f'{ch_name}_B_shape_1f_exponent'] = np.nan
            kpi_row[f'{ch_name}_B_shape_1f_offset'] = np.nan
            
        # --- B-4. 밴드 간 비율 특징 ---
        p_theta = band_powers_per_channel[ch_name].get('Theta', 0)
        p_alpha = band_powers_per_channel[ch_name].get('Alpha', 0)
        p_beta = band_powers_per_channel[ch_name].get('Beta', 0)
        p_delta = band_powers_per_channel[ch_name].get('Delta', 0)

        kpi_row[f'{ch_name}_B_ratio_tbr'] = p_theta / (p_beta + epsilon)
        kpi_row[f'{ch_name}_B_ratio_engagement'] = p_beta / (p_alpha + p_theta + epsilon)
        kpi_row[f'{ch_name}_B_ratio_dar'] = p_delta / (p_alpha + epsilon)


    # --- 2. 채널 간 비대칭성 계산 ---
    if len(ch_names) == 2:
        ch1_name = ch_names[0] 
        ch2_name = ch_names[1] 
        
        alpha_L = band_powers_per_channel[ch1_name].get('Alpha', 0)
        alpha_R = band_powers_per_channel[ch2_name].get('Alpha', 0)
        kpi_row['B_asym_alpha_ln_R-L'] = safe_log(alpha_R) - safe_log(alpha_L)
        
        beta_L = band_powers_per_channel[ch1_name].get('Beta', 0)
        beta_R = band_powers_per_channel[ch2_name].get('Beta', 0)
        kpi_row['B_asym_beta_ln_R-L'] = safe_log(beta_R) - safe_log(beta_L)