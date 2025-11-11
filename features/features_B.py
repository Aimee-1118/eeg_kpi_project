# 📜 features/features_B.py
# 🧮 [모듈 5-B] 주파수축 KPI 계산 함수들

import numpy as np
import config
from scipy.signal import welch
# antropy는 스펙트럼 엔트로피 등 계산에 사용됩니다.
import antropy as ant

def get_B_features(epoch_data: np.ndarray, cfg: config, kpi_row: dict):
    """
    'B/C' 유형 Epoch 데이터(단일 Epoch)에서 B 카테고리의 모든 KPI를 추출합니다.
    추출된 KPI는 'kpi_row' 딕셔너리에 직접 추가됩니다.

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
    
    # 밴드 파워 계산 결과를 임시 저장 (채널 간 비율/비대칭 계산용)
    band_powers_per_channel = {ch_name: {} for ch_name in ch_names}

    # --- 1. 각 채널(Fp1, Fp2)을 순회하며 PSD 기반 KPI 계산 ---
    for i, ch_name in enumerate(ch_names):
        x = epoch_data[i, :]  # (n_samples,) 1D 배열
        
        # --- 1a. PSD 계산 (Welch's Method) ---
        # 2초(sfreq*2) 윈도우를 사용하여 0.5Hz의 주파수 해상도를 확보합니다.
        nperseg = int(sfreq * 2)
        if len(x) < nperseg:
            nperseg = len(x) # Epoch가 2초보다 짧은 경우
            
        freqs, psd = welch(x, sfreq=sfreq, nperseg=nperseg, nfft=nperseg)
        freq_res = freqs[1] - freqs[0] # 주파수 해상도 (적분 시 사용)

        # --- B-1. 스펙트럼 파워/크기 특징 ---
        abs_powers = {}
        for band_name, (f_low, f_high) in bands.items():
            band_mask = (freqs >= f_low) & (freqs < f_high)
            if np.sum(band_mask) == 0:
                abs_powers[band_name] = 0.0
            else:
                # np.trapz: 주파수 해상도(freq_res)를 고려한 면적(적분) 계산
                abs_powers[band_name] = np.trapz(psd[band_mask], dx=freq_res)
        
        total_power = np.sum(list(abs_powers.values()))

        # KPI 딕셔너리에 저장
        kpi_row[f'{ch_name}_B_pow_total'] = total_power
        for band_name, abs_p in abs_powers.items():
            rel_p = (abs_p / (total_power + epsilon)) * 100.0
            kpi_row[f'{ch_name}_B_pow_abs_{band_name.lower()}'] = abs_p
            kpi_row[f'{ch_name}_B_pow_rel_{band_name.lower()}'] = rel_p
            # 비율/비대칭 계산을 위해 임시 저장
            band_powers_per_channel[ch_name][band_name] = abs_p

        # --- B-2. 스펙트럼 주파수/위치 특징 ---
        # 🏔️ 피크 주파수 (Alpha)
        alpha_mask = (freqs >= bands['Alpha'][0]) & (freqs < bands['Alpha'][1])
        if np.sum(alpha_mask) > 0:
            kpi_row[f'{ch_name}_B_loc_peak_alpha_hz'] = freqs[alpha_mask][np.argmax(psd[alpha_mask])]
        
        # 🔪 스펙트럼 엣지 주파수 (SEF90)
        psd_cumsum = np.cumsum(psd) * freq_res
        total_power_psd = psd_cumsum[-1]
        try:
            sef90_idx = np.searchsorted(psd_cumsum, 0.90 * total_power_psd)
            kpi_row[f'{ch_name}_B_loc_sef90_hz'] = freqs[sef90_idx]
        except IndexError:
            kpi_row[f'{ch_name}_B_loc_sef90_hz'] = np.nan
        
        # 🧭 스펙트럼 중심 (Spectral Centroid)
        kpi_row[f'{ch_name}_B_loc_centroid_hz'] = np.sum(freqs * psd) / (np.sum(psd) + epsilon)

        # --- B-3. 스펙트럼 형태/분포 특징 ---
        # 📉 스펙트럼 엔트로피 (antropy 사용)
        kpi_row[f'{ch_name}_B_shape_spec_ent'] = ant.spectral_entropy(psd, sfreq=sfreq, method='welch', normalize=True)
        
        # 📉 1/f 지수 (Aperiodic Exponent / Slope) - (FOOOF 대신 간단한 Polyfit 사용)
        # (주의: 이 방식은 FOOOF 라이브러리보다 덜 정교한 추정치입니다.)
        log_freqs = np.log10(freqs[1:]) # f=0 제외
        log_psd = np.log10(psd[1:] + epsilon)
        # 1Hz ~ 30Hz 범위에서만 기울기 계산 (저/고주파 아티팩트 회피)
        fit_mask = (freqs[1:] >= 1) & (freqs[1:] <= 30)
        if np.sum(fit_mask) > 1:
            slope, _ = np.polyfit(log_freqs[fit_mask], log_psd[fit_mask], 1)
            kpi_row[f'{ch_name}_B_shape_1f_slope'] = -slope # Exponent(χ)는 보통 양수로 표현
        else:
            kpi_row[f'{ch_name}_B_shape_1f_slope'] = np.nan
            
        # --- B-4. 밴드 간 비율 특징 (채널 내부) ---
        p_theta = band_powers_per_channel[ch_name].get('Theta', 0)
        p_alpha = band_powers_per_channel[ch_name].get('Alpha', 0)
        p_beta = band_powers_per_channel[ch_name].get('Beta', 0)
        p_delta = band_powers_per_channel[ch_name].get('Delta', 0)

        # 🧠 세타/베타 비율 (TBR)
        kpi_row[f'{ch_name}_B_ratio_tbr'] = p_theta / (p_beta + epsilon)
        # 🚀 몰입 지수 (Engagement Index)
        kpi_row[f'{ch_name}_B_ratio_engagement'] = p_beta / (p_alpha + p_theta + epsilon)
        # 🛌 델타/알파 비율 (DAR)
        kpi_row[f'{ch_name}_B_ratio_dar'] = p_delta / (p_alpha + epsilon)


    # --- 2. 채널 간 비대칭성 계산 (B-4의 일부) ---
    # (Fp1 = Left, Fp2 = Right 라고 가정)
    if len(ch_names) == 2:
        ch1_name = ch_names[0] # Left (Fp1)
        ch2_name = ch_names[1] # Right (Fp2)
        
        # Alpha Asymmetry: ln(Right) - ln(Left)
        alpha_L = band_powers_per_channel[ch1_name].get('Alpha', 0)
        alpha_R = band_powers_per_channel[ch2_name].get('Alpha', 0)
        kpi_row['B_asym_alpha_ln_R-L'] = np.log(alpha_R + epsilon) - np.log(alpha_L + epsilon)
        
        # Beta Asymmetry
        beta_L = band_powers_per_channel[ch1_name].get('Beta', 0)
        beta_R = band_powers_per_channel[ch2_name].get('Beta', 0)
        kpi_row['B_asym_beta_ln_R-L'] = np.log(beta_R + epsilon) - np.log(beta_L + epsilon)
    
    # (kpi_row 딕셔너리가 수정되었으므로, 별도 반환값 없음)