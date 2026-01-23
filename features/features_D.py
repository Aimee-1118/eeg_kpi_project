"""
features_D.py
=============
Cross-Channel Connectivity & Asymmetry Feature Extraction

주요 기능:
- Connectivity (Coherence): 5 bands (Delta, Theta, Alpha, Beta, Gamma)
- Correlation: Pearson correlation between channels
- Asymmetry (Power): ln(Ch2_Power) - ln(Ch1_Power) for 5 bands

총 11개 Cross-Channel KPI 추출
"""

import numpy as np
from scipy import signal
from scipy.integrate import simpson
from typing import Dict


def compute_cross_features(data_ch1: np.ndarray, data_ch2: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Cross-Channel 특징 추출 (Connectivity & Asymmetry).

    Parameters
    ----------
    data_ch1 : np.ndarray
        1D array, Channel 1 신호.
    data_ch2 : np.ndarray
        1D array, Channel 2 신호.
    sr : int
        샘플링 레이트 (Hz).

    Returns
    -------
    dict
        Cross-Channel KPI 딕셔너리 (11개).
    """
    features = {}
    epsilon = 1e-10

    # 주파수 대역 정의
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 13.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 50.0),
    }

    try:
        # ===== 1. Connectivity: Coherence (5개 대역) =====
        # scipy.signal.coherence 계산
        nperseg = int(sr * 2)  # 2초 윈도우
        freqs, coh = signal.coherence(data_ch1, data_ch2, fs=sr, nperseg=nperseg)

        for band_name, (fmin, fmax) in bands.items():
            try:
                # 해당 대역의 주파수 인덱스
                band_idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                if np.sum(band_idx) > 0:
                    # 해당 대역 Coherence의 평균
                    features[f'coh_{band_name}'] = np.mean(coh[band_idx])
                else:
                    features[f'coh_{band_name}'] = np.nan
            except Exception:
                features[f'coh_{band_name}'] = np.nan

        # ===== 2. Correlation: Pearson Correlation =====
        try:
            corr_matrix = np.corrcoef(data_ch1, data_ch2)
            features['pearson_corr'] = corr_matrix[0, 1]
        except Exception:
            features['pearson_corr'] = np.nan

        # ===== 3. Asymmetry: Power Asymmetry (5개 대역) =====
        # 각 채널에 대해 Welch PSD 계산
        freqs_psd, psd_ch1 = signal.welch(data_ch1, fs=sr, nperseg=nperseg)
        _, psd_ch2 = signal.welch(data_ch2, fs=sr, nperseg=nperseg)

        for band_name, (fmin, fmax) in bands.items():
            try:
                # 해당 대역의 주파수 인덱스
                band_idx = np.logical_and(freqs_psd >= fmin, freqs_psd <= fmax)
                if np.sum(band_idx) > 0:
                    # 절대 파워 (적분)
                    power_ch1 = simpson(psd_ch1[band_idx], x=freqs_psd[band_idx])
                    power_ch2 = simpson(psd_ch2[band_idx], x=freqs_psd[band_idx])

                    # Power Asymmetry: ln(Ch2) - ln(Ch1)
                    # 0 또는 음수 파워 방지
                    if power_ch1 > 0 and power_ch2 > 0:
                        features[f'asym_power_{band_name}'] = np.log(power_ch2) - np.log(power_ch1)
                    else:
                        features[f'asym_power_{band_name}'] = np.nan
                else:
                    features[f'asym_power_{band_name}'] = np.nan
            except Exception:
                features[f'asym_power_{band_name}'] = np.nan

    except Exception as e:
        # 전체 실패 시 모든 값 NaN
        for band_name in bands.keys():
            features[f'coh_{band_name}'] = np.nan
            features[f'asym_power_{band_name}'] = np.nan
        features['pearson_corr'] = np.nan

    return features
