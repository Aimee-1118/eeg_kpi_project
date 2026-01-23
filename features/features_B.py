"""
features_B.py
=============
Frequency-Domain Feature Extraction (Full Feature Set)

주요 기능:
- Power Features (12개: Total, Abs×5, Rel×5)
- Spectral Shape (5개: Peak, Centroid, SEF90, Entropy, Flatness)
- FOOOF Features (2개: Aperiodic Exp, Offset)
- Ratios (2개: Alpha/Beta, Theta/Beta)

총 21개 Frequency-Domain KPI 추출
"""

import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson
from typing import Dict

# FOOOF import (실패해도 괜찮음)
try:
    from fooof import FOOOF
    FOOOF_AVAILABLE = True
except ImportError:
    FOOOF_AVAILABLE = False


def compute_freq_features(data: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Frequency-Domain 특징 추출.

    Parameters
    ----------
    data : np.ndarray
        1D array, EEG 신호 (단일 채널).
    sr : int
        샘플링 레이트 (Hz).

    Returns
    -------
    dict
        Frequency-Domain KPI 딕셔너리.
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
        # ===== PSD 계산 (Welch's Method) =====
        nperseg = min(int(sr * 2), len(data))  # 2초 window
        freqs, psd = welch(data, fs=sr, nperseg=nperseg)

        # ===== 1. Power Features (12개) =====
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            if np.sum(idx) > 0:
                # Absolute Power (적분)
                band_powers[band_name] = simpson(psd[idx], x=freqs[idx])
            else:
                band_powers[band_name] = 0.0

        # Total Power
        total_power = sum(band_powers.values())
        features['pow_total'] = total_power

        # Absolute Powers
        for band_name, power in band_powers.items():
            features[f'pow_abs_{band_name}'] = power

        # Relative Powers (%)
        for band_name, power in band_powers.items():
            features[f'pow_rel_{band_name}'] = (power / (total_power + epsilon)) * 100.0

        # ===== 2. Spectral Shape (5개) =====
        # Peak Frequency (전체 스펙트럼)
        peak_idx = np.argmax(psd)
        features['peak_freq_hz'] = freqs[peak_idx]

        # Spectral Centroid
        features['centroid_hz'] = np.sum(freqs * psd) / (np.sum(psd) + epsilon)

        # SEF90 (Spectral Edge Frequency - 90%)
        cumsum = np.cumsum(psd)
        total = cumsum[-1]
        sef90_idx = np.searchsorted(cumsum, 0.9 * total)
        features['sef90_hz'] = freqs[min(sef90_idx, len(freqs) - 1)]

        # Spectral Entropy
        psd_norm = psd / (np.sum(psd) + epsilon)
        features['spec_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + epsilon))

        # Spectral Flatness
        geometric_mean = np.exp(np.mean(np.log(psd + epsilon)))
        arithmetic_mean = np.mean(psd)
        features['spec_flatness'] = geometric_mean / (arithmetic_mean + epsilon)

        # ===== 3. FOOOF Features (2개) =====
        if FOOOF_AVAILABLE:
            try:
                fm = FOOOF(
                    peak_width_limits=[0.5, 12.0],
                    max_n_peaks=8,
                    min_peak_height=0.0,
                    aperiodic_mode='fixed',
                    verbose=False
                )
                fm.fit(freqs, psd, freq_range=[0.5, 50.0])
                ap_params = fm.aperiodic_params_
                features['aperiodic_exponent'] = ap_params[1]  # Exponent
                features['aperiodic_offset'] = ap_params[0]    # Offset
            except Exception:
                features['aperiodic_exponent'] = np.nan
                features['aperiodic_offset'] = np.nan
        else:
            features['aperiodic_exponent'] = np.nan
            features['aperiodic_offset'] = np.nan

        # ===== 4. Ratios (2개) =====
        alpha_power = band_powers.get('alpha', 0.0)
        beta_power = band_powers.get('beta', 0.0)
        theta_power = band_powers.get('theta', 0.0)

        features['alpha_beta_ratio'] = alpha_power / (beta_power + epsilon)
        features['theta_beta_ratio'] = theta_power / (beta_power + epsilon)

    except Exception as e:
        # 전체 실패 시 모든 값 NaN
        for key in [
            'pow_total',
            'pow_abs_delta', 'pow_abs_theta', 'pow_abs_alpha', 'pow_abs_beta', 'pow_abs_gamma',
            'pow_rel_delta', 'pow_rel_theta', 'pow_rel_alpha', 'pow_rel_beta', 'pow_rel_gamma',
            'peak_freq_hz', 'centroid_hz', 'sef90_hz', 'spec_entropy', 'spec_flatness',
            'aperiodic_exponent', 'aperiodic_offset',
            'alpha_beta_ratio', 'theta_beta_ratio'
        ]:
            features[key] = np.nan

    return features
