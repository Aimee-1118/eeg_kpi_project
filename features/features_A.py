"""
features_A.py
=============
Time-Domain Feature Extraction (Full Feature Set)

주요 기능:
- Amplitude Features (5개)
- Statistical Features (6개)
- Pattern Features (4개)
- Hjorth Parameters (2개)

총 17개 Time-Domain KPI 추출
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from typing import Dict


def compute_time_features(data: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Time-Domain 특징 추출.

    Parameters
    ----------
    data : np.ndarray
        1D array, EEG 신호 (단일 채널).
    sr : int
        샘플링 레이트 (Hz).

    Returns
    -------
    dict
        Time-Domain KPI 딕셔너리.
    """
    features = {}
    epsilon = 1e-10  # Division by zero 방지

    try:
        # ===== 1. Amplitude Features (5개) =====
        features['amp_max'] = np.max(data)
        features['amp_min'] = np.min(data)
        features['amp_p2p'] = np.ptp(data)  # Peak-to-Peak
        features['amp_mean'] = np.mean(data)
        features['amp_rms'] = np.sqrt(np.mean(np.square(data)))

        # ===== 2. Statistical Features (6개) =====
        features['stat_mean'] = np.mean(data)
        features['stat_std'] = np.std(data)
        features['stat_variance'] = np.var(data)
        features['stat_median'] = np.median(data)
        features['stat_skewness'] = stats.skew(data)
        features['stat_kurtosis'] = stats.kurtosis(data)

        # ===== 3. Pattern Features (4개) =====
        # Zero Crossing Rate
        zero_crossings = np.where(np.diff(np.sign(data)))[0]
        features['zcr'] = len(zero_crossings) / len(data)

        # Slope Mean (평균 기울기)
        dx = np.diff(data)
        features['slope_mean'] = np.mean(np.abs(dx))

        # Peak Detection
        peaks, properties = find_peaks(data, height=0)
        features['peak_count'] = len(peaks)
        if len(peaks) > 0:
            features['peak_mean_height'] = np.mean(properties['peak_heights'])
        else:
            features['peak_mean_height'] = 0.0

        # ===== 4. Hjorth Parameters (2개) =====
        # Hjorth Mobility = sqrt(var(dx) / var(x))
        var_x = np.var(data)
        var_dx = np.var(dx)
        features['hjorth_mobility'] = np.sqrt(var_dx / (var_x + epsilon))

        # Hjorth Complexity = Mobility(dx) / Mobility(x)
        ddx = np.diff(dx)
        var_ddx = np.var(ddx)
        mobility_dx = np.sqrt(var_ddx / (var_dx + epsilon))
        features['hjorth_complexity'] = mobility_dx / (features['hjorth_mobility'] + epsilon)

    except Exception as e:
        # 전체 실패 시 모든 값 NaN
        for key in [
            'amp_max', 'amp_min', 'amp_p2p', 'amp_mean', 'amp_rms',
            'stat_mean', 'stat_std', 'stat_variance', 'stat_median', 'stat_skewness', 'stat_kurtosis',
            'zcr', 'slope_mean', 'peak_count', 'peak_mean_height',
            'hjorth_mobility', 'hjorth_complexity'
        ]:
            features[key] = np.nan

    return features
