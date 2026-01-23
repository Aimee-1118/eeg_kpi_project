"""
feature_extractor.py
====================
Clean Epochs로부터 모든 KPI(Band Power, Asymmetry, Connectivity 등)를 추출하는 모듈.

주요 기능:
- Band Powers: Delta, Theta, Alpha, Beta, Gamma (Welch's method)
- Basic Stats: Mean, Std, Skewness, Kurtosis
- Cross-Channel: Asymmetry, Coherence
- Optional: TBR, Engagement 등
- 에러 처리: 개별 KPI 실패 시 np.nan
- 반환: Epoch별 평균낸 최종 딕셔너리
"""

import logging
from typing import Dict, Optional

import mne
import numpy as np
from omegaconf import DictConfig
from scipy import signal, stats

logger = logging.getLogger(__name__)


def extract_features(
    epochs: mne.Epochs, cfg: DictConfig
) -> Optional[Dict[str, float]]:
    """
    Clean Epochs로부터 모든 KPI를 추출하여 평균값 딕셔너리 반환.

    Parameters
    ----------
    epochs : mne.Epochs
        정제된 Epochs 객체.
    cfg : DictConfig
        분석 설정 (BANDS, PREPROCESSING.sampling_rate 등 참조).

    Returns
    -------
    dict or None
        KPI 딕셔너리 (평균값), 실패 시 None.
    """
    try:
        logger.info(f"Feature Extraction 시작: {len(epochs)}개 Epoch")

        # Epoch별 KPI 수집
        epoch_features = []
        for epoch_idx in range(len(epochs)):
            epoch_data = epochs[epoch_idx].get_data()[0]  # (n_channels, n_times)
            features = _extract_epoch_features(epoch_data, cfg)
            epoch_features.append(features)

        # Epoch별 평균 계산
        avg_features = _average_features(epoch_features)

        logger.info(f"Feature Extraction 완료: {len(avg_features)}개 KPI")
        return avg_features

    except Exception as e:
        logger.error(f"Feature Extraction 중 오류 발생: {e}", exc_info=True)
        return None


def _extract_epoch_features(
    epoch_data: np.ndarray, cfg: DictConfig
) -> Dict[str, float]:
    """
    단일 Epoch에서 모든 KPI 추출.

    Parameters
    ----------
    epoch_data : np.ndarray
        (n_channels, n_times) 형태의 Epoch 데이터.
    cfg : DictConfig
        분석 설정.

    Returns
    -------
    dict
        KPI 딕셔너리.
    """
    features = {}
    sfreq = cfg.PREPROCESSING.sampling_rate
    ch1_data = epoch_data[0, :]
    ch2_data = epoch_data[1, :]

    # 1. Band Powers (Welch's method)
    band_powers_ch1 = _compute_band_powers(ch1_data, sfreq, cfg.BANDS)
    band_powers_ch2 = _compute_band_powers(ch2_data, sfreq, cfg.BANDS)

    for band_name, power in band_powers_ch1.items():
        features[f"Ch1_Band_{band_name}"] = power
    for band_name, power in band_powers_ch2.items():
        features[f"Ch2_Band_{band_name}"] = power

    # 2. Basic Stats
    stats_ch1 = _compute_basic_stats(ch1_data)
    stats_ch2 = _compute_basic_stats(ch2_data)

    for stat_name, value in stats_ch1.items():
        features[f"Ch1_Stat_{stat_name}"] = value
    for stat_name, value in stats_ch2.items():
        features[f"Ch2_Stat_{stat_name}"] = value

    # 3. Cross-Channel Features
    # Asymmetry (Alpha)
    alpha_ch1 = band_powers_ch1.get("Alpha", np.nan)
    alpha_ch2 = band_powers_ch2.get("Alpha", np.nan)
    features["Asym_Band_Alpha"] = _compute_asymmetry(alpha_ch1, alpha_ch2)

    # Coherence
    coherence_vals = _compute_coherence(ch1_data, ch2_data, sfreq, cfg.BANDS)
    for band_name, coh in coherence_vals.items():
        features[f"Conn_Coh_{band_name}"] = coh

    # 4. Optional Ratios
    theta_ch1 = band_powers_ch1.get("Theta", np.nan)
    beta_ch1 = band_powers_ch1.get("Beta", np.nan)
    theta_ch2 = band_powers_ch2.get("Theta", np.nan)
    beta_ch2 = band_powers_ch2.get("Beta", np.nan)

    features["Ch1_Ratio_TBR"] = _compute_ratio(theta_ch1, beta_ch1)
    features["Ch2_Ratio_TBR"] = _compute_ratio(theta_ch2, beta_ch2)

    # Engagement (Beta / (Alpha + Theta))
    engagement_ch1 = _compute_engagement(
        beta_ch1, alpha_ch1, theta_ch1
    )
    engagement_ch2 = _compute_engagement(
        beta_ch2, alpha_ch2, theta_ch2
    )
    features["Ch1_Ratio_Engagement"] = engagement_ch1
    features["Ch2_Ratio_Engagement"] = engagement_ch2

    return features


def _compute_band_powers(
    data: np.ndarray, sfreq: float, bands: DictConfig
) -> Dict[str, float]:
    """
    Welch's method로 Band Power 계산.

    Parameters
    ----------
    data : np.ndarray
        (n_times,) 형태의 채널 데이터.
    sfreq : float
        샘플링 레이트.
    bands : DictConfig
        주파수 대역 설정.

    Returns
    -------
    dict
        {band_name: power} 딕셔너리.
    """
    powers = {}
    try:
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=min(256, len(data)))

        for band_name, band_range in bands.items():
            low, high = band_range
            idx = np.logical_and(freqs >= low, freqs <= high)
            power = np.trapz(psd[idx], freqs[idx])
            powers[band_name] = power

    except Exception as e:
        logger.warning(f"Band Power 계산 실패: {e}")
        for band_name in bands.keys():
            powers[band_name] = np.nan

    return powers


def _compute_basic_stats(data: np.ndarray) -> Dict[str, float]:
    """
    기본 통계량 계산 (Mean, Std, Skewness, Kurtosis).

    Parameters
    ----------
    data : np.ndarray
        (n_times,) 형태의 채널 데이터.

    Returns
    -------
    dict
        {stat_name: value} 딕셔너리.
    """
    stat_dict = {}
    try:
        stat_dict["Mean"] = np.mean(data)
        stat_dict["Std"] = np.std(data)
        stat_dict["Skewness"] = stats.skew(data)
        stat_dict["Kurtosis"] = stats.kurtosis(data)
    except Exception as e:
        logger.warning(f"Basic Stats 계산 실패: {e}")
        stat_dict["Mean"] = np.nan
        stat_dict["Std"] = np.nan
        stat_dict["Skewness"] = np.nan
        stat_dict["Kurtosis"] = np.nan

    return stat_dict


def _compute_asymmetry(ch1_power: float, ch2_power: float) -> float:
    """
    비대칭도 계산: ln(Ch2) - ln(Ch1).

    Parameters
    ----------
    ch1_power : float
        Ch1의 Band Power.
    ch2_power : float
        Ch2의 Band Power.

    Returns
    -------
    float
        비대칭도 값.
    """
    try:
        if ch1_power > 0 and ch2_power > 0:
            return np.log(ch2_power) - np.log(ch1_power)
        else:
            return np.nan
    except Exception as e:
        logger.warning(f"Asymmetry 계산 실패: {e}")
        return np.nan


def _compute_coherence(
    ch1_data: np.ndarray,
    ch2_data: np.ndarray,
    sfreq: float,
    bands: DictConfig,
) -> Dict[str, float]:
    """
    채널 간 Coherence 계산.

    Parameters
    ----------
    ch1_data : np.ndarray
        Ch1 데이터.
    ch2_data : np.ndarray
        Ch2 데이터.
    sfreq : float
        샘플링 레이트.
    bands : DictConfig
        주파수 대역 설정.

    Returns
    -------
    dict
        {band_name: coherence} 딕셔너리.
    """
    coherence_dict = {}
    try:
        freqs, coh = signal.coherence(ch1_data, ch2_data, fs=sfreq, nperseg=min(256, len(ch1_data)))

        for band_name, band_range in bands.items():
            low, high = band_range
            idx = np.logical_and(freqs >= low, freqs <= high)
            avg_coh = np.mean(coh[idx]) if np.any(idx) else np.nan
            coherence_dict[band_name] = avg_coh

    except Exception as e:
        logger.warning(f"Coherence 계산 실패: {e}")
        for band_name in bands.keys():
            coherence_dict[band_name] = np.nan

    return coherence_dict


def _compute_ratio(numerator: float, denominator: float) -> float:
    """
    비율 계산 (분모가 0이면 np.nan).

    Parameters
    ----------
    numerator : float
        분자.
    denominator : float
        분모.

    Returns
    -------
    float
        비율 값.
    """
    try:
        if denominator > 0:
            return numerator / denominator
        else:
            return np.nan
    except Exception:
        return np.nan


def _compute_engagement(
    beta: float, alpha: float, theta: float
) -> float:
    """
    Engagement 지수 계산: Beta / (Alpha + Theta).

    Parameters
    ----------
    beta : float
        Beta Power.
    alpha : float
        Alpha Power.
    theta : float
        Theta Power.

    Returns
    -------
    float
        Engagement 값.
    """
    try:
        denominator = alpha + theta
        if denominator > 0:
            return beta / denominator
        else:
            return np.nan
    except Exception:
        return np.nan


def _average_features(
    epoch_features: list[Dict[str, float]]
) -> Dict[str, float]:
    """
    Epoch별 KPI를 평균 내어 최종 딕셔너리 생성.

    Parameters
    ----------
    epoch_features : list of dict
        Epoch별 KPI 딕셔너리 리스트.

    Returns
    -------
    dict
        평균 KPI 딕셔너리.
    """
    if not epoch_features:
        return {}

    # 모든 키 수집
    all_keys = set()
    for features in epoch_features:
        all_keys.update(features.keys())

    # 키별 평균 계산
    avg_features = {}
    for key in all_keys:
        values = [f.get(key, np.nan) for f in epoch_features]
        # NaN 제외하고 평균
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            avg_features[key] = np.mean(valid_values)
        else:
            avg_features[key] = np.nan

    return avg_features
