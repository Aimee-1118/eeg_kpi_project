"""
features_C.py
=============
Nonlinear/Dynamics Feature Extraction (Full Feature Set)

주요 기능:
- Entropy Features (4개): Sample, Spectral, Permutation, SVD
- Complexity Features (3개): Higuchi FD, Petrosian FD, Katz FD
- Dynamics Features (2개): Lempel-Ziv Complexity, Detrended Fluctuation Analysis

총 9개 Nonlinear KPI 추출
"""

import numpy as np
import antropy as ant
from typing import Dict


def compute_nonlinear_features(data: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Nonlinear/Dynamics 특징 추출.

    Parameters
    ----------
    data : np.ndarray
        1D array, EEG 신호 (단일 채널).
    sr : int
        샘플링 레이트 (Hz).

    Returns
    -------
    dict
        Nonlinear KPI 딕셔너리 (9개).
    """
    features = {}

    # ===== 1. Entropy Features (4개) =====

    # Sample Entropy
    try:
        features['sampen'] = ant.sample_entropy(data, order=2)
    except Exception:
        features['sampen'] = np.nan

    # Spectral Entropy (normalized)
    try:
        features['spec_ent'] = ant.spectral_entropy(
            data, sf=sr, method='welch', normalize=True
        )
    except Exception:
        features['spec_ent'] = np.nan

    # Permutation Entropy (normalized)
    try:
        features['perm_ent'] = ant.perm_entropy(data, order=3, normalize=True)
    except Exception:
        features['perm_ent'] = np.nan

    # SVD Entropy (normalized)
    try:
        features['svd_ent'] = ant.svd_entropy(data, order=3, normalize=True)
    except Exception:
        features['svd_ent'] = np.nan

    # ===== 2. Complexity Features (3개) =====

    # Higuchi Fractal Dimension
    try:
        features['higuchi_fd'] = ant.higuchi_fd(data, kmax=10)
    except Exception:
        features['higuchi_fd'] = np.nan

    # Petrosian Fractal Dimension
    try:
        features['petrosian_fd'] = ant.petrosian_fd(data)
    except Exception:
        features['petrosian_fd'] = np.nan

    # Katz Fractal Dimension
    try:
        features['katz_fd'] = ant.katz_fd(data)
    except Exception:
        features['katz_fd'] = np.nan

    # ===== 3. Dynamics Features (2개) =====

    # Lempel-Ziv Complexity (이진화 필수)
    try:
        # Step 1: 이진화 (threshold = mean)
        binary_data = (data > np.mean(data)).astype(int)
        # Step 2: LZC 계산 (normalized)
        features['lzc'] = ant.lziv_complexity(binary_data, normalize=True)
    except Exception:
        features['lzc'] = np.nan

    # Detrended Fluctuation Analysis
    try:
        features['dfa'] = ant.detrended_fluctuation(data)
    except Exception:
        features['dfa'] = np.nan

    return features