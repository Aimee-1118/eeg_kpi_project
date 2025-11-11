# 📜 features/utils.py
# 🛠️ [유틸리티] 여러 특징 추출 모듈에서 공통으로 사용하는 도우미 함수들

import numpy as np

def safe_z_score(data: np.ndarray) -> np.ndarray:
    """
    데이터를 Z-score로 표준화합니다.
    데이터가 상수(표준편차가 0)일 경우, 0으로 채워진 배열을 반환하여
    NaN/무한대 오류를 방지합니다.

    Args:
        data (np.ndarray): 1D 배열

    Returns:
        np.ndarray: Z-score로 표준화된 배열
    """
    std_val = np.std(data)
    if std_val < 1e-10:  # 표준편차가 0에 가까우면 (상수 데이터)
        return np.zeros_like(data)  # 0으로 채운 배열 반환
    
    return (data - np.mean(data)) / std_val

def safe_log(data: np.ndarray) -> np.ndarray:
    """
    0 또는 음수 값에 대한 로그 오류를 방지하는 '안전한' 로그 함수입니다.
    매우 작은 값(1e-10)을 더한 후 로그를 계산합니다.

    Args:
        data (np.ndarray): 1D 또는 스칼라 값

    Returns:
        np.ndarray: 로그가 적용된 배열
    """
    epsilon = 1e-10
    # 데이터가 0보다 작은 경우를 대비해 절대값을 취하고 epsilon을 더함
    # (PSD 등 음수가 없는 데이터는 np.log(data + epsilon)만으로도 충분)
    return np.log(np.abs(data) + epsilon)

def get_band_mask(freqs: np.ndarray, f_low: float, f_high: float) -> np.ndarray:
    """
    주파수 배열(freqs)에서 특정 대역(f_low ~ f_high)에 해당하는
    불리언 마스크(Boolean mask)를 생성합니다.

    Args:
        freqs (np.ndarray): 전체 주파수 축 배열
        f_low (float): 밴드의 시작 주파수
        f_high (float): 밴드의 끝 주파수

    Returns:
        np.ndarray: True/False 값으로 채워진 불리언 마스크
    """
    return (freqs >= f_low) & (freqs < f_high)

# (필요에 따라 향후 공통으로 사용될 다른 함수들을 이곳에 추가할 수 있습니다.)
# 예: def custom_filter(data, sfreq, f_low, f_high)...