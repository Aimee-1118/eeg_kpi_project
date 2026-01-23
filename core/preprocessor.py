"""
preprocessor.py
===============
MNE Raw 객체에 노이즈 제거 필터를 적용하는 전처리 모듈.

주요 기능:
- Notch Filter: 전원 잡음 제거 (60Hz)
- Bandpass Filter: 관심 주파수 대역 추출 (0.5 ~ 50.0Hz)

적용 순서 (Q11 반영):
1. Notch Filter
2. Bandpass Filter
"""

import logging
from typing import Optional

import mne
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def preprocess_raw(raw: mne.io.RawArray, cfg: DictConfig) -> Optional[mne.io.RawArray]:
    """
    Raw 객체에 Notch + Bandpass 필터 적용.

    Parameters
    ----------
    raw : mne.io.RawArray
        전처리할 MNE Raw 객체.
    cfg : DictConfig
        분석 설정 (PREPROCESSING.notch_freq, filter_band 참조).

    Returns
    -------
    mne.io.RawArray or None
        필터링된 Raw 객체 (in-place 수정됨), 실패 시 None.
    """
    try:
        # 1. Notch Filter (전원 잡음 제거)
        notch_freq = cfg.PREPROCESSING.notch_freq
        logger.info(f"Notch Filter 적용: {notch_freq}Hz")
        raw.notch_filter(
            freqs=notch_freq,
            verbose=False,
        )

        # 2. Bandpass Filter (관심 주파수 대역 추출)
        low_freq = cfg.PREPROCESSING.filter_band.low
        high_freq = cfg.PREPROCESSING.filter_band.high
        logger.info(f"Bandpass Filter 적용: {low_freq} ~ {high_freq}Hz")
        raw.filter(
            l_freq=low_freq,
            h_freq=high_freq,
            verbose=False,
        )

        logger.info("전처리 완료")
        return raw

    except Exception as e:
        logger.error(f"전처리 중 오류 발생: {e}", exc_info=True)
        return None
