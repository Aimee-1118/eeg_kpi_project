"""
epocher.py
==========
전처리된 Continuous Raw 데이터를 고정 길이 Epochs로 분할하는 모듈.

주요 기능:
- 고정 길이 Epoch 생성 (window_sec, overlap_sec 설정 적용)
- MNE make_fixed_length_epochs 활용
"""

import logging
from typing import Optional

import mne
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def create_epochs(
    raw: mne.io.RawArray, cfg: DictConfig
) -> Optional[mne.Epochs]:
    """
    Raw 객체를 고정 길이 Epochs로 분할.

    Parameters
    ----------
    raw : mne.io.RawArray
        전처리된 MNE Raw 객체.
    cfg : DictConfig
        분석 설정 (EPOCH.window_sec, overlap_sec 참조).

    Returns
    -------
    mne.Epochs or None
        Epochs 객체, 실패 시 None.
    """
    try:
        window_sec = cfg.EPOCH.window_sec
        overlap_sec = cfg.EPOCH.overlap_sec

        logger.info(
            f"Epoch 생성 시작: window={window_sec}s, overlap={overlap_sec}s"
        )

        # 고정 길이 Epochs 생성
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=window_sec,
            overlap=overlap_sec,
            preload=True,  # 필수: 메모리에 로드
            verbose=False,
        )

        logger.info(f"Epoch 생성 완료: 총 {len(epochs)}개 Epoch")
        return epochs

    except Exception as e:
        logger.error(f"Epoch 생성 중 오류 발생: {e}", exc_info=True)
        return None
