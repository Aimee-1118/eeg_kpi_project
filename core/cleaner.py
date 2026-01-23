"""
cleaner.py
==========
Epochs 객체에서 진폭 기준을 초과하는 Bad Epochs를 제거하는 모듈.

주요 기능:
- 진폭 임계값 기반 Artifact Rejection
- Q16 규칙: 3개 미만 남으면 분석 불가 판정
"""

import logging
from typing import Optional

import mne
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def clean_epochs(
    epochs: mne.Epochs, cfg: DictConfig
) -> Optional[mne.Epochs]:
    """
    진폭 기준을 초과하는 Bad Epochs 제거.

    Parameters
    ----------
    epochs : mne.Epochs
        정제할 Epochs 객체.
    cfg : DictConfig
        분석 설정 (PREPROCESSING.artifact_threshold_uv 참조).

    Returns
    -------
    mne.Epochs or None
        정제된 Epochs 객체 (3개 이상), 실패 시 None.
    """
    try:
        # 진폭 임계값 (uV → V 변환)
        threshold_uv = cfg.PREPROCESSING.artifact_threshold_uv
        threshold_v = threshold_uv * 1e-6

        total_epochs = len(epochs)
        logger.info(f"Artifact Rejection 시작: 임계값={threshold_uv}µV")

        # Bad Epochs 제거
        # reject: 채널 타입별 최대 허용 진폭 (peak-to-peak)
        epochs.drop_bad(reject=dict(eeg=threshold_v), verbose=False)

        clean_count = len(epochs)
        dropped_count = total_epochs - clean_count
        drop_rate = (dropped_count / total_epochs * 100) if total_epochs > 0 else 0

        logger.info(
            f"총 {total_epochs}개 Epoch 중 {dropped_count}개 제거됨 "
            f"(Drop Rate: {drop_rate:.1f}%)"
        )

        # Q16 규칙: 3개 미만이면 분석 불가
        if clean_count < 3:
            logger.warning(
                f"유효한 Epoch 수 부족 ({clean_count}개 < 3개). "
                "분석을 수행할 수 없습니다."
            )
            return None

        logger.info(f"정제 완료: {clean_count}개 Clean Epochs")
        return epochs

    except Exception as e:
        logger.error(f"Artifact Rejection 중 오류 발생: {e}", exc_info=True)
        return None
