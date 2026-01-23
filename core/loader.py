"""
loader.py
=========
텍스트 파일(.txt, CSV 형식)을 읽어 MNE RawArray 객체로 변환하는 모듈.

주요 기능:
- pandas로 CSV 읽기 (delimiter=',', header=0)
- 단위 변환: uV → V (1e-6 곱하기)
- 샘플링 레이트 검증: Timestamp 컬럼 파싱하여 실제 SR 계산
- MNE RawArray 객체 생성 (2채널: Ch1, Ch2)
"""

import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_raw_data(file_path: str, cfg: DictConfig) -> Optional[mne.io.RawArray]:
    """
    텍스트 파일(.txt)을 MNE RawArray 객체로 로드.

    Parameters
    ----------
    file_path : str
        로드할 데이터 파일 경로 (.txt, CSV 포맷).
    cfg : DictConfig
        분석 설정 (PREPROCESSING.sampling_rate 등 참조).

    Returns
    -------
    mne.io.RawArray or None
        성공 시 MNE RawArray, 실패 시 None.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None

    try:
        # CSV 읽기 (delimiter=',', header=0)
        df = pd.read_csv(file_path, delimiter=",", header=0)
        logger.info(f"파일 로드: {file_path.name} (행={len(df)})")

        # 컬럼명 정규화 (실제 파일 형식: 'Timestamp(HH:mm:ss.SSS)', 'Ch1(uV)', 'Ch2(uV)')
        column_mapping = {}
        for col in df.columns:
            if 'timestamp' in col.lower():
                column_mapping[col] = 'Timestamp'
            elif 'ch1' in col.lower():
                column_mapping[col] = 'Ch1'
            elif 'ch2' in col.lower():
                column_mapping[col] = 'Ch2'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.info(f"컬럼명 정규화: {list(column_mapping.keys())} → {list(column_mapping.values())}")

        # 필수 컬럼 검증
        required_cols = ["Timestamp", "Ch1", "Ch2"]
        if not all(col in df.columns for col in required_cols):
            logger.error(f"필수 컬럼 누락: {required_cols}. 발견된 컬럼: {df.columns.tolist()}")
            return None

        # Timestamp로 실제 샘플링 레이트 계산
        # 형식: "HH:mm:ss.SSS" → 초 단위로 변환
        timestamps = df["Timestamp"].values
        if len(timestamps) < 2:
            logger.error("데이터 샘플 수 부족 (최소 2개 필요).")
            return None

        # 시간 문자열을 초 단위로 변환
        def parse_timestamp(ts_str: str) -> float:
            """HH:mm:ss.SSS 형식을 초 단위로 변환."""
            parts = ts_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds

        timestamps_sec = np.array([parse_timestamp(ts) for ts in timestamps])
        time_diffs = np.diff(timestamps_sec)
        mean_interval = np.mean(time_diffs)
        actual_sr = 1.0 / mean_interval if mean_interval > 0 else 0.0

        # Config SR과 비교 (10% 이상 차이나면 경고)
        config_sr = cfg.PREPROCESSING.sampling_rate
        sr_diff_pct = abs(actual_sr - config_sr) / config_sr * 100

        if sr_diff_pct > 10.0:
            logger.warning(
                f"샘플링 레이트 불일치: 실제={actual_sr:.2f}Hz, "
                f"설정={config_sr}Hz (차이={sr_diff_pct:.1f}%)"
            )
        else:
            logger.info(f"샘플링 레이트 검증 통과: {actual_sr:.2f}Hz ≈ {config_sr}Hz")

        # 채널 데이터 추출 및 단위 변환 (uV → V)
        ch1_data = df["Ch1"].values * 1e-6  # uV → V
        ch2_data = df["Ch2"].values * 1e-6  # uV → V
        data = np.vstack([ch1_data, ch2_data])  # (2, n_samples)

        # MNE Info 생성 (2채널)
        ch_names = ["Ch1", "Ch2"]
        ch_types = ["eeg", "eeg"]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=config_sr,
            ch_types=ch_types,
            verbose=False,
        )

        # RawArray 생성
        raw = mne.io.RawArray(data, info, verbose=False)
        logger.info(f"MNE Raw 객체 생성 완료: {raw}")

        return raw

    except Exception as e:
        logger.error(f"파일 로드 중 오류 발생 ({file_path.name}): {e}", exc_info=True)
        return None
