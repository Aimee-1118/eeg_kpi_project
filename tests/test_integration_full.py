"""
test_integration_full.py
========================
최종 통합 테스트: 103개 KPI 추출 검증

동작:
1. 랜덤 데이터로 가짜 EEG 신호 생성
2. MNE Epochs 객체 생성
3. feature_extractor로 103개 KPI 추출
4. 컬럼 개수 및 구조 검증
"""

import sys
from pathlib import Path
import numpy as np
import mne
import logging

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core_pipeline.feature_extractor import extract_features, _get_all_kpi_columns


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_dummy_epochs(sr: int = 250, n_epochs: int = 5, duration_sec: float = 4.0) -> mne.Epochs:
    """
    생성 더미 EEG Epochs.
    
    Parameters
    ----------
    sr : int
        샘플링 레이트 (Hz)
    n_epochs : int
        에포크 개수
    duration_sec : float
        에포크당 길이 (초)
    
    Returns
    -------
    mne.Epochs
    """
    n_samples = int(sr * duration_sec)
    
    # 2채널, 다중 에포크 데이터 생성
    data = np.random.randn(n_epochs, 2, n_samples) * 10  # 10µV scale
    
    # Alpha (10Hz) + Beta (20Hz) 신호 추가 (좀 더 현실적)
    t = np.linspace(0, duration_sec, n_samples)
    for i in range(n_epochs):
        alpha = 5 * np.sin(2 * np.pi * 10 * t)
        beta = 3 * np.sin(2 * np.pi * 20 * t)
        data[i, 0, :] += alpha + beta
        data[i, 1, :] += alpha * 0.8 + beta * 1.2  # 약간 다른 혼합
    
    # MNE Info 생성
    info = mne.create_info(
        ch_names=['Ch1', 'Ch2'],
        sfreq=sr,
        ch_types='eeg'
    )
    
    # Raw 데이터 생성
    raw_data = data.reshape(2, -1)  # [2, n_epochs*n_samples]
    raw = mne.io.RawArray(raw_data, info)
    
    # Epochs 생성 (각 에포크의 시작점)
    events = np.array([[i * n_samples, 0, 1] for i in range(n_epochs)])
    epochs = mne.Epochs(
        raw, events,
        event_id={'stim': 1},
        tmin=0,
        tmax=(n_samples - 1) / sr,
        baseline=None,
        verbose=False
    )
    
    return epochs


def main():
    """테스트 메인 함수."""
    logger.info("=" * 80)
    logger.info("Full System Integration Test: 103 KPI Extraction")
    logger.info("=" * 80)
    
    # ===== Step 1: Create Dummy Epochs =====
    logger.info("\n[Step 1] Creating dummy EEG Epochs...")
    epochs = create_dummy_epochs(sr=250, n_epochs=5, duration_sec=4.0)
    logger.info(f"✓ Created epochs: shape {epochs.get_data().shape}")
    logger.info(f"  - n_epochs: {len(epochs)}")
    logger.info(f"  - n_channels: {len(epochs.ch_names)}")
    logger.info(f"  - n_times per epoch: {epochs.get_data().shape[2]}")
    
    # ===== Step 2: Extract Features =====
    logger.info("\n[Step 2] Extracting 103 KPIs...")
    try:
        result_dict = extract_features(
            epochs,
            subject="TestSubject",
            condition=1,
            trial_no=1
        )
        logger.info("✓ Feature extraction completed")
    except Exception as e:
        logger.error(f"✗ Feature extraction failed: {e}")
        return
    
    # ===== Step 3: Verify KPI Count & Structure =====
    logger.info("\n[Step 3] Verifying KPI structure...")
    
    metadata_cols = ['Subject', 'Condition', 'Trial_No']
    kpi_cols = [k for k in result_dict.keys() if k not in metadata_cols]
    
    logger.info(f"✓ Metadata columns: {len(metadata_cols)}")
    for col in metadata_cols:
        logger.info(f"    - {col}: {result_dict[col]}")
    
    logger.info(f"✓ KPI columns: {len(kpi_cols)}")
    
    # 각 채널별 KPI 개수 확인
    ch1_kpis = [k for k in kpi_cols if k.startswith('Ch1_')]
    ch2_kpis = [k for k in kpi_cols if k.startswith('Ch2_')]
    cross_kpis = [k for k in kpi_cols if k.startswith('Cross_')]
    
    logger.info(f"    - Ch1 KPIs: {len(ch1_kpis)}")
    logger.info(f"    - Ch2 KPIs: {len(ch2_kpis)}")
    logger.info(f"    - Cross-Channel KPIs: {len(cross_kpis)}")
    logger.info(f"    - Total: {len(ch1_kpis) + len(ch2_kpis) + len(cross_kpis)}")
    
    # ===== Step 4: Expected vs Actual Count =====
    logger.info("\n[Step 4] Verifying expected counts...")
    expected_ch_kpis = 46  # 17 (A) + 20 (B) + 9 (C)
    expected_cross_kpis = 11  # D
    expected_total = 46 + 46 + 11  # 103
    
    actual_total = len(ch1_kpis) + len(ch2_kpis) + len(cross_kpis)
    
    if len(ch1_kpis) == expected_ch_kpis and len(ch2_kpis) == expected_ch_kpis:
        logger.info(f"✓ Channel KPI counts correct: {len(ch1_kpis)} + {len(ch2_kpis)} = {len(ch1_kpis) + len(ch2_kpis)}")
    else:
        logger.warning(f"✗ Channel KPI count mismatch:")
        logger.warning(f"  Expected: {expected_ch_kpis} each")
        logger.warning(f"  Actual: Ch1={len(ch1_kpis)}, Ch2={len(ch2_kpis)}")
    
    if len(cross_kpis) == expected_cross_kpis:
        logger.info(f"✓ Cross-Channel KPI count correct: {len(cross_kpis)}")
    else:
        logger.warning(f"✗ Cross-Channel KPI count mismatch:")
        logger.warning(f"  Expected: {expected_cross_kpis}")
        logger.warning(f"  Actual: {len(cross_kpis)}")
    
    if actual_total == expected_total:
        logger.info(f"✓ TOTAL KPI COUNT CORRECT: {actual_total} == {expected_total}")
    else:
        logger.warning(f"✗ Total KPI count mismatch:")
        logger.warning(f"  Expected: {expected_total}")
        logger.warning(f"  Actual: {actual_total}")
    
    # ===== Step 5: Sample KPI Values =====
    logger.info("\n[Step 5] Sample KPI values (first 10 KPIs)...")
    kpi_items = list(result_dict.items())[3:13]  # Skip metadata, get first 10 KPIs
    for key, val in kpi_items:
        nan_mark = " (NaN)" if isinstance(val, float) and np.isnan(val) else ""
        logger.info(f"    - {key}: {val:.6f}{nan_mark}" if isinstance(val, (int, float)) and not np.isnan(val) else f"    - {key}: {val}{nan_mark}")
    
    # ===== Step 6: NaN Check =====
    logger.info("\n[Step 6] Checking for NaN values...")
    nan_count = sum(1 for v in result_dict.values() if isinstance(v, float) and np.isnan(v))
    total_count = len(result_dict)
    logger.info(f"✓ NaN count: {nan_count}/{total_count}")
    
    if nan_count == 0:
        logger.info("✓ All KPIs successfully computed (no NaN)")
    else:
        logger.warning(f"⚠️  {nan_count} KPIs are NaN (possible calculation failures)")
    
    # ===== Final Summary =====
    logger.info("\n" + "=" * 80)
    logger.info("✅ Integration Test Complete!")
    logger.info("=" * 80)
    logger.info(f"Result Summary:")
    logger.info(f"  - Metadata: {len(metadata_cols)} columns")
    logger.info(f"  - Ch1: {len(ch1_kpis)} KPIs")
    logger.info(f"  - Ch2: {len(ch2_kpis)} KPIs")
    logger.info(f"  - Cross-Channel: {len(cross_kpis)} KPIs")
    logger.info(f"  - TOTAL: {actual_total} KPIs")
    if actual_total == 103:
        logger.info(f"  ✓ ✓ ✓ **103 KPI TARGET ACHIEVED!** ✓ ✓ ✓")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
