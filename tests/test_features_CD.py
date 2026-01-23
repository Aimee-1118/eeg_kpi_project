"""
test_features_CD.py
===================
Nonlinear & Cross-Channel Feature 단위 테스트.

동작:
1. 두 가지 시나리오로 테스트 신호 생성
   - 시나리오 1: 독립적인 두 채널 (낮은 상관관계)
   - 시나리오 2: 상관관계 높은 두 채널 (Ch2 = Ch1 + 작은 노이즈)
2. features_C.compute_nonlinear_features 호출 (각 채널)
3. features_D.compute_cross_features 호출
4. 결과 검증
"""

import sys
from pathlib import Path
import numpy as np
import logging

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.features_C import compute_nonlinear_features
from features.features_D import compute_cross_features


def generate_test_signals(sr=250, duration=4.0, scenario='correlated'):
    """
    테스트용 EEG-like 신호 쌍 생성.
    
    Parameters
    ----------
    sr : int
        샘플링 레이트 (Hz)
    duration : float
        신호 길이 (초)
    scenario : str
        'independent': 독립적인 두 채널 (낮은 상관)
        'correlated': 상관관계 높은 두 채널 (Ch2 ≈ Ch1 + noise)
    
    Returns
    -------
    data_ch1, data_ch2 : np.ndarray
    """
    t = np.arange(0, duration, 1/sr)
    
    if scenario == 'independent':
        # 채널 1: Alpha (10Hz)
        ch1 = 5 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 1, len(t))
        
        # 채널 2: Beta (20Hz)
        ch2 = 3 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 1, len(t))
        
    elif scenario == 'correlated':
        # 채널 1: Alpha + Beta + Noise
        ch1 = 5 * np.sin(2 * np.pi * 10 * t) + 3 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.5, len(t))
        
        # 채널 2: Ch1의 스케일 변형 + 약간의 추가 노이즈
        ch2 = 1.1 * ch1 + np.random.normal(0, 0.3, len(t))
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return ch1, ch2


def main():
    """테스트 메인 함수."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Nonlinear & Cross-Channel Feature 테스트")
    logger.info("=" * 80)

    sr = 250
    
    # ===== 시나리오 1: Independent Channels =====
    logger.info("\n[SCENARIO 1] Independent Channels")
    logger.info("-" * 80)
    
    ch1_indep, ch2_indep = generate_test_signals(sr=sr, duration=4.0, scenario='independent')
    logger.info(f"✓ 테스트 신호 생성: 각 {len(ch1_indep)}개 샘플")
    
    # Nonlinear Features for Ch1
    logger.info("\n  [Step 1] Channel 1 - Nonlinear Features 추출 중...")
    nl_ch1_indep = compute_nonlinear_features(ch1_indep, sr)
    nan_count_c1 = sum(1 for v in nl_ch1_indep.values() if np.isnan(v))
    logger.info(f"    ✓ 추출된 Nonlinear KPI: {len(nl_ch1_indep)}개 (NaN: {nan_count_c1}개)")
    logger.info(f"    샘플: sampen={nl_ch1_indep['sampen']:.6f}, higuchi_fd={nl_ch1_indep['higuchi_fd']:.6f}, lzc={nl_ch1_indep['lzc']:.6f}")

    # Nonlinear Features for Ch2
    logger.info("\n  [Step 2] Channel 2 - Nonlinear Features 추출 중...")
    nl_ch2_indep = compute_nonlinear_features(ch2_indep, sr)
    nan_count_c2 = sum(1 for v in nl_ch2_indep.values() if np.isnan(v))
    logger.info(f"    ✓ 추출된 Nonlinear KPI: {len(nl_ch2_indep)}개 (NaN: {nan_count_c2}개)")

    # Cross-Channel Features
    logger.info("\n  [Step 3] Cross-Channel Features 추출 중...")
    cross_indep = compute_cross_features(ch1_indep, ch2_indep, sr)
    nan_count_d = sum(1 for v in cross_indep.values() if np.isnan(v))
    logger.info(f"    ✓ 추출된 Cross-Channel KPI: {len(cross_indep)}개 (NaN: {nan_count_d}개)")
    logger.info(f"    샘플: pearson_corr={cross_indep['pearson_corr']:.6f}, coh_alpha={cross_indep['coh_alpha']:.6f}")

    # ===== 시나리오 2: Correlated Channels =====
    logger.info("\n[SCENARIO 2] Correlated Channels (Ch2 ≈ Ch1 + noise)")
    logger.info("-" * 80)
    
    ch1_corr, ch2_corr = generate_test_signals(sr=sr, duration=4.0, scenario='correlated')
    logger.info(f"✓ 테스트 신호 생성: 각 {len(ch1_corr)}개 샘플")
    
    # Nonlinear Features for Ch1
    logger.info("\n  [Step 1] Channel 1 - Nonlinear Features 추출 중...")
    nl_ch1_corr = compute_nonlinear_features(ch1_corr, sr)
    nan_count_c1 = sum(1 for v in nl_ch1_corr.values() if np.isnan(v))
    logger.info(f"    ✓ 추출된 Nonlinear KPI: {len(nl_ch1_corr)}개 (NaN: {nan_count_c1}개)")

    # Nonlinear Features for Ch2
    logger.info("\n  [Step 2] Channel 2 - Nonlinear Features 추출 중...")
    nl_ch2_corr = compute_nonlinear_features(ch2_corr, sr)
    nan_count_c2 = sum(1 for v in nl_ch2_corr.values() if np.isnan(v))
    logger.info(f"    ✓ 추출된 Nonlinear KPI: {len(nl_ch2_corr)}개 (NaN: {nan_count_c2}개)")

    # Cross-Channel Features
    logger.info("\n  [Step 3] Cross-Channel Features 추출 중...")
    cross_corr = compute_cross_features(ch1_corr, ch2_corr, sr)
    nan_count_d = sum(1 for v in cross_corr.values() if np.isnan(v))
    logger.info(f"    ✓ 추출된 Cross-Channel KPI: {len(cross_corr)}개 (NaN: {nan_count_d}개)")
    
    # 상관관계 높은 채널에서는 pearson_corr가 높아야 함
    corr_value = cross_corr['pearson_corr']
    logger.info(f"    샘플: pearson_corr={corr_value:.6f} (기대값 >0.8)")
    if corr_value > 0.8:
        logger.info(f"    ✓ 예상대로 높은 상관관계!")
    else:
        logger.info(f"    ⚠️  상관관계가 예상보다 낮음")

    # ===== 종합 결과 =====
    logger.info("\n" + "=" * 80)
    logger.info("✅ 모든 테스트 완료!")
    logger.info(f"  - Nonlinear (C) 기본 개수: {len(nl_ch1_indep)}개")
    logger.info(f"  - Cross-Channel (D) 기본 개수: {len(cross_indep)}개")
    logger.info(f"  - 총 파일별 특징: 2채널 × {len(nl_ch1_indep)} + {len(cross_indep)} = {2*len(nl_ch1_indep) + len(cross_indep)}개")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
