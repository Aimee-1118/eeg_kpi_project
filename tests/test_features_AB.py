"""
test_features_AB.py
===================
Time & Frequency Domain Feature 단위 테스트.

동작:
1. 랜덤 신호 생성 (Sine waves + Noise)
2. features_A.compute_time_features 호출
3. features_B.compute_freq_features 호출
4. 결과 검증 (Key 개수, NaN 체크)
"""

import sys
from pathlib import Path
import numpy as np
import logging

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.features_A import compute_time_features
from features.features_B import compute_freq_features


def generate_test_signal(sr=250, duration=4.0):
    """
    테스트용 EEG-like 신호 생성.
    
    - Alpha (10Hz) + Beta (20Hz) + Noise
    """
    t = np.arange(0, duration, 1/sr)
    
    # Alpha wave (10Hz, amplitude=5µV)
    alpha = 5 * np.sin(2 * np.pi * 10 * t)
    
    # Beta wave (20Hz, amplitude=3µV)
    beta = 3 * np.sin(2 * np.pi * 20 * t)
    
    # White noise
    noise = np.random.normal(0, 1, len(t))
    
    signal = alpha + beta + noise
    return signal


def main():
    """테스트 메인 함수."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Time & Frequency Domain Feature 테스트")
    logger.info("=" * 70)

    # 테스트 신호 생성
    sr = 250
    signal = generate_test_signal(sr=sr, duration=4.0)
    logger.info(f"\n✓ 테스트 신호 생성: {len(signal)}개 샘플 ({len(signal)/sr:.1f}초)")

    # ===== Time-Domain 테스트 =====
    logger.info("\n[STEP 1] Time-Domain Features 추출 중...")
    time_features = compute_time_features(signal, sr)
    
    logger.info(f"  ✓ 추출된 Time-Domain KPI: {len(time_features)}개")
    
    # NaN 체크
    nan_count = sum(1 for v in time_features.values() if np.isnan(v))
    logger.info(f"  ✓ NaN 개수: {nan_count}/{len(time_features)}")
    
    # 샘플 값 출력
    logger.info("\n  샘플 Time-Domain KPI:")
    for key in ['amp_mean', 'stat_std', 'zcr', 'hjorth_mobility']:
        if key in time_features:
            logger.info(f"    - {key}: {time_features[key]:.6f}")

    # ===== Frequency-Domain 테스트 =====
    logger.info("\n[STEP 2] Frequency-Domain Features 추출 중...")
    freq_features = compute_freq_features(signal, sr)
    
    logger.info(f"  ✓ 추출된 Frequency-Domain KPI: {len(freq_features)}개")
    
    # NaN 체크
    nan_count = sum(1 for v in freq_features.values() if np.isnan(v))
    logger.info(f"  ✓ NaN 개수: {nan_count}/{len(freq_features)}")
    
    # 샘플 값 출력
    logger.info("\n  샘플 Frequency-Domain KPI:")
    for key in ['pow_total', 'pow_abs_alpha', 'pow_rel_alpha', 'peak_freq_hz', 'aperiodic_exponent']:
        if key in freq_features:
            logger.info(f"    - {key}: {freq_features[key]:.6f}")

    # ===== 종합 결과 =====
    total_features = len(time_features) + len(freq_features)
    logger.info("\n" + "=" * 70)
    logger.info("✅ 테스트 완료!")
    logger.info(f"총 추출된 KPI: {total_features}개")
    logger.info(f"  - Time-Domain: {len(time_features)}개")
    logger.info(f"  - Frequency-Domain: {len(freq_features)}개")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
