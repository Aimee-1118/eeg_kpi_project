"""
test_phase5.py
==============
Phase 5 검증 스크립트: Feature Extraction 테스트.

동작:
1. raw_data/ 폴더에서 유효한 파일 하나 선택
2. loader.py로 MNE Raw 객체 로드
3. preprocessor.py로 전처리 적용
4. epocher.py로 Epochs 생성
5. cleaner.py로 Artifact Rejection 수행
6. feature_extractor.py로 KPI 추출
7. 결과 확인 (딕셔너리 키 개수, NaN 처리 등)
"""

import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.cleaner import clean_epochs
from core.data_scanner import scan_raw_data
from core.epocher import create_epochs
from core.feature_extractor import extract_features
from core.loader import load_raw_data
from core.preprocessor import preprocess_raw
from utils.config_loader import load_and_validate_config


def main():
    """Phase 5 테스트 메인 함수."""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Phase 5 테스트: Feature Extraction")
    logger.info("=" * 70)

    # 설정 로드
    config_path = project_root / "configs" / "analysis_config.yaml"
    cfg = load_and_validate_config(config_path=str(config_path), cli_args=[])

    # raw_data 폴더에서 유효한 파일 찾기
    valid_files, skipped = scan_raw_data(cfg.PATHS.data_dir)

    if not valid_files:
        logger.warning("유효한 데이터 파일이 없습니다. 테스트를 종료합니다.")
        return

    # 첫 번째 파일 선택
    test_file = valid_files[0]
    logger.info(f"\n테스트 파일 선택: {test_file['filename']}")
    logger.info(f"  - Subject: {test_file['subject']}")
    logger.info(f"  - Condition: {test_file['condition']}")
    logger.info(f"  - Trial: {test_file['trial']}")

    # 1. 데이터 로드
    logger.info("\n[STEP 1] 데이터 로드 중...")
    raw = load_raw_data(test_file["path"], cfg)
    if raw is None:
        logger.error("데이터 로드 실패. 테스트를 종료합니다.")
        return
    logger.info(f"  ✓ Raw 객체 생성")

    # 2. 전처리
    logger.info("\n[STEP 2] 전처리 적용 중...")
    raw_filtered = preprocess_raw(raw, cfg)
    if raw_filtered is None:
        logger.error("전처리 실패. 테스트를 종료합니다.")
        return
    logger.info(f"  ✓ 전처리 완료")

    # 3. Epoching
    logger.info("\n[STEP 3] Epoch 생성 중...")
    epochs = create_epochs(raw_filtered, cfg)
    if epochs is None:
        logger.error("Epoch 생성 실패. 테스트를 종료합니다.")
        return
    logger.info(f"  ✓ Epochs 생성: {len(epochs)}개")

    # 4. Artifact Rejection
    logger.info("\n[STEP 4] Artifact Rejection 수행 중...")
    clean_epochs_obj = clean_epochs(epochs, cfg)
    if clean_epochs_obj is None:
        logger.error("Artifact Rejection 후 유효한 Epoch이 부족합니다.")
        return
    logger.info(f"  ✓ Clean Epochs: {len(clean_epochs_obj)}개")

    # 5. Feature Extraction
    logger.info("\n[STEP 5] Feature Extraction 수행 중...")
    features = extract_features(clean_epochs_obj, cfg)
    if features is None:
        logger.error("Feature Extraction 실패. 테스트를 종료합니다.")
        return

    logger.info(f"  ✓ Feature Extraction 완료: {len(features)}개 KPI")

    # 결과 분석
    logger.info("\n" + "=" * 70)
    logger.info("📊 추출된 KPI 요약:")
    logger.info("=" * 70)

    # KPI 카테고리별 분류
    band_powers = {k: v for k, v in features.items() if "_Band_" in k}
    stats = {k: v for k, v in features.items() if "_Stat_" in k}
    asymmetry = {k: v for k, v in features.items() if "Asym_" in k}
    coherence = {k: v for k, v in features.items() if "Conn_Coh_" in k}
    ratios = {k: v for k, v in features.items() if "_Ratio_" in k}

    logger.info(f"  - Band Powers: {len(band_powers)}개")
    logger.info(f"  - Basic Stats: {len(stats)}개")
    logger.info(f"  - Asymmetry: {len(asymmetry)}개")
    logger.info(f"  - Coherence: {len(coherence)}개")
    logger.info(f"  - Ratios: {len(ratios)}개")

    # NaN 체크
    nan_count = sum(1 for v in features.values() if v != v)  # NaN check
    logger.info(f"  - NaN 값 개수: {nan_count}/{len(features)}")

    # 샘플 KPI 출력
    logger.info("\n샘플 KPI 값:")
    sample_keys = [
        "Ch1_Band_Alpha",
        "Ch2_Band_Alpha",
        "Asym_Band_Alpha",
        "Ch1_Stat_Mean",
        "Ch1_Ratio_TBR",
        "Conn_Coh_Alpha",
    ]
    for key in sample_keys:
        if key in features:
            logger.info(f"  - {key}: {features[key]:.6f}")

    # 완료
    logger.info("\n" + "=" * 70)
    logger.info("✅ Phase 5 테스트 성공!")
    logger.info(f"총 {len(features)}개 KPI 추출 완료 (예상: 40~50개)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
