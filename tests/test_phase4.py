"""
test_phase4.py
==============
Phase 4 검증 스크립트: Epoching & Artifact Rejection 테스트.

동작:
1. raw_data/ 폴더에서 유효한 파일 하나 선택
2. loader.py로 MNE Raw 객체 로드
3. preprocessor.py로 전처리 적용
4. epocher.py로 Epochs 생성
5. cleaner.py로 Artifact Rejection 수행
6. 결과 확인 (에러 없이 실행되는지 검증)
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
from core.loader import load_raw_data
from core.preprocessor import preprocess_raw
from utils.config_loader import load_and_validate_config


def main():
    """Phase 4 테스트 메인 함수."""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Phase 4 테스트: Epoching & Artifact Rejection")
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

    logger.info(f"  ✓ Raw 객체 생성: {raw}")

    # 2. 전처리
    logger.info("\n[STEP 2] 전처리 적용 중...")
    raw_filtered = preprocess_raw(raw, cfg)

    if raw_filtered is None:
        logger.error("전처리 실패. 테스트를 종료합니다.")
        return

    logger.info(f"  ✓ 전처리 완료: {raw_filtered}")

    # 3. Epoching
    logger.info("\n[STEP 3] Epoch 생성 중...")
    epochs = create_epochs(raw_filtered, cfg)

    if epochs is None:
        logger.error("Epoch 생성 실패. 테스트를 종료합니다.")
        return

    logger.info(f"  ✓ Epochs 생성: {epochs}")
    logger.info(f"  ✓ Epoch 수: {len(epochs)}개")
    logger.info(f"  ✓ Epoch 길이: {cfg.EPOCH.window_sec}초")

    # 4. Artifact Rejection
    logger.info("\n[STEP 4] Artifact Rejection 수행 중...")
    clean_epochs_obj = clean_epochs(epochs, cfg)

    if clean_epochs_obj is None:
        logger.error("Artifact Rejection 후 유효한 Epoch이 부족합니다.")
        return

    logger.info(f"  ✓ 정제 완료: {clean_epochs_obj}")
    logger.info(f"  ✓ Clean Epochs 수: {len(clean_epochs_obj)}개")

    # 완료
    logger.info("\n" + "=" * 70)
    logger.info("✅ Phase 4 테스트 성공!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
