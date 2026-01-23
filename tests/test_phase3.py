"""
test_phase3.py
==============
Phase 3 검증 스크립트: Data Loading & Preprocessing 테스트.

동작:
1. raw_data/ 폴더에서 유효한 파일 하나 선택
2. loader.py로 MNE Raw 객체 로드
3. preprocessor.py로 전처리 적용
4. 결과 확인 (에러 없이 실행되는지 검증)
"""

import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_scanner import scan_raw_data
from core.loader import load_raw_data
from core.preprocessor import preprocess_raw
from utils.config_loader import load_and_validate_config


def main():
    """Phase 3 테스트 메인 함수."""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("Phase 3 테스트: Data Loading & Preprocessing")
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
    logger.info(f"  ✓ 채널 수: {len(raw.ch_names)}, 샘플 수: {len(raw.times)}")
    logger.info(f"  ✓ 지속 시간: {raw.times[-1]:.2f}초")

    # 2. 전처리
    logger.info("\n[STEP 2] 전처리 적용 중...")
    raw_filtered = preprocess_raw(raw, cfg)

    if raw_filtered is None:
        logger.error("전처리 실패. 테스트를 종료합니다.")
        return

    logger.info(f"  ✓ 전처리 완료: {raw_filtered}")

    # 완료
    logger.info("\n" + "=" * 70)
    logger.info("✅ Phase 3 테스트 성공!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
