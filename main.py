# 📜 main.py
# 💥 이 파일 하나만 실행하면 전체 파이프라인이 작동합니다.
# (🔥 config.py 대신 OmegaConf/argparse를 사용하도록 전문 수정됨)

import time
import argparse  # (🔥 신규) 터미널 인자 파싱을 위해
from omegaconf import OmegaConf  # (🔥 신규) YAML 및 인자 병합을 위해

# 🏭 핵심 파이프라인 함수는 그대로 가져옵니다.
from core_pipeline.run_pipeline import run_full_pipeline

def main():
    """
    메인 실행 함수:
    1. (🔥 신규) Argparse와 OmegaConf를 사용해 설정을 로드합니다.
        - 기본 YAML 설정 파일을 로드합니다.
        - 터미널 인자(arg)를 로드합니다.
        - 두 설정을 병합(merge)하여 최종 'cfg' 객체를 생성합니다.
    2. core_pipeline의 run_full_pipeline 함수를 호출합니다.
    3. 완료 메시지 및 실행 시간을 출력합니다.
    """

    # --- 1. (🔥 신규) 설정 로드 (Argparse + OmegaConf) ---
    parser = argparse.ArgumentParser(description="EEG KPI Extraction Pipeline")
    parser.add_argument(
        '-c', '--config_path',
        type=str,
        default='./configs/base_config.yaml',  # (🔥 신규) 기본 설정 파일 경로
        help="Path to the base YAML config file."
    )
    # 기본 설정 파일 경로(-c)만 argparse로 파싱합니다.
    # 나머지 (예: --use_ica=True)는 unknown_args로 받아서 OmegaConf가 처리합니다.
    args, unknown_args = parser.parse_known_args()

    # --- 2. (🔥 신규) 기본 YAML 설정 로드 ---
    try:
        base_cfg = OmegaConf.load(args.config_path)
    except FileNotFoundError:
        print(f"❌ 기본 설정 파일({args.config_path})을 찾을 수 없습니다.")
        print("    'configs/base_config.yaml' 파일을 생성했는지 확인하세요.")
        return

    # --- 3. (🔥 신규) 터미널에서 받은 추가 인자(override) 로드 ---
    cli_cfg = OmegaConf.from_cli(unknown_args)

    # --- 4. (🔥 신규) 설정 병합 (터미널 인자가 YAML 파일보다 우선함) ---
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    
    # --- (이하는 기존 main.py와 거의 동일) ---
    
    print("="*70)
    print("🧠 EEG KPI 추출 파이프라인을 시작합니다.")
    # (🔥 수정) config.py 대신 로드된 YAML 파일 경로와 오버라이드 내용 출력
    print(f"▶️ 기본 설정 파일: {args.config_path}")
    if unknown_args:
        print(f"▶️ 런타임 설정 (Override): {unknown_args}")
    print(f"▶️ 데이터 폴더: {cfg.DATA_PATH}") # YAML 파일의 DATA_PATH 값
    print(f"◀️ 결과 폴더: {cfg.RESULTS_PATH}") # YAML 파일의 RESULTS_PATH 값
    print("="*70)

    start_time = time.time()  # 시작 시간 기록

    try:
        # 5. (🔥 수정) 'import config' 대신 'OmegaConf'로 생성된 cfg 객체 전달
        run_full_pipeline(cfg=cfg)

        end_time = time.time()  # 종료 시간 기록
        total_time = end_time - start_time

        print("\n" + "="*70)
        print(f"✅ 파이프라인이 성공적으로 완료되었습니다.")
        print(f"⏱️ 총 실행 시간: {total_time:.2f} 초")
        print(f"📊 최종 결과물은 '{cfg.RESULTS_PATH}' 폴더에 저장되었습니다.")
        print("="*70)

    except Exception as e:
        print("\n" + "!"*70)
        print(f"❌ 오류가 발생하여 파이프라인이 중단되었습니다.")
        print(f"오류 상세: {e}")
        # 디버깅을 위해 전체 오류 추적을 보려면 아래 줄의 주석을 해제하세요.
        # import traceback
        # traceback.print_exc()
        print("!"*70)


if __name__ == "__main__":
    # 이 파일이 직접 실행되었을 때만 main() 함수를 호출합니다.
    main()