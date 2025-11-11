# 📜 main.py
# 💥 이 파일 하나만 실행하면 전체 파이프라인이 작동합니다.

import time  # 실행 시간 측정을 위해
import config  # ⚙️ config.py 파일에서 모든 설정값 로드

# 🏭 핵심 파이프라인 함수를 가져옵니다.
from core_pipeline.run_pipeline import run_full_pipeline

def main():
    """
    메인 실행 함수:
    1. 시작 메시지를 출력합니다.
    2. core_pipeline의 run_full_pipeline 함수를 호출합니다.
    3. 완료 메시지 및 실행 시간을 출력합니다.
    """
    print("="*70)
    print("🧠 EEG KPI 추출 파이프라인을 시작합니다.")
    print(f"▶️ 데이터 폴더: {config.DATA_PATH}")
    print(f"◀️ 결과 폴더: {config.RESULTS_PATH}")
    print("="*70)

    start_time = time.time()  # 시작 시간 기록

    try:
        # config 모듈 자체를 파이프라인 함수에 인수로 전달합니다.
        # 이렇게 하면 파이프라인의 모든 하위 모듈이 설정값(예: SAMPLE_RATE)에
        # 쉽게 접근할 수 있습니다.
        run_full_pipeline(cfg=config)

        end_time = time.time()  # 종료 시간 기록
        total_time = end_time - start_time

        print("\n" + "="*70)
        print(f"✅ 파이프라인이 성공적으로 완료되었습니다.")
        print(f"⏱️ 총 실행 시간: {total_time:.2f} 초")
        print(f"📊 최종 결과물은 '{config.RESULTS_PATH}' 폴더에 저장되었습니다.")
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