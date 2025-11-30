# 📜 core_pipeline/run_pipeline.py
# 🚀 이 파일은 전체 6모듈 파이프라인의 실행을 총괄 지휘합니다.
# (🔥 ICA 옵션 처리 로직 및 예외 처리 강화됨)

import os
import pandas as pd
from omegaconf import DictConfig

# --- 1. 각 모듈의 핵심 기능 임포트 ---
from core_pipeline.m1_load import load_data_from_csv
from core_pipeline.m2_preprocess import filter_data
from core_pipeline.m3_ica import run_ica_and_clean
from core_pipeline.m4_epoch import create_epochs
from features.m5_extract_features import extract_features_from_epochs
from core_pipeline.m6_save import save_dataframe_to_csv

def run_full_pipeline(cfg: DictConfig):
    """
    M1부터 M6까지 전체 파이프라인을 순차적으로 실행합니다.
    config.py에서 설정된 경로의 모든 CSV 파일을 처리합니다.

    Args:
        cfg (module): main.py로부터 전달받은 config 모듈 객체
        
    Returns:
        tuple: (final_kpi_df, metrics)
            - final_kpi_df (pd.DataFrame): 모든 파일에서 추출된 KPI가 담긴 데이터프레임
            - metrics (dict): 파이프라인 실행 결과 요약 정보 (현재는 빈 딕셔너리 반환)
            - 만약 처리된 데이터가 없다면 (None, None)을 반환합니다.
    """
    
    print(f"[INFO] 파이프라인 매니저: 작업을 시작합니다...")
    
    # 1. 📥 원본 데이터 파일 목록 가져오기
    try:
        # 하위 폴더까지 포함하여 모든 .csv 파일 탐색 (재귀적 탐색)
        # (convert_data.py로 변환된 파일들이 data_raw 바로 아래에 있을 수도 있고, 하위 폴더에 있을 수도 있음)
        raw_files = []
        for root, dirs, files in os.walk(cfg.DATA_PATH):
            for file in files:
                if file.endswith('.csv'):
                    raw_files.append(os.path.join(root, file))
                    
        if not raw_files:
            print(f"[WARNING] '{cfg.DATA_PATH}' 폴더 및 하위 폴더에 처리할 CSV 파일이 없습니다.")
            return None, None # (🔥 수정) 빈 데이터 반환 시 None, None 반환
            
    except FileNotFoundError:
        print(f"[ERROR] '{cfg.DATA_PATH}' 폴더를 찾을 수 없습니다. config.py를 확인하세요.")
        return None, None # (🔥 수정)

    # 2. 🧮 모든 파일의 KPI 결과를 취합할 리스트
    all_kpi_results = []

    # 3. 🔁 각 파일을 순차적으로 처리
    for file_path in raw_files:
        file_name = os.path.basename(file_path)
        print(f"\n--- 🔄 {file_name} 처리 중 ---")

        try:
            # --- M1. 데이터 로드 ---
            # CSV 파일을 MNE Raw 객체로 변환 (EEG + STIM 채널 포함)
            raw = load_data_from_csv(file_path, cfg)
            
            # --- M2. 전처리 & 필터링 ---
            # 노치 필터 및 대역통과 필터 적용
            raw_filtered = filter_data(raw, cfg)
            
            # --- (🔥 수정) M3. 핵심 노이즈 제거 (ICA 옵션) ---
            # config.py의 USE_ICA 플래그 확인
            if cfg.USE_ICA:
                print("[M3] config.USE_ICA=True이므로 ICA를 실행합니다.")
                raw_cleaned = run_ica_and_clean(raw_filtered, cfg)
            else:
                print("[M3] config.USE_ICA=False이므로 ICA를 건너뜁니다.")
                # M2(필터링) 결과를 M4로 바로 전달
                raw_cleaned = raw_filtered.copy() 
            
            # --- M4. 데이터 분할 & 정제 ---
            # '교회/시장' 블록을 5초 Epochs로 생성
            epochs_A, epochs_BC = create_epochs(raw_cleaned, cfg)
            
            # (🔥 수정) M4 로직 변경에 따라 epochs_BC만 확인
            if epochs_BC is None or len(epochs_BC) == 0:
                print(f"[WARNING] {file_name}에서 유효한 Epoch를 찾지 못해 건너뜁니다.")
                continue

            # --- M5. 핵심 변수 추출 ---
            # A, B, C 유형의 모든 KPI를 계산 (epochs_BC만 사용)
            kpi_rows_for_file = extract_features_from_epochs(epochs_A, epochs_BC, cfg)
            
            # 각 행에 파일 식별자 추가
            for row in kpi_rows_for_file:
                row['source_file'] = file_name
                all_kpi_results.append(row)

            print(f"[INFO] {file_name}: {len(kpi_rows_for_file)}개의 유효 Epoch에서 KPI 추출 완료.")

        except Exception as e:
            print(f"[ERROR] {file_name} 처리 중 오류 발생: {e}")
            # 디버깅 시 아래 코드 주석 해제
            # import traceback
            # traceback.print_exc()

    # 4. 📊 모든 결과를 하나의 DataFrame으로 통합
    if not all_kpi_results:
        print("[INFO] 처리된 데이터가 없어 파이프라인을 종료합니다.")
        return None, None # (🔥 수정) 데이터 없음 반환
        
    final_kpi_df = pd.DataFrame(all_kpi_results)
    
    # 5. --- M6. 데이터 테이블 생성 & 저장 ---
    save_dataframe_to_csv(final_kpi_df, cfg)
    
    print(f"\n[SUCCESS] 모든 파일 처리가 완료되었습니다.")
    print(f"총 {len(final_kpi_df)}개의 Epoch(행)과 {len(final_kpi_df.columns)}개의 KPI(열)가 저장되었습니다.")
    
    # (🔥 수정) 정상 완료 시 DataFrame과 빈 Metrics 반환
    return final_kpi_df, {}