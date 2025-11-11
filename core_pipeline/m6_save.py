# 📜 core_pipeline/m6_save.py
# 모듈 6: 최종 KPI 데이터프레임(DataFrame)을 CSV 파일로 저장합니다.

import pandas as pd
import os
import config  # config.py를 타입 힌팅 및 설정값 로드를 위해 임포트

def save_dataframe_to_csv(df: pd.DataFrame, cfg: config):
    """
    Pandas DataFrame을 config에 지정된 경로와 파일 이름으로 저장합니다.

    - config.py의 RESULTS_PATH 폴더가 없으면 자동으로 생성합니다.
    - config.py의 RESULT_FILENAME으로 CSV 파일을 저장합니다.
    - CSV 저장 시, pandas가 자동으로 추가하는 행 인덱스는 저장하지 않습니다 (index=False).

    Args:
        df (pd.DataFrame): M5 모듈에서 생성되고 M0에서 취합된 최종 KPI DataFrame
        cfg (config): config.py 모듈 객체
    """
    
    print(f"\n[M6] 최종 KPI 테이블 저장 시작...")

    try:
        # 1. 결과 폴더가 존재하는지 확인하고, 없으면 생성
        results_dir = cfg.RESULTS_PATH
        os.makedirs(results_dir, exist_ok=True)  # exist_ok=True: 폴더가 이미 있어도 오류 X

        # 2. 최종 저장 경로 설정
        file_name = cfg.RESULT_FILENAME
        save_path = os.path.join(results_dir, file_name)

        # 3. DataFrame을 CSV 파일로 저장
        # index=False 옵션: pandas의 기본 행 인덱스(0, 1, 2...)가
        #                   CSV 파일에 별도 열로 저장되는 것을 방지합니다.
        df.to_csv(save_path, index=False, encoding='utf-8-sig') 
        # (encoding='utf-8-sig'는 Excel에서 한글이 깨지지 않게 엽니다)

        print(f"[M6] 저장 완료: '{save_path}'")

    except PermissionError:
        print(f"[ERROR M6] 파일 쓰기 권한이 없습니다. '{save_path}'")
        print(f"    (혹시 해당 CSV 파일을 Excel 등에서 열어두지 않았는지 확인하세요.)")
        raise
    except Exception as e:
        print(f"[ERROR M6] 파일 저장 중 알 수 없는 오류 발생: {e}")
        raise