# 📜 convert_data.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- 설정 ---
SOURCE_DIR = "./data_raw/user_sample"  # 원본 txt 파일들이 있는 최상위 폴더
TARGET_DIR = "./data_raw"              # 변환된 csv 파일이 저장될 폴더
EVENT_MAP = {'_A_': 1, '_B_': 2}       # 파일명 패턴에 따른 이벤트 ID

def convert_txt_to_compatible_csv():
    print(f"🚀 데이터 변환 시작: {SOURCE_DIR} -> {TARGET_DIR}")
    
    # 1. 대상 파일 수집
    target_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith(".txt") and (('_A_' in file) or ('_B_' in file)):
                target_files.append(os.path.join(root, file))
    
    if not target_files:
        print("❌ 변환할 .txt 파일을 찾지 못했습니다.")
        return

    print(f"📄 총 {len(target_files)}개의 파일을 발견했습니다. 변환을 시작합니다...")

    # 2. 파일 변환 루프
    for file_path in tqdm(target_files, desc="Converting"):
        try:
            # --- A. 메타데이터 파싱 ---
            subject_id = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)
            
            # 이벤트 ID 결정 (A=1, B=2)
            event_id = 0
            for key, val in EVENT_MAP.items():
                if key in file_name:
                    event_id = val
                    break
            
            # --- B. 데이터 로드 ---
            # 첫 줄(...) 제외, 탭 구분, 헤더 없음
            df = pd.read_csv(file_path, sep='\t', header=None, skiprows=1)
            
            # --- C. 데이터 가공 ---
            # 1) 컬럼명 지정
            df.columns = ['Fp1', 'Fp2']
            
            # 2) 'stim' 채널 생성
            # (🔥 중요 수정) 0번 인덱스가 아니라 50번 인덱스(0.2초 지점)에 이벤트를 찍습니다.
            # 이렇게 해야 0 -> 1 로 변하는 '상승 엣지'가 생겨 MNE가 이벤트를 감지합니다.
            df['stim'] = 0
            
            # 데이터 길이가 충분한지 확인 후 마킹
            marker_idx = 50 
            if len(df) > marker_idx:
                df.loc[marker_idx, 'stim'] = event_id
            else:
                # 데이터가 너무 짧으면 마지막에 찍음
                df.loc[len(df)-1, 'stim'] = event_id
            
            # --- D. 저장 ---
            new_file_name = f"{subject_id}_{file_name.replace('.txt', '.csv')}"
            save_path = os.path.join(TARGET_DIR, new_file_name)
            
            df.to_csv(save_path, index=False)
            
        except Exception as e:
            print(f"\n[ERROR] {file_name} 변환 중 오류: {e}")

    print("\n✅ 모든 변환이 완료되었습니다! 이제 main.py를 실행하세요.")

if __name__ == "__main__":
    convert_txt_to_compatible_csv()