# 📜 features/m5_extract_features.py
# 🧠 [모듈 5] 특징 추출 매니저 (Manager)
# M4에서 받은 Epochs 객체를 순회하며 A, B, C 특징을 추출하도록 지시합니다.

import mne
import config
import numpy as np

# --- 1. 각 특징별 '일꾼' 함수들을 임포트합니다 ---
# (이 파일들은 아직 생성 전이지만, 곧 만들 파일들입니다.)
# 이 함수들은 (epoch_data, cfg, kpi_row)를 인수로 받아,
# kpi_row 딕셔너리에 계산된 KPI 값들을 '채워 넣는' 역할을 합니다.
from .features_A import get_A_features
from .features_B import get_B_features
from .features_C import get_C_features


def extract_features_from_epochs(epochs_A: mne.Epochs, epochs_BC: mne.Epochs, cfg: config) -> list:
    """
    [M5] M4에서 받은 'A' Epochs와 'B/C' Epochs를 순회하며
    features_A, B, C 파일에 있는 함수들을 호출하여 모든 KPI를 추출합니다.

    Args:
        epochs_A (mne.Epochs): A 유형 (형태학적) Epochs 객체
        epochs_BC (mne.Epochs): B/C 유형 (주파수/비선형) Epochs 객체
        cfg (config): config.py 모듈 객체

    Returns:
        list: 각 Epoch의 KPI가 담긴 딕셔너리들의 리스트
              (이 리스트가 최종 CSV 파일의 내용이 됩니다.)
    """
    
    print(f"[M5] 핵심 변수 추출(KPI) 시작...")
    all_kpi_rows = [] # 최종 CSV 파일이 될 딕셔너리 리스트

    # --- A. '첫 대면' (형태학적/시간축) Epoch 처리 ---
    if epochs_A is not None:
        # .get_data()는 (n_epochs, n_channels, n_samples) 3D 배열을 반환합니다.
        # (예: 20개 Epochs, 2개 채널, 1000개 샘플)
        all_data_A = epochs_A.get_data(picks='eeg')
        
        # (n_epochs) 만큼 반복
        for i in range(len(all_data_A)):
            # (n_channels, n_samples) 2D 배열을 함수에 전달
            epoch_data = all_data_A[i] 
            
            # CSV의 한 '행(row)'이 될 기본 딕셔너리 생성
            kpi_row = {
                'epoch_id': f"A_{i}",          # Epoch 식별자 (A타입, 0번째)
                'epoch_type': 'A_first_glimpse'
            }
            
            # features_A.py의 함수가 이 딕셔너리를 받아 KPI를 채워넣음
            try:
                get_A_features(epoch_data, cfg, kpi_row)
                all_kpi_rows.append(kpi_row)
            except Exception as e:
                print(f"[ERROR M5-A] 'A' Epoch {i} 처리 중 오류: {e}")

    
    # --- B/C. '연속 거닐기' (주파수/비선형) Epoch 처리 ---
    if epochs_BC is not None:
        # (n_epochs, n_channels, n_samples) 3D 배열 반환
        all_data_BC = epochs_BC.get_data(picks='eeg')
        
        # (n_epochs) 만큼 반복
        for i in range(len(all_data_BC)):
            # (n_channels, n_samples) 2D 배열 전달
            epoch_data = all_data_BC[i]
            
            kpi_row = {
                'epoch_id': f"BC_{i}",         # Epoch 식별자 (BC타입, 0번째)
                'epoch_type': 'BC_judgment'
            }
            
            # features_B.py와 features_C.py의 함수가 이 딕셔너리를 채워넣음
            try:
                # 1. 주파수축 변수(B) 계산
                get_B_features(epoch_data, cfg, kpi_row)
                # 2. 동적/비선형 변수(C) 계산
                get_C_features(epoch_data, cfg, kpi_row)
                
                all_kpi_rows.append(kpi_row)
            except Exception as e:
                print(f"[ERROR M5-BC] 'B/C' Epoch {i} 처리 중 오류: {e}")

    print(f"[M5] KPI 추출 완료: 총 {len(all_kpi_rows)}개의 유효 Epoch 처리.")
    return all_kpi_rows