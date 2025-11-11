# 📜 features/m5_extract_features.py
# 🧠 [모듈 5] 특징 추출 매니저 (Manager)
# (🔥 "교회 vs 시장" 목표에 맞게 논리 오류 수정됨)

import mne
import config
import numpy as np
from typing import List, Dict, Any, Optional

# --- 1. 각 특징별 '일꾼' 함수들을 임포트합니다 ---
from .features_A import get_A_features
from .features_B import get_B_features
from .features_C import get_C_features


def extract_features_from_epochs(epochs_A: Optional[mne.Epochs], epochs_BC: Optional[mne.Epochs], cfg: config) -> List[Dict[str, Any]]:
    """
    [M5] M4에서 받은 Epochs 객체를 순회하며 A, B, C 특징을 추출합니다.
    (🔥 수정됨: 'epochs_BC'만 순회하며 A, B, C를 모두 추출하고, 숫자 라벨을 추가합니다.)

    Args:
        epochs_A (mne.Epochs | None): (사용 안 함) M4의 반환값을 받기 위해 인자는 남겨둠.
        epochs_BC (mne.Epochs | None): 'church'(1), 'market'(2) 라벨이 붙은 5초짜리 Epochs
        cfg (config): config.py 모듈 객체

    Returns:
        list: 각 Epoch의 KPI가 담긴 딕셔너리들의 리스트
    """
    
    print(f"[M5] 핵심 변수 추출(KPI) 시작...")
    all_kpi_rows = [] 

    # --- (🔥 수정됨) '첫 대면'(A) Epoch 루프 삭제 ---
    # '교회 vs 시장' 목표에서는 A(형태학적) 특징도
    # 'B/C' Epochs(5초 상태)에서 함께 추출합니다.
    if epochs_A is not None:
        print("[M5-WARN] 'epochs_A'가 None이 아닙니다. '교회 vs 시장' 목표에서는 이 Epoch가 무시됩니다.")

    
    # --- (🔥 수정됨) 'B/C' Epoch 루프가 모든 작업을 처리 ---
    if epochs_BC is not None:
        # (n_epochs, n_channels, n_samples) 3D 배열 반환
        all_data_BC = epochs_BC.get_data(picks='eeg')
        
        # (n_epochs) 만큼 반복
        for i in range(len(all_data_BC)):
            # (n_channels, n_samples) 2D 배열 전달
            epoch_data = all_data_BC[i]
            
            # (🔥 신규) MNE Epochs 객체에서 숫자 라벨(1, 2 등) 가져오기
            # epochs.events는 [샘플번호, 이전ID, 현재ID] 3열로 구성됨
            numeric_label = epochs_BC.events[i, 2] 
            
            kpi_row = {
                'epoch_id': i,                 # Epoch 순번 (0, 1, 2...)
                'label': numeric_label         # (🔥 신규) 1(church) 또는 2(market)
            }
            
            # (🔥 수정됨) 5초 Epoch에 대해 A, B, C 특징 모두 계산
            try:
                # 1. 형태학적 변수(A) 계산
                get_A_features(epoch_data, cfg, kpi_row)
                # 2. 주파수축 변수(B) 계산
                get_B_features(epoch_data, cfg, kpi_row)
                # 3. 동적/비선형 변수(C) 계산
                get_C_features(epoch_data, cfg, kpi_row)
                
                all_kpi_rows.append(kpi_row)
                
            except Exception as e:
                print(f"[ERROR M5] Epoch {i} (Label: {numeric_label}) 처리 중 오류: {e}")
                # 디버깅 시 아래 코드 주석 해제
                # import traceback 
                # traceback.print_exc()

    print(f"[M5] KPI 추출 완료: 총 {len(all_kpi_rows)}개의 유효 Epoch 처리.")
    return all_kpi_rows