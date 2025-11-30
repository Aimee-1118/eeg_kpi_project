# 📜 core_pipeline/m4_epoch.py
# 모듈 4: 정제된 Raw 데이터를 '상태(Block)' 기준으로 분할(Epoching)하고
#         최종적으로 아티팩트를 제거(Rejection)합니다.
# (🔥 수정됨: 파일의 조건(A/B)에 따라 Epoch 라벨(1/2)을 올바르게 지정하도록 로직 변경)

import mne
from typing import Tuple, Optional
from omegaconf import DictConfig

def create_epochs(raw: mne.io.RawArray, cfg: DictConfig) -> Tuple[Optional[mne.Epochs], Optional[mne.Epochs]]:
    """
    M3에서 정제된 Raw 객체로부터 '교회', '시장' 등 상태(Block)별로
    고정된 길이(예: 5초)의 Epochs 객체를 생성합니다.

    - 'stim' 채널에서 config.EVENT_IDS에 정의된 (예: 1='church', 2='market') 
      블록 시작 이벤트를 찾습니다.
    - (🔥 핵심 수정) 해당 파일이 어떤 조건(1 또는 2)인지 파악하여, 
      make_fixed_length_epochs의 id 파라미터로 넘겨줍니다.
    - Epoch 생성 후 drop_bad()를 호출하여 아티팩트를 제거합니다.

    Args:
        raw (mne.io.RawArray): M3 모듈에서 ICA로 정제된 Raw 객체
        cfg (config): config.py 모듈 객체

    Returns:
        tuple (None, mne.Epochs | None):
            - epochs_A: None (ERP 분석을 사용하지 않음)
            - epochs_BC: '교회', '시장' 라벨이 붙은 5초짜리 Epochs 객체
    """
    
    print(f"[M4] 데이터 분할(Block Epoching) 및 정제 시작...")

    # --- 1. MNE Raw 객체에서 모든 이벤트(트리거) 찾기 ---
    try:
        events = mne.find_events(raw, stim_channel=cfg.STIM_CHANNEL, shortest_event=1, verbose=False)
    except Exception as e:
        print(f"[ERROR M4] 'stim' 채널('{cfg.STIM_CHANNEL}')에서 이벤트를 찾는 데 실패했습니다: {e}")
        return None, None

    if events.shape[0] == 0:
        print(f"[WARNING M4] 'stim' 채널에서 어떠한 이벤트도 찾지 못했습니다.")
        return None, None
        
    # --- 2. 대표 이벤트 ID 식별 (🔥 핵심) ---
    # 현재 파일 구조상, 하나의 파일에는 하나의 조건(A 또는 B)만 존재한다고 가정합니다.
    # 따라서 감지된 첫 번째 이벤트 ID를 이 파일 전체의 라벨로 사용합니다.
    main_event_id = int(events[0, 2])
    
    event_ids_map = cfg.EVENT_IDS # 예: {'church': 1, 'market': 2}
    event_desc_map = {v: k for k, v in event_ids_map.items()} 
    
    block_name = event_desc_map.get(main_event_id, str(main_event_id))
    print(f"[M4] 이 파일의 주요 조건: '{block_name}' (ID: {main_event_id})")

    # --- 3. 고정 길이 Epochs 생성 및 정제 ---
    epochs_BC = None
    try:
        # Epoch 정제(Rejection) 기준 설정
        reject_threshold_volts = cfg.REJECT_THRESHOLD_UV * 1e-6
        reject_criteria = dict(eeg=reject_threshold_volts)

        # (🔥 수정됨) id=main_event_id 를 전달하여 올바른 라벨(1 또는 2)을 부여
        epochs_BC = mne.make_fixed_length_epochs(
            raw,
            duration=cfg.EPOCH_DURATION_SEC,
            overlap=cfg.EPOCH_OVERLAP_SEC,
            id=main_event_id,  # <--- 여기가 핵심 수정 사항입니다!
            preload=True,
            verbose=False
        )
        
        # 생성 후 drop_bad() 메서드에 reject 기준 전달
        # print(f"[M4] 아티팩트 제거 중 (기준: {cfg.REJECT_THRESHOLD_UV} µV)...")
        epochs_BC.drop_bad(reject=reject_criteria, verbose=False)
        
        print(f"[M4] '{block_name}' Epochs 생성 완료: 총 {len(epochs_BC)}개 생존.")
        # print(f"    Epochs 라벨 분포: {epochs_BC.event_id}")

    except Exception as e:
        print(f"[ERROR M4] 고정 길이 Epochs 생성 중 오류 발생: {e}")
        return None, None

    return None, epochs_BC