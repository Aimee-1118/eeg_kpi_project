# 📜 core_pipeline/m4_epoch.py
# 모듈 4: 정제된 Raw 데이터를 '상태(Block)' 기준으로 분할(Epoching)하고
#         최종적으로 아티팩트를 제거(Rejection)합니다.
# (🔥 "교회 vs 시장" 목표에 맞게 전문 수정됨)

import mne
from typing import Tuple, Optional
from omegaconf import DictConfig

def create_epochs(raw: mne.io.RawArray, cfg: DictConfig) -> Tuple[Optional[mne.Epochs], Optional[mne.Epochs]]:
    """
    M3에서 정제된 Raw 객체로부터 '교회', '시장' 등 상태(Block)별로
    고정된 길이(예: 5초)의 Epochs 객체를 생성합니다.

    - 'stim' 채널에서 config.EVENT_IDS에 정의된 (예: 1='church', 2='market') 
      블록 시작 이벤트를 찾습니다.
    - MNE Annotations를 생성하여 각 블록의 (시작, 지속시간, 라벨)을 정의합니다.
    - MNE make_fixed_length_epochs를 사용해 이 블록들을 
      config의 EPOCH_DURATION_SEC (예: 5초) 단위로 분할합니다.
    - A(ERP)용 Epochs는 None을 반환하고, B/C(상태 분석)용 Epochs만 반환합니다.

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
        
    print(f"[M4] 총 {events.shape[0]}개의 이벤트를 '{cfg.STIM_CHANNEL}' 채널에서 감지했습니다.")

    # --- 2. 이벤트를 MNE Annotations로 변환 ---
    # (MNE에서 블록(Block)을 다루는 표준 방식)
    event_ids_map = cfg.EVENT_IDS # 예: {'church': 1, 'market': 2}
    # {1: 'church', 2: 'market'} 형태로 뒤집기
    event_desc_map = {v: k for k, v in event_ids_map.items()} 
    
    onsets = []
    durations = []
    descriptions = []
    sfreq = cfg.SAMPLE_RATE

    for i in range(len(events)):
        event_sample, _, event_id = events[i]
        
        # config에 정의된 이벤트 ID만 처리
        if event_id in event_desc_map:
            description = event_desc_map[event_id]
            onset_sec = event_sample / sfreq
            
            # 이 이벤트의 지속시간(duration) 계산
            # (다음 이벤트 시작 전까지, 또는 파일 끝까지)
            if i + 1 < len(events):
                next_event_sample = events[i+1, 0]
            else:
                next_event_sample = raw.n_times # 파일 끝
            
            duration_sample = next_event_sample - event_sample
            duration_sec = duration_sample / sfreq
            
            onsets.append(onset_sec)
            durations.append(duration_sec)
            descriptions.append(description)
            
            print(f"[M4] '{description}' 블록 감지: {onset_sec:.2f}초 시작, {duration_sec:.2f}초 지속.")

    if not descriptions:
        print(f"[WARNING M4] config.EVENT_IDS {event_ids_map}에 해당하는 이벤트를 찾지 못했습니다.")
        return None, None

    # 생성된 Annotations을 Raw 객체에 적용
    annotations = mne.Annotations(onsets, durations, descriptions)
    raw_with_annots = raw.copy().set_annotations(annotations)

    # --- 3. 고정 길이 Epochs (Fixed Length Epochs) 생성 ---
    # (Annotations이 적용된 Raw 객체에서 Epochs를 생성하면
    #  각 Epoch는 자동으로 'church' 또는 'market' 라벨을 갖게 됩니다.)
    
    epochs_BC = None
    try:
        # Epoch 정제(Rejection) 기준 설정
        reject_threshold_volts = cfg.REJECT_THRESHOLD_UV * 1e-6
        reject_criteria = dict(eeg=reject_threshold_volts)

        epochs_BC = mne.make_fixed_length_epochs(
            raw_with_annots,
            duration=cfg.EPOCH_DURATION_SEC,      # 예: 5.0초
            overlap=cfg.EPOCH_OVERLAP_SEC,        # 예: 0.0초
            reject=reject_criteria,               # 100µV 초과 Epoch 제외
            preload=True,                         # KPI 추출을 위해 메모리에 즉시 로드
            verbose=False
        )
        
        # (중요) 베이스라인 보정(baseline=None)을 하지 않습니다.
        #      주파수/상태 분석에는 베이스라인 보정이 필요 없습니다.
        
        epochs_BC.drop_bad() # 리젝 기준에 걸린 Epoch 최종 드랍
        
        print(f"[M4] 'B/C' Epochs 생성 완료: 총 {len(epochs_BC)}개 생존.")
        print(f"    Epochs 라벨 분포: {epochs_BC.event_id}")

    except Exception as e:
        print(f"[ERROR M4] 고정 길이 Epochs 생성 중 오류 발생: {e}")
        return None, None

    # --- 4. 최종 반환 ---
    # A(ERP)용 Epochs는 없으므로 None 반환, B/C(상태)용 Epochs만 반환
    return None, epochs_BC