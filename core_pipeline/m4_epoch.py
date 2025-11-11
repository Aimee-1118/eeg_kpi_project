# 📜 core_pipeline/m4_epoch.py
# 모듈 4: 정제된 Raw 데이터를 '사건(Event)' 기준으로 분할(Epoching)하고
#         최종적으로 아티팩트를 제거(Rejection)합니다.

import mne
import config  # config.py를 타입 힌팅 및 설정값 로드를 위해 임포트
from typing import Tuple

def create_epochs(raw: mne.io.RawArray, cfg: config) -> Tuple[mne.Epochs, mne.Epochs]:
    """
    M3에서 정제된 Raw 객체로부터 A와 B/C 두 종류의 Epochs 객체를 생성합니다.
    
    - 'stim' 채널에서 모든 이벤트를 찾습니다.
    - (가정) '첫 대면' 이벤트 ID(예: 1)로 A Epochs (ERP/형태학적)를 생성합니다.
    - (가정) '판단' 이벤트 ID(예: 2)로 B/C Epochs (주파수/비선형)를 생성합니다.
    - config에 설정된 REJECT_THRESHOLD_UV 기준으로 아티팩트 Epoch를 제거합니다.

    Args:
        raw (mne.io.RawArray): M3 모듈에서 ICA로 정제된 Raw 객체
        cfg (config): config.py 모듈 객체

    Returns:
        tuple (mne.Epochs, mne.Epochs):
            - epochs_A: '첫 대면' 기준 Epochs 객체 (베이스라인 보정 O)
            - epochs_BC: '연속 거닐기' 기준 Epochs 객체 (베이스라인 보정 X)
            - 이벤트가 없을 경우 None을 반환합니다.
    """
    
    print(f"[M4] 데이터 분할 및 정제 시작...")

    # --- ❗ 중요: 이 부분은 사용자의 실제 트리거 코드에 맞게 수정해야 합니다 ---
    # (config.py에 이 변수들을 추가하는 것을 강력히 권장합니다.)
    EVENT_ID_A = {'first_glimpse': 1}  # '첫 대면' (A)을 유발한 트리거 코드 (예시)
    EVENT_ID_BC = {'judgment_button': 2} # '연속 거닐기'(BC)의 판단 마커 트리거 코드 (예시)
    # -------------------------------------------------------------------

    # 1. MNE Raw 객체에서 모든 이벤트(트리거) 찾기
    try:
        # MNE가 'stim' 채널을 자동으로 찾아 이벤트를 추출합니다.
        events = mne.find_events(raw, shortest_event=1, verbose=False)
    except Exception as e:
        print(f"[ERROR M4] 'stim' 채널에서 이벤트를 찾는 데 실패했습니다: {e}")
        print(f"    M1 로드 시 'STIM' 또는 'TRIGGER' 채널이 포함되었는지,")
        print(f"    config.py의 CHANNELS 목록에 *포함되지 않았는지* 확인하세요.")
        return None, None

    if events.shape[0] == 0:
        print(f"[WARNING M4] 'stim' 채널에서 어떠한 이벤트도 찾지 못했습니다.")
        return None, None
        
    print(f"[M4] 총 {events.shape[0]}개의 이벤트를 'stim' 채널에서 감지했습니다.")

    # 2. Epoch 정제(Rejection) 기준 설정
    # config에서 µV 단위의 임계값을 V 단위로 변환 (MNE 기본 단위는 V)
    reject_threshold_volts = cfg.REJECT_THRESHOLD_UV * 1e-6
    reject_criteria = dict(eeg=reject_threshold_volts)

    # 3. --- Epoch A (형태학적/시간축) 생성 ---
    epochs_A = None
    try:
        # event_id_A(예: 1)에 해당하는 이벤트만 필터링
        events_A = mne.pick_events(events, include=list(EVENT_ID_A.values()))
        
        if len(events_A) > 0:
            epochs_A = mne.Epochs(
                raw,
                events=events_A,
                event_id=EVENT_ID_A,
                tmin=cfg.EPOCH_A_TMIN,      # 예: -1.0초
                tmax=cfg.EPOCH_A_TMAX,      # 예: 3.0초
                reject=reject_criteria,     # 100µV 초과 Epoch 제외
                baseline=(cfg.EPOCH_A_TMIN, 0), # 💥 ERP 분석: 베이스라인 보정 필수
                preload=True,               # KPI 추출을 위해 메모리에 즉시 로드
                verbose=False
            )
            epochs_A.drop_bad() # 리젝 기준에 걸린 Epoch 최종 드랍
            print(f"[M4] 'A' Epochs 생성 완료: {len(events_A)}개 이벤트 중 {len(epochs_A)}개 생존.")
        else:
            print(f"[M4-INFO] 'A' 유형({EVENT_ID_A})의 이벤트를 찾지 못했습니다.")

    except Exception as e:
        print(f"[ERROR M4] 'A' Epochs 생성 중 오류 발생: {e}")

    # 4. --- Epoch B/C (주파수/비선형) 생성 ---
    epochs_BC = None
    try:
        # event_id_BC(예: 2)에 해당하는 이벤트만 필터링
        events_BC = mne.pick_events(events, include=list(EVENT_ID_BC.values()))
        
        if len(events_BC) > 0:
            epochs_BC = mne.Epochs(
                raw,
                events=events_BC,
                event_id=EVENT_ID_BC,
                tmin=cfg.EPOCH_BC_TMIN,     # 예: -10.0초
                tmax=cfg.EPOCH_BC_TMAX,     # 예: 0.0초
                reject=reject_criteria,     # 100µV 초과 Epoch 제외
                baseline=None,              # 💥 주파수/상태 분석: 베이스라인 보정 안 함
                preload=True,               # KPI 추출을 위해 메모리에 즉시 로드
                verbose=False
            )
            epochs_BC.drop_bad() # 리젝 기준에 걸린 Epoch 최종 드랍
            print(f"[M4] 'B/C' Epochs 생성 완료: {len(events_BC)}개 이벤트 중 {len(epochs_BC)}개 생존.")
        else:
            print(f"[M4-INFO] 'B/C' 유형({EVENT_ID_BC})의 이벤트를 찾지 못했습니다.")

    except Exception as e:
        print(f"[ERROR M4] 'B/C' Epochs 생성 중 오류 발생: {e}")

    print(f"[M4] 데이터 분할 및 정제 완료.")
    return epochs_A, epochs_BC