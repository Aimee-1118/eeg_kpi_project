# 📜 core_pipeline/m2_preprocess.py
# 모듈 2: MNE Raw 객체에 대해 전처리(필터링)를 수행합니다.

import mne
import config  # config.py를 타입 힌팅 및 설정값 로드를 위해 임포트

def filter_data(raw: mne.io.RawArray, cfg: config) -> mne.io.RawArray:
    """
    MNE Raw 객체에 대해 노치 필터와 대역통과 필터를 순차적으로 적용합니다.

    - config.py의 NOTCH_FREQ 설정에 따라 노치 필터(전원 노이즈)를 적용합니다.
    - config.py의 FILTER_L_FREQ, FILTER_H_FREQ 설정에 따라 대역통과 필터를 적용합니다.
    - 원본(raw) 객체를 덮어쓰지 않고, 필터링된 복사본을 반환합니다.

    Args:
        raw (mne.io.RawArray): M1 모듈에서 로드된 원본 Raw 객체
        cfg (config): config.py 모듈 객체

    Returns:
        mne.io.RawArray: 필터링이 적용된 새로운 Raw 객체
    """
    
    print(f"[M2] 데이터 전처리(필터링) 시작...")

    # MNE는 필터링 시 원본을 직접 수정할 수 있으므로,
    # 원본을 보존하기 위해 .copy()를 명시적으로 사용합니다.
    raw_filtered = raw.copy()

    try:
        # 1. ⚡️ 노치 필터 (Notch Filter) 적용
        # 전원 노이즈(예: 60Hz) 제거
        notch_freq = cfg.NOTCH_FREQ
        if notch_freq is not None:
            print(f"[M2] {notch_freq}Hz 노치 필터 적용 중...")
            raw_filtered.notch_filter(freqs=notch_freq, verbose=False)
        
        # 2. 📉📈 대역통과 필터 (Band-pass Filter) 적용
        # 뇌파 신호 외의 저주파(DC 쏠림) 및 고주파(근육 노이즈) 제거
        l_freq = cfg.FILTER_L_FREQ
        h_freq = cfg.FILTER_H_FREQ
        
        if l_freq is not None or h_freq is not None:
            print(f"[M2] {l_freq}Hz - {h_freq}Hz 대역통과 필터 적용 중...")
            raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        
        print(f"[M2] 필터링 완료.")
        
        return raw_filtered

    except Exception as e:
        print(f"[ERROR M2] 필터링 중 오류 발생: {e}")
        raise