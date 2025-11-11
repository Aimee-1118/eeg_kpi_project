# 📜 core_pipeline/m3_ica.py
# 모듈 3: ICA(독립성분분석)를 실행하여 핵심 노이즈(특히 EOG)를 제거합니다.
# (🔥 config.py 설정값을 참조하도록 하드코딩 수정됨)

import mne
import config  # config.py를 타입 힌팅 및 설정값 로드를 위해 임포트

def run_ica_and_clean(raw: mne.io.RawArray, cfg: config) -> mne.io.RawArray:
    """
    MNE Raw 객체에 ICA를 적용하여 EOG(눈 깜빡임) 아티팩트를 제거합니다.

    - 2채널(Fp1, Fp2) 환경에서는 EOG 전용 채널이 없습니다.
    - (🔥 수정) config의 EOG_CHANNEL_NAME(예: 'Fp1')을 EOG 감지용 채널로 사용합니다.
    - ICA '학습'은 1.0Hz로 필터링된 임시 데이터로 수행합니다.
    - ICA '적용'은 모듈 2에서 받은 원본(0.5Hz 필터링) 데이터에 수행합니다.

    Args:
        raw (mne.io.RawArray): M2 모듈에서 필터링된 Raw 객체
        cfg (config): config.py 모듈 객체

    Returns:
        mne.io.RawArray: ICA가 적용되어 노이즈가 제거된 새로운 Raw 객체
    """
    
    print(f"[M3] ICA 노이즈 제거 시작...")

    try:
        # 1. ICA "학습"을 위한 임시 데이터 생성
        # MNE 공식 문서에서는 ICA 학습(fit) 시 1.0Hz high-pass 필터를 권장합니다.
        # 모듈 2에서 0.5Hz로 필터링했지만, ICA 학습만을 위해 임시 복사본을 만듭니다.
        raw_for_ica_fit = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        
        # 2. ICA 객체 설정
        n_comps = len(cfg.CHANNELS)
        if n_comps < 2:
            print(f"[M3-WARN] ICA는 2개 이상의 채널이 필요합니다. ICA를 건너뜁니다.")
            return raw.copy()
            
        ica = mne.preprocessing.ICA(
            n_components=n_comps,  # 2채널이므로 2개의 컴포넌트
            method='fastica',      # 'picard'가 설치되어 있다면 'picard'가 더 좋음
            # (🔥 수정) config.py의 ICA_RANDOM_STATE 값 참조
            random_state=cfg.ICA_RANDOM_STATE, 
            max_iter='auto'
        )
        
        # 3. ICA "학습" (Fitting)
        print(f"[M3] {n_comps}개 컴포넌트로 ICA 학습(fit) 중...")
        ica.fit(raw_for_ica_fit, verbose=False)
        
        # 4. EOG (눈 깜빡임) 아티팩트 자동 탐지
        eog_indices = []
        
        # (🔥 수정) config.py에서 EOG 탐지용 채널 이름(예: 'Fp1') 가져오기
        eog_ch_name = cfg.EOG_CHANNEL_NAME
        
        # 설정된 EOG 채널이 실제 EEG 채널 목록에 있는지 확인
        if eog_ch_name not in raw.ch_names:
            print(f"[M3-ERROR] config의 EOG_CHANNEL_NAME('{eog_ch_name}')이(가) 로드된 채널({raw.ch_names})에 없습니다.")
            print(f"    EOG 자동 탐지를 건너뜁니다.")
            eog_events = []
        else:
            print(f"[M3] '{eog_ch_name}' 채널을 기준으로 EOG 이벤트 탐색 중...")
            # 4a. EOG 이벤트(깜빡임 시점) 찾기
            eog_events = mne.preprocessing.find_eog_events(raw, ch_name=eog_ch_name, l_freq=1.0, h_freq=10.0, verbose=False)
        
        if len(eog_events) < 5:
            print(f"[M3-WARN] EOG 이벤트가 5개 미만({len(eog_events)}개)으로 감지되어, EOG 컴포넌트 탐지에 실패할 수 있습니다.")
        
        # 4b. EOG Epoch (평균 깜빡임 파형) 생성
        if len(eog_events) > 0:
            eog_epochs = mne.preprocessing.create_eog_epochs(raw, events=eog_events, ch_name=eog_ch_name, verbose=False)
        else:
            eog_epochs = None # 빈 Epochs 객체 대신 None으로 초기화

        # 4c. EOG Epoch와 가장 상관관계가 높은 ICA 컴포넌트 찾기
        if eog_epochs is not None and eog_epochs.get_data().shape[0] > 0: # 생성된 Epoch가 있다면
            eog_indices, eog_scores = ica.find_bads_eog(eog_epochs, ch_name=eog_ch_name, verbose=False)
        
            if eog_indices:
                # 찾은 컴포넌트를 '제외 목록'에 추가
                ica.exclude = eog_indices
                print(f"[M3] EOG 컴포넌트 {eog_indices}번을 '제외 목록'에 추가했습니다.")
            else:
                print(f"[M3-WARN] EOG와 일치하는 ICA 컴포넌트를 찾지 못했습니다.")
        else:
             print(f"[M3-WARN] EOG Epoch를 생성하지 못했거나 이벤트가 부족합니다. EOG 자동 탐지를 건너뜁니다.")

        # 5. 정제된 뇌파 데이터 생성 (Applying)
        # ICA를 '적용'할 때는 모듈 2에서 받은 원본(0.5Hz 필터링) 'raw' 객체를 사용합니다.
        # ica.apply()는 원본을 수정할 수 있으므로 .copy()를 명시적으로 사용합니다.
        print(f"[M3] ICA 적용하여 노이즈 제거 중...")
        raw_cleaned = ica.apply(raw.copy(), exclude=ica.exclude, verbose=False)
        
        print(f"[M3] ICA 노이즈 제거 완료.")
        
        return raw_cleaned

    except Exception as e:
        print(f"[ERROR M3] ICA 처리 중 오류 발생: {e}")
        # 디버깅 시 아래 코드 주석 해제
        # import traceback
        # traceback.print_exc()
        # 오류가 발생하면, 최소한 필터링된 데이터라도 반환하기 위해 원본 복사본을 반환
        return raw.copy()