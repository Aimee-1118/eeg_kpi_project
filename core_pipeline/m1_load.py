# 📜 core_pipeline/m1_load.py
# 모듈 1: CSV 파일에서 EEG 및 STIM 데이터를 로드하고 MNE Raw 객체로 변환합니다.
# (🔥 "교회 vs 시장" 목표 및 'stim' 채널 로직을 반영하여 수정됨)

import pandas as pd
import numpy as np
import mne
from omegaconf import DictConfig

def load_data_from_csv(file_path: str, cfg: DictConfig) -> mne.io.RawArray:
    """
    CSV 파일에서 EEG 데이터와 STIM 데이터를 로드하고 MNE Raw 객체로 변환합니다.
    
    - (수정) CSV 헤더에서 config.py의 'CHANNELS'(EEG)와 'STIM_CHANNEL'(이벤트)을 모두 선택합니다.
    - (수정) 'EEG 채널'만 Microvolts(µV) -> Volts(V)로 변환합니다.
    - (수정) 'STIM 채널'은 이벤트 코드로 간주하여 변환하지 않습니다.
    - MNE RawArray 객체를 생성하여 반환합니다.

    Args:
        file_path (str): 로드할 .csv 파일의 전체 경로
        cfg (config): config.py 모듈 객체

    Returns:
        mne.io.RawArray: MNE Raw 객체 (EEG + STIM 채널 포함)
    
    Raises:
        FileNotFoundError: file_path에 파일이 없는 경우
        KeyError: CSV 파일에 config.py에 정의된 필수 채널이 없는 경우
    """
    
    print(f"[M1] '{file_path}'에서 데이터 로드 중...")

    try:
        # 1. Pandas로 CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 2. (🔥 수정) config에 정의된 필수 채널(EEG + STIM) 확인
        # OmegaConf ListConfig 객체를 파이썬 기본 list로 변환해야 MNE에서 오류가 나지 않습니다.
        eeg_ch_names = list(cfg.CHANNELS) 
        stim_ch_name = cfg.STIM_CHANNEL
        
        # 2a. EEG 채널 확인
        for ch in eeg_ch_names:
            if ch not in df.columns:
                raise KeyError(f"CSV 파일에 config.py의 EEG 채널({ch})이 없습니다.")
        
        # 2b. STIM 채널 확인
        if stim_ch_name not in df.columns:
            raise KeyError(f"CSV 파일에 config.py의 STIM 채널({stim_ch_name})이 없습니다. M4 Epoching에 필수입니다.")

        # 3. (🔥 수정) MNE에 필요한 채널 이름 및 타입 리스트 생성
        final_ch_names = eeg_ch_names + [stim_ch_name]
        ch_types = ['eeg'] * len(eeg_ch_names) + ['stim']

        # 4. (🔥 수정) MNE (n_channels, n_samples) 형태로 데이터 추출
        data_transposed = df[final_ch_names].values.T

        # 5. (🔥 수정) 단위 변환 (!!!)
        # *EEG 채널만* µV -> V로 변환. STIM 채널은 변환하지 않음.
        
        # 5a. float으로 타입 변환 (STIM 채널도 숫자이므로)
        data_transposed_float = data_transposed.astype(float)

        # 5b. EEG 채널 인덱스만 찾기
        eeg_indices = [final_ch_names.index(ch) for ch in eeg_ch_names]
        
        # 5c. EEG 인덱스의 데이터에만 1e-6 곱하기
        data_transposed_float[eeg_indices, :] *= 1e-6
        
        # 6. MNE Info 객체 생성
        sfreq = cfg.SAMPLE_RATE
        info = mne.create_info(ch_names=final_ch_names, sfreq=sfreq, ch_types=ch_types)

        # 7. MNE RawArray 객체 생성
        raw = mne.io.RawArray(data_transposed_float, info)
        
        # (선택) 센서 위치(Montage) 설정 (EEG 채널에 대해서만)
        try:
            # .set_montage는 'eeg' 타입 채널만 알아서 설정합니다.
            raw.set_montage('standard_1020', on_missing='warn')
        except ValueError:
            print(f"[M1-WARN] 표준 10-20 몬타주에 {cfg.CHANNELS} 채널이 없습니다. 몬타주 설정을 건너뜁니다.")
        
        print(f"[M1] 로드 완료: {len(eeg_ch_names)}개 EEG 채널, 1개 STIM 채널.")
        print(f"    총 {raw.n_times}개 샘플 ({raw.n_times / sfreq:.2f}초)")
        
        return raw

    except FileNotFoundError:
        print(f"[ERROR M1] 파일을 찾을 수 없습니다: {file_path}")
        raise
    except KeyError as e:
        print(f"[ERROR M1] {e}")
        print(f"    CSV에 있는 헤더: {list(df.columns)}")
        raise
    except Exception as e:
        print(f"[ERROR M1] 데이터 로드 중 알 수 없는 오류 발생: {e}")
        raise