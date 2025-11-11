# 📜 core_pipeline/m1_load.py
# 모듈 1: CSV 파일에서 EEG 데이터를 로드하고 MNE Raw 객체로 변환합니다.

import pandas as pd
import numpy as np
import mne
import config  # config.py를 타입 힌팅 및 설정값 로드를 위해 임포트

def load_data_from_csv(file_path: str, cfg: config) -> mne.io.RawArray:
    """
    CSV 파일에서 EEG 데이터를 로드하고 MNE Raw 객체로 변환합니다.
    
    - CSV 헤더에서 config.py에 정의된 채널(CHANNELS)을 선택합니다.
    - 데이터 단위를 Microvolts(µV)에서 Volts(V)로 변환합니다.
    - MNE RawArray 객체를 생성하여 반환합니다.

    Args:
        file_path (str): 로드할 .csv 파일의 전체 경로
        cfg (config): config.py 모듈 객체

    Returns:
        mne.io.RawArray: MNE Raw 객체
    
    Raises:
        FileNotFoundError: file_path에 파일이 없는 경우
        KeyError: CSV 파일에 config.CHANNELS에 정의된 채널 이름이 없는 경우
    """
    
    print(f"[M1] '{file_path}'에서 데이터 로드 중...")

    try:
        # 1. Pandas로 CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 2. config에 정의된 채널만 선택 (예: ['Fp1', 'Fp2'])
        # CSV에 'Timestamp' 등 다른 열이 있어도 무시됩니다.
        eeg_data = df[cfg.CHANNELS].values

        # 3. 데이터 전치 (Transpose)
        # MNE는 (n_channels, n_samples) 형태를 기대합니다.
        # Pandas .values는 (n_samples, n_channels) 형태이므로 .T로 축을 변경합니다.
        eeg_data_transposed = eeg_data.T

        # 4. 단위 변환 (매우 중요!)
        # CSV 데이터가 µV (마이크로볼트) 단위라고 가정합니다.
        # MNE의 기본 단위는 V (볼트)이므로 1e-6 (0.000001)을 곱해줍니다.
        data_in_volts = eeg_data_transposed * 1e-6

        # 5. MNE Info 객체 생성
        ch_names = cfg.CHANNELS
        ch_types = ['eeg'] * len(ch_names)  # 모든 채널을 'eeg' 타입으로 지정
        sfreq = cfg.SAMPLE_RATE
        
        # MNE Info 객체 생성
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # 6. MNE RawArray 객체 생성
        raw = mne.io.RawArray(data_in_volts, info)
        
        # (선택) 센서 위치(Montage) 설정 (Fp1, Fp2는 표준 위치에 있음)
        # 2채널만으로는 위치 정보가 큰 의미가 없을 수 있으나,
        # 향후 시각화를 위해 표준 10-20 몬타주를 설정할 수 있습니다.
        try:
            raw.set_montage('standard_1020', on_missing='warn')
        except ValueError:
            print(f"[M1-WARN] 표준 10-20 몬타주에 {cfg.CHANNELS} 채널이 없습니다. 몬타주 설정을 건너뜁니다.")
        
        print(f"[M1] 로드 완료: {len(ch_names)}개 채널, {raw.n_times}개 샘플 ({raw.n_times / sfreq:.2f}초)")
        
        return raw

    except FileNotFoundError:
        print(f"[ERROR M1] 파일을 찾을 수 없습니다: {file_path}")
        raise
    except KeyError:
        print(f"[ERROR M1] CSV 파일 헤더에 config.py에 정의된 채널({cfg.CHANNELS})이 없습니다.")
        print(f"    CSV에 있는 헤더: {list(df.columns)}")
        raise
    except Exception as e:
        print(f"[ERROR M1] 데이터 로드 중 알 수 없는 오류 발생: {e}")
        raise