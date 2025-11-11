# 📜 features/features_C.py
# 🧮 [모듈 5-C] 동적/비선형 KPI 계산 함수들

import numpy as np
import config
import antropy as ant
from scipy.signal import butter, filtfilt, hilbert, spectrogram, coherence
import importlib

# Try to dynamically import mne.connectivity to avoid static import resolution errors
spectral_connectivity = None
try:
    mne_conn = importlib.import_module('mne.connectivity')
    spectral_connectivity = getattr(mne_conn, 'spectral_connectivity', None)
except Exception:
    spectral_connectivity = None

from warnings import filterwarnings

# MNE의 connectivity 함수가 때때로 경고(warning)를 발생시킬 수 있으므로,
# (예: "Only 1 epoch... estimating connectivity_")
# 불필요한 경고 메시지를 숨깁니다.
filterwarnings("ignore", category=UserWarning, module='mne')

def get_C_features(epoch_data: np.ndarray, cfg: config, kpi_row: dict):
    """
    'B/C' 유형 Epoch 데이터(단일 Epoch)에서 C 카테고리의 모든 KPI를 추출합니다.
    추출된 KPI는 'kpi_row' 딕셔너리에 직접 추가됩니다.

    Args:
        epoch_data (np.ndarray): (n_channels, n_samples) 형태의 2D 배열.
        cfg (config): config.py 모듈 객체
        kpi_row (dict): KPI 결과를 누적할 딕셔너리 (수정됨)
    """
    
    sfreq = cfg.SAMPLE_RATE
    ch_names = cfg.CHANNELS
    bands = cfg.BANDS
    epoch_duration_sec = epoch_data.shape[1] / sfreq
    epsilon = 1e-10 

    # --- C-1. 시간-주파수 동역학 (Per Channel) ---
    for i, ch_name in enumerate(ch_names):
        x = epoch_data[i, :]
        
        # 💥 알파 버스트율 (Alpha Burst Rate)
        try:
            # 1. 알파 밴드 필터링
            b, a = butter(N=3, Wn=bands['Alpha'], btype='bandpass', fs=sfreq)
            alpha_filtered = filtfilt(b, a, x)
            # 2. 힐버트 변환으로 순간 진폭(Envelope) 추출
            alpha_envelope = np.abs(hilbert(alpha_filtered))
            # 3. 임계값 설정 (예: 평균 + 1*표준편차)
            threshold = np.mean(alpha_envelope) + 1.0 * np.std(alpha_envelope)
            # 4. 임계값을 넘는 '시작점' 카운트
            burst_starts = np.where((alpha_envelope[:-1] < threshold) & (alpha_envelope[1:] >= threshold))[0]
            # 5. 초당 횟수로 변환
            kpi_row[f'{ch_name}_C_dyn_alpha_burst_rate_hz'] = len(burst_starts) / epoch_duration_sec
        except Exception as e:
            print(f"[ERROR M5-C] Alpha Burst Rate 계산 실패: {e}")
            kpi_row[f'{ch_name}_C_dyn_alpha_burst_rate_hz'] = np.nan

        # 📉 시간에 따른 파워 변동성 (Variance of bandpower over time)
        try:
            # 1초 윈도우, 50% 오버랩으로 스펙트로그램 계산
            nperseg = min(int(sfreq * 1.0), len(x)) # 1초 윈도우
            noverlap = int(nperseg * 0.5) # 50% 오버랩
            freqs, t, Sxx = spectrogram(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
            
            for band_name, (fmin, fmax) in bands.items():
                band_mask = (freqs >= fmin) & (freqs < fmax)
                if np.sum(band_mask) > 0:
                    # 시간에 따른 밴드 파워 시계열
                    power_over_time = Sxx[band_mask, :].mean(axis=0)
                    # 밴드 파워의 변동성(분산)
                    kpi_row[f'{ch_name}_C_dyn_var_{band_name.lower()}'] = np.var(power_over_time)
                else:
                    kpi_row[f'{ch_name}_C_dyn_var_{band_name.lower()}'] = 0.0
        except Exception as e:
            print(f"[ERROR M5-C] Power Variability 계산 실패: {e}")

    # --- C-2. 비선형 복잡도 (Per Channel) ---
    for i, ch_name in enumerate(ch_names):
        x = epoch_data[i, :]
        std_x = np.std(x)
        
        # 🌀 샘플 엔트로피 (Sample Entropy)
        r = cfg.SAMPEN_R_RATIO * std_x # 허용 반경 r
        kpi_row[f'{ch_name}_C_comp_sampen'] = ant.sample_entropy(x, order=cfg.SAMPEN_M, radius=r)
        
        # 📐 프랙탈 차원 (Higuchi)
        kpi_row[f'{ch_name}_C_comp_higuchi_fd'] = ant.higuchi_fd(x, kmax=10)
        
        # (빠진 지표) LZC (Lempel-Ziv Complexity)
        # (평균값을 기준으로 0/1 이진화)
        x_bin = (x > np.mean(x)).astype(int)
        kpi_row[f'{ch_name}_C_comp_lzc_norm'] = ant.lziv_complexity("".join(x_bin.astype(str)), normalize=True)
        
        # (빠진 지표) 📈 DFA (Detrended Fluctuation Analysis)
        # (장기 기억 특성의 기울기(scaling exponent)를 반환)
        kpi_row[f'{ch_name}_C_comp_dfa_exp'] = ant.detrended_fluctuation(x)

    # --- C-3. 기능적 연결성 (Between Channels) ---
    if len(ch_names) == 2: # 2채널일 때만 실행
        x1 = epoch_data[0, :]
        x2 = epoch_data[1, :]
        
        # 🔗 채널 간 코히런스 (Coherence)
        # (2초 윈도우 사용)
        nperseg_coh = min(int(sfreq * 2.0), len(x1))
        f_coh, Cxy = coherence(x1, x2, fs=sfreq, nperseg=nperseg_coh)
        for band_name, (fmin, fmax) in bands.items():
            band_mask = (f_coh >= fmin) & (f_coh < fmax)
            if np.sum(band_mask) > 0:
                kpi_row[f'C_conn_coh_{band_name.lower()}'] = np.mean(Cxy[band_mask])
            else:
                kpi_row[f'C_conn_coh_{band_name.lower()}'] = 0.0

        # 🔗🔗 PLV & wPLI (MNE 사용)
        # MNE 함수는 (n_epochs, n_channels, n_samples) 3D 배열을 기대
        # 현재 (2, n_samples) -> (1, 2, n_samples)로 차원 변경
        epoch_data_mne = epoch_data[np.newaxis, :, :] 
        
        for band_name, (fmin, fmax) in bands.items():
            try:
                # 🔗 위상 동기화 (PLV)
                con_plv = spectral_connectivity(
                    epoch_data_mne, method='plv', sfreq=sfreq, 
                    fmin=fmin, fmax=fmax, faverage=True, verbose=False
                )
                # con_plv[0]는 (n_channels, n_channels) 배열, [0, 1]은 Fp1-Fp2 연결
                kpi_row[f'C_conn_plv_{band_name.lower()}'] = con_plv[0].get_data()[0, 1]
                
                # 🖇️ wPLI (Weighted Phase Lag Index)
                con_wpli = spectral_connectivity(
                    epoch_data_mne, method='wpli', sfreq=sfreq, 
                    fmin=fmin, fmax=fmax, faverage=True, verbose=False
                )
                kpi_row[f'C_C_conn_wpli_{band_name.lower()}'] = con_wpli[0].get_data()[0, 1]
            
            except Exception as e:
                # (Epoch가 너무 짧거나 하면 MNE에서 오류가 날 수 있음)
                print(f"[ERROR M5-C] MNE Connectivity ({band_name}) 계산 실패: {e}")
                kpi_row[f'C_conn_plv_{band_name.lower()}'] = np.nan
                kpi_row[f'C_conn_wpli_{band_name.lower()}'] = np.nan
    
    # (kpi_row 딕셔너리가 수정되었으므로, 별도 반환값 없음)