# 📜 features/features_C.py
# 🧮 [모듈 5-C] 동적/비선형 KPI 계산 함수들
# (🔥 수정됨: sample_entropy 인자명을 radius -> tolerance로 변경)

import numpy as np
import antropy as ant
from scipy.signal import butter, filtfilt, hilbert, spectrogram, coherence
from warnings import filterwarnings
from omegaconf import DictConfig

# MNE Connectivity 임포트 시도
try:
    from mne.connectivity import spectral_connectivity
except ImportError:
    # print("[M5-C WARN] 'mne.connectivity.spectral_connectivity'를 임포트할 수 없습니다. PLV/wPLI 계산을 건너뜁니다.")
    spectral_connectivity = None

filterwarnings("ignore", category=UserWarning, module='mne')

def get_C_features(epoch_data: np.ndarray, cfg: DictConfig, kpi_row: dict):
    """
    'B/C' 유형 Epoch 데이터(단일 Epoch)에서 C 카테고리의 모든 KPI를 추출합니다.
    """
    
    sfreq = cfg.SAMPLE_RATE
    ch_names = cfg.CHANNELS
    bands = cfg.BANDS
    epoch_duration_sec = epoch_data.shape[1] / sfreq
    
    # --- C-1. 시간-주파수 동역학 (Per Channel) ---
    for i, ch_name in enumerate(ch_names):
        x = epoch_data[i, :]
        
        # 💥 알파 버스트율
        try:
            b, a = butter(N=3, Wn=bands['Alpha'], btype='bandpass', fs=sfreq)
            alpha_filtered = filtfilt(b, a, x)
            alpha_envelope = np.abs(hilbert(alpha_filtered))
            threshold = np.mean(alpha_envelope) + cfg.ALPHA_BURST_THRESHOLD_SD * np.std(alpha_envelope)
            burst_starts = np.where((alpha_envelope[:-1] < threshold) & (alpha_envelope[1:] >= threshold))[0]
            kpi_row[f'{ch_name}_C_dyn_alpha_burst_rate_hz'] = len(burst_starts) / epoch_duration_sec
        except Exception:
            kpi_row[f'{ch_name}_C_dyn_alpha_burst_rate_hz'] = np.nan

        # 📉 파워 변동성
        try:
            nperseg = min(int(sfreq * cfg.POWER_VAR_WINDOW_SEC), len(x))
            noverlap = int(nperseg * cfg.POWER_VAR_OVERLAP_RATIO)
            freqs, t, Sxx = spectrogram(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
            
            for band_name, (fmin, fmax) in bands.items():
                band_mask = (freqs >= fmin) & (freqs < fmax)
                if np.sum(band_mask) > 0:
                    power_over_time = Sxx[band_mask, :].mean(axis=0)
                    kpi_row[f'{ch_name}_C_dyn_var_{band_name.lower()}'] = np.var(power_over_time)
                else:
                    kpi_row[f'{ch_name}_C_dyn_var_{band_name.lower()}'] = 0.0
        except Exception:
            pass

    # --- C-2. 비선형 복잡도 (Per Channel) ---
    for i, ch_name in enumerate(ch_names):
        x = epoch_data[i, :]
        std_x = np.std(x)
        
        # 🌀 샘플 엔트로피 (Sample Entropy)
        r = cfg.SAMPEN_R_RATIO * std_x
        try:
            # 🔥 [핵심 수정] radius -> tolerance
            kpi_row[f'{ch_name}_C_comp_sampen'] = ant.sample_entropy(x, order=cfg.SAMPEN_M, tolerance=r)
        except Exception:
             kpi_row[f'{ch_name}_C_comp_sampen'] = np.nan

        # 📐 프랙탈 차원 (Higuchi)
        try:
            kpi_row[f'{ch_name}_C_comp_higuchi_fd'] = ant.higuchi_fd(x, kmax=10)
        except Exception:
            kpi_row[f'{ch_name}_C_comp_higuchi_fd'] = np.nan
        
        # LZC (Lempel-Ziv Complexity)
        try:
            x_bin = (x > np.mean(x)).astype(int)
            # map 대신 "".join 사용
            kpi_row[f'{ch_name}_C_comp_lzc_norm'] = ant.lziv_complexity("".join(x_bin.astype(str)), normalize=True)
        except Exception:
             kpi_row[f'{ch_name}_C_comp_lzc_norm'] = np.nan
        
        # 📈 DFA
        try:
            kpi_row[f'{ch_name}_C_comp_dfa_exp'] = ant.detrended_fluctuation(x)
        except Exception:
            kpi_row[f'{ch_name}_C_comp_dfa_exp'] = np.nan

    # --- C-3. 기능적 연결성 (Between Channels) ---
    if len(ch_names) == 2: 
        x1 = epoch_data[0, :]
        x2 = epoch_data[1, :]
        
        # 코히런스
        try:
            nperseg_coh = min(int(sfreq * cfg.CONN_WINDOW_SEC), len(x1))
            if nperseg_coh > 0:
                f_coh, Cxy = coherence(x1, x2, fs=sfreq, nperseg=nperseg_coh)
                for band_name, (fmin, fmax) in bands.items():
                    band_mask = (f_coh >= fmin) & (f_coh < fmax)
                    if np.sum(band_mask) > 0:
                        kpi_row[f'C_conn_coh_{band_name.lower()}'] = np.mean(Cxy[band_mask])
                    else:
                        kpi_row[f'C_conn_coh_{band_name.lower()}'] = 0.0
        except Exception:
            pass

        # PLV & wPLI
        if spectral_connectivity is not None: 
            epoch_data_mne = epoch_data[np.newaxis, :, :] 
            for band_name, (fmin, fmax) in bands.items():
                try:
                    con_plv = spectral_connectivity(epoch_data_mne, method='plv', sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
                    kpi_row[f'C_conn_plv_{band_name.lower()}'] = con_plv[0].get_data()[0, 1]
                    
                    con_wpli = spectral_connectivity(epoch_data_mne, method='wpli', sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, verbose=False)
                    kpi_row[f'C_conn_wpli_{band_name.lower()}'] = con_wpli[0].get_data()[0, 1]
                except Exception:
                    kpi_row[f'C_conn_plv_{band_name.lower()}'] = np.nan
                    kpi_row[f'C_conn_wpli_{band_name.lower()}'] = np.nan