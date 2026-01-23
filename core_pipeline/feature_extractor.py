"""
feature_extractor.py
====================
The Engine: Converts MNE Epochs → 103 KPI Dictionary

Role: Takes 1 MNE Epochs object, runs A-B-C per-channel,
D cross-channel, aggregates across epochs, returns 1 Row Dict.

Total KPIs: 
- Ch1: 17 (A) + 20 (B) + 9 (C) = 46
- Ch2: 17 (A) + 20 (B) + 9 (C) = 46
- Cross: 11 (D)
= 103 total
"""

import numpy as np
import pandas as pd
from typing import Dict
import logging

from features.features_A import compute_time_features
from features.features_B import compute_freq_features
from features.features_C import compute_nonlinear_features
from features.features_D import compute_cross_features


logger = logging.getLogger(__name__)


def extract_features(epochs, subject: str, condition: int, trial_no: int) -> Dict:
    """
    Extract all 103 KPIs from an MNE Epochs object.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object (shape: [n_epochs, n_channels, n_times])
    subject : str
        Subject name (e.g., "한석")
    condition : int
        Condition code (1=positive/G, 2=negative/B)
    trial_no : int
        Trial number from filename
    
    Returns
    -------
    dict
        Single row dictionary with 103 KPI columns
    """
    
    # Extract metadata
    metadata_dict = {
        'Subject': subject,
        'Condition': condition,
        'Trial_No': trial_no,
    }
    
    # Get raw data
    data = epochs.get_data()  # Shape: [n_epochs, n_channels, n_times]
    sr = int(epochs.info['sfreq'])
    n_epochs = data.shape[0]
    
    if n_epochs == 0:
        logger.warning(f"[{subject}_{condition}_{trial_no}] No epochs found. Returning all NaN.")
        return _return_nan_row(metadata_dict)
    
    # ===== Loop over epochs and collect KPIs =====
    epoch_dicts = []
    
    for epoch_idx in range(n_epochs):
        try:
            epoch_data = data[epoch_idx]  # Shape: [2, n_times]
            ch1_data = epoch_data[0]
            ch2_data = epoch_data[1]
            
            # Per-channel features
            ch1_a = compute_time_features(ch1_data, sr)
            ch1_b = compute_freq_features(ch1_data, sr)
            ch1_c = compute_nonlinear_features(ch1_data, sr)
            
            ch2_a = compute_time_features(ch2_data, sr)
            ch2_b = compute_freq_features(ch2_data, sr)
            ch2_c = compute_nonlinear_features(ch2_data, sr)
            
            # Cross-channel features
            cross_d = compute_cross_features(ch1_data, ch2_data, sr)
            
            # Combine with prefixes
            epoch_dict = {}
            
            # Ch1
            for key, val in ch1_a.items():
                epoch_dict[f'Ch1_{key}'] = val
            for key, val in ch1_b.items():
                epoch_dict[f'Ch1_{key}'] = val
            for key, val in ch1_c.items():
                epoch_dict[f'Ch1_{key}'] = val
            
            # Ch2
            for key, val in ch2_a.items():
                epoch_dict[f'Ch2_{key}'] = val
            for key, val in ch2_b.items():
                epoch_dict[f'Ch2_{key}'] = val
            for key, val in ch2_c.items():
                epoch_dict[f'Ch2_{key}'] = val
            
            # Cross
            for key, val in cross_d.items():
                epoch_dict[f'Cross_{key}'] = val
            
            epoch_dicts.append(epoch_dict)
        
        except Exception as e:
            logger.warning(f"[{subject}_{condition}_{trial_no}] Epoch {epoch_idx} failed: {e}")
            continue
    
    # ===== Aggregation: Average across epochs =====
    if not epoch_dicts:
        logger.warning(f"[{subject}_{condition}_{trial_no}] All epochs failed. Returning all NaN.")
        return _return_nan_row(metadata_dict)
    
    try:
        df_epochs = pd.DataFrame(epoch_dicts)
        aggregated = df_epochs.mean(axis=0).to_dict()
    except Exception as e:
        logger.error(f"[{subject}_{condition}_{trial_no}] Aggregation failed: {e}")
        return _return_nan_row(metadata_dict)
    
    # ===== Merge metadata + aggregated KPIs =====
    result_dict = {**metadata_dict, **aggregated}
    
    return result_dict


def _return_nan_row(metadata_dict: Dict) -> Dict:
    """
    Return a row with metadata + all KPI columns as NaN.
    
    Parameters
    ----------
    metadata_dict : dict
        Metadata (Subject, Condition, Trial_No)
    
    Returns
    -------
    dict
        Full row with NaN KPIs
    """
    kpi_columns = _get_all_kpi_columns()
    result_dict = {**metadata_dict}
    for col in kpi_columns:
        result_dict[col] = np.nan
    return result_dict


def _get_all_kpi_columns() -> list:
    """
    Generate all possible KPI column names.
    Used for creating NaN rows or column ordering.
    
    Returns
    -------
    list
        All KPI column names (without metadata)
    """
    # Manually define expected KPI names based on features_A~D
    
    # Time-Domain (A): 17 per channel
    a_cols = [
        'amp_max', 'amp_min', 'amp_p2p', 'amp_mean', 'amp_rms',
        'stat_mean', 'stat_std', 'stat_variance', 'stat_median', 'stat_skewness', 'stat_kurtosis',
        'zcr', 'slope_mean', 'peak_count', 'peak_mean_height',
        'hjorth_mobility', 'hjorth_complexity'
    ]
    
    # Frequency-Domain (B): 20 per channel
    b_cols = [
        'pow_total',
        'pow_abs_delta', 'pow_abs_theta', 'pow_abs_alpha', 'pow_abs_beta', 'pow_abs_gamma',
        'pow_rel_delta', 'pow_rel_theta', 'pow_rel_alpha', 'pow_rel_beta', 'pow_rel_gamma',
        'peak_freq_hz', 'centroid_hz', 'sef90_hz', 'spec_entropy', 'spec_flatness',
        'aperiodic_exponent', 'aperiodic_offset',
        'alpha_beta_ratio', 'theta_beta_ratio'
    ]
    
    # Nonlinear (C): 9 per channel
    c_cols = [
        'sampen', 'spec_ent', 'perm_ent', 'svd_ent',
        'higuchi_fd', 'petrosian_fd', 'katz_fd',
        'lzc', 'dfa'
    ]
    
    # Cross-Channel (D): 11 total
    d_cols = [
        'coh_delta', 'coh_theta', 'coh_alpha', 'coh_beta', 'coh_gamma',
        'pearson_corr',
        'asym_power_delta', 'asym_power_theta', 'asym_power_alpha', 'asym_power_beta', 'asym_power_gamma'
    ]
    
    # Build full column list with prefixes
    all_cols = []
    
    # Ch1
    for col in a_cols + b_cols + c_cols:
        all_cols.append(f'Ch1_{col}')
    
    # Ch2
    for col in a_cols + b_cols + c_cols:
        all_cols.append(f'Ch2_{col}')
    
    # Cross
    for col in d_cols:
        all_cols.append(f'Cross_{col}')
    
    return all_cols
