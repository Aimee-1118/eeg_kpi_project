"""
main.py
=======
The Driver: Orchestrates parallel file processing + KPI extraction.

Flow:
1. Load config (analysis_config.yaml)
2. Scan raw_data/ for EEG files
3. Parallel process each file (joblib Parallel, n_jobs=-1)
4. Aggregate results into DataFrame
5. Sort columns (Subject → Condition → Cross → Ch1 → Ch2)
6. Save CSV + summary report
"""

import os
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# MNE for filtering and epoching
import mne
from mne.filter import notch_filter, filter_data

# Parallel processing
from joblib import Parallel, delayed
from tqdm import tqdm

# OmegaConf for config
from omegaconf import OmegaConf

# Local modules
from core_pipeline.feature_extractor import extract_features


# ===== Setup Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===== Configuration =====
def load_config(config_path: str = "configs/analysis_config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    """
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    cfg = OmegaConf.load(config_path)
    logger.info(f"Config loaded from {config_path}")
    return cfg


# ===== File Scanning =====
def scan_raw_data(data_dir: str, pattern: str = r"(.+?)_([GB])_(\d{3})\.txt") -> List[Dict]:
    """
    Scan raw_data/ directory for EEG files matching pattern.
    
    Parameters
    ----------
    data_dir : str
        Path to raw_data directory
    pattern : str
        Regex pattern for filename (Subject_[G/B]_NNN.txt)
    
    Returns
    -------
    list of dict
        File metadata (path, subject, condition, trial_no)
    """
    file_list = []
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return file_list
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.txt'):
            continue
        
        match = re.match(pattern, filename, re.IGNORECASE)
        if not match:
            logger.warning(f"Skipped (invalid filename): {filename}")
            continue
        
        subject = match.group(1)
        condition_str = match.group(2).upper()
        trial_no_str = match.group(3)
        
        # Map G/B to condition codes
        condition = 1 if condition_str == 'G' else 2
        trial_no = int(trial_no_str)
        
        file_path = os.path.join(data_dir, filename)
        
        file_list.append({
            'file_path': file_path,
            'subject': subject,
            'condition': condition,
            'trial_no': trial_no,
            'filename': filename,
        })
    
    logger.info(f"Found {len(file_list)} valid files")
    return file_list


# ===== Single File Processing =====
def process_file_wrapper(file_info: Dict, cfg: Dict) -> Dict:
    """
    Process a single EEG file: Load → Filter → Epoch → Extract KPIs.
    
    Parameters
    ----------
    file_info : dict
        File metadata (path, subject, condition, trial_no)
    cfg : dict
        Configuration object
    
    Returns
    -------
    dict
        Single row dictionary with all KPIs (or NaN if failed)
    """
    file_path = file_info['file_path']
    subject = file_info['subject']
    condition = file_info['condition']
    trial_no = file_info['trial_no']
    
    metadata = {
        'Subject': subject,
        'Condition': condition,
        'Trial_No': trial_no,
    }
    
    try:
        # ===== Step 1: Load CSV =====
        df = pd.read_csv(file_path)
        if len(df) < 10:  # Too short
            logger.warning(f"[{subject}_{condition}_{trial_no}] File too short (<10 samples)")
            return _return_nan_metadata(metadata)
        
        # Extract Ch1, Ch2 (assuming columns are named Ch1(uV), Ch2(uV) or similar)
        ch_names = ['Ch1(uV)', 'Ch2(uV)'] if 'Ch1(uV)' in df.columns else ['Ch1', 'Ch2']
        if ch_names[0] not in df.columns:
            # Try alternative column names
            cols = df.columns.tolist()
            if len(cols) >= 3:  # Skip timestamp, use next 2
                ch_names = [cols[1], cols[2]]
            else:
                logger.error(f"[{subject}_{condition}_{trial_no}] Cannot find channel columns")
                return _return_nan_metadata(metadata)
        
        ch1_data = df[ch_names[0]].values
        ch2_data = df[ch_names[1]].values
        
        sr = int(cfg.PREPROCESSING.sr)
        
        # ===== Step 2: Preprocess (Filter) =====
        try:
            # Notch filter 60Hz
            ch1_filtered = notch_filter(ch1_data, sr, freqs=60.0, verbose=False)
            ch2_filtered = notch_filter(ch2_data, sr, freqs=60.0, verbose=False)
            
            # Bandpass 0.5-50Hz
            ch1_filtered = filter_data(
                ch1_filtered, sr, 
                l_freq=cfg.PREPROCESSING.filter_band[0],
                h_freq=cfg.PREPROCESSING.filter_band[1],
                verbose=False
            )
            ch2_filtered = filter_data(
                ch2_filtered, sr,
                l_freq=cfg.PREPROCESSING.filter_band[0],
                h_freq=cfg.PREPROCESSING.filter_band[1],
                verbose=False
            )
        except Exception as e:
            logger.error(f"[{subject}_{condition}_{trial_no}] Filtering failed: {e}")
            return _return_nan_metadata(metadata)
        
        # Stack channels for MNE
        data_filtered = np.array([ch1_filtered, ch2_filtered])
        
        # ===== Step 3: Create MNE Info & Epochs =====
        try:
            info = mne.create_info(
                ch_names=['Ch1', 'Ch2'],
                sfreq=sr,
                ch_types='eeg'
            )
            raw = mne.io.RawArray(data_filtered, info)
            
            # Epoching: 4sec window, 50% overlap
            window_sec = cfg.EPOCH.window_sec
            overlap_sec = cfg.EPOCH.overlap_sec
            window_samples = int(window_sec * sr)
            overlap_samples = int(overlap_sec * sr)
            step_samples = window_samples - overlap_samples
            
            events = []
            for i in range(0, len(data_filtered[0]) - window_samples, step_samples):
                events.append([i, 0, 1])
            
            if not events:
                logger.warning(f"[{subject}_{condition}_{trial_no}] No epochs created")
                return _return_nan_metadata(metadata)
            
            events = np.array(events)
            epochs = mne.Epochs(
                raw, events,
                event_id={'stim': 1},
                tmin=0,
                tmax=(window_samples - 1) / sr,
                baseline=None,
                verbose=False
            )
            
            # ===== Step 4: Artifact Rejection =====
            # Drop epochs with peak-to-peak > threshold
            threshold_uv = cfg.PREPROCESSING.artifact_threshold_uv
            epochs_to_drop = []
            for epoch_idx in range(len(epochs)):
                epoch_data = epochs[epoch_idx][0]
                p2p_ch1 = np.ptp(epoch_data[0])
                p2p_ch2 = np.ptp(epoch_data[1])
                if p2p_ch1 > threshold_uv or p2p_ch2 > threshold_uv:
                    epochs_to_drop.append(epoch_idx)
            
            if epochs_to_drop:
                epochs.drop(epochs_to_drop)
            
            if len(epochs) < 3:
                logger.warning(f"[{subject}_{condition}_{trial_no}] Too few clean epochs (<3)")
                return _return_nan_metadata(metadata)
            
            # ===== Step 5: Feature Extraction =====
            kpi_row = extract_features(epochs, subject, condition, trial_no)
            
            return kpi_row
        
        except Exception as e:
            logger.error(f"[{subject}_{condition}_{trial_no}] Epoching/Feature extraction failed: {e}")
            return _return_nan_metadata(metadata)
    
    except Exception as e:
        logger.error(f"[{subject}_{condition}_{trial_no}] File processing failed: {e}")
        return _return_nan_metadata(metadata)



def _return_nan_metadata(metadata: Dict) -> Dict:
    """Return metadata with NaN placeholders for KPIs."""
    # Generate dummy KPI columns (all NaN)
    kpi_cols = _get_kpi_columns()
    row = {**metadata}
    for col in kpi_cols:
        row[col] = np.nan
    return row


def _get_kpi_columns() -> List[str]:
    """Get list of all KPI column names."""
    a_cols = [
        'amp_max', 'amp_min', 'amp_p2p', 'amp_mean', 'amp_rms',
        'stat_mean', 'stat_std', 'stat_variance', 'stat_median', 'stat_skewness', 'stat_kurtosis',
        'zcr', 'slope_mean', 'peak_count', 'peak_mean_height',
        'hjorth_mobility', 'hjorth_complexity'
    ]
    b_cols = [
        'pow_total',
        'pow_abs_delta', 'pow_abs_theta', 'pow_abs_alpha', 'pow_abs_beta', 'pow_abs_gamma',
        'pow_rel_delta', 'pow_rel_theta', 'pow_rel_alpha', 'pow_rel_beta', 'pow_rel_gamma',
        'peak_freq_hz', 'centroid_hz', 'sef90_hz', 'spec_entropy', 'spec_flatness',
        'aperiodic_exponent', 'aperiodic_offset',
        'alpha_beta_ratio', 'theta_beta_ratio'
    ]
    c_cols = [
        'sampen', 'spec_ent', 'perm_ent', 'svd_ent',
        'higuchi_fd', 'petrosian_fd', 'katz_fd',
        'lzc', 'dfa'
    ]
    d_cols = [
        'coh_delta', 'coh_theta', 'coh_alpha', 'coh_beta', 'coh_gamma',
        'pearson_corr',
        'asym_power_delta', 'asym_power_theta', 'asym_power_alpha', 'asym_power_beta', 'asym_power_gamma'
    ]
    
    kpi_cols = []
    for col in a_cols + b_cols + c_cols:
        kpi_cols.append(f'Ch1_{col}')
    for col in a_cols + b_cols + c_cols:
        kpi_cols.append(f'Ch2_{col}')
    for col in d_cols:
        kpi_cols.append(f'Cross_{col}')
    
    return kpi_cols


# ===== Column Ordering =====
def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order columns: Subject, Condition, Trial_No → Cross_* → Ch1_* → Ch2_*.
    """
    metadata_cols = ['Subject', 'Condition', 'Trial_No']
    cross_cols = sorted([c for c in df.columns if c.startswith('Cross_')])
    ch1_cols = sorted([c for c in df.columns if c.startswith('Ch1_')])
    ch2_cols = sorted([c for c in df.columns if c.startswith('Ch2_')])
    
    ordered_cols = metadata_cols + cross_cols + ch1_cols + ch2_cols
    return df[ordered_cols]


# ===== Main =====
def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("EEG KPI Analysis Pipeline - START")
    logger.info("=" * 80)
    
    # Load config
    cfg = load_config()
    
    # Create output directory
    output_dir = cfg.PATHS.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Scan files
    data_dir = cfg.PATHS.data_dir
    file_list = scan_raw_data(data_dir)
    
    if not file_list:
        logger.error("No files found to process")
        return
    
    # ===== Parallel Processing =====
    logger.info(f"Processing {len(file_list)} files with all CPU cores...")
    
    results = Parallel(n_jobs=-1)(
        delayed(process_file_wrapper)(file_info, cfg)
        for file_info in tqdm(file_list, desc="Processing files")
    )
    
    # ===== Create DataFrame =====
    df_results = pd.DataFrame(results)
    logger.info(f"Processed {len(df_results)} files. Total columns: {len(df_results.columns)}")
    
    # ===== Order Columns =====
    df_results = order_columns(df_results)
    
    # ===== Save CSV =====
    output_csv = os.path.join(output_dir, "eeg_kpi_analysis_results.csv")
    df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"✓ CSV saved: {output_csv}")
    
    # ===== Generate Summary Report =====
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EEG KPI Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files processed: {len(df_results)}\n")
        f.write(f"Total KPI columns: {len(df_results.columns) - 3}\n")  # Subtract metadata
        f.write(f"\nColumns per channel:\n")
        f.write(f"  - Time-Domain (A): 17\n")
        f.write(f"  - Frequency-Domain (B): 20\n")
        f.write(f"  - Nonlinear (C): 9\n")
        f.write(f"  - Ch1 Total: 46\n")
        f.write(f"  - Ch2 Total: 46\n")
        f.write(f"  - Cross-Channel (D): 11\n")
        f.write(f"  - Grand Total: 103\n")
        f.write(f"\nFirst few rows:\n")
        f.write(df_results.head().to_string())
    logger.info(f"✓ Summary saved: {summary_file}")
    
    logger.info("=" * 80)
    logger.info("✅ Pipeline Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()