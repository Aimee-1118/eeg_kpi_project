# EEG KPI Rebuild Spec

## 1. Project Overview
- Goal: Rebuild EEG processing to compare positive vs negative listening (Condition 1 = Good music + church, Condition 2 = Bad music + parking lot) and output SPSS-ready CSV.
- Input: 2-channel EEG text files (`raw_data/*.txt`) per session; naming `[Subject]_[G/B]_[NNN].txt`.
- Output: `output/eeg_kpi_analysis_results.csv` (utf-8-sig), plus `analysis.log` and `analysis_summary.txt`.

## 2. Directory Structure
```
project-root/
├── raw_data/                           # flat folder with all .txt sessions
├── output/                             # results + logs (create if absent)
│   ├── eeg_kpi_analysis_results.csv
│   ├── analysis.log
│   └── analysis_summary.txt
├── configs/
│   ├── base_config.yaml                # legacy (reference only)
│   └── analysis_config.yaml            # new active config (OmegaConf)
├── core_pipeline/                      # modules (to be rebuilt)
├── features/                           # KPI feature implementations (to be rebuilt)
├── main.py                             # entrypoint using new pipeline
├── validate_kpi.py                     # optional validation utilities
└── new_analysis_spec.md                # this spec
```

## 3. Feature Extraction Specs

Total estimated KPIs: approx. **100+ features** per file (flattened, averaged across clean epochs).

### 3.1 A. Time-Domain (`features_A.py`) [Complete]
- **Stats** (6): mean, std, variance, median, skewness, kurtosis
- **Amplitude** (5): max, min, p2p, mean, rms
- **Pattern** (4): ZCR (zero-crossing rate), slope_mean, peak_count, peak_mean_height
- **Hjorth** (2): mobility, complexity
- **Total per channel: 17 features**

### 3.2 B. Frequency-Domain (`features_B.py`) [Complete]
- **Powers** (11): total_power, abs_delta–gamma (5), rel_delta–gamma (5)
- **Spectral Shape** (5): peak_freq_hz, centroid_hz, sef90_hz, spec_entropy, spec_flatness
- **FOOOF** (2): aperiodic_exponent, aperiodic_offset
- **Ratios** (2): alpha_beta_ratio, theta_beta_ratio
- **Total per channel: 20 features**

### 3.3 C. Nonlinear/Dynamics (`features_C.py`) [Planned]
- **Entropy** (4): Sample Entropy, Spectral Entropy (via antropy), Permutation Entropy, SVD Entropy
- **Complexity** (3): Higuchi Fractal Dimension, Petrosian FD, Katz FD
- **Dynamics** (2): Lempel-Ziv Complexity (LZC), Detrended Fluctuation Analysis (DFA)
- **Total per channel: ~9 features**

### 3.4 D. Cross-Channel Connectivity & Asymmetry (`features_D.py`) [Planned]
- **Connectivity** (6): Coherence (delta, theta, alpha, beta, gamma) + Pearson Correlation
- **Asymmetry** (5): Power Asymmetry (5 bands: delta–gamma)
- **Total cross-channel: ~11 features**

**Grand Total**: ~57 per-channel features (A+B+C) + ~11 cross-channel features (D) = **~79 features** (further expandable with multi-band variants).

## 4. Data & Config
- File naming: `[Subject]_[G/B]_[NNN].txt` (case-insensitive for G/B). Examples: `한석_G_001.txt`, `나현_B_007.txt`.
- Contents: CSV with header `Timestamp(HH:mm:ss.SSS),Ch1(uV),Ch2(uV)`; 2 channels; nominal SR 250 Hz.
- Condition mapping: `_G_` → 1 (positive), `_B_` → 2 (negative).
- Trial_No: zero-padded number from filename (`NNN`).
- Config file: `configs/analysis_config.yaml` using OmegaConf with validation.
  - PATHS: `data_dir`, `output_dir`, `log_file`.
  - PREPROCESSING: `sr=250`, `filter_band=[0.5, 50.0]`, `notch_freq=60.0`, `artifact_threshold_uv=150.0`.
  - EPOCH: `window_sec=4.0`, `overlap_sec=2.0`.
  - BANDS: Delta 0.5-4, Theta 4-8, Alpha 8-13, Beta 13-30, Gamma 30-50.
  - KPI_SELECT: list of KPI groups (A/B/C/D) to compute; mark core vs optional.
- Priority: CLI args > config file > no hardcoded defaults.
- Validation: fail fast if config inconsistent (e.g., low >= high, overlap >= window).

## 5. Pipeline Logic (Spec-First)
1) Load files from `data_dir` matching pattern; skip invalid filenames with warning.
2) Parse subject, condition, trial from filename (case-insensitive), allow new subjects with warning.
3) Load CSV (Timestamp, Ch1, Ch2). Compute observed SR from timestamps; warn if outside 250 Hz ±10%.
4) Length check: if total duration < 10 s → mark all KPI as NaN, log `[Short_File_Error]`.
5) Preprocess per channel:
   - Notch filter 60 Hz.
   - Bandpass 0.5–50 Hz.
6) Epoching: 4 s windows, 50% overlap. For each epoch, drop if peak-to-peak > 150 µV.
7) After artifact rejection: if clean epochs < 3 → all KPI NaN, log `[Insufficient_Epochs]`.
8) Per-channel checks: if std < 1e-6 or flat → mark that channel’s KPIs NaN; log `[Channel_Flat_Warning]`.
9) KPI computation on remaining clean epochs:
   - Morphological (A): amp max/min/mean/rms, zcr, slope, hjorth mobility/complexity, skewness, kurtosis.
   - Spectral (B): band powers (Delta…Gamma), SEF90, center freq, spectral entropy, ratios (TBR, Engagement, DAR).
   - Dynamic/Nonlinear (C): alpha burst, SampEn, Higuchi FD, LZC, coherence/PLV (Ch1-Ch2).
   - Asymmetry: ln(Ch2_Alpha) - ln(Ch1_Alpha).
   - Connectivity: coherence Ch1-Ch2 (per band where applicable).
10) Epoch aggregation: average KPI across clean epochs → one row per file.
11) Error handling during KPI calc: core failure → NaN for that KPI (not full drop); optional failure → NaN. Only drop whole row if all core KPIs for all channels are NaN.
12) Output row columns: metadata then cross-channel then per-channel KPIs. Numeric precision: 6 decimals.

## 6. Validation & Error Handling
- File parsing: skip invalid names with `[Skipped_Invalid_Filename]` warning.
- Subjects: whitelist (한석, 나현, 기윤); new names allowed with `[New_Subject_Detected]` warning.
- Duration <10s → all NaN; log `[Short_File_Error]`.
- Clean epochs <3 → all NaN; log `[Insufficient_Epochs]`.
- SR off by >±10% → warn, continue.
- Channel flatline → channel KPIs NaN; continue.
- Core KPI failure policy:
  - Per-band per-channel failures result in column-wise NaN.
  - Drop entire row only if all core KPIs (all bands × both channels) are NaN.
- Logging: write to `analysis.log`; include KPI-specific failure reasons and file context.

## 7. Implementation Plan

### Phase 5: Feature Extraction & Integration (Ongoing)

**Step 4: Time-Domain Features** `[✅ Done]`
- Implement `features/features_A.py` with `compute_time_features(data, sr) → Dict`
- 17 KPIs: amplitude, stats, patterns, Hjorth parameters.
- Test with synthetic sine + noise signal.

**Step 5: Frequency-Domain Features** `[✅ Done]`
- Implement `features/features_B.py` with `compute_freq_features(data, sr) → Dict`
- 20 KPIs: powers, spectral shape, FOOOF aperiodic, band ratios.
- Test with synthetic signal; verify band power distributions.

**Step 6: Nonlinear/Dynamics Features** `[→ Next]`
- Implement `features/features_C.py` with `compute_nonlinear_features(data, sr) → Dict`
- 9 KPIs: entropy (sample, spectral, permutation, SVD), complexity (Higuchi, Petrosian, Katz), dynamics (LZC, DFA).
- Use antropy library for entropies; scipy for fractal dimensions.
- Unit test with test signal; validate NaN handling.

**Step 7: Cross-Channel Connectivity & Asymmetry** `[→ Next]`
- Implement `features/features_D.py` with `compute_cross_channel_features(data_ch1, data_ch2, sr) → Dict`
- 11 KPIs: coherence (5 bands), correlation, power asymmetry (5 bands).
- Use scipy.signal.coherence for band-wise coherence; manual calculation for asymmetry.
- Unit test with paired signals (phase-locked vs orthogonal).

**Step 8: Feature Extractor Refactoring** `[→ Pending]`
- Rewrite `core/feature_extractor.py` as orchestrator:
  - Call `compute_time_features()`, `compute_freq_features()`, `compute_nonlinear_features()`, `compute_cross_channel_features()`
  - Average across clean epochs per file → 1 row per file.
  - Handle NaN propagation per core/optional policy.
- Integrate joblib parallelization: file-level parallelism (`n_jobs=-1`), epoch-level sequential.

**Step 9: Integration & Validation** `[→ Pending]`
- Update `main.py` to use refactored `feature_extractor.py`.
- Wire parallelization with joblib in batch processing.
- Full pipeline test on sample raw_data files.
- Generate CSV, logs, summary report.
- Verify output columns, precision, NaN handling.

**Step 10: Edge Cases & Reporting** `[→ Pending]`
- Dry run on diverse samples (short files, noisy, clean).
- Review `analysis.log` and `analysis_summary.txt` for clarity.
- Adjust artifact thresholds if needed.
- Final validation with SPSS import test.

---

### Legacy Micro-Steps (Infrastructure, mostly complete)
1. ✅ Create `configs/analysis_config.yaml`
2. ✅ Implement config loader/validator (OmegaConf)
3. ✅ Implement filename parser (subject/condition/trial)
4. ✅ Build raw loader for CSV
5. ✅ Implement filters (notch, bandpass)
6. ✅ Implement epoching & artifact rejection
7. ✅ Implement channel flatline detection
8. → In Progress: Core KPI calculators (A, B done; C, D next)
9. → Pending: KPI aggregation & error handling (after C, D)
10. → Pending: Column ordering & naming
11. → Pending: CSV writer & summary report
12. → Pending: Logging to `analysis.log`
13. → Pending: main.py orchestration
14. → Pending: Unit/smoke tests
15. → Pending: Dry run & tuning
