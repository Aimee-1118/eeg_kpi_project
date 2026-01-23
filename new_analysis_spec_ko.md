# EEG KPI 재구축 명세서 (Korean)

## 1. 프로젝트 개요
- 목적: 긍정 환경(좋은 음악 + 교회) vs 부정 환경(싫은 음악 + 지하주차장) 청취 시 EEG를 비교하여 SPSS용 CSV를 생성.
- 입력: 2채널 EEG 텍스트 파일(`raw_data/*.txt`), 파일명 `[Subject]_[G/B]_[NNN].txt`.
- 출력: `output/eeg_kpi_analysis_results.csv`(utf-8-sig), `analysis.log`, `analysis_summary.txt`.

## 2. 디렉터리 구조
```
project-root/
├── raw_data/                           # 모든 .txt 세션을 평평하게 저장
├── output/                             # 결과/로그 (없으면 생성)
│   ├── eeg_kpi_analysis_results.csv
│   ├── analysis.log
│   └── analysis_summary.txt
├── configs/
│   ├── base_config.yaml                # 레거시 (참조용)
│   └── analysis_config.yaml            # 신규 활성 설정 (OmegaConf)
├── core_pipeline/                      # 파이프라인 모듈 (재작성 예정)
├── features/                           # KPI 구현 모듈 (재작성 예정)
├── main.py                             # 신규 파이프라인 엔트리포인트
├── validate_kpi.py                     # 선택적 검증 유틸
└── new_analysis_spec.md                # 영문 스펙
```

## 3. KPI 추출 명세

총 예상 KPI: 파일당 약 **100+ 특징** (평탄화, 클린 에포크 평균).

### 3.1 A. 시간축 특징 (`features_A.py`) [완료]
- **통계** (6): mean, std, variance, median, skewness, kurtosis
- **진폭** (5): max, min, p2p, mean, rms
- **패턴** (4): ZCR (영점 교차율), slope_mean, peak_count, peak_mean_height
- **Hjorth** (2): mobility, complexity
- **채널당 총 17개 특징**

### 3.2 B. 주파수축 특징 (`features_B.py`) [완료]
- **파워** (11): total_power, abs_delta–gamma (5), rel_delta–gamma (5)
- **스펙트럼 형태** (5): peak_freq_hz, centroid_hz, sef90_hz, spec_entropy, spec_flatness
- **FOOOF** (2): aperiodic_exponent, aperiodic_offset
- **비율** (2): alpha_beta_ratio, theta_beta_ratio
- **채널당 총 20개 특징**

### 3.3 C. 비선형/동적 특징 (`features_C.py`) [계획 중]
- **엔트로피** (4): Sample Entropy, Spectral Entropy (antropy via), Permutation Entropy, SVD Entropy
- **복잡도** (3): Higuchi Fractal Dimension, Petrosian FD, Katz FD
- **동역학** (2): Lempel-Ziv Complexity (LZC), Detrended Fluctuation Analysis (DFA)
- **채널당 총 ~9개 특징**

### 3.4 D. 채널 간 연결성 & 비대칭성 (`features_D.py`) [계획 중]
- **연결성** (6): Coherence (delta, theta, alpha, beta, gamma) + Pearson Correlation
- **비대칭성** (5): Power Asymmetry (5 대역: delta–gamma)
- **채널 간 총 ~11개 특징**

**최종 합계**: ~57개 채널당 특징 (A+B+C) + ~11개 채널 간 특징 (D) = **~79개 특징** (다중 대역 변형으로 추가 가능).

## 4. 데이터 & 설정
- 파일명 규칙: `[Subject]_[G/B]_[NNN].txt` (G/B는 대소문자 무시). 예: `한석_G_001.txt`, `나현_B_007.txt`.
- 파일 내용: CSV 헤더 `Timestamp(HH:mm:ss.SSS),Ch1(uV),Ch2(uV)`; 2채널; 기준 SR 250 Hz.
- Condition 매핑: `_G_` → 1(긍정), `_B_` → 2(부정).
- Trial_No: 파일명 번호 `NNN` 사용.
- 설정 파일: `configs/analysis_config.yaml` (OmegaConf) + 검증.
  - PATHS: `data_dir`, `output_dir`, `log_file`.
  - PREPROCESSING: `sr=250`, `filter_band=[0.5, 50.0]`, `notch_freq=60.0`, `artifact_threshold_uv=150.0`.
  - EPOCH: `window_sec=4.0`, `overlap_sec=2.0`.
  - BANDS: Delta 0.5-4, Theta 4-8, Alpha 8-13, Beta 13-30, Gamma 30-50.
  - KPI_SELECT: KPI 그룹(A/B/C/D) 리스트, core/optional 표시.
- 우선순위: CLI 인자 > config 파일 > 코드 기본값 없음.
- 설정 검증: 값 불일치(예: low >= high, overlap >= window) 시 시작 전에 즉시 실패(fail fast).

## 5. 파이프라인 로직 (Spec-First)
1) `data_dir`에서 패턴 매칭 파일 로드; 잘못된 파일명은 경고 후 스킵.
2) 파일명에서 subject/condition/trial 파싱(대소문자 무시); 화이트리스트 외 새 subject는 경고만.
3) CSV 로드(Timestamp, Ch1, Ch2); 타임스탬프로 실제 SR 계산, 250 Hz ±10% 벗어나면 경고.
4) 길이 체크: 전체 길이 < 10초면 모든 KPI NaN, `[Short_File_Error]` 로그.
5) 채널별 전처리: 노치 60 Hz → 밴드패스 0.5–50 Hz.
6) 에포킹: 4초 윈도우, 50% 오버랩; peak-to-peak > 150 µV인 에포크 드롭.
7) 클린 에포크 < 3개면 모든 KPI NaN, `[Insufficient_Epochs]` 로그.
8) 채널 평탄(std < 1e-6) 시 해당 채널 KPI NaN, `[Channel_Flat_Warning]` 로그.
9) KPI 계산(클린 에포크 평균 기반):
   - 형태학(A): amp max/min/mean/rms, zcr, slope, hjorth mobility/complexity, skewness, kurtosis.
   - 스펙트럼(B): 대역 파워(Delta~Gamma), SEF90, center freq, spectral entropy, 비율(TBR, Engagement, DAR).
   - 동적/비선형(C): Alpha burst, SampEn, Higuchi FD, LZC, coherence/PLV(Ch1-Ch2).
   - Asymmetry: ln(Ch2_Alpha) - ln(Ch1_Alpha).
   - Connectivity: Ch1-Ch2 coherence (대역별 적용 가능).
10) 에포크 집계: 클린 에포크 KPI 평균 → 파일당 1행.
11) KPI 오류 처리: core 실패 시 해당 KPI만 NaN; optional 실패 시 NaN; 모든 core(모든 밴드×양 채널)가 NaN일 때만 행 Drop.
12) 컬럼 순서/정밀도: 메타데이터 → 크로스채널 → Ch1_* → Ch2_* (각 알파벳순); 소수점 6자리.

## 6. 검증 & 에러 처리
- 파일명 파싱 실패: `[Skipped_Invalid_Filename]` 경고 후 스킵.
- Subject: 화이트리스트(한석, 나현, 기윤) 외는 `[New_Subject_Detected]` 경고 후 포함.
- 길이 <10초 → 전부 NaN; `[Short_File_Error]`.
- 클린 에포크 <3 → 전부 NaN; `[Insufficient_Epochs]`.
- SR 편차 >±10% → 경고 후 진행.
- 채널 평탄 → 해당 채널 KPI만 NaN.
- Core KPI 정책: 밴드/채널 단위 실패는 컬럼 NaN; 양 채널 모든 core가 NaN일 때만 행 Drop.
- 로깅: `analysis.log`에 파일/subject/condition/trial/이벤트/사유를 구조적으로 기록.

## 7. 구현 계획

### Phase 5: 특징 추출 & 통합 (진행 중)

**Step 4: 시간축 특징** `[✅ 완료]`
- `features/features_A.py` 구현: `compute_time_features(data, sr) → Dict`
- 17개 KPI: 진폭, 통계, 패턴, Hjorth 파라미터.
- 합성 정현파 + 노이즈로 테스트.

**Step 5: 주파수축 특징** `[✅ 완료]`
- `features/features_B.py` 구현: `compute_freq_features(data, sr) → Dict`
- 20개 KPI: 파워, 스펙트럼 형태, FOOOF 비주기성, 밴드 비율.
- 합성 신호로 테스트; 대역 파워 분포 검증.

**Step 6: 비선형/동적 특징** `[→ 다음]`
- `features/features_C.py` 구현: `compute_nonlinear_features(data, sr) → Dict`
- 9개 KPI: 엔트로피 (샘플, 스펙트럼, 순열, SVD), 복잡도 (Higuchi, Petrosian, Katz), 동역학 (LZC, DFA).
- antropy 라이브러리 사용; scipy를 이용한 프랙탈 차원.
- 테스트 신호로 단위 테스트; NaN 처리 검증.

**Step 7: 채널 간 연결성 & 비대칭성** `[→ 다음]`
- `features/features_D.py` 구현: `compute_cross_channel_features(data_ch1, data_ch2, sr) → Dict`
- 11개 KPI: 코히어런스 (5 대역), 상관관계, 파워 비대칭 (5 대역).
- scipy.signal.coherence 사용; 비대칭성은 수동 계산.
- 쌍을 이룬 신호 (위상 잠금 vs 직교)로 단위 테스트.

**Step 8: 특징 추출기 리팩토링** `[→ 보류 중]`
- `core/feature_extractor.py` 재작성 (오케스트레이터):
  - `compute_time_features()`, `compute_freq_features()`, `compute_nonlinear_features()`, `compute_cross_channel_features()` 호출
  - 클린 에포크 평균 → 파일당 1행.
  - core/optional 정책에 따른 NaN 전파 처리.
- joblib 병렬화 통합: 파일 수준 병렬 (`n_jobs=-1`), 에포크 수준 순차.

**Step 9: 통합 & 검증** `[→ 보류 중]`
- `main.py` 리팩토링된 `feature_extractor.py` 사용하도록 업데이트.
- 배치 처리에 joblib 병렬화 연결.
- 샘플 raw_data 파일로 전체 파이프라인 테스트.
- CSV, 로그, 요약 보고서 생성.
- 출력 컬럼, 정밀도, NaN 처리 검증.

**Step 10: 엣지 케이스 & 보고** `[→ 보류 중]`
- 다양한 샘플 (짧은 파일, 노이즈, 깨끗한 파일) 드라이런.
- `analysis.log` 및 `analysis_summary.txt` 명확성 검토.
- 필요 시 아티팩트 임계값 조정.
- SPSS 임포트 테스트로 최종 검증.

---

### 레거시 마이크로 스텝 (인프라, 대부분 완료)
1. ✅ `configs/analysis_config.yaml` 생성
2. ✅ 설정 로더/검증기 구현 (OmegaConf)
3. ✅ 파일명 파서 구현 (subject/condition/trial)
4. ✅ CSV 로더 구현
5. ✅ 필터 구현 (노치, 밴드패스)
6. ✅ 에포킹 & 아티팩트 제거 구현
7. ✅ 채널 평탄 감지 구현
8. → 진행 중: 핵심 KPI 계산기 (A, B 완료; C, D 다음)
9. → 보류 중: KPI 집계 & 에러 처리 (C, D 후)
10. → 보류 중: 컬럼 순서 & 네이밍
11. → 보류 중: CSV 라이터 & 요약 보고서
12. → 보류 중: `analysis.log` 로깅
13. → 보류 중: main.py 오케스트레이션
14. → 보류 중: 단위/스모크 테스트
15. → 보류 중: 드라이런 & 튜닝
