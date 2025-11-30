# 📜 validate_kpi.py
# (🔥 업그레이드: 물리적 타당성, 파일 누락, 중복 검사 추가)

import pandas as pd
import numpy as np
import os
import glob

def validate_kpi_table(
    kpi_path="./results/final_kpi_table.csv", 
    raw_data_dir="./data_raw"
):
    print("="*60)
    print(f"🔬 [심화] KPI 데이터 무결성 및 타당성 검증 시작")
    print("="*60)
    
    if not os.path.exists(kpi_path):
        print("❌ [CRITICAL] 결과 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(kpi_path)
    n_rows, n_cols = df.shape
    
    # --- 1. 기본 구조 확인 ---
    print(f"✅ 데이터 형태: {n_rows} Epochs x {n_cols} Features")
    if n_rows == 0:
        print("❌ [CRITICAL] 데이터가 비어있습니다! (0 rows)")
        return

    # --- 2. 파일 누락(Data Loss) 확인 ---
    # 원본 csv 파일 목록 (재귀적 탐색)
    raw_files = glob.glob(os.path.join(raw_data_dir, "**/*.csv"), recursive=True)
    # 파일명만 추출 (폴더 경로 제거)
    raw_filenames = set(os.path.basename(f) for f in raw_files)
    # KPI 테이블에 있는 파일명
    processed_filenames = set(df['source_file'].unique())
    
    missing_files = raw_filenames - processed_filenames
    
    print(f"\n📁 [파일 처리 현황]")
    print(f"   - 원본 파일 수: {len(raw_filenames)}개")
    print(f"   - 처리된 파일 수: {len(processed_filenames)}개")
    if len(missing_files) > 0:
        print(f"⚠️ [WARN] {len(missing_files)}개 파일이 결과에서 누락되었습니다.")
        print(f"   -> 예: {list(missing_files)[:3]} ...")
    else:
        print("✅ [PASS] 모든 원본 파일이 처리되었습니다.")

    # --- 3. 라벨 및 식별자 검증 ---
    print(f"\n🏷️ [라벨 및 ID 검증]")
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"   - 라벨 분포: {label_counts.to_dict()}")
        if len(label_counts) < 2:
            print("❌ [FAIL] 라벨이 1가지뿐입니다. 분류 모델 학습 불가!")
    else:
        print("❌ [FAIL] 'label' 컬럼 누락.")

    # Epoch ID 중복 검사
    if 'source_file' in df.columns and 'epoch_id' in df.columns:
        duplicates = df.duplicated(subset=['source_file', 'epoch_id']).sum()
        if duplicates > 0:
            print(f"❌ [FAIL] 중복된 Epoch ID가 {duplicates}개 발견되었습니다.")
        else:
            print("✅ [PASS] Epoch ID 중복 없음.")

    # --- 4. 결측치(NaN/Inf) 심층 확인 ---
    print(f"\n🕳️ [결측치 점검]")
    nan_rows = df.isna().any(axis=1).sum()
    inf_rows = np.isinf(df.select_dtypes(include=np.number)).any(axis=1).sum()
    
    if nan_rows > 0:
        print(f"⚠️ [WARN] NaN 포함 행: {nan_rows}개 ({nan_rows/n_rows*100:.1f}%) -> 분석 시 삭제됨")
    if inf_rows > 0:
        print(f"⚠️ [WARN] Inf 포함 행: {inf_rows}개 ({inf_rows/n_rows*100:.1f}%) -> 분석 시 삭제됨")
    
    if nan_rows == 0 and inf_rows == 0:
        print("✅ [PASS] 결측치(NaN/Inf) 완전 없음 (Clean!).")

    # --- 5. 물리적 타당성 검증 (Feature Sanity Check) ---
    print(f"\n🧠 [물리적 타당성 검증]")
    
    # (1) 파워 스펙트럼은 음수일 수 없음
    pow_cols = [c for c in df.columns if '_B_pow_' in c]
    if pow_cols:
        negative_pow = (df[pow_cols] < 0).sum().sum()
        if negative_pow > 0:
            print(f"❌ [FAIL] 스펙트럼 파워(Power)에 음수 값이 {negative_pow}개 있습니다. (계산 로직 오류 가능성)")
        else:
            print("✅ [PASS] 스펙트럼 파워 값 정상 (모두 >= 0)")
    
    # (2) 모든 특징값이 0인 '유령 행' 확인
    feature_cols = df.select_dtypes(include=[np.number]).columns.drop(['label', 'epoch_id'], errors='ignore')
    zeros_rows = (df[feature_cols] == 0).all(axis=1).sum()
    if zeros_rows > 0:
        print(f"⚠️ [WARN] 모든 특징값이 0인 행이 {zeros_rows}개 있습니다. (신호가 없거나 계산 실패)")
    else:
        print("✅ [PASS] 모든 행에 유효한 특징값이 존재함.")

    # (3) 상수 컬럼 (분산 0) 확인
    std_vals = df[feature_cols].std()
    constant_cols = std_vals[std_vals == 0].index.tolist()
    if constant_cols:
        print(f"⚠️ [WARN] 값이 전혀 변하지 않는 특징(상수)이 {len(constant_cols)}개 있습니다.")
        print(f"   -> {constant_cols[:3]} ...")
    else:
        print("✅ [PASS] 모든 특징이 변별력을 가짐 (상수 컬럼 없음).")

    print("\n" + "="*60)
    print("🏁 검증 완료. [FAIL] 항목이 없다면 M7 분석으로 넘어가셔도 좋습니다.")
    print("="*60)

if __name__ == "__main__":
    validate_kpi_table()