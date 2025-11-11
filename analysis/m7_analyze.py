# 📜 analysis/m7_analyze.py
# 🔬 [모듈 7] KPI 테이블 분석 및 Metrics 생성
# (readme.md의 "최종 분석 가이드"를 기반으로 MLflow 로깅용 지표를 계산)

import pandas as pd
import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, Any

def run_analysis(df: pd.DataFrame, cfg: DictConfig) -> Dict[str, Any]:
    """
    M5에서 생성된 최종 KPI DataFrame을 입력받아,
    readme.md 가이드라인에 따라 머신러닝 분석을 수행하고,
    MLflow에 로깅할 Metrics 딕셔너리를 반환합니다.

    - 결측치(NaN/Inf)가 포함된 행(Epoch)은 통계 왜곡을 막기 위해 제거합니다.
    - StandardScaler로 표준화를 수행합니다 (데이터 누수 방지 위해 Pipeline 사용).
    - GroupKFold(groups=df['source_file'])를 사용하여 피험자 독립성을 보장합니다.
    - LassoCV (L1 규제)를 사용하여 환경 분류(church=1 vs market=2)에
      유의미한 KPI를 선별(Feature Selection)하고 교차 검증 점수를 계산합니다.

    Args:
        df (pd.DataFrame): M5에서 생성된 KPI 테이블 (final_kpi_df)
        cfg (DictConfig): OmegaConf 설정 객체

    Returns:
        Dict[str, Any]: MLflow에 로깅할 지표(metrics) 딕셔너리
    """
    
    print(f"[M7] KPI 테이블 분석 및 Metrics 계산 시작...")
    metrics = {}
    
    try:
        # --- 1. (가이드 1) 결측치(NaN/Inf) 처리 ---
        # 0이나 평균으로 대체하지 않고, 해당 행(Epoch)을 제거
        initial_rows = len(df)
        # Inf 값을 NaN으로 먼저 변환
        df_clean = df.replace([np.inf, -np.inf], np.nan)
        # NaN 값을 포함한 모든 행 제거
        df_clean = df_clean.dropna()
        final_rows = len(df_clean)
        
        metrics['analysis_initial_rows'] = initial_rows
        metrics['analysis_rows_after_nan_drop'] = final_rows
        metrics['analysis_rows_dropped_ratio'] = (initial_rows - final_rows) / (initial_rows + 1e-10)

        if final_rows < 50: # (임계값, 예: 50개)
            print(f"[M7-WARN] 유효 데이터(Epoch)가 {final_rows}개로 너무 적어 분석을 건너뜁니다.")
            metrics['analysis_status'] = "skipped_insufficient_data"
            return metrics

        # --- 2. X (특징), y (라벨), groups (파일) 분리 ---
        
        # y: 라벨 (1 또는 2)
        y = df_clean['label']
        
        # groups: (가이드 3) 피험자 독립성을 위한 파일 이름
        groups = df_clean['source_file']
        
        # X: KPI 특징들 (라벨, ID, 파일명 등 비-특징 열 제외)
        # (errors='ignore'는 혹시 열이 없더라도 오류를 내지 않음)
        X = df_clean.drop(columns=['label', 'epoch_id', 'source_file'], errors='ignore') 
        
        # (중요) 라벨이 2개(church, market)인지 확인
        if y.nunique() < 2:
            print(f"[M7-WARN] 라벨이 1개 종류({y.unique()})만 존재하여 분류 분석을 건너뜁니다.")
            metrics['analysis_status'] = "skipped_single_class"
            return metrics

        # --- 3. (가이드 2 & 4) Scikit-learn 파이프라인 및 GroupKFold 설정 ---
        
        # (가이드 2) 데이터 누수 방지를 위해 Pipeline 내에서 Scaler 사용
        # (가이드 4) LassoCV로 자동 피처 선별
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', LassoCV(
                # (1 vs 2 라벨이므로) 1.5를 기준으로 분류하는 회귀 모델 사용
                cv=5, # Lasso 내부의 최적 alpha 탐색용 CV
                random_state=cfg.get('ICA_RANDOM_STATE', 97), # config에서 랜덤 시드 가져오기
                max_iter=3000, # 수렴을 위한 반복 횟수 증가
                n_jobs=-1 # 모든 CPU 사용
            ))
        ])
        
        # (가이드 3) 피험자(파일) 독립적 교차 검증
        # (최소 2개, 최대 5개 또는 파일 개수만큼 KFold 수행)
        n_splits = min(max(2, groups.nunique()), 5)
        gkf = GroupKFold(n_splits=n_splits)
        
        f1_scores = []
        acc_scores = []

        print(f"[M7] GroupKFold (n_splits={n_splits}) 교차 검증 시작...")
        
        for train_idx, test_idx in gkf.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            pipeline.fit(X_train, y_train)
            
            # Lasso는 회귀 모델이므로 예측값(예: 1.1, 1.9)을 라벨(1, 2)로 변환
            preds_float = pipeline.predict(X_test)
            # 1.5를 기준으로 1(church)과 2(market)로 변환
            preds_binary_label = [2 if p > 1.5 else 1 for p in preds_float]
            
            f1_scores.append(f1_score(y_test, preds_binary_label, average='weighted', zero_division=0))
            acc_scores.append(accuracy_score(y_test, preds_binary_label))

        # --- 4. 최종 Metrics 계산 ---
        metrics['analysis_cv_f1_mean'] = np.mean(f1_scores)
        metrics['analysis_cv_f1_std'] = np.std(f1_scores)
        metrics['analysis_cv_accuracy_mean'] = np.mean(acc_scores)
        metrics['analysis_cv_accuracy_std'] = np.std(acc_scores)

        print(f"[M7] CV F1-Score (Mean): {metrics['analysis_cv_f1_mean']:.4f}")

        # --- 5. (가이드 4) 최종 모델 피처 선별 ---
        # 전체 데이터로 파이프라인 재학습 (최적의 피처 찾기 위해)
        pipeline.fit(X, y) 
        
        # 학습된 Lasso 모델 접근
        lasso_model = pipeline.named_steps['lasso']
        
        # (Lasso가 0이 아닌 가중치를 부여한) 중요 피처 추출
        importances = np.abs(lasso_model.coef_)
        selected_features_mask = importances > 1e-5 # (0에 매우 가까운 값 제외)
        n_selected = np.sum(selected_features_mask)
        
        metrics['analysis_lasso_features_selected'] = int(n_selected)
        metrics['analysis_lasso_total_features'] = len(importances)
        
        print(f"[M7] Lasso 선별 피처 개수: {n_selected} / {len(importances)}")

        # (선택) MLflow에 상위 5개 피처 이름 로깅 (log_param은 문자열 250자 제한, log_text는 무제한)
        try:
            top_5_indices = np.argsort(importances)[-5:][::-1]
            top_5_features = X.columns[top_5_indices].tolist()
            # (MLflow에서는 log_text를 사용해야 함. 여기서는 print로 대체)
            print(f"[M7] Top 5 Features: {top_5_features}")
            # (main.py에서 mlflow.log_text(..., "top_features.txt")를 호출하거나
            #  metrics 딕셔너리에 문자열로 추가 - 단, 250자 제한 주의)
            metrics['analysis_top_1_feature'] = top_5_features[0] if n_selected > 0 else "None"
            metrics['analysis_top_2_feature'] = top_5_features[1] if n_selected > 1 else "None"
            
        except Exception as e:
            print(f"[M7-WARN] Top 피처 이름 저장 중 오류: {e}")

        metrics['analysis_status'] = "completed"

    except Exception as e:
        print(f"[ERROR M7] KPI 분석 중 심각한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        metrics['analysis_status'] = "failed"
        
    return metrics