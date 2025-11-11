# 📜 main.py
# 💥 이 파일 하나만 실행하면 전체 파이프라인이 작동합니다.
# (🔥 MLflow 실험 로깅 기능이 통합됨)

import os      # (🔥 신규) PYTHONHASHSEED 고정을 위해
import random  # (🔥 신규) Python 기본 random 시드 고정을 위해
import numpy as np # (🔥 신규) NumPy 시드 고정을 위해
import time
import argparse  # 터미널 인자 파싱
import os      # MLflow URI 설정을 위해
import tempfile # Artifact 저장을 위한 임시 폴더
from omegaconf import OmegaConf, DictConfig # YAML 및 인자 병합
import mlflow  # (🔥 신규) MLOps 실험 로깅

# 🏭 핵심 파이프라인 함수
from core_pipeline.run_pipeline import run_full_pipeline

def main():
    """
    메인 실행 함수:
    1. (🔥 수정) Argparse와 OmegaConf를 사용해 설정을 로드합니다.
    2. (🔥 신규) MLflow 실험(Run)을 시작합니다.
    3. (🔥 신규) 병합된 최종 Config를 MLflow에 로깅합니다.
    4. core_pipeline의 run_full_pipeline 함수를 호출합니다.
       (반환값: final_kpi_df, metrics)
    5. (🔥 신규) 반환된 Metrics와 Parquet 파일을 MLflow에 로깅합니다.
    6. 완료 메시지 및 실행 시간을 출력합니다.
    """

    # --- 1. 설정 로드 (Argparse + OmegaConf) ---
    parser = argparse.ArgumentParser(description="EEG KPI Extraction Pipeline")
    parser.add_argument(
        '-c', '--config_path',
        type=str,
        default='./configs/base_config.yaml',
        help="Path to the base YAML config file."
    )
    args, unknown_args = parser.parse_known_args()

    # --- 2. 기본 YAML 설정 로드 ---
    try:
        base_cfg = OmegaConf.load(args.config_path)
    except FileNotFoundError:
        print(f"❌ 기본 설정 파일({args.config_path})을 찾을 수 없습니다.")
        return

    # --- 3. 터미널 인자(override) 로드 ---
    cli_cfg = OmegaConf.from_cli(unknown_args)

    # --- 4. 설정 병합 (터미널 인자가 YAML보다 우선함) ---
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    # --- 4-B. (🔥 신규) 재현성을 위한 글로벌 시드 고정 ---
    # (Config 로드 직후, 다른 모든 작업 시작 전)
    try:
        seed = cfg.GLOBAL_RANDOM_SEED
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"🧬 [INFO] Global random seed-를 {seed}로 고정합니다.")

        # (주석: 향후 PyTorch 사용 시)
        # try:
        #     import torch
        #     torch.manual_seed(seed)
        #     if torch.cuda.is_available():
        #         torch.cuda.manual_seed_all(seed) # if use multi-GPU
        #         # (🔥 신규) Deterministic 연산 플래그
        #         torch.use_deterministic_algorithms(True)
        #         torch.backends.cudnn.deterministic = True
        #         torch.backends.cudnn.benchmark = False
        # except ImportError:
        #     pass # PyTorch가 설치되지 않음

    except Exception as e:
        print(f"[WARN] 시드 고정 중 오류 발생 (config에 GLOBAL_RANDOM_SEED가 없는지 확인): {e}")
    # --- 5. (🔥 신규) MLflow 설정 및 실험 시작 ---
    # (프로젝트 루트에 'mlruns' 폴더를 생성하여 로그 저장)
    mlflow.set_tracking_uri(f"file:{os.path.abspath('mlruns')}")
    
    # config.yaml의 EXPERIMENT_NAME 값을 사용
    # (값이 없으면 'EEG_KPI_Analysis'를 기본값으로 사용)
    experiment_name = cfg.get("EXPERIMENT_NAME", "EEG_KPI_Analysis")
    mlflow.set_experiment(experiment_name)

    # MLflow 실험(Run) 시작
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"🚀 MLflow 실험 시작. Run ID: {run_id}")
        
        # --- 6. (🔥 신규) Config 로깅 ---
        # OmegaConf 객체를 딕셔너리로 변환하여 로깅
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            # MLflow는 중첩 딕셔너리를 'BANDS.Delta'처럼 자동으로 펼쳐서 저장
            mlflow.log_params(cfg_dict)
            print(f"    MLflow: Config 파라미터 로깅 완료.")
        except Exception as e:
            print(f"[WARN] MLflow Config 로깅 중 오류 발생: {e}")

        
        # --- 7. (이하 기존) 파이프라인 실행 ---
        print("="*70)
        print("🧠 EEG KPI 추출 파이프라인을 시작합니다.")
        print(f"▶️ 기본 설정 파일: {args.config_path}")
        if unknown_args:
            print(f"▶️ 런타임 설정 (Override): {unknown_args}")
        print(f"▶️ MLflow 실험명: {experiment_name}")
        print("="*70)

        start_time = time.time()  # 시작 시간 기록

        try:
            # (🔥 수정) run_full_pipeline이 (df, metrics)를 반환
            final_kpi_df, metrics = run_full_pipeline(cfg=cfg)

            if final_kpi_df is None:
                print("\n[INFO] 처리된 유효 데이터가 없습니다. 파이프라인을 종료합니다.")
                mlflow.log_param("status", "no_valid_data")
                return

            end_time = time.time()  # 종료 시간 기록
            total_time = end_time - start_time

            # --- 8. (🔥 신규) Metrics 로깅 (metrics.json 대체) ---
            print(f"    MLflow: Metrics 로깅 중...")
            if metrics:
                mlflow.log_metrics(metrics)
            
            # 파이프라인 기본 지표 로깅
            mlflow.log_metric("pipeline_duration_sec", total_time)
            mlflow.log_metric("total_epochs_processed", len(final_kpi_df))
            mlflow.log_metric("total_kpis_generated", len(final_kpi_df.columns))

            # --- 9. (🔥 신규) Artifact (Parquet) 로깅 (features.parquet 대체) ---
            print(f"    MLflow: Artifact (features.parquet) 로깅 중...")
            with tempfile.TemporaryDirectory() as tmpdir:
                # 'features.parquet'라는 이름으로 임시 폴더에 저장
                parquet_path = os.path.join(tmpdir, "features.parquet")
                
                # (와이드 포맷 Parquet 파일로 저장)
                final_kpi_df.to_parquet(parquet_path, index=False)
                
                # MLflow에 "features"라는 하위 폴더 이름으로 아티팩트 저장
                mlflow.log_artifact(parquet_path, artifact_path="features")

            print("\n" + "="*70)
            print(f"✅ 파이프라인이 성공적으로 완료되었습니다.")
            print(f"⏱️ 총 실행 시간: {total_time:.2f} 초")
            print(f"📊 MLflow UI에서 Run ID '{run_id}'를 확인하세요.")
            print("="*70)

        except Exception as e:
            print("\n" + "!"*70)
            print(f"❌ 오류가 발생하여 파이프라인이 중단되었습니다.")
            print(f"오류 상세: {e}")
            mlflow.log_param("status", "pipeline_failed")
            mlflow.log_text(str(e), "error_details.txt")
            import traceback
            traceback.print_exc()
            print("!"*70)


if __name__ == "__main__":
    # 이 파일이 직접 실행되었을 때만 main() 함수를 호출합니다.
    main()