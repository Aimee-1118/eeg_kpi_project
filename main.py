# 📜 main.py
# 💥 이 파일 하나만 실행하면 전체 파이프라인이 작동합니다.
# (🔥 MLflow, 시드 고정, Logging 기능이 모두 통합됨)

import os      # PYTHONHASHSEED 고정 및 MLflow URI 설정
import random  # Python 기본 random 시드 고정
import numpy as np # NumPy 시드 고정
import time
import argparse  # 터미널 인자 파싱
import tempfile # Artifact 저장을 위한 임시 폴더
from omegaconf import OmegaConf, DictConfig # YAML 및 인자 병합
import mlflow  # MLOps 실험 로깅
import logging # (🔥 신규) Tqdm과 호환되는 로깅

# 🏭 핵심 파이프라인 함수
from core_pipeline.run_pipeline import run_full_pipeline

# (🔥 신규) main 함수 밖에 로거 설정
# (프로젝트의 모든 모듈이 이 설정을 상속받아 사용)
logger = logging.getLogger(__name__)

def main():
    """
    메인 실행 함수:
    1. (🔥 신규) 로깅(Logging) 기본 설정을 수행합니다.
    2. Argparse와 OmegaConf를 사용해 설정을 로드합니다.
    3. (🔥 신규) 재현성을 위한 글로벌 시드를 고정합니다.
    4. MLflow 실험(Run)을 시작하고 Config를 로깅합니다.
    5. core_pipeline의 run_full_pipeline 함수를 호출합니다.
       (반환값: final_kpi_df, metrics)
    6. 반환된 Metrics와 Parquet 파일을 MLflow에 로깅합니다.
    7. 완료 메시지 및 실행 시간을 로깅합니다.
    """

    # --- 1. (🔥 신규) 로깅(Logging) 중앙 설정 ---
    # (다른 모든 작업보다 먼저 실행)
    logging.basicConfig(
        level=logging.INFO, # INFO 레벨 이상만 출력
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler() # 콘솔(터미널)로 출력
            # (선택) 파일로도 저장하려면 아래 핸들러 주석 해제
            # logging.FileHandler("pipeline.log", mode='w') 
        ]
    )

    # --- 2. 설정 로드 (Argparse + OmegaConf) ---
    parser = argparse.ArgumentParser(description="EEG KPI Extraction Pipeline")
    parser.add_argument(
        '-c', '--config_path',
        type=str,
        default='./configs/base_config.yaml',
        help="Path to the base YAML config file."
    )
    args, unknown_args = parser.parse_known_args()

    # --- 3. 기본 YAML 설정 로드 ---
    try:
        base_cfg = OmegaConf.load(args.config_path)
    except FileNotFoundError:
        # (🔥 수정) print -> logger.error
        logger.error(f"❌ 기본 설정 파일({args.config_path})을 찾을 수 없습니다.")
        return

    # --- 4. 터미널 인자(override) 로드 ---
    cli_cfg = OmegaConf.from_cli(unknown_args)

    # --- 5. 설정 병합 ---
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    
    # --- 6. (🔥 신규) 재현성을 위한 글로벌 시드 고정 ---
    try:
        seed = cfg.GLOBAL_RANDOM_SEED
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        # (🔥 수정) print -> logger.info
        logger.info(f"🧬 [INFO] Global random seed를 {seed}로 고정합니다.")

        # (주석: 향후 PyTorch 사용 시)
        # try:
        #     import torch
        #     torch.manual_seed(seed)
        #     if torch.cuda.is_available():
        #         torch.cuda.manual_seed_all(seed)
        #         torch.use_deterministic_algorithms(True)
        #         torch.backends.cudnn.deterministic = True
        #         torch.backends.cudnn.benchmark = False
        # except ImportError:
        #     pass # PyTorch가 설치되지 않음

    except Exception as e:
        # (🔥 수정) print -> logger.warning
        logger.warning(f"[WARN] 시드 고정 중 오류 발생 (config에 GLOBAL_RANDOM_SEED가 없는지 확인): {e}")

    # --- 7. (🔥 신규) MLflow 설정 및 실험 시작 ---
    mlflow.set_tracking_uri(f"file:{os.path.abspath('mlruns')}")
    experiment_name = cfg.get("EXPERIMENT_NAME", "EEG_KPI_Analysis")
    mlflow.set_experiment(experiment_name)

    # MLflow 실험(Run) 시작
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        # (🔥 수정) print -> logger.info
        logger.info(f"🚀 MLflow 실험 시작. Run ID: {run_id}")
        
        # --- 8. (🔥 신규) Config 로깅 ---
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            mlflow.log_params(cfg_dict)
            # (🔥 수정) print -> logger.info
            logger.info(f"    MLflow: Config 파라미터 로깅 완료.")
        except Exception as e:
            # (🔥 수정) print -> logger.warning
            logger.warning(f"[WARN] MLflow Config 로깅 중 오류 발생: {e}")

        
        # --- 9. 파이프라인 실행 ---
        # (🔥 수정) print -> logger.info
        logger.info("="*70)
        logger.info("🧠 EEG KPI 추출 파이프라인을 시작합니다.")
        logger.info(f"▶️ 기본 설정 파일: {args.config_path}")
        if unknown_args:
            logger.info(f"▶️ 런타임 설정 (Override): {unknown_args}")
        logger.info(f"▶️ MLflow 실험명: {experiment_name}")
        logger.info("="*70)

        start_time = time.time()  # 시작 시간 기록

        try:
            # (🔥 수정) run_full_pipeline이 (df, metrics)를 반환
            final_kpi_df, metrics = run_full_pipeline(cfg=cfg)

            if final_kpi_df is None:
                # (🔥 수정) print -> logger.info
                logger.info("\n[INFO] 처리된 유효 데이터가 없습니다. 파이프라인을 종료합니다.")
                mlflow.log_param("status", "no_valid_data")
                return

            end_time = time.time()  # 종료 시간 기록
            total_time = end_time - start_time

            # --- 10. (🔥 신규) Metrics 로깅 (metrics.json 대체) ---
            # (🔥 수정) print -> logger.info
            logger.info(f"    MLflow: Metrics 로깅 중...")
            if metrics:
                mlflow.log_metrics(metrics)
            
            mlflow.log_metric("pipeline_duration_sec", total_time)
            mlflow.log_metric("total_epochs_processed", len(final_kpi_df))
            mlflow.log_metric("total_kpis_generated", len(final_kpi_df.columns))

            # --- 11. (🔥 신규) Artifact (Parquet) 로깅 (features.parquet 대체) ---
            # (🔥 수정) print -> logger.info
            logger.info(f"    MLflow: Artifact (features.parquet) 로깅 중...")
            with tempfile.TemporaryDirectory() as tmpdir:
                parquet_path = os.path.join(tmpdir, "features.parquet")
                final_kpi_df.to_parquet(parquet_path, index=False)
                mlflow.log_artifact(parquet_path, artifact_path="features")

            # (🔥 수정) print -> logger.info
            logger.info("\n" + "="*70)
            logger.info(f"✅ 파이프라인이 성공적으로 완료되었습니다.")
            logger.info(f"⏱️ 총 실행 시간: {total_time:.2f} 초")
            logger.info(f"📊 MLflow UI에서 Run ID '{run_id}'를 확인하세요.")
            logger.info("="*70)

        except Exception as e:
            # (🔥 수정) print -> logger.critical
            logger.critical("\n" + "!"*70)
            logger.critical(f"❌ 오류가 발생하여 파이프라인이 중단되었습니다.")
            logger.critical(f"오류 상세: {e}")
            mlflow.log_param("status", "pipeline_failed")
            mlflow.log_text(str(e), "error_details.txt")
            import traceback
            # (🔥 수정) traceback.print_exc() -> logger.error()
            logger.error(traceback.format_exc())
            logger.critical("!"*70)


if __name__ == "__main__":
    # 이 파일이 직접 실행되었을 때만 main() 함수를 호출합니다.
    main()