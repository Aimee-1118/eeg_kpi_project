from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class ConfigPaths:
    data_dir: str
    output_dir: str
    log_file: str



def load_config(
    config_path: str = "./configs/analysis_config.yaml",
    cli_args: Optional[list[str]] = None,
) -> DictConfig:
    """Load YAML config, merge CLI overrides, and run fail-fast validation."""
    cli_args = cli_args or []
    base_cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    validate_config(cfg)
    return cfg


def load_and_validate_config(
    config_path: str = "./configs/analysis_config.yaml",
    cli_args: Optional[list[str]] = None,
) -> DictConfig:
    """Alias for clarity in main pipeline."""
    return load_config(config_path=config_path, cli_args=cli_args)


def validate_config(cfg: DictConfig) -> None:
    """Fail-fast config validation."""
    # Paths
    assert cfg.PATHS.data_dir, "PATHS.data_dir is required"
    assert cfg.PATHS.output_dir, "PATHS.output_dir is required"
    assert cfg.PATHS.log_file, "PATHS.log_file is required"

    os.makedirs(cfg.PATHS.output_dir, exist_ok=True)
    log_dir = os.path.dirname(cfg.PATHS.log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Preprocessing
    low = cfg.PREPROCESSING.filter_band.low
    high = cfg.PREPROCESSING.filter_band.high
    assert low < high, "PREPROCESSING.filter_band low must be < high"
    assert cfg.PREPROCESSING.sampling_rate > 0, "PREPROCESSING.sampling_rate must be > 0"
    assert cfg.PREPROCESSING.notch_freq > 0, "PREPROCESSING.notch_freq must be > 0"
    assert cfg.PREPROCESSING.artifact_threshold_uv > 0, "artifact_threshold_uv must be > 0"

    # Epoch
    assert cfg.EPOCH.window_sec > 0, "EPOCH.window_sec must be > 0"
    assert cfg.EPOCH.overlap_sec >= 0, "EPOCH.overlap_sec must be >= 0"
    assert cfg.EPOCH.overlap_sec < cfg.EPOCH.window_sec, "EPOCH.overlap_sec must be < EPOCH.window_sec"

    # Bands
    for name, band in cfg.BANDS.items():
        assert hasattr(band, '__len__') and len(band) == 2, f"BANDS.{name} must have [low, high]"
        b_low, b_high = band
        assert b_low < b_high, f"BANDS.{name}: low must be < high"
        assert b_low >= 0, f"BANDS.{name}: low must be >= 0"
        assert b_high <= high, f"BANDS.{name}: high exceeds preprocessing high cutoff"

    # KPI selection
    assert "core" in cfg.KPI_SELECT, "KPI_SELECT.core missing"
    assert "optional" in cfg.KPI_SELECT, "KPI_SELECT.optional missing"
    core_list = OmegaConf.to_container(cfg.KPI_SELECT.core, resolve=True)
    optional_list = OmegaConf.to_container(cfg.KPI_SELECT.optional, resolve=True)
    assert isinstance(core_list, list), "KPI_SELECT.core must be a list"
    assert isinstance(optional_list, list), "KPI_SELECT.optional must be a list"
