import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_PATTERN = re.compile(r"^(.+?)_([gGbB])_(\d{3})\.txt$")


def scan_raw_data(data_dir: str) -> Tuple[List[Dict], int]:
    """Scan data_dir for valid EEG txt files and parse metadata.

    Returns a tuple of (valid_items, skipped_count).
    Each valid item: {path, filename, subject, condition, trial}.
    """
    root = Path(data_dir)
    valid: List[Dict] = []
    skipped = 0

    if not root.exists():
        logger.warning("data_dir does not exist: %s", root)
        return valid, 0

    for path in root.rglob("*.txt"):
        fname = path.name
        m = _PATTERN.match(fname)
        if not m:
            skipped += 1
            logger.warning("[Skipped_Invalid_Filename] %s", fname)
            continue

        subject, cond_raw, trial_str = m.groups()
        cond = 1 if cond_raw.lower() == "g" else 2
        trial_no = int(trial_str)

        valid.append(
            {
                "path": str(path),
                "filename": fname,
                "subject": subject,
                "condition": cond,
                "trial": trial_no,
            }
        )

    return valid, skipped
