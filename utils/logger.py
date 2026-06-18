import csv
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class TrainingLogger:
  
    def __init__(self, log_dir: str, name: str = "train"):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"{name}_{ts}.log"
        self.csv_file = log_dir / f"{name}_{ts}_metrics.csv"

        # Console + file handler
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        fh = logging.FileHandler(self.log_file)
        fh.setFormatter(fmt)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

        self._csv_fields = None
        self._csv_writer = None
        self._csv_handle = None

    def info(self, msg: str):
        self.logger.info(msg)

    def log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Write a row to the CSV and log to console."""
        row = {"epoch": epoch, **metrics}

        # Initialize CSV on first call
        if self._csv_fields is None:
            self._csv_fields = list(row.keys())
            self._csv_handle = open(self.csv_file, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=self._csv_fields)
            self._csv_writer.writeheader()

        self._csv_writer.writerow(row)
        self._csv_handle.flush()

        # Pretty console line
        parts = [f"Epoch {epoch:3d}"] + [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                           for k, v in metrics.items()]
        self.logger.info(" | ".join(parts))

    def close(self):
        if self._csv_handle:
            self._csv_handle.close()