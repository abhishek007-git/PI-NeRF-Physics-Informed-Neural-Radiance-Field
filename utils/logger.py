"""Logger: console + CSV + optional TensorBoard."""

import os
import csv
import time
from typing import Dict


class Logger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self._csv_file = None
        self._csv_writer = None
        self._headers_written = False

        # TensorBoard (optional)
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
            print(f"TensorBoard logging to: {log_dir}/tb")
        except ImportError:
            pass

    def log(self, step: int, metrics: Dict[str, float]):
        row = {"step": step, "time": time.time(), **metrics}

        # CSV
        if not self._headers_written:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
            self._headers_written = True

        # Only write keys we know about
        safe_row = {k: row.get(k, "") for k in self._csv_writer.fieldnames}
        self._csv_writer.writerow(safe_row)
        self._csv_file.flush()

        # TensorBoard
        if self.tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step)

    def close(self):
        if self._csv_file:
            self._csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()
