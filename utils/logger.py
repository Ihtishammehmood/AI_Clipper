"""Logging setup with job-id aware loggers."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, job_id: str | None = None) -> logging.LoggerAdapter:
    root = logging.getLogger("clipper")
    if not root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(job)s] %(name)s: %(message)s"))
        root.addHandler(h)
        root.setLevel(logging.INFO)
    return logging.LoggerAdapter(logging.getLogger(f"clipper.{name}"), {"job": job_id or "-"})
