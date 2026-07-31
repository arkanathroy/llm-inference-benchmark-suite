"""Structured logging helper shared by every phase script.

Emits one JSON object per log line to stdout (line-buffered, flushed
immediately) so:
  1. Colab shows live progress instead of a long silent gap, since JSON
     lines print incrementally rather than only appearing once the
     subprocess buffer flushes at exit.
  2. A Promtail agent tailing the notebook's log file (see
     docs/observability.md) can parse each line as structured JSON and
     ship it to Loki, letting Grafana render live per-technique,
     per-batch dashboards while phases are still running.

Falls back to human-readable plain text if PLAIN_LOGS=1 is set, useful
for quick manual debugging without a JSON-line wall of text.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time


class JSONLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in getattr(record, "__dict__", {}).items():
            if key in ("args", "msg", "levelname", "levelno", "pathname", "filename",
                       "module", "exc_info", "exc_text", "stack_info", "lineno",
                       "funcName", "created", "msecs", "relativeCreated", "thread",
                       "threadName", "processName", "process", "name", "message",
                       "getMessage"):
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    if os.environ.get("PLAIN_LOGS") == "1":
        stdout_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    else:
        stdout_handler.setFormatter(JSONLineFormatter())
    logger.addHandler(stdout_handler)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(repo_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(JSONLineFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
