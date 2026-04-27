"""
logging_setup.py
----------------
Structured JSON logging for pipeline stages.
"""

import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in ("layer", "status", "records", "latency_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def log_stage(
    logger: logging.Logger,
    *,
    layer: str,
    status: str,
    records: int,
    latency_ms: float,
    level: int = logging.INFO,
) -> None:
    logger.log(
        level,
        f"{layer} {status}",
        extra={
            "layer": layer,
            "status": status,
            "records": records,
            "latency_ms": round(latency_ms, 2),
        },
    )
