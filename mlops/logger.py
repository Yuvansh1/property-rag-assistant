"""
Structured JSON Logger

Outputs one JSON object per line to stdout for each log event.
Fields: timestamp, level, event, query_id, step, latency_ms, extra.
"""

import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
            "query_id": getattr(record, "query_id", None),
            "step": getattr(record, "step", None),
            "latency_ms": getattr(record, "latency_ms", None),
            "extra": getattr(record, "log_extra", {}),
        }
        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
