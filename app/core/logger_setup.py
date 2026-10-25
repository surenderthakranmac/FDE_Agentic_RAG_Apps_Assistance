import logging
import sys
import json


class SIEMJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "component": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "telemetry"):
            log_payload["telemetry"] = record.telemetry
        return json.dumps(log_payload)


# Initialize production logging configuration matrix pointing straight to stdout
logger = logging.getLogger("EnterpriseITCopilot")
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(SIEMJsonFormatter())
    logger.addHandler(stream_handler)
