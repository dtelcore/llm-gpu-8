"""Shared logging configuration for the Kepler GT 730 training workspace."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from paths import DEFAULT_TRAINING_LOG, OUTPUT_LOGS, ensure_output_dirs

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_LOGGER_NAME = "llm_gpu"


class SafeConsoleHandler(logging.StreamHandler):
    """Console handler that degrades unsupported Unicode instead of crashing."""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe_msg = msg.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe_msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def _prepare_console_stream():
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return stream


def setup_logging(
    log_level: int = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    log_filename: Optional[str] = None,
    also_append_to: Optional[Union[str, Path]] = DEFAULT_TRAINING_LOG,
) -> logging.Logger:
    """Configure logging to console + per-run file under output/logs/.

    Args:
        log_level: Console log level.
        log_dir: Directory for log files (default: output/logs).
        log_filename: Base name without .log extension. If None, uses a timestamp.
        also_append_to: Optional aggregate log file (default: output/logs/training.log).
    """
    ensure_output_dirs()
    log_dir = Path(log_dir) if log_dir is not None else OUTPUT_LOGS
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_filename:
        log_file = log_dir / f"{log_filename}.log"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{stamp}.log"

    log = logging.getLogger(_LOGGER_NAME)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    log.handlers.clear()

    console = SafeConsoleHandler(stream=_prepare_console_stream())
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    log.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    log.addHandler(file_handler)

    if also_append_to is not None:
        aggregate = Path(also_append_to)
        aggregate.parent.mkdir(parents=True, exist_ok=True)
        agg_handler = logging.FileHandler(aggregate, encoding="utf-8", mode="a")
        agg_handler.setLevel(logging.DEBUG)
        agg_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        log.addHandler(agg_handler)

    log.info("Logging initialized | Level: %s | File: %s", logging.getLevelName(log_level), log_file)
    return log


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


# Module import: default logger writing to output/logs/training.log (+ per-session file).
logger = setup_logging()
