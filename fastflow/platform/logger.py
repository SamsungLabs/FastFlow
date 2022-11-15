# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import logging
import os
import sys
import threading

LEVELS = {
    "ALL": 0,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "SILENCE": 60
}

FASTFLOW_LOG_FORMAT = (
    "[%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d] "
    "%(levelname)s - %(message)s"
)

ENV_LEVEL = os.getenv("FASTFLOW_LOG_LEVEL", "INFO").upper()


# Don't use this directly. Use get_logger() instead.
class _GlobalLogger:
    logger = None
    logger_lock = threading.Lock()


def get_logger():
    """Return FastFlow logger instance."""

    # Use double-checked locking to avoid taking lock unnecessarily.
    if _GlobalLogger.logger:
        return _GlobalLogger.logger

    with _GlobalLogger.logger_lock:
        if _GlobalLogger.logger:
            return _GlobalLogger.logger

        # Scope the FastFlow logger to not conflict with users' loggers.
        logger = logging.getLogger('fastflow')

        # Don't further configure the FastFlow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        # For pytest, interactive always False.
        interactive = False
        if not logging.getLogger().handlers:
            # Determine whether we are in an interactive environment
            try:
                # This is only defined in interactive shells.
                if sys.ps1:
                    interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                interactive = sys.flags.interactive

        # If we are in an interactive environment (like Jupyter), set loglevel
        # to INFO and pipe the output to stdout.
        # but you can configure level with environment variables
        if interactive:
            logging_target = sys.stdout
        else:
            logging_target = sys.stderr
        logger.setLevel(LEVELS.get(ENV_LEVEL, "INFO"))

        # Add the output handler.
        handler = logging.StreamHandler(logging_target)
        handler.setFormatter(logging.Formatter(FASTFLOW_LOG_FORMAT,
                                               'UTC%z %Y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)

        _GlobalLogger.logger = logger
        return _GlobalLogger.logger
