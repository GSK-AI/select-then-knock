"""Plain console logging for local experiments."""

import logging
import os


def get_logger(name: str = "") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, os.getenv("STK_LOG_LEVEL", "WARNING").upper(), logging.WARNING))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger
