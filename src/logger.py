from loguru import logger
import sys
import os

def setup_logger(log_file=None):
    logger.remove()  # Clear default logger
    logger.add(sys.stderr, level="INFO")
    if log_file:
        log_dir = os.path.dirname(log_file)
        os.makedirs(log_dir, exist_ok=True)
        logger.add(log_file, rotation="500 MB", level="DEBUG")