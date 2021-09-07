import logging
from logging import StreamHandler
import os

logger_instances = {}


def get_logger(logger_name, filename=None):
    """
    Returns a handy logger with both printing to std output and file
    """
    global logger_instances
    LOGGING_MODE = os.environ.get("LOGGING_MODE", "INFO")
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if logger_name not in logger_instances:
        logger_instances[logger_name] = logging.getLogger(logger_name)
        if filename is None:
            filename = os.path.join(
                os.environ.get("LOG_DIR", "/tmp"), logger_name + ".log"
            )

        file_handler = logging.FileHandler(filename=filename, mode="a+")
        file_handler.setFormatter(log_format)
        logger_instances[logger_name].addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)

        logger_instances[logger_name].addHandler(console_handler)
        logger_instances[logger_name].setLevel(level=getattr(logging, LOGGING_MODE))
    return logger_instances[logger_name]


def equal_dicts(dict1, dict2):
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for k1, v1 in dict1.items():
        v2 = dict2[k1]
        if isinstance(v1, dict) ^ isinstance(v2, dict):
            return False
        if isinstance(v1, dict):
            if not equal_dicts(v1, v2):
                return False
        else:
            if v1 != v2:
                return False
    return True