# ref to optuna logging

import logging
from typing import Optional, TextIO

try:
    import colorlog

    _has_colorlog = True
except ImportError:
    _has_colorlog = False

# private APIs
_LOG_LEVEL = {"debug": logging.DEBUG,
              "info": logging.INFO,
              "warning": logging.WARNING,
              "error": logging.ERROR,
              "critical": logging.CRITICAL}

_default_handler = None


def _name() -> str:
    return __name__.split('.')[0]


def _create_default_formatter() -> logging.Formatter:
    datefmt = "%Y-%m-%d %H:%M:%S"
    if _has_colorlog:
        return colorlog.ColoredFormatter('%(log_color)s[%(name)s|%(asctime)s|%(levelname)s] %(message)s',
                                         datefmt=datefmt)
    return logging.Formatter("[%(name)s|%(asctime)s|%(levelname)s] %(message)s", datefmt=datefmt)


def _get_root_logger() -> logging.Logger:
    return logging.getLogger(_name())


def _configure_root_logger() -> None:
    global _default_handler
    if _default_handler is not None:
        return None
    _default_handler = logging.StreamHandler()
    _default_handler.setFormatter(_create_default_formatter())
    _user_root_logger = logging.getLogger()
    if len(_user_root_logger.handlers) > 0:
        # if user already defines their own root logger
        return None
    root_logger = _get_root_logger()
    root_logger.addHandler(_default_handler)
    root_logger.setLevel(logging.INFO)


def _reset_root_logger() -> None:
    global _default_handler
    if _default_handler is None:
        return None
    root_logger = _get_root_logger()
    root_logger.removeHandler(_default_handler)
    root_logger.setLevel(logging.NOTSET)
    _default_handler = None


# public APIs

def get_logger(name: str):
    _configure_root_logger()
    return logging.getLogger(name)


def get_verb_level() -> int:
    _configure_root_logger()
    return _get_root_logger().getEffectiveLevel()


def set_verb_level(level: str or int) -> None:
    if isinstance(level, str):
        level = _LOG_LEVEL[level]
    _configure_root_logger()
    _get_root_logger().setLevel(level)


def enable_default_handler() -> None:
    _configure_root_logger()
    if _default_handler is None:
        raise RuntimeWarning()
    _get_root_logger().addHandler(_default_handler)


def disable_default_handler() -> None:
    _configure_root_logger()
    if _default_handler is not None:
        raise RuntimeWarning()
    _get_root_logger().removeHandler(_default_handler)


def set_file_handler(log_file: str or TextIO, level: str or int = logging.DEBUG,
                     formatter: Optional[logging.Formatter] = None) -> None:
    _configure_root_logger()
    fh = logging.FileHandler(log_file)
    if isinstance(level, str):
        level = _LOG_LEVEL[level]
    fh.setLevel(level)
    if formatter is None:
        formatter = _create_default_formatter()
    fh.setFormatter(formatter)
    _get_root_logger().addHandler(fh)


def _set_tqdm_handler(level: str or int = logging.INFO,
                      formatter: Optional[logging.Formatter] = None) -> None:
    """ An alternative handler to avoid disturbing tqdm
    """
    from tqdm import tqdm

    class TQDMHandler(logging.StreamHandler):
        def __init__(self):
            logging.StreamHandler.__init__(self)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    _configure_root_logger()
    th = TQDMHandler()
    if isinstance(level, str):
        level = _LOG_LEVEL[level]
    th.setLevel(level)
    if _default_handler is not None:
        # to avoid multiple logs!
        _get_root_logger().removeHandler(_default_handler)
    if formatter is None:
        formatter = _create_default_formatter()
    th.setFormatter(formatter)
    _get_root_logger().addHandler(th)
