import io
import logging
import re

from colorlog.escape_codes import escape_codes

from ..utils.contants import LOGGER_DATE_TIME_FORMAT, LOGGER_FORMAT, LOGGER_LEVEL
from ..utils.files import relpath

URL_REGEX = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
PATH_REGEX = r"(\/.*?\.[\w:]+)"


class CustomFormatter(logging.Formatter):
    """Logging colored formatter based on https://stackoverflow.com/a/56944256/3638629.

    Handles FileHandler and StreamHandler logging objects.
    """

    _reset = "\x1b[0m"
    _msg = "%(message)s"
    _level = "%(levelname)s"

    def __init__(
        self,
        fmt: str = LOGGER_FORMAT,
        datetime_format: str = LOGGER_DATE_TIME_FORMAT,
        is_file: bool = False,
        datetime_format_color: str | None = "light_black",
        url_color: str | None = "purple",
    ):
        if is_file:
            datetime_format_color = None
            url_color = None
        if datetime_format_color is not None:
            fmt = fmt.replace("%(asctime)s", f"{escape_codes[datetime_format_color]} %(asctime)s")
        super().__init__(fmt, datetime_format)

        for keyword in escape_codes:
            fmt = fmt.replace(f"%({keyword})s", escape_codes[keyword])

        self.fmt = fmt
        self.datetime_format = datetime_format
        self.is_file = is_file
        self.url_color = url_color

        self.FORMATS = {
            logging.DEBUG: self.add_color_to_levelname(self.fmt, escape_codes["light_cyan"]),
            logging.INFO: self.add_color_to_levelname(self.fmt, escape_codes["green"]),
            logging.WARNING: self.add_color_to_levelname(self.fmt, escape_codes["yellow"]),
            logging.ERROR: self.add_color_to_levelname(self.fmt, escape_codes["red"]),
            logging.CRITICAL: self.add_color_to_levelname(self.fmt, escape_codes["bg_bold_red"]),
        }
        names = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL",
        }
        max_len = max([len(name) for name in names.values()])
        num_spaces = {level_id: max_len - len(names[level_id]) for level_id in names}
        # spaces to the left
        # self.LEVEL_NAMES = {
        #     level_id: f"{' ' * num_spaces[level_id]}{names[level_id]}" for level_id in names
        # }
        # centered with spaces around
        self.LEVEL_NAMES = {
            level_id: f"{names[level_id].center(2 + len(names[level_id]) + num_spaces[level_id])}" for level_id in names
        }
        self._set_device("cpu", None)

    def _set_device(self, device: str, device_id: int | None):
        self.device = device
        self.device_id = device_id

    @property
    def device_info(self) -> str:
        _device_info = f"{self.device}{':' + str(self.device_id) if self.device_id is not None else ''}"
        len_center = 1 + len(_device_info) + 1
        return f"[{_device_info.center(len_center)}] "

    @classmethod
    def add_color_to_levelname(cls, fmt: str, color: str):
        return fmt.replace(f"{cls._level} {cls._msg}", f"{color}{cls._level}{cls._reset} {cls._msg}")

    @classmethod
    def add_color_to_regex(cls, record: logging.LogRecord, regex: str, color: str):
        color_code = escape_codes[color]
        record.msg = re.sub(regex, rf"{color_code}\1{cls._reset}", record.msg)

    def format(self, record: logging.LogRecord):
        if self.is_file:
            log_fmt = self.fmt  # no formating for files
        else:
            log_fmt = self.FORMATS.get(record.levelno)
            if self.url_color is not None and isinstance(record.msg, str):
                self.add_color_to_regex(record, URL_REGEX, self.url_color)

        record.levelname = self.LEVEL_NAMES[record.levelno]
        if isinstance(record.msg, str) and self.device_info not in record.msg:
            record.msg = self.device_info + record.msg
        if self.is_file:
            for code in escape_codes.values():
                if not isinstance(record.msg, Exception) and code in record.msg:
                    record.msg = record.msg.replace(code, "")
        formatter = logging.Formatter(log_fmt, self.datetime_format)
        return formatter.format(record)


def get_cmd_pylogger(name: str = __name__) -> logging.Logger:
    """Initialize command line logger"""
    formatter = CustomFormatter(
        fmt=LOGGER_FORMAT,
        datetime_format=LOGGER_DATE_TIME_FORMAT,
        is_file=False,
        datetime_format_color="light_black",
        url_color="purple",
    )
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(LOGGER_LEVEL)
    logger.propagate = False
    return logger


def get_file_pylogger(filepath: str, name: str = __name__) -> logging.Logger:
    """Initialize .log file logger"""
    formatter = CustomFormatter(fmt=LOGGER_FORMAT, datetime_format=LOGGER_DATE_TIME_FORMAT, is_file=True)
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(filepath, "a+")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(LOGGER_LEVEL)
    logger.propagate = False
    return logger


def add_file_handler_to_logger(filepath: str, device: str, device_id: int):
    """Add file handler to logger"""
    global log
    file_log = get_file_pylogger(filepath, "log_file")
    # insert handler to enable file logs in command line aswell
    log.handlers.insert(0, file_log.handlers[0])
    log.info(f"Saving logs to '{relpath(filepath)}'")
    for handler in log.handlers:
        formatter = handler.formatter
        if formatter is not None and hasattr(formatter, "_set_device"):
            handler.formatter._set_device(device, device_id)
    return file_log  # file log handler


def remove_last_logged_line(file_log: logging.Logger):
    """Remove the last line of log file"""
    file: io.TextIOWrapper = file_log.handlers[0].stream
    file.seek(0)
    lines = file.readlines()
    file.seek(0)
    file.truncate()
    file.writelines(lines[:-1])
    file.seek(0, 2)


def log_breaking_point(
    msg: str,
    n_top: int = 0,
    n_bottom: int = 0,
    top_char: str = " ",
    bottom_char: str = " ",
    fill_char: str = " ",
    num_chars: int = 100,
):
    top_line = top_char * num_chars
    bottom_line = bottom_char * num_chars
    for _ in range(n_top):
        log.info(top_line)
    log.info(msg.center(num_chars, fill_char))
    for _ in range(n_bottom):
        log.info(bottom_line)


log = get_cmd_pylogger(__name__)
