# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Ignis Logging
"""

import logging
import logging.handlers
import logging.config
from logging import Logger
import os
import glob
from datetime import datetime
import re
from typing import List, Union, Optional


class IgnisLogger(logging.getLoggerClass()):
    """
    A logger class for Ignis

    IgnisLogger is a like any other :class:`logging.Logger` object except it has an
    additional method, :meth:`log_to_file`, used to log data in the form of
    key:value pairs to a log file. Logging configuration is performed via a
    configuration file and is handled by IgnisLogging.

    Refer to Python's logging documentation for more details on how to use
    logging in Python

    """

    def __init__(self, name: str, level: Optional[int] = logging.NOTSET):
        """
        Initialize the IgnisLogger object

        Args:
            name: name of the logger. Usually set to package name using __name__
            level(logging.NOTSET): Verbosity level (use logging package enums)
        """
        Logger.__init__(self, name, level)
        self._file_logging_enabled = False
        self._file_handler = None
        self._stream_handler = None
        self._conf_file_exists = False
        self._warning_omitted = False

    def configure(self, sh: logging.StreamHandler, conf_file_exists: bool):
        """
        Internal configuration method of IgnisLogger. Should only be called
        by IgnisLogger

        Args:
            sh: StreamHandler object
            conf_file_exists: Whether or not a file config exists
        """
        self._stream_handler = sh
        self.addHandler(sh)
        self._conf_file_exists = conf_file_exists

    def log_to_file(self, **kargs: str):
        """
        Log key:value pairs to a log file.

        Note: Logger name in the log file is fixed (ignis_logging)

        Args:
            kwargs: key/value pairs to be logged, e.g t1=0.02, qubits=[1,2,4]
        """
        if not self._file_logging_enabled:
            if not self._warning_omitted:  # Omitting this warning only once
                msg = "File logging is disabled"
                if not self._conf_file_exists:
                    msg += ": no config file"
                logger = logging.getLogger(__name__)
                logger.warning(msg)
                self._warning_omitted = True
            return

        # We defer setting up the file handler, since its __init__ method
        # has the side effect of creating the file
        if self._file_handler is None:
            self._file_handler = IgnisLogging().get_file_handler()

        assert(self._file_handler is not None), "file_handler is not set"

        Logger.removeHandler(self, self._stream_handler)
        Logger.addHandler(self, self._file_handler)
        logstr = ""
        for k, v in kargs.items():
            logstr += "'{}':'{}' ".format(k, v)

        Logger.log(self, 100, logstr)

        Logger.removeHandler(self, self._file_handler)
        Logger.addHandler(self, self._stream_handler)

    def enable_file_logging(self):
        """
        Enable file logging for this logger object (note there is a single
        object for a given logger name
        """
        self._file_logging_enabled = True

    def disable_file_logging(self):
        """
        Disable file logging for this logger object (note there is a single
        object for a given logger name
        """
        self._file_logging_enabled = False


class IgnisLogging:
    """Singleton class to configure file logging via IgnisLogger


    Logging to file is enabled only if there is a config file present. Otherwise
    IgnisLogger will behave as a regular logger.

    Config file is assumed to be in <user home>/.qiskit/logging.yaml

    **Config file fields:**

    file_logging: {true/false}      - Specifies whether file logging is enabled

    log_file: <path>                - path to the log file. If not specified,
    ignis.log will be used

    max_size:  <# bytes>            - maximum size limit for a given log file.
    If not specified file size is unlimited

    max_rotations: <count>          - maximum number of log files to rotate
    (oldest file is deleted in case count is reached)
    """

    # TODO: Should we allow to override file settings programmatically ?
    # (e.g. enable logging)
    _instance = None
    _file_logging_enabled = False
    _log_file = None
    _max_bytes = 0
    _max_rotations = 0
    _log_label = "ignis_logging"
    _default_datefmt = '%Y/%m/%d %H:%M:%S'
    _config_file_exists = False

    # Making the class a Singleton
    def __new__(cls):
        if IgnisLogging._instance is None:
            IgnisLogging._instance = object.__new__(cls)
            IgnisLogging._initialize()

        return IgnisLogging._instance

    @staticmethod
    def _load_config_file():
        """
        Load and parse the config file
        Returns:
            A dictionary containing all the settings
        """
        config_file_path = os.path.join(os.path.expanduser('~'),
                                        ".qiskit", "logging.yaml")
        config = dict()
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as log_file:
                for line in log_file:
                    # removing comments
                    line = line[:line.find('#') if "#" in line else None]
                    line = line.split(':')  # Splitting to key value
                    if len(line) < 2:
                        continue
                    config[line[0].strip().lower()] = line[1].strip().lower()
            IgnisLogging._config_file_exists = True

        return config

    @staticmethod
    def _initialize():
        """
        Initialize the logging facility for Ignis
        """
        logging.setLoggerClass(IgnisLogger)

        log_config = IgnisLogging._load_config_file()
        # Reading the config file content
        IgnisLogging._file_logging_enabled = \
            log_config.get('file_logging') == "true"
        IgnisLogging._log_file = log_config.get('log_file') if \
            log_config.get('log_file') is not None else "ignis.log"
        max_size = log_config.get('max_size')
        IgnisLogging._max_bytes = int(max_size) if \
            max_size is not None and max_size.isdigit() else 0
        max_rotations = log_config.get('max_rotations')
        IgnisLogging._max_rotations = int(max_rotations) if \
            max_rotations is not None and max_rotations.isdigit() else 0

    def get_logger(self, __name__: str) -> IgnisLogger:
        """
        Return an IgnisLogger object

        To be used in by the code which needs logging.

        Args:
            __name__: Name of the module being logged

        Returns:
            An IgnisLogger object
        """
        logger = logging.getLogger(__name__)
        assert(isinstance(logger, IgnisLogger)), \
            "IgnisLogger class was not registered"
        self.configure_logger(logger)

        return logger

    def get_file_handler(self):
        """
        Configure and retrieve the RotatingFileHandler object. Called on
        demand the first time IgnisLoggers needs to write to a file

        Returns:
            The configured RotatingFileHandler
        """
        # Configuring the file handling aspect
        fh = logging.handlers.RotatingFileHandler(
            IgnisLogging._log_file, maxBytes=IgnisLogging._max_bytes,
            backupCount=IgnisLogging._max_rotations)

        # Formatting
        formatter = logging.Formatter(
            '%(asctime)s {} %(message)s'.format(IgnisLogging._log_label),
            datefmt=IgnisLogging._default_datefmt)
        fh.setFormatter(formatter)

        return fh

    def _configure_logger(self, logger):
        """
        Configure the stream handler of the logger

        Args:
            logger: the logger to be configured
        """
        # Configuring the stream handler
        sh = logging.StreamHandler()
        sh.setLevel(logging.NOTSET)
        stream_fmt = logging.Formatter('%(levelname)s: %(name)s - %(message)s')
        sh.setFormatter(stream_fmt)
        # This will enable limiting file size and rotating once file size
        # is exhausted

        logger.configure(sh, IgnisLogging._config_file_exists)

        if IgnisLogging._file_logging_enabled:
            logger.enable_file_logging()

    def get_log_file(self) -> str:
        """
        Get the name of the log file

        Returns:
            Name of the log file
        """
        return IgnisLogging._log_file

    def default_datetime_fmt(self) -> str:
        """
        Get the default date time format used for writing log entries

        Returns:
             Default date time format
        """
        return IgnisLogging._default_datefmt


class IgnisLogReader:
    """
    Class to read from Ignis log files

    Reads and constructs tabular representation of logged data based on
    date/time and key criteria
    """

    def get_log_files(self) -> List[str]:
        """
        Get Names of all log files (several may be present due to logging
        file rotation). File names are sorted by modification time.

        Returns:
            list of all log file names

        """

        file_name = IgnisLogging().get_log_file()
        search_path = os.path.abspath(file_name + "*")
        files = sorted(glob.glob(search_path), key=os.path.getmtime)

        result = list()
        m = re.compile(
            os.path.abspath(file_name).replace('\\', r'\\') + r"$|" +
            os.path.abspath(file_name).replace('\\', r'\\') + r".\d+$")
        for f in files:
            if m.match(f):
                result.append(f)

        return result

    def read_values(self, log_files: Optional[List[str]] = None,
                    keys: Optional[List[str]] = None,
                    from_datetime: Optional[str] = None,
                    from_datetime_format: Optional[Union[str, datetime]] = None,
                    to_datetime: Optional[str] = None,
                    to_datetime_format: Optional[Union[str, datetime]] = None)\
            -> List[List[str]]:
        """
        Retrieve log lines using key and date/time filtering criteria

        Params:
            log_files: List of log files to read from
            keys: Retrieve only key value pairs of corresponding to keys A row
            with no matching keys will not be retrieved. If not specified,
            all keys are retrieved (optional)

            from_datetime(None): Retrieve only rows newer than the given date and
            time

            from_datetime_format(None): datetime format string. If not specified
            will assume "%Y/%m/%d %H:%M:%S"

            to_datetime(None): Retrieve only rows older than the given date and
            time

            to_datetime_format(None): datetime format string. If not specified
            will assume "%Y/%m/%d %H:%M:%S"

        Returns:
            A list containing the retrieved rows of key pair values
        """

        if log_files is not None:
            files = [log_files] if isinstance(log_files, str) else log_files
        else:
            files = self.get_log_files()
        retrieved_date = list()

        for file in files:
            with open(file, "r") as f:
                for line in f:
                    terms = line.split()
                    date_time = terms[0:2]

                    dt_filterd = self._filter_by_datetime(date_time,
                                                          from_datetime,
                                                          from_datetime_format,
                                                          to_datetime,
                                                          to_datetime_format)
                    if dt_filterd:
                        continue

                    key_values = terms[3:]
                    if keys is not None:
                        key_values = self._filter_keys(key_values, keys)
                    if not key_values:
                        continue

                    retrieved_date.append(date_time + key_values)

        return retrieved_date

    def _filter_keys(self, key_values, keys):
        """
        Retrieve key value pairs matching the given keys

        Params:
            key_values: list of key value pairs
            keys: list of keys to retrieve key value pair of

        Returns:
            A list of key value pairs according to keys
        """

        result = list()
        assert(isinstance(key_values, list)), "key_values is not a list"

        for kv in key_values:
            if kv.split(":")[0].strip("'") in keys:
                result.append(kv)

        return result

    def _filter_by_datetime(self, row_datetime, from_dt,
                            from_dt_fmt, to_dt,
                            to_dt_fmt):

        """
        Determine whether the given datetime should be filtered

        Params:
            row_datetime: the date/time in question
            from_dt: starting date/time
            from_dt_fmt: format of the starting date/time
            to_dt: ending date/time
            to_dt_fmt: format of the ending date/time

        Returns:
            True if the row should be filtered out
        """
        if from_dt is not None and not isinstance(from_dt, datetime):
            try:
                if from_dt_fmt is None:
                    from_dt_fmt = IgnisLogging().default_datetime_fmt()
                from_dt = datetime.strptime(from_dt, from_dt_fmt)
            except ValueError as ve:
                raise ve

        if to_dt is not None and not isinstance(to_dt, datetime):
            try:
                if to_dt_fmt is None:
                    to_dt_fmt = IgnisLogging().default_datetime_fmt()
                to_dt = datetime.strptime(to_dt, to_dt_fmt)
            except ValueError as ve:
                raise ve

        if from_dt is None and to_dt is None:
            return False

        row_datetime = datetime.strptime("%s %s" % (row_datetime[0],
                                                    row_datetime[1]),
                                         IgnisLogging().default_datetime_fmt())

        if from_dt is not None:
            if row_datetime < from_dt:
                return True

        if to_dt is not None:
            if row_datetime > to_dt:
                return True

        return False
