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
import yaml


class IgnisLogger(logging.getLoggerClass()):
    """
    A logger class for Ignis. IgnisLogger is a like any other logging.Logger object except it has an additional method:
    log_to_file, used to log data in the form of key:value pairs to a log file. Logging configuration is performed via
    a configuration file and is handled by IgnisLogging.

    """
    def __init__(self, name, level=logging.NOTSET):
        Logger.__init__(self, name,level)
        self._file_logging_enabled = True
        self._file_handler = None

    def _set_file_handler(self, fh):
        self._file_handler = fh

    def log_to_file(self, **kargs):
        """
        This function logs key:value pairs to a log file.
        Note: Logger name in the log file is fixed (ignis_logging)

        :param kargs: Keyword parameters to be logged (e.g t1=0.02, qubits=[1,2,4])
        """
        if not self._file_logging_enabled or self._file_handler is None:
            return

        Logger.addHandler(self, self._file_handler)
        logstr = ""
        for k,v in kargs.items():
            logstr += "'{}':'{}' ".format(k, v)

        Logger.info(self, logstr)

        Logger.removeHandler(self, self._file_handler)

    def enable_file_logging(self):
        self._file_logging_enabled = True

    def disable_file_logging(self):
        self._file_logging_enabled = False


class IgnisLogging:
    """
    Singleton class to configure file logging via IgnisLogger.  Logging to file is enabled only if there is a
    config file present. Otherwise IgnisLogger will behave as a regular logger.

    Config file is assumed to be <user home>/.qiskit/logging.yaml

    Config file fields:
    ===================
    file_logging: {true/false}      - Specifies whether file logging is enabled
    log_file_path: <path>           - path to the log file. If not specified, ignis.log will be used
    max_size:  <# bytes>            - maximum size limit for a given log file. If not specified file size is unlimited
    max_rotations: <count>          - maximum number of log files to rotate (oldest file is deleted in case count is reached)
    """
    _instance = None
    _file_logging_enabled = False
    _log_file = None
    _max_bytes = 0
    _max_rotations = 0

    # Making the class a Singleton
    def __new__(cls):
        if IgnisLogging._instance is None:
            IgnisLogging._instance = object.__new__(cls)
            IgnisLogging._initialize()

        return IgnisLogging._instance

    def _initialize():
        """
        Initializes the logging facility for Ignis.
        """
        logging.setLoggerClass(IgnisLogger)

        # Loading and handling the config file
        config_file_path = os.path.join(os.path.expanduser('~'), ".qiskit", "logging.yaml")
        if os.path.exists(config_file_path):
            IgnisLogging._file_logging_enabled = True

        config_file = open(config_file_path, 'r')
        log_config = yaml.load(config_file, Loader=yaml.FullLoader)

        # Reading the config file content
        IgnisLogging._file_logging_enabled = True if log_config.get('file_logging') else False
        IgnisLogging._log_file = log_config.get('log_file') if log_config.get('log_file') is not None else "ignis.log"
        IgnisLogging._max_bytes = log_config.get('max_size') if log_config.get('max_size') is not None else 0
        IgnisLogging._max_rotations = log_config.get('max_rotations') if log_config.get('max_rotations') is not None else 0


    def get_logger(self, __name__):
        """
        Return an IgnisLogger object
        :param __name__: Name of the module being logged
        :return: IgnisLogger
        """
        logger = logging.getLogger(__name__)
        assert(isinstance(logger, IgnisLogger)), "IgnisLogger class was not registered"
        self._configure_logger(logger)

        return logger

    def _configure_logger(self, logger):
        if IgnisLogging._file_logging_enabled:
            # This will enable limiting file size and rotating once file size is exhausted
            fh = logging.handlers.RotatingFileHandler(IgnisLogging._log_file, maxBytes=IgnisLogging._max_bytes, backupCount=IgnisLogging._max_rotations)  # make limits configurable

            # Formatting
            formatter = logging.Formatter('%(asctime)s - ignis_logging %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
            fh.setFormatter(formatter)
            logger._set_file_handler(fh)

    def get_log_file(self):
        """
        :return: name of the log file
        """
        return IgnisLogging._log_file


class IgnisLogReader:
    """
    Class to read from Ignis log and construct tabular representation based on date/time and key criteria
    """

    def read_values(self, time_range=None, keys=None):
        pass




