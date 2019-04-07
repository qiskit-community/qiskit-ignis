# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
logger wrapper for the tomography package
"""

import logging

class TomoLogger:
    _instance = None
    _first = True

    def __new__(cls, val):
        if TomoLogger._instance is None:
            TomoLogger._instance = object.__new__(cls)
        TomoLogger._instance.val = val
        return TomoLogger._instance

    def get_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler("test.log")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        if TomoLogger._first:
            logger.info("### Logging started for Tomography  ###")
            TomoLogger._first = False

        return logger
