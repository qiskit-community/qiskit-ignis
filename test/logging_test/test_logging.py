# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name


"""
Unit testing of the Ignis Logging facility. Covering the following specs:
1) Configuration aspects
======================
1.1: No config file entails logging is disabled
1.2: A typo in one of the params is ignored
1.3: Supported params are affecting the logging
1.4: Programmatic settings of the logger overrides the config file

2) File logging:
================
2.1: Data is saved to the log
2.2: The message is in the right format
2.3: Data is accumulated across log_to_file calls

3) Log reader:
==============
3.1: Values are read properly
3.2: Key filtering is working properly
3.3: Date filtering is working properly


"""
import os
import tempfile
import unittest

from qiskit.ignis.logging import IgnisLogging, IgnisLogReader


class TestLogging(unittest.TestCase):
    """Test logging module"""
    _default_log = "ignis.log"

    def setUp(self):
        """
        Basic setup - making the .qiskit dir and preserving any existing files
        :return:
        """
        super().setUp()
        _, self.temp_config_path = tempfile.mkstemp()
        self.addCleanup(os.remove, self.temp_config_path)

    def tearDown(self):
        """
        Remove auto-generated files, resurrecting original files, and
        resetting the IgnisLogging singleton state
        """
        IgnisLogging._reset_to_defaults()
        if os.path.isfile(self._default_log):
            os.remove(self._default_log)
        super().tearDown()

    def test_no_config_file(self):
        """
        Test there are no config file
        :return:
        """
        logger = IgnisLogging().get_logger(__name__)
        logger.log_to_file(test="test")

        self.assertFalse(os.path.exists(self._default_log))

    def test_file_logging_typo(self):
        """
        Only tests the main param: file_logging
        :return:
        """
        with open(self.temp_config_path, 'w') as fd:
            fd.write("file_logging1: true")

        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)
        logger.log_to_file(test="test")

        self.assertFalse(os.path.exists(self._default_log))

    def test_file_true_typo(self):
        """
        Only tests the main param: file_logging
        :return:
        """
        with open(self.temp_config_path, "w") as fd:
            fd.write("file_logging: tru")

        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)
        logger.log_to_file(test="test")

        self.assertFalse(os.path.exists(self._default_log))

    def test_log_file_path(self):
        """
        test that a custom log file path is honored
        :return:
        """
        log_path = os.path.join(tempfile.gettempdir(), 'test_log.log')
        with open(self.temp_config_path, "w") as fd:
            fd.write("file_logging: true\nlog_file: %s" % log_path)

        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)
        self.addClassCleanup(os.remove, log_path)
        logger.log_to_file(test="test")

        self.assertTrue(os.path.exists, log_path)

    def test_file_rotation(self):
        """
        Test that the file rotation is working
        """
        log_path = os.path.join(tempfile.gettempdir(), 'test_log_rotate.log')
        with open(self.temp_config_path, "w") as fd:
            fd.write("file_logging: true\n"
                     "max_size: 10\n"
                     "max_rotations: 3\n"
                     "log_file: %s" % log_path)

        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)
        self.addCleanup(os.remove, log_path)
        self.addCleanup(os.remove, log_path + '.1')
        self.addCleanup(os.remove, log_path + '.2')
        self.addCleanup(os.remove, log_path + '.3')

        for i in range(100):
            logger.log_to_file(test="test%d" % i)

        self.assertTrue(os.path.exists(log_path))
        self.assertTrue(os.path.exists(log_path + ".1"))
        self.assertTrue(os.path.exists(log_path + ".2"))
        self.assertTrue(os.path.exists(log_path + ".3"))

    def test_manual_enabling(self):
        """
        Test that enabling the logging manually works
        """
        log_path = os.path.join(tempfile.gettempdir(), 'test_log_manual.log')
        with open(self.temp_config_path, "w") as fd:
            fd.write("log_file: %s" % log_path)

        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)
        logger.enable_file_logging()
        self.addCleanup(os.remove, log_path)
        logger.log_to_file(test="test")

        self.assertTrue(os.path.exists(log_path))

    def test_manual_disabling(self):
        """
        Test that disabling the logging manually works
        """
        log_path = os.path.join(tempfile.gettempdir(), 'test_log_manual.log')
        with open(self.temp_config_path, "w") as fd:
            fd.write("file_logging: true\nlog_file: %s" % log_path)

        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)
        logger.disable_file_logging()
        logger.log_to_file(test="test")

        self.assertFalse(os.path.exists(log_path))

    def test_save_line(self):
        """
        Test basic log operations
        """
        log_path = os.path.join(tempfile.gettempdir(), 'test_log_save_log.log')
        with open(self.temp_config_path, "w") as fd:
            fd.write("log_file: %s" % log_path)
        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)

        logger.enable_file_logging()
        self.addCleanup(os.remove, log_path)
        logger.log_to_file(test="test")

        self.assertTrue(os.path.exists(log_path))
        with open(log_path, 'r') as file:
            self.assertIn("\'test\':\'test\'", file.read())

    def test_format(self):
        """
        Test format of the saved line
        """
        log_path = os.path.join(tempfile.gettempdir(), 'test_log_format.log')
        with open(self.temp_config_path, "w") as fd:
            fd.write("log_file: %s" % log_path)
        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)

        logger.enable_file_logging()
        self.addCleanup(os.remove, log_path)
        logger.log_to_file(test="test")

        with open(log_path, 'r') as file:
            self.assertRegex(
                file.read(),
                r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} ignis_logging \S+")

    def test_multiple_lines(self):
        """
        Test logging multiple lines
        """
        log_path = os.path.join(tempfile.gettempdir(), 'test_log_multiple_lines.log')
        with open(self.temp_config_path, "w") as fd:
            fd.write("log_file: %s" % log_path)
        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)
        logger = IgnisLogging().get_logger(__name__)

        logger.enable_file_logging()
        self.addCleanup(os.remove, log_path)
        logger.log_to_file(test="test1")
        logger.log_to_file(test="test2")

        with open(log_path, 'r') as file:
            lines = file.read().split('\n')

        self.assertGreaterEqual(len(lines), 2)

    def test_read_multiple_files(self):
        """
        Test reading multiple lines
        """
        log_path = os.path.join(tempfile.gettempdir(), 'test_log_read_multiple.log')
        with open(self.temp_config_path, "w") as fd:
            fd.write("file_logging: true\nmax_size: 40\nmax_rotations: 5\nlog_file: %s" % log_path)

        logger = IgnisLogging(self.temp_config_path).get_logger(__name__)
        for i in range(10):
            logger.log_to_file(k="data%d" % i)

        reader = IgnisLogReader()
        files = reader.get_log_files()
        self.assertEqual(len(files), 6)

        for file in reader.get_log_files():
            try:
                os.remove(file)
            except OSError:
                pass

    def test_filtering(self):
        """
        Test filtering operations
        """
        log_path = os.path.join(tempfile.gettempdir(), 'test_log_filtering')
        with open(log_path, 'w') as log:
            log.write(
                "2019/08/04 13:27:04 ignis_logging \'k1\':\'d1\'\n"
                "2019/08/04 13:27:05 ignis_logging \'k1\':\'d2\'\n"
                "2019/08/04 13:27:06 ignis_logging \'k1\':\'d3\'\n"
                "2019/08/05 13:27:04 ignis_logging \'k2\':\'d4\'\n"
                "2019/08/06 13:27:04 ignis_logging \'k2\':\'d5\'\n"
                "2019/09/02 13:27:04 ignis_logging \'k3\':\'d6\'\n"
                "2019/09/04 13:27:04 ignis_logging \'k4\':\'d7\'\n")
        self.addCleanup(os.remove, log_path)

        with open(self.temp_config_path, "w") as fd:
            fd.write("log_file: %s" % log_path)

        IgnisLogging(self.temp_config_path).get_logger(__name__)

        reader = IgnisLogReader()

        self.assertEqual(len(reader.read_values(keys=["k1", "k2"])), 5)
        self.assertEqual(
            len(reader.read_values(from_datetime="2019/08/05 00:00:00")), 4)
        self.assertEqual(
            len(reader.read_values(to_datetime="2019/08/04 13:27:06")), 3)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
