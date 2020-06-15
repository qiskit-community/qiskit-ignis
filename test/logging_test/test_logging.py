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
import unittest
import os
from qiskit.ignis.logging import IgnisLogging, IgnisLogReader


class TestLoggingBase(unittest.TestCase):
    """
    Base class for the logging test classes
    """
    _qiskit_dir = ""
    _default_log = "ignis.log"

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self._qiskit_dir = os.path.join(os.path.expanduser('~'), ".qiskit")

    def setUp(self):
        """
        Basic setup - making the .qiskit dir and preserving any existing files
        :return:
        """
        os.makedirs(self._qiskit_dir, exist_ok=True)

        # Protecting the original files, if exist
        _safe_rename_file(os.path.join(self._qiskit_dir, "logging.yaml"),
                          os.path.join(self._qiskit_dir, "logging.yaml.orig"))
        # Assuming isnis.log for the default log file, as hard coded
        _safe_rename_file(self._default_log, self._default_log + ".orig")

    def tearDown(self):
        """
        Remove auto-generated files, resurrecting original files, and
        resetting the IgnisLogging singleton state

        :return:
        """
        try:
            os.remove("logging.yaml")
        except OSError:
            pass

        # Resurrecting the original files
        _safe_rename_file(
            os.path.join(self._qiskit_dir, "logging.yaml.orig"),
            os.path.join(self._qiskit_dir, "logging.yaml"))

        _safe_rename_file(self._default_log + ".orig", self._default_log)

        # Resetting the following attributes, to make the singleton reset
        IgnisLogging().get_logger(__name__).__init__(__name__)
        IgnisLogging._instance = None  # pylint: disable=W0212
        IgnisLogging._file_logging_enabled = False  # pylint: disable=W0212
        IgnisLogging._log_file = None  # pylint: disable=W0212
        IgnisLogging._config_file_exists = False  # pylint: disable=W0212


def _safe_rename_file(src, dst):
    try:
        os.replace(src, dst)
    except FileNotFoundError:
        pass
    except OSError:
        pass


class TestLoggingConfiguration(TestLoggingBase):
    """
    Testing configuration file handling
    """
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
        with open(os.path.join(self._qiskit_dir, "logging.yaml"), "w") as file:
            file.write("file_logging1: true")

        logger = IgnisLogging().get_logger(__name__)
        logger.log_to_file(test="test")

        self.assertFalse(os.path.exists(self._default_log))

    def test_file_true_typo(self):
        """
        Only tests the main param: file_logging
        :return:
        """
        with open(os.path.join(self._qiskit_dir, "logging.yaml"), "w") as file:
            file.write("file_logging: tru")

        logger = IgnisLogging().get_logger(__name__)
        logger.log_to_file(test="test")

        self.assertFalse(os.path.exists(self._default_log))

    def test_log_file_path(self):
        """
        test that a custom log file path is honored
        :return:
        """
        with open(os.path.join(self._qiskit_dir, "logging.yaml"), "w") as file:
            file.write("file_logging: true\nlog_file: test_log.log")

        logger = IgnisLogging().get_logger(__name__)
        logger.log_to_file(test="test")

        self.assertTrue(os.path.exists("test_log.log"))
        try:
            os.remove("test_log.log")
        except OSError:
            pass

    def test_file_rotation(self):
        """
        Test that the file rotation is working
        :return:
        """
        with open(os.path.join(self._qiskit_dir, "logging.yaml"), "w") as file:
            file.write("file_logging: true\n"
                       "max_size: 10\n"
                       "max_rotations: 3")

        logger = IgnisLogging().get_logger(__name__)
        for i in range(100):
            logger.log_to_file(test="test%d" % i)

        self.assertTrue(os.path.exists(self._default_log))
        self.assertTrue(os.path.exists(self._default_log + ".1"))
        self.assertTrue(os.path.exists(self._default_log + ".2"))
        self.assertTrue(os.path.exists(self._default_log + ".3"))

        list(map(os.remove, [self._default_log + ".1",
                             self._default_log + ".2",
                             self._default_log + ".3"]))

    def test_manual_enabling(self):
        """
        Test that enabling the logging manually works
        :return:
        """
        logger = IgnisLogging().get_logger(__name__)
        logger.enable_file_logging()
        logger.log_to_file(test="test")

        self.assertTrue(os.path.exists(self._default_log))

    def test_manual_disabling(self):
        """
        Test that disabling the logging manually works
        :return:
        """
        with open(os.path.join(self._qiskit_dir, "logging.yaml"), "w") as file:
            file.write("file_logging: true\n")

        logger = IgnisLogging().get_logger(__name__)
        logger.disable_file_logging()
        logger.log_to_file(test="test")

        self.assertFalse(os.path.exists(self._default_log))


class TestFileLogging(TestLoggingBase):
    """
    Testing logging actions
    """
    def test_save_line(self):
        """
        Test basic log operations
        """
        logger = IgnisLogging().get_logger(__name__)

        logger.enable_file_logging()
        logger.log_to_file(test="test")

        self.assertTrue(os.path.exists(self._default_log))
        with open(self._default_log, 'r') as file:
            self.assertIn("\'test\':\'test\'", file.read())

    def test_format(self):
        """
        Test format of the saved line
        """
        logger = IgnisLogging().get_logger(__name__)

        logger.enable_file_logging()
        logger.log_to_file(test="test")

        with open(self._default_log, 'r') as file:
            self.assertRegex(
                file.read(),
                r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} ignis_logging \S+")

    def test_multiple_lines(self):
        """
        Test logging multiple lines
        """
        logger = IgnisLogging().get_logger(__name__)

        logger.enable_file_logging()
        logger.log_to_file(test="test1")
        logger.log_to_file(test="test2")

        with open(self._default_log, 'r') as file:
            lines = file.read().split('\n')

        self.assertGreaterEqual(len(lines), 2)


class TestLogReader(TestLoggingBase):
    """
    Testing the IgnisLogReader class
    """
    def setUp(self):
        TestLoggingBase.setUp(self)

    def tearDown(self):
        TestLoggingBase.tearDown(self)

    def test_read_multiple_files(self):
        """
        Test reading multiple lines
        """
        with open(os.path.join(self._qiskit_dir, "logging.yaml"), "w") as file:
            file.write("file_logging: true\nmax_size: 40\nmax_rotations: 5")

        logger = IgnisLogging().get_logger(__name__)
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
        with open(self._default_log, 'w') as log:
            log.write(
                "2019/08/04 13:27:04 ignis_logging \'k1\':\'d1\'\n"
                "2019/08/04 13:27:05 ignis_logging \'k1\':\'d2\'\n"
                "2019/08/04 13:27:06 ignis_logging \'k1\':\'d3\'\n"
                "2019/08/05 13:27:04 ignis_logging \'k2\':\'d4\'\n"
                "2019/08/06 13:27:04 ignis_logging \'k2\':\'d5\'\n"
                "2019/09/02 13:27:04 ignis_logging \'k3\':\'d6\'\n"
                "2019/09/04 13:27:04 ignis_logging \'k4\':\'d7\'\n")

        reader = IgnisLogReader()

        self.assertEqual(len(reader.read_values(keys=["k1", "k2"])), 5)
        self.assertEqual(
            len(reader.read_values(from_datetime="2019/08/05 00:00:00")), 4)
        self.assertEqual(
            len(reader.read_values(to_datetime="2019/08/04 13:27:06")), 3)


if __name__ == '__main__':
    unittest.main(warnings='ignore')
