# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Logging for Ignis
"""

import yaml
import os
import datetime


# TODO: Update docstrings

###########################################################################
# Logging for Ignis:
# Looks for a config file in home/.qiskit/logging.yaml
# logging: true
# logfile: XXX.csv
###########################################################################

class IgnisLogging():
    """Class to read from the log congfig file and write to the log file"""

    def __init__(self):

        """Initialize logging.

        """

        #Path to the config file
        self._log_config_file = os.path.join(os.path.expanduser("~"),
                                             '.qiskit', 'logging.yaml')

        self._load_config()



    def _load_config(self):

        """Load the logging configuration

        """

        if os.file.exists(self._log_config_file):
            #open
            fo = open(log_config_file, 'r')
            log_yaml = yaml.load(fo)
            fo.close()

            if log_yaml['logging']:
                self.logging = True
                self.logfile = log_yaml['logfile']
            else:
                self.logging = False

        else:
            self.logging = False


    def enable_logging(self, filename=None):

        """
        Enable logging if not already enabled in the config file.
        This will enable logging for any future experiments as well.

        Args:
            filename: name for the logging file (uses default if None)

        """

        log_yaml = {'logging': True}

        if filename is None:
            filename = os.path.join(os.path.expanduser("~"),
                                    '.qiskit', 'logfile.csv')

        log_yaml['logfile'] = filename

        #open
        fo = open(self._log_config_file, 'w')
        yaml.dump(fo, log_yaml)
        fo.close()

        self._load_config()


    def disable_logging(self):

        """
        Disable logging if not already disabled in the config file.
        This will disable logging for any future experiments as well.


        """

        if os.file.exists(log_config_file):
            #open
            fo = open(log_config_file)
            log_yaml = yaml.load(fo)
            fo.close()

            log_yaml['logging'] = False

        else:
            log_yaml['logging'] = False


        #open
        fo = open(self._log_config_file, 'w')
        yaml.dump(fo, log_yaml)
        fo.close()

        self._load_config()


    def logvalue(self, qubit_list, valuename, value, valueerr=0, valueunits=''):
        """
        Write a single value to the log file. If logging is disabled this
        will just be ignored.

        Args:
            qubit_list: qubits associated with this value
            valuename: name of the value (e.g. 't1')
            value: the value (a float)
            valuerr: error in the value
            valueunits: string of the units

        """

        if self.logging:
            #log to the file
            fo = open(self.logfile,'a')
            fo.write('%s,%s,%s,%s,%f'%(
                    datetime.datetime.now().strftime('%Y-%M-%d %H:%M:%S'),
                    qubit_list, valuename, valueunits, value))
            fo.close()



class IgnisLogReader():

    """Class to read from the log"""

    def __init__(self):
        pass


    def get_values(self, qubit_list, valuename, daterange=None, valuerange=None):
        """
        Get values from the log file corresponding to the given
        qubit_list and valuename

        Args:
            qubit_list: qubits
            valuename: name of the value (e.g. 't1')
            daterange: optional date range
            valuerange: optional value range

        Return:
            dates
            values

        """

        pass




