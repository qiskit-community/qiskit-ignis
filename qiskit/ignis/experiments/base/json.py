# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
JSON serialization and deserialization
"""
# pylint: disable=missing-function-docstring, arguments-differ, method-hidden

import base64
import io
import json
import numpy


def numpy_to_binary(arr):
    """Convert a Numpy array to a JSON serializable binary string."""
    with io.BytesIO() as tmp:
        numpy.save(tmp, arr, allow_pickle=False, fix_imports=False)
        return base64.b64encode(tmp.getvalue()).decode()


def numpy_from_binary(obj):
    """Convert binary string to a Numpy array."""
    with io.BytesIO() as tmp:
        tmp.write(base64.b64decode(obj))
        tmp.seek(0)
        return numpy.load(tmp, allow_pickle=False, fix_imports=False)


class NumpyEncoder(json.JSONEncoder):
    """JSON Encoder for Numpy arrays and complex numbers."""

    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return {'type': 'array', 'value': numpy_to_binary(obj)}
        if isinstance(obj, complex):
            return {'type': 'complex', 'value': [obj.real, obj.imag]}
        return super(NumpyEncoder, self).default(obj)


class NumpyDecoder(json.JSONDecoder):
    """JSON Decoder for Numpy arrays and complex numbers."""

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if 'type' in obj:
            if obj['type'] == 'complex':
                val = obj['value']
                return val[0] + 1j * val[1]
            if obj['type'] == 'array':
                return numpy_from_binary(obj['value'])
        return obj
