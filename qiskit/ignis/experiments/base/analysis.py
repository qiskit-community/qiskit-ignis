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
Experiment Analysis class.
"""

import json
from typing import Optional, Union, Dict, List, Callable

from qiskit.providers import BaseJob
from qiskit.result import Result, Counts
from qiskit.exceptions import QiskitError
from .json import NumpyDecoder, NumpyEncoder


class Analysis:
    """Experiment result analysis class."""

    def __init__(self,
                 data: Optional[Union[BaseJob, Result, List[any], any]] = None,
                 metadata: Optional[Union[List[Dict[str, any]], Dict[str, any]]] = None,
                 name: Optional[str] = None,
                 exp_id: Optional[str] = None,
                 analysis_fn: Optional[Callable] = None):
        """Initialize the analysis object."""
        # Experiment identification metadata
        self._name = name
        self._exp_id = exp_id

        # Experiment result data
        self._exp_data = []
        self._exp_metadata = []
        self._result = None

        # Optionally initialize with data
        self.add_data(data, metadata)

        # Analysis function
        # NOTE: subclasses can optionally override the run method to not
        # use this callable function. If it is used it must have signature
        # (data: List[any], metadata: List[Dict[str, any]], **kwargs: Dict[str, any])
        self._analysis_fn = analysis_fn

    def run(self, **params) -> any:
        """Analyze the stored data.

        Returns:
            any: the output of the analysis,
        """
        self._result = self._analysis_fn(self.data, self.metadata, **params)
        return self._result

    @property
    def result(self) -> any:
        """Return the analysis result"""
        if self._result is None:
            raise QiskitError("No analysis results are stored. Run analysis first.")
        return self._result

    def plot(self, *args, **kwargs) -> Union[None, any, List[any]]:  # any = Matplotlib figure
        """Generate a plot of analysis result.

        Args:
            args: Optional plot arguments
            kwargs: Optional plot kwargs.

        Additional Information:
            This is a base class method that should be overridden by any
            experiment Analysis subclasses that generate plots.
        """
        # pylint: disable=unused-argument
        return None

    @property
    def exp_id(self) -> str:
        """Return the experiment id used to filter added data."""
        return self._exp_id

    @exp_id.setter
    def exp_id(self, value: str):
        """Set the experiment id  used to filter added data."""
        self._exp_id = str(value)

    @property
    def name(self) -> str:
        """Return the experiment name used to filter added data."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the experiment name used to filter added data."""
        self._name = str(value)

    @property
    def data(self) -> List[any]:
        """Return stored data"""
        return self._exp_data

    @property
    def metadata(self) -> List[Dict[str, any]]:
        """Return stored metadata"""
        return self._exp_metadata

    def clear_data(self):
        """Clear stored data"""
        self._exp_data = []
        self._exp_metadata = []
        self._result = None

    def add_data(self,
                 data: Union[BaseJob, Result, any],
                 metadata: Optional[Dict[str, any]] = None):
        """Add additional data to the fitter.

        Args:
                data: input data for the fitter.
                metadata: Optional, list of metadata dicts for input data.
                          if None will be taken from data Result object.

        Raises:
            QiskitError: if input data is incorrectly formatted.
        """
        if data is None:
            return

        if isinstance(data, BaseJob):
            data = data.result()

        if isinstance(data, Result):
            # Extract metadata from result object if not provided
            if metadata is None:
                if not hasattr(data.header, "metadata"):
                    raise QiskitError("Experiment is missing metadata.")
                metadata = data.header.metadata

            # Get data from result
            new_data = []
            new_meta = []
            for i, meta in enumerate(metadata):
                if self._accept_data(meta):
                    new_data.append(self._format_data(data, meta, i))
                    new_meta.append(meta)
        else:
            # Add general preformatted data
            if not isinstance(data, list):
                data = [data]

            if metadata is None:
                # Empty metadata incase it is not needed for a given experiment
                metadata = len(data) * [{}]
            elif not isinstance(metadata, list):
                metadata = [metadata]

            # Filter data
            new_data = []
            new_meta = []
            for i, meta in enumerate(metadata):
                if self._accept_data(meta):
                    new_data.append(data)
                    new_meta.append(meta)

        # Add extra data
        self._exp_data += new_data
        self._exp_metadata += new_meta

        # Check metadata and data are same length
        if len(self._exp_metadata) != len(self._exp_data):
            raise QiskitError("data and metadata lists must be the same length")

    def serialize_result(self, encoder: Optional[json.JSONEncoder] = NumpyEncoder) -> str:
        """Serialize the analysis result for storing in a database.

        Args:
            encoder: Optional, a custom JSON encoder to use for serialization.

        Returns:
            str: A JSON string for serialization
        """
        # TODO: Write custom JSON encoder for complex / Numpy types (or use Terra's)
        return json.dumps(self.result, cls=encoder)

    @staticmethod
    def deserialize_result(result: str, decoder: json.JSONDecoder = NumpyDecoder) -> any:
        """Deserialize a JSON string to an analysis result.

        Args:
            result: The JSON string to be deserialized.
            decoder: Optional, a custom JSON decoder to use for deserialization.

        Returns:
            any: the deserialized Analysis result.
        """
        # TODO: Write a custom JSON decoder
        return json.loads(result, cls=decoder)

    def _format_data(self, data: Result,
                     metadata: Dict[str, any],
                     index: int) -> Counts:
        """Format the required data from a Result.data dict

        Additional Information:
            This extracts counts from the experiment result data.
            Analysis subclasses can override this method to extract
            different data types from circuit results.
        """
        # Derived classes should override this method to filter
        # only the required data.
        # The default behavior filters on counts.
        return data.get_counts(index)

    def _accept_data(self, metadata: Dict[str, any]) -> bool:
        """Return True if a data should be added based on metadata.

        Filtering is based on experiment name, id, and tags.
        """
        # Check experiment name matches
        if self._name is not None and metadata.get("name") != self._name:
            return False
        # Check experiment id matches
        if self._exp_id is not None and metadata.get("exp_id") != self._exp_id:
            return False

        return True

    def __len__(self) -> int:
        """Return the number of stored results"""
        return len(self._exp_data)
