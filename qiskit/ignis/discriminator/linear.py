
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from qiskit.result import Result
from qiskit.ignis.discriminator.discriminator import AbstractDiscriminator


class LinearDiscriminator(AbstractDiscriminator):
    """
    A linear discriminant analysis based on scikit learn's LinearDiscriminantAnalysis.
    """

    def __init__(self, result: Result, **discriminator_parameters):
        super().__init__(result, **discriminator_parameters)

        self.lda = LinearDiscriminantAnalysis(
            solver=discriminator_parameters.get('solver', 'svd'),
            shrinkage=discriminator_parameters.get('shrinkage', None),
            store_covariance=discriminator_parameters.get('store_covariance', False),
            tol=discriminator_parameters.get('tol', 1.0e-4))

    def _extract_data(self):

        x = []

        for circuit in self._cal_circuits:
            x_point = []
            for result in self.result.get_memory(circuit):
                x_point.append(np.real(result))
                x_point.append(np.imag(result))

            x.append(x_point)

        return x

    def fit(self):

        # 1: extract the cals into X and y. -> how to do this?
        # 2: use the cals to fit the discriminator

        x = self._extract_data()
        self.lda.fit(x, self._cal_circuits_expected)
