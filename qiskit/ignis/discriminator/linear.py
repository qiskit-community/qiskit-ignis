
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from qiskit.result import Result
from qiskit.result.models import ExperimentResult
from qiskit.ignis.discriminator.discriminator import AbstractDiscriminator


class LinearDiscriminator(AbstractDiscriminator):
    """
    Linear discriminant analysis based on scikit learn's LinearDiscriminantAnalysis.
    """

    def __init__(self, result: Result, **discriminator_parameters):
        super().__init__(result, **discriminator_parameters)

        self.lda = LinearDiscriminantAnalysis(
            solver=discriminator_parameters.get('solver', 'svd'),
            shrinkage=discriminator_parameters.get('shrinkage', None),
            store_covariance=discriminator_parameters.get('store_covariance', False),
            tol=discriminator_parameters.get('tol', 1.0e-4))

    def _extract_data(self, ):

        exp_result = self.result.memory()

    def fit(self):
        X = None
        y = None
        self.lda.fit(X, y)
