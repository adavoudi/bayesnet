from fitter import Fitter
from scipy import stats
import numpy as np
from .cpd import CPD
from .utils import get_logger
from time import time
import warnings

warnings.filterwarnings("ignore")

logger = get_logger("BayesNet")


class SingleVariate(CPD):
    def __init__(self, targets=None, node_data=None):
        super(SingleVariate, self).__init__([], targets, node_data)

        self.dist_name = None
        self.dist_params = None
        self.dist = None

    def fit(self, data, input_logger=None, force=False):
        if self.is_fit and not force:
            return

        if input_logger is None:
            input_logger = logger

        self._find_node_range(data, input_logger)
        self._reset_marginal_dist(data, input_logger)
        self._is_fit = True

    def log_likelihood(self, data, return_expected_value=False):
        assert self.is_fit, "Model is not fit. Did you call fit()?"

        llks = []
        expected_values = []
        for index in range(data.shape[0]):
            row = data[self.targets].iloc[index : index + 1].values
            if np.any(np.isnan(row)):
                llks.append(0)
                expected_values.append(0)
                continue
            llk = self._marginal_llk(row)
            llks.append(llk[0])
            expected_values.append(self.marginal_expected_val)

        if return_expected_value:
            return np.asarray(llks, dtype=np.float64).reshape(-1), expected_values
        return np.asarray(llks, dtype=np.float64).reshape(-1)

    def sample(self, n_samples=1, data=None):
        assert self.is_fit, "Model is not fit. Did you call fit()?"
        return self._marginal_sample(n_samples).reshape(-1)

    @property
    def is_fit(self):
        return self._is_fit

    def get_params(self):
        return {
            "class": "SingleVariate",
            "dist_type": self.dist_type,
            "dist_params": self.dist_params,
            "dist_name": self.dist_name,
            "targets": self.targets,
            "target_range": self.target_range,
            "kde": self.kde,
            "marginal_expected_val": self.marginal_expected_val,
        }

    @staticmethod
    def load(parameters):
        model = SingleVariate()
        model.dist_type = parameters["dist_type"]
        model.dist_params = parameters["dist_params"]
        model.dist_name = parameters["dist_name"]
        if parameters["dist_type"] == "parametric":
            model.dist = getattr(stats, model.dist_name)
        model.targets = parameters["targets"]
        model.target_range = parameters["target_range"]
        model.kde = parameters["kde"]
        model.marginal_expected_val = parameters["marginal_expected_val"]
        model._is_fit = True
        return model
