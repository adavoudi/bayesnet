from scipy import stats
from fitter import Fitter
import numpy as np
from .cpd import CPD
from .utils import get_logger
from time import time
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import warnings

warnings.filterwarnings("ignore")
# import xgboost as xgb

max_samples = 100000

logger = get_logger("BayesNet")


class MultiVariate(CPD):
    def __init__(self, observations=None, targets=None, node_data=None, func_type=None):
        super(MultiVariate, self).__init__(observations, targets, node_data)
        self.cpds = {}
        self.func_type = func_type

    def fit(self, data, input_logger=None, force=False):
        if self.is_fit and not force:
            return

        if input_logger is None:
            input_logger = logger

        self._find_node_range(data, input_logger)

        input_logger.info(
            "Fitting P("
            + ",".join(self.targets)
            + " | "
            + ",".join(self.observations)
            + ")"
        )

        for r in range(1, len(self.observations) + 1):
            for comb in combinations(self.observations, r):
                comb = list(comb)
                input_logger.info(
                    "-> Fitting P("
                    + ",".join(self.targets)
                    + " | "
                    + ",".join(comb)
                    + ")"
                )

                data_all = data[comb + self.targets].dropna()
                if data_all.shape[0] > max_samples:
                    data_all = data_all.sample(n=max_samples)

                start_time = time()

                if self.func_type == "nonlinear":
                    input_logger.info(
                        "---> (1) Fitting non-linear model on {} data ...".format(
                            data_all.shape[0]
                        )
                    )
                    # Non-Linear Gaussian
                    # xgb.XGBRegressor(objective="reg:squarederror")
                    model = MLPRegressor(hidden_layer_sizes=(2 * len(comb)))
                elif self.func_type == "linear":
                    input_logger.info(
                        "---> (1) Fitting linear model on {} data ...".format(
                            data_all.shape[0]
                        )
                    )
                    # Linear Gaussian
                    model = LinearRegression()

                model.fit(X=data_all[comb].values, y=data_all[self.targets].values)
                input_logger.info(
                    "--------> Finished in {0:.2f} seconds.".format(time() - start_time)
                )

                input_logger.info("---> (2) Fitting standard deviation ...")
                start_time = time()
                a = model.predict(data_all[comb].values)
                b = data_all[self.targets].values.reshape(-1)
                residual = np.square(a - b)
                std = np.sqrt(np.mean(residual))
                input_logger.info(
                    "--------> Finished in {0:.2f} seconds.".format(time() - start_time)
                )
                input_logger.info("--------> Standard deviation {0:.2f} ".format(std))

                self.cpds[",".join(comb)] = {"model": model, "std": std}

        self._reset_marginal_dist(data, input_logger)
        self._is_fit = True

    def log_likelihood(self, data, return_expected_value=False):
        # assert self.is_fit, 'Model is not fit. Did you call fit()?'

        nodes = self.observations + self.targets

        llks = []
        expected_values = []
        for index in range(data.shape[0]):
            row = data.iloc[index : index + 1]
            # print(row[self.targets].values)
            if np.all(np.isnan(row[self.targets].values)):
                llks.append(0)
                expected_values.append(0)
                continue

            # print(row[self.observations].values)
            if np.all(np.isnan(row[self.observations].values)):
                # use the marginal
                llk = self._marginal_llk(row[self.targets].values[0])
                expected_values.append(self.marginal_expected_val)
            else:
                # find not-nan observations
                valid_observations = []
                for observ in self.observations:
                    if not np.any(np.isnan(row[observ].values)):
                        valid_observations.append(observ)

                cpd_name = ",".join(valid_observations)
                std = self.cpds[cpd_name]["std"]
                model = self.cpds[cpd_name]["model"]

                mus = model.predict(row[valid_observations].values).reshape(-1)[0]
                mus = max(mus, self.target_range[0])
                mus = min(mus, self.target_range[1])

                a, b = (self.target_range[0] - mus) / std, (
                    self.target_range[1] - mus
                ) / std
                c = (row[self.targets].values[0] - mus) / std
                llk = stats.truncnorm.logpdf(c, a, b)
                # llk -= 0.5 * np.square((row[self.targets].values[0] - mus) / (std + epsilon))[0]
                expected_values.append(mus)

            llks.append(llk)

        llks = np.asarray(llks, dtype=np.float64).reshape(-1)
        if return_expected_value:
            return llks, expected_values
        return llks

    def sample(self, data):
        assert self.is_fit, "Model is not fit. Did you call fit()?"

        all_samples = []
        for index in range(data.shape[0]):
            row = data.iloc[index : index + 1]

            if np.all(np.isnan(row[self.observations].values)):
                # use the marginal
                smpl = self._marginal_sample(1)[0]
            else:
                # find not-nan observations
                valid_observations = []
                for observ in self.observations:
                    if not np.any(np.isnan(row[observ].values)):
                        valid_observations.append(observ)

                cpd_name = ",".join(valid_observations)
                std = self.cpds[cpd_name]["std"]
                model = self.cpds[cpd_name]["model"]

                mus = model.predict(row[valid_observations].values).reshape(-1)[0]
                rnd = np.random.normal(0, 1, data.shape[0])
                smpl = rnd * std + mus

            all_samples.append(smpl)

        return np.asarray(all_samples, dtype=np.float64).reshape(-1)

    @property
    def is_fit(self):
        return self._is_fit

    def get_params(self):
        return {
            "class": "MultiVariate",
            "cpds": self.cpds,
            "dist_type": self.dist_type,
            "dist_params": self.dist_params,
            "dist_name": self.dist_name,
            "observations": self.observations,
            "targets": self.targets,
            "target_range": self.target_range,
            "func_type": self.func_type,
            "kde": self.kde,
            "marginal_expected_val": self.marginal_expected_val,
        }

    @staticmethod
    def load(parameters):
        model = MultiVariate()
        model.cpds = parameters["cpds"]
        model.observations = parameters["observations"]
        model.dist_type = parameters["dist_type"]
        model.dist_params = parameters["dist_params"]
        model.dist_name = parameters["dist_name"]
        model.func_type = parameters["func_type"]
        if parameters["dist_type"] == "parametric":
            model.dist = getattr(stats, model.dist_name)
        model.targets = parameters["targets"]
        model.target_range = parameters["target_range"]
        model.kde = parameters["kde"]
        model.marginal_expected_val = parameters["marginal_expected_val"]
        model._is_fit = True
        return model
