from abc import ABCMeta
from abc import abstractmethod
from .utils import get_logger
from scipy import stats
from fitter import Fitter
from time import time
from sklearn.neighbors import KernelDensity
import numpy as np

max_samples = 100000
logger = get_logger("BayesNet")


class CPD:
    __metaclass__ = ABCMeta

    def __init__(self, observations, targets, node_data):
        super(CPD, self).__init__()

        self.observations = observations
        self.targets = targets
        self.node_data = {} if node_data is None else node_data

        self.target_range = None
        self.dist_name = None
        self.dist_type = None
        self.dist_params = None
        self.func_type = None
        self.kde = None
        self.marginal_expected_val = None

        self._is_fit = False

    def _find_node_range(self, data, input_logger=None):
        if input_logger is None:
            input_logger = logger
        node_range = self.node_data["range"] if "range" in self.node_data else None
        if node_range is None:
            input_logger.info("Finiding `{}` range ...".format(",".join(self.targets)))
            data_all = data[self.targets]
            node_range = [data_all.min().values[0], data_all.max().values[0]]
            input_logger.info("----> min: {:.2f}, max: {:.2f}".format(*node_range))
        self.target_range = node_range

    def _parametric_fitter(
        self,
        data,
        dists=[
            "norm",
            "expon",
            "exponpow",
            "dweibull",
            "gamma",
            "genlogistic",
            "halfnorm",
            "invgauss",
            "invweibull",
            "logistic",
        ],
    ):
        f = Fitter(data.values, timeout=20, verbose=False, distributions=dists)
        f.fit()
        dist = f.get_best()
        dist_name = list(dist.keys())[0]
        dist_params = dist[dist_name]
        return dist_name, dist_params

    def get_node_info(self):
        info = {
            "range": self.target_range,
            "dist_type": self.dist_type,
            "dist_info": {"dist_name": self.dist_name, "dist_params": self.dist_params},
        }
        if self.func_type is not None:
            info["func_type"] = self.func_type
        return info

    def _marginal_llk(self, data):
        if self.dist_type == "parametric":
            llk = self.dist.logpdf(data, *self.dist_params)
        else:
            llk = self.kde.score_samples(data)
        return llk

    def _marginal_pdf(self, data):
        if self.dist_type == "parametric":
            pdf = self.dist.pdf(data, *self.dist_params)
        else:
            pdf = np.exp(self.kde.score_samples(data))
        return pdf

    def _marginal_sample(self, n_samples):
        if self.dist_type == "parametric":
            samples = self.dist.rvs(*self.dist_params, n_samples)
        else:
            samples = self.kde.sample(n_samples)
        return samples

    def _reset_marginal_dist(self, data, input_logger=None):
        if input_logger is None:
            input_logger = logger

        if self.node_data is None:
            self.node_data = {}

        self.dist_type = "parametric"
        if "dist_type" in self.node_data and self.node_data["dist_type"] is not None:
            self.dist_type = self.node_data["dist_type"]

        dist_info = {}
        if "dist_info" in self.node_data:
            dist_info = self.node_data["dist_info"]

        data_all = data[self.targets].dropna()
        if data_all.shape[0] > max_samples:
            data_all = data_all.sample(n=max_samples)

        if self.dist_type == "parametric":
            if "dist_name" in dist_info and dist_info["dist_name"] is not None:
                try:
                    self.dist = getattr(stats, dist_info["dist_name"])

                    self.dist_name = dist_info["dist_name"]
                    if (
                        "dist_params" in dist_info
                        and dist_info["dist_params"] is not None
                    ):
                        self.dist_params = dist_info["dist_params"]
                    else:
                        input_logger.info(
                            "Fitting P("
                            + ",".join(self.targets)
                            + ") with dist `{}`".format(self.dist_name)
                        )

                        start_time = time()
                        _, params = self._parametric_fitter(
                            data_all, dists=[self.dist_name]
                        )
                        input_logger.info(
                            "---> Finished in {0:.2f} seconds.".format(
                                time() - start_time
                            )
                        )
                except:
                    input_logger.info(
                        "The distribution `{}` does not exist".format(
                            self.node_data["dist_name"]
                        )
                    )
            else:
                input_logger.info("Fitting P(" + ",".join(self.targets) + ")")
                start_time = time()
                self.dist_name, self.dist_params = self._parametric_fitter(data_all)
                self.dist = getattr(stats, self.dist_name)
                input_logger.info(
                    "---> Finished in {0:.2f} seconds.".format(time() - start_time)
                )
                input_logger.info("---> Best distribution: {}".format(self.dist_name))
        else:  # "non-parametric":
            self.dist_name = "kde"
            self.dist_params = [0.75]
            if "dist_params" in dist_info and dist_info["dist_params"] is not None:
                self.dist_params = [max(0.01, dist_info["dist_params"][0])]
            input_logger.info(
                "Fitting P("
                + ",".join(self.targets)
                + ") with Kernel Density Estimaton and bandwidth `{}`".format(
                    self.dist_params[0]
                )
            )
            start_time = time()
            self.kde = KernelDensity(
                kernel="gaussian", bandwidth=self.dist_params[0]
            ).fit(data_all.values)
            input_logger.info(
                "---> Finished in {0:.2f} seconds.".format(time() - start_time)
            )

        self.marginal_expected_val = np.mean(self._marginal_sample(100))
        self.marginal_expected_val = max(
            self.marginal_expected_val, self.target_range[0]
        )
        self.marginal_expected_val = min(
            self.marginal_expected_val, self.target_range[1]
        )

    def get_marginal_pdf(self, start=None, end=None, splits=100):
        start = self.target_range[0] if start is None else start
        end = self.target_range[1] if end is None else end
        x = np.linspace(start, end, splits).reshape(-1, 1)
        pdf = self._marginal_pdf(x).reshape(-1)
        return {"x": x.reshape(-1).tolist(), "pdf": pdf.tolist()}

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def log_likelihood(self, data):
        pass

    @abstractmethod
    def n_params(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def __repr__(self):
        s = self.__class__.__name__ + "P("
        s += ",".join(self.targets)
        if len(self.observations) > 0:
            s += " | "
            s += ",".join(self.observations)
        s += ")"
        return s
