import numpy as np
from .single_variate_cpd import SingleVariate
from .multi_variate_cpd import MultiVariate
import pandas as pd
import pickle
from time import time
import networkx as nx
import psopy
from .utils import equal_dicts
import sys
import traceback


class BayesianNetwork:
    def __init__(self, graph=None):

        self.graph = nx.DiGraph() if graph is None else graph
        self.cpds = {}
        self.sorted_nodes = None

        if graph is not None:
            self.sorted_nodes = list(nx.topological_sort(self.graph))
            for node in self.graph.nodes():
                self.cpds[node] = self._createCpd(self.graph, node)

    def _createCpd(self, graph, node):
        data = dict(graph.nodes(data=True))[node]
        if graph.in_degree(node) > 0:
            func_type = "nonlinear"
            if "func_type" in data:
                func_type = data["func_type"]
            else:
                data["func_type"] = func_type

            cpd = MultiVariate(
                list(graph.predecessors(node)), [node], data, func_type=func_type
            )
        else:
            cpd = SingleVariate([node], data)
        return cpd

    def fit(self, data, logger=None, force=False):
        try:
            for node, cpd in self.cpds.items():
                cpd.fit(data, logger, force=force)
            for node in list(self.graph.nodes()):
                cpd = self.cpds[node]
                self.graph.add_node(node, **cpd.get_node_info())
        except Exception as e:
            logger.error(str(e))

    def compare_graph(self, new_graph):
        if not nx.is_isomorphic(self.graph, new_graph):
            return False
        nodes = dict(new_graph.nodes(data=True))
        for node, data in nodes.items():
            current_node_data = dict(self.graph.nodes(data=True))[node]
            if not equal_dicts(current_node_data, data):
                return False
        return True

    def update_graph(self, new_graph):
        nodes = dict(new_graph.nodes(data=True))
        for node, data in nodes.items():
            if node not in self.cpds:
                self.cpds[node] = self._createCpd(new_graph, node)
            else:
                cond1 = new_graph.in_degree(node) != self.graph.in_degree(node)
                set1 = set(self.graph.predecessors(node))
                set2 = set(new_graph.predecessors(node))
                cond2 = set1 != set2
                current_node_data = dict(self.graph.nodes(data=True))[node]
                cond3 = not equal_dicts(current_node_data, data)
                if cond1 or cond2 or cond3:
                    self.cpds[node] = self._createCpd(new_graph, node)

        for node in list(self.cpds.keys()):
            if node not in nodes:
                del self.cpds[node]

        self.graph = new_graph
        self.sorted_nodes = list(nx.topological_sort(self.graph))

    def log_likelihood(self, data, return_all=False, nodes=None, normalize=False):
        llk_nodes = {}
        expected_values = {}
        llk = 0
        num_nodes = np.zeros(data.shape[0], dtype=np.float64) + 0.000001
        for node, cpd in self.cpds.items():
            if nodes is not None and node not in nodes:
                continue
            log_lik, expected_value = cpd.log_likelihood(
                data, return_expected_value=True
            )
            num_nodes += (log_lik != 0).astype(np.float32)
            llk += log_lik
            llk_nodes[node] = log_lik
            expected_values[node] = expected_value

        if normalize:
            llk /= num_nodes

        if return_all:
            return llk, llk_nodes, expected_values
        else:
            return llk

    def get_marginal_pdfs(self, node=None, splits=100):
        pdfs = {}
        if node is None:
            for n in list(self.graph.nodes()):
                pdfs[n] = self.cpds[n].get_marginal_pdf(splits=splits)
        elif node in self.cpds:
            pdfs[node] = self.cpds[node].get_marginal_pdf(splits=splits)
        return pdfs

    def sample(self, n_samples=1):
        samples = []
        for row_index in range(n_samples):
            all_sample = []
            col_names = []
            for col_index, node in enumerate(self.sorted_nodes):
                sample_data = (
                    pd.DataFrame(
                        np.asarray(all_sample).reshape(1, len(col_names)),
                        columns=col_names,
                    )
                    if col_index > 0
                    else None
                )
                gen_val = self.cpds[node].sample(data=sample_data).reshape(-1)[0]
                all_sample.append(gen_val)
                col_names.append(node)
            samples.append(all_sample)

        samples = pd.DataFrame(np.asarray(samples), columns=self.sorted_nodes)
        return samples

    def infer(
        self,
        nodes,
        data,
        n_rnd_samples=5,
        initial_values={},
        node_ranges={},
        normalize=False,
    ):

        if len(nodes) == 0:
            return data, None

        samples = {}
        for row_index, row in enumerate(data[self.sorted_nodes].values):
            random_samples = []
            not_infer_values = []
            infering_nodes = []
            for ind in range(n_rnd_samples):
                sample = []
                all_sample = []
                col_names = []
                for col_index, (val, node) in enumerate(zip(row, self.sorted_nodes)):
                    if node in nodes:
                        sample_data = (
                            pd.DataFrame(
                                np.asarray(all_sample).reshape(1, len(col_names)),
                                columns=col_names,
                            )
                            if col_index > 0
                            else None
                        )
                        gen_val = (
                            self.cpds[node].sample(data=sample_data).reshape(-1)[0]
                        )
                        sample.append(gen_val)
                        all_sample.append(gen_val)
                        if ind == 0:
                            infering_nodes.append(node)
                    else:
                        all_sample.append(val)
                        if ind == 0:
                            not_infer_values.append((node, val))

                    col_names.append(node)

                random_samples.append(sample)

            dep_nodes = self._get_dependent_nodes(
                infering_nodes, {k: v for k, v in zip(self.sorted_nodes, all_sample)}
            )

            def func(x):
                penalty = 0
                df = {}
                for not_infer_val in not_infer_values:
                    df[not_infer_val[0]] = [not_infer_val[1]]
                for xi, infer_val in zip(x, infering_nodes):
                    df[infer_val] = [xi]
                    if infer_val in node_ranges:
                        if (
                            xi < node_ranges[infer_val][0]
                            or xi > node_ranges[infer_val][1]
                        ):
                            penalty += 100 * (
                                1
                                + min(
                                    abs(xi - node_ranges[infer_val][1]),
                                    abs(xi - node_ranges[infer_val][0]),
                                )
                            )
                df = pd.DataFrame.from_dict(df)
                llk = -self.log_likelihood(df, nodes=dep_nodes)[0]
                return llk + penalty

            res = psopy.minimize(
                func,
                random_samples,
                options={
                    "l_rate": 0.5,
                    "max_iter": 5,
                    "ptol": 1e-6,
                    "verbose": False,
                    "stable_iter": 3,
                },
            )
            final_sample = res.x

            for not_infer_val in not_infer_values:
                if not_infer_val[0] not in samples:
                    samples[not_infer_val[0]] = []
                samples[not_infer_val[0]].append(not_infer_val[1])
            for xi, infer_val in zip(res.x, infering_nodes):
                if infer_val not in samples:
                    samples[infer_val] = []
                samples[infer_val].append(xi)

        samples = pd.DataFrame.from_dict(samples)
        llks = self.log_likelihood(samples, normalize=normalize)
        return samples, llks

    def forward_infer(self, nodes, data):
        infered_values = []
        for row_index, row in enumerate(data[self.sorted_nodes].values):
            all_sample = []
            col_names = []
            for col_index, (val, node) in enumerate(zip(row, self.sorted_nodes)):
                if node in nodes:
                    sample_data = (
                        pd.DataFrame(
                            np.asarray(all_sample).reshape(1, len(col_names)),
                            columns=col_names,
                        )
                        if col_index > 0
                        else None
                    )
                    gen_val = self.cpds[node].sample(data=sample_data).reshape(-1)[0]
                    all_sample.append(gen_val)
                else:
                    all_sample.append(val)
                col_names.append(node)
            infered_values.append(all_sample)
        infered_values = pd.DataFrame(
            np.asarray(infered_values), columns=self.sorted_nodes
        )
        return infered_values

    def _get_dependent_nodes(self, nodes, data=None):
        dependent_nodes = nodes
        for node in nodes:
            for succ_n in self.graph.successors(node):
                if succ_n not in dependent_nodes:
                    if data is None or not np.isnan(data[succ_n]):
                        dependent_nodes.append(succ_n)

        return dependent_nodes

    def save(self, path):
        params = {}
        for node, cpd in self.cpds.items():
            params[node] = cpd.get_params()
        with open(path, "wb") as fp:
            pickle.dump({"params": params, "graph": self.graph}, fp)

    @staticmethod
    def load(path):
        model = BayesianNetwork()
        with open(path, "rb") as fp:
            network = pickle.load(fp)
        model.graph = network["graph"]
        # model.node_ranges = network['ranges']
        model.sorted_nodes = list(nx.topological_sort(model.graph))
        for node, params in network["params"].items():
            if params["class"] == "MultiVariate":
                model.cpds[node] = MultiVariate.load(params)
            elif params["class"] == "SingleVariate":
                model.cpds[node] = SingleVariate.load(params)
            else:
                raise ("Error: Class {} does not exist.".format(params["class"]))
        return model

    @property
    def is_fit(self):
        return np.all([cpd.is_fit for cpd in self.cpds])

    @property
    def independent_params(self):
        return np.sum([cpd.n_params for cpd in self.cpds])
