"""
Abstract classes for test.
Overwrite the class to write a test.
"""

import csv
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import utils
import generators
import bases
import representations
import eigensolvers
from functools import partial


class TestBase(object):
    def __init__(self, *args, **kwargs) -> None:
        self.root = os.path.join(os.path.dirname(__file__), 'results')
        self.n_iter = kwargs.get('n_iter', 10)
        self.ncols = kwargs.get('ncols', 2)

    def write_results_to_csv(self, filename, header):
        writer = csv.writer(open(os.path.join(self.root, filename), "a", newline=''))
        writer.writerow(header)
        results = []
        for _ in tqdm(range(self.n_iter)):
            result = self.run()
            writer.writerow(result)
            results.append(result)
        return results

    def read_results_from_csv(self, filename):
        results = []
        with open(os.path.join(self.root, filename), 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                results.append([float(x) for x in row])
        return header, results

    def show_stats(self, results, header):
        np_results = np.array(results)
        N = np_results.shape[1]
        np_mean = np.round(np.mean(np_results, axis=0), decimals=2)
        np_std = np.round(np.std(np_results, axis=0), decimals=2)
        np_max = np.round(np.max(np_results, axis=0), decimals=2)
        np_min = np.round(np.min(np_results, axis=0), decimals=2)
        for i in range(N):
            print("{:>15} mean {:>10.2f}, std {:>10.2f}, max {:>10.2f}, min {:>10.2f}".format(header[i], np_mean[i], np_std[i], np_max[i], np_min[i]))
        return np_mean.tolist(), np_std.tolist(), np_max.tolist(), np_min.tolist()

    def show_plot(self, results, header):
        np_results = np.array(results)
        np_max = np.round(np.max(np_results, axis=0), decimals=2)
        fig, axs = plt.subplots(nrows=1, ncols=self.ncols, figsize=(8*self.ncols, 3))
        for c in range(self.ncols):
            axs[c].boxplot(np_results[:,c], vert=True, patch_artist=True)
            # axs[r, c].scatter(np_results[:,r*self.ncols+c], vert=True)
            axs[c].set_title(header[c])
            axs[c].set_ylim([0, max(100, np_max[c])])
        plt.show()

    def test_all(self, filename, header):
        results = self.write_results_to_csv(filename, header)
        self.show_plot(results, header)
        self.show_stats(results, header)
        return results

    def read_and_show_plot(self, filename):
        header, results = self.read_results_from_csv(filename)
        self.show_plot(results, header)
        self.show_stats(results, header)
        return results

    def run(self):
        # override
        raise Exception("runner not implemented.")


class TestNonzeroRatio(TestBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.degree = kwargs.get('degree', 10)
        self.basis = kwargs.get('basis', partial(bases.fourier_basis, degree=self.degree))
        self.input_size = kwargs.get('input_size', 1)
        self.representation = kwargs.get('representation', partial(representations.EDMD_matrix_representation, basis=self.basis))
        self.eigensolver = kwargs.get('eigensolver', partial(eigensolvers.power_orthogonal_QR_algorithm, basis=self.basis, input_size=self.input_size))

    def _get_matrix(self, initial_state=0, num_col=200):
        Ut = np.expand_dims(
            utils.generate_krylov(generators.angle_evolution, generators.full_state_observable, initial_state, num_col), 
            axis=1)
        return Ut

    def run(self):
        U = self._get_matrix()
        K, V, L = self.representation(U)
        new_L, new_V, n = self.eigensolver(K)
        r = utils.get_zero_norm_ratio(new_V)*100
        return [r, n]

    def get_non_zero_mean(self, results):
        np_results = np.array(results)
        np_mean = np.round(np.mean(np_results, axis=0), decimals=2)
        return np_mean[0]

class TestNDTorus(TestBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_dim = kwargs.get('n_dim', 2)
        self.degree = kwargs.get('degree', 10)
        self.input_size = kwargs.get('input_size', 1)
        self.combinations = kwargs.get('combinations', False)
        self.basis = kwargs.get('basis', partial(bases.fourier_basis, degree=self.degree, combinations=self.combinations))
        self.representation = kwargs.get('representation', partial(representations.mpEDMD_matrix_representation, basis=self.basis))
        self.eigensolver = kwargs.get('eigensolver', partial(eigensolvers.power_orthogonal_QR_algorithm, basis=self.basis, input_size=self.input_size))

    def _get_matrix(self, num_col=200):
        frequencies, initial_points = generators.n_torus_initial_points(n=self.n_dim)
        Xr, Er = utils.generate_krylov(partial(generators.n_torus_evolution_function, frequencies=frequencies), generators.full_state_observable, initial_points, num_col)
        if self.n_dim==2: utils.plot_on_unit_square(Er)
        K, V, D = self.representation(Xr)
        return K, V, D

    def run(self):
        self.K, self.V, self.D = self._get_matrix()
        new_L, self.new_V, self.n = self.eigensolver(self.K)
        self.r = utils.get_zero_norm_ratio(self.new_V)*100
        return [self.r, self.n]
