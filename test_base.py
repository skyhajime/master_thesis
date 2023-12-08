"""
Abstract classes for test.
Overwrite the class to write a test.
"""

import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from functools import partial
import pickle
from copy import deepcopy
import time

import utils
import generators
import bases
import functionals
import representations
import reconstructions
import eigensolvers
import eigenfunctions

pd.set_option("display.precision", 2)


class TestBase(object):
    def __init__(self, *args, **kwargs) -> None:
        self.root = os.path.join(os.path.dirname(__file__), 'results')
        self.n_iter = kwargs.get('n_iter', 10)
        self.ncols = kwargs.get('ncols', 2)

    def save_results_to_pickle(self, results, filename):
        return pickle.dump(results, open(os.path.join(self.root, filename), "wb"))
    
    def read_results_from_pickl(self, filename):
        return pickle.load(open(os.path.join(self.root, filename), "rb"))

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
                try:
                    results.append([float(x) for x in row])
                except ValueError:
                    pass
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

class TestTwoDTorus(TestBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_dim = 2
        self.degree = kwargs.get('degree', 10)
        self.input_size = 1
        self.n_samples = kwargs.get('n_samples', 100)
        self.combinations = kwargs.get('combinations', True)
        self.basis = kwargs.get('basis', partial(bases.fourier_basis, degree=self.degree, combinations=self.combinations))
        self.representation = kwargs.get('representation', partial(representations.mpEDMD_matrix_representation, basis=self.basis))
        self.eigensolver = kwargs.get('eigensolver', partial(eigensolvers.power_orthogonal_QR_algorithm, basis=self.basis, input_size=self.input_size))

    def _get_matrix(self):
        frequencies, initial_points = generators.n_torus_initial_points(n=self.n_dim)
        self.Xr, self.Er = utils.generate_krylov(partial(generators.n_torus_evolution_function, frequencies=frequencies), generators.twoD_state_to_complex, initial_points, self.n_samples)
        utils.plot_on_unit_square(self.Er)
        K, V, D = self.representation(self.Xr)
        return K, V, D

    def _get_koopman_modes(self):
        return reconstructions.get_koopman_modes(self.Xr, np.linalg.inv(self.V), self.basis)

    def run(self):
        self.K, self.V, self.D = self._get_matrix()
        new_L, self.new_V, self.n = self.eigensolver(self.K)
        self.r = utils.get_zero_norm_ratio(self.new_V)*100
        return [self.r, self.n]

class TestNDTorus(TestBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_show_plot = kwargs.get('is_show_plot', False)
        self.n_dim = kwargs.get('n_dim', 2)
        self.degree = kwargs.get('degree', 10)
        self.input_size = kwargs.get('n_dim', 1)
        self.num_col = kwargs.get('num_col', 10**3)
        self.basis = kwargs.get('basis', partial(bases.fourier_basis, degree=self.degree, combinations=True))
        self.representation = kwargs.get('representation', partial(representations.mpEDMD_matrix_representation, basis=self.basis))
        self.eigensolver = kwargs.get('eigensolver', partial(eigensolvers.power_orthogonal_QR_algorithm, basis=self.basis, input_size=self.input_size))

    def _get_settings(self):
        self.frequencies, self.initial_state = generators.n_torus_initial_points(n=self.n_dim)
        Yr, Xr = utils.generate_krylov(partial(generators.n_torus_evolution_function, frequencies=self.frequencies), generators.flat_torus_observable_function, self.initial_state, self.num_col)
        if self.n_dim ==1 and self.is_show_plot: utils.plot_complex_on_unit_circle(generators.unit_circle_observable_function(Yr))
        elif self.n_dim==2 and self.is_show_plot: utils.plot_on_unit_square(Xr)
        return Yr, Xr

    def _get_matrix(self):
        self.Xr, self.Er = self._get_settings()
        if self.n_dim==2 and self.is_show_plot: utils.plot_on_unit_square(self.Er)
        K, V, D = self.representation(self.Xr)
        return K, V, D

    def _get_koopman_modes(self):
        return reconstructions.get_koopman_modes(self.Xr, np.linalg.inv(self.V), self.basis)

    def run(self):
        self.K, self.V, self.D = self._get_matrix()
        new_L, self.new_V, self.n = self.eigensolver(self.K)
        self.r = utils.get_zero_norm_ratio(self.new_V)*100
        return [self.r, self.n]

class TestEigenfuncNDTorus(TestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_show_plot = kwargs.get('is_show_plot', True)
        self.n_dim = kwargs.get('n_dim', 1)
        self.frequencies, self.initial_state = generators.n_torus_initial_points(n=self.n_dim)
        if 'frequencies' in kwargs:
            self.frequencies = kwargs['frequencies']  # update frequencies if it is given
        if 'initial_state' in kwargs:
            self.initial_state = kwargs['initial_state']  # update initial_state if it is given
        self.degree = kwargs.get('degree', 10)
        self.basis = partial(bases.fourier_basis, degree=self.degree, combinations=True)
        self.num_col = kwargs.get('num_col', 10**3)
        self.representation_func = representations.mpEDMD_matrix_representation
        self.filename = kwargs.get('filename', f'test_eigenfunction_{self.n_dim}Dtorus_degree{self.degree}.pickle')

    def get_settings(self):
        self.Yr, self.Xr = utils.generate_krylov(partial(generators.n_torus_evolution_function, frequencies=self.frequencies), generators.flat_torus_observable_function, self.initial_state, self.num_col)
        if self.n_dim ==1 and self.is_show_plot: utils.plot_complex_on_unit_circle(generators.unit_circle_observable_function(self.Yr))
        elif self.n_dim==2 and self.is_show_plot: utils.plot_on_unit_square(self.Xr)
        return self.Yr, self.Xr
    
    def plot_eigenfunctions(self, domain, vals, n_row, labels):
        n_col = 1
        LINE_STYLES = ['b--', 'm:', 'r-.', 'g-']
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(16, 6*n_row))
        for idx in range(n_row):
            ax = axes[idx]
            for j, val in enumerate(vals):
                ax.plot(domain, val[:,idx].real, LINE_STYLES[j])
            ax.legend(labels)
        plt.show()
        return

    def compare_on_unit_circle(self, vals, col_names=['mpEDMD', 'True', 'QR+EIG', 'QR+Power'], n_row=3, N_max=20):
        n_col = len(vals)
        fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(3*n_col, 3*n_row))
        for c, name in enumerate(col_names):
            axs[0, c].set_title(name)
        for i in range(n_row):
            for j, val in enumerate(vals):
                utils.set_subplot_complex_on_unit_circle(val[:N_max,2*i], axs[i, j], add_color=True)
        plt.show()

    def show_E1_value(self, V1, V2, eigenvalues):
        """
        V1: true vectors
        V2: reconstructed vectors
        """
        r = V1.shape[0]
        self.ip = [1-abs(np.inner(V1[:,k], V2[:,k].conj()))/(2*np.pi) for k in range(r)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ev_angles = np.angle(eigenvalues)
        self.p = [ev_angles[k]/ev_angles[0] for k in range(r)]
        ax.scatter(self.p, self.ip)
        ax.set_ylim([0,1])
        plt.show()
        self.special_idxes = [i for i, e in enumerate(self.ip) if e < 0.1]
        return
    
    def get_E2_error(self, f1, f2, X, N=100):
        alpha = sum([np.divide(np.squeeze(f1(X[0])), np.squeeze(f2(X[0]))) for i in range(N) if any(abs(f2(X[i])) > 1e-3)])
        f = lambda x: f1(x) - np.multiply(alpha, np.squeeze(f2(x)))
        norm = functionals.functional_inner_product(f, f, -np.pi, np.pi, input_size=1)
        return np.squeeze(np.sqrt(abs(norm)))

    def show_E2_error(self, true_func, funcs, labels, X, n_rows=5):
        fig, axes = plt.subplots(nrows=n_rows, figsize=(10, 5*n_rows))
        y = np.array([self.get_E2_error(func, true_func, X) for func in funcs])
        for i in range(n_rows):
            ax = axes[i]
            ax.scatter(labels, y[:,i])
        plt.show()
        return y

    def get_E3_error(self, Ph, L, k=0, N=20, M=None):
        if M is None:
            M = self.Yr.shape[0]-1
        results = []
        for n in range(N+1):
            Phn = lambda x: Ph(x)**n
            Ln = pow(L[k], n)
            diff = sum(np.linalg.norm(Phn(self.Yr[i+1]) - Ln*Phn(self.Yr[i])) for i in range(M))/M
            results.append([n, Ln in L, np.isclose(diff, 0).item(), diff])
        cols = ['n', 'is_in_eigenvalue', 'is_same', 'diff_norm']
        df = pd.DataFrame(results, columns=cols)
        return df

    def get_eigenfunction_reconstruction(self, V):
        return eigenfunctions.get_eigenfunctions_from_matrix(self.basis, V)

    def get_reconstruction(self, L, V, M):
        KM = reconstructions.get_koopman_modes(self.Yr, np.linalg.inv(V), self.basis)
        Ph = self.get_eigenfunction_reconstruction(V)(self.Yr[0])
        return np.array([np.squeeze(Ph @ np.diag(np.power(L, k)) @ KM) for k in range(M)])

    def get_E4_error(self, L, V, M=None):
        if M is None:
            M = self.Yr.shape[0]-1
        original_data = np.squeeze(self.Yr)[:M]
        reconstruction = self.get_reconstruction(L, V, M)
        return np.linalg.norm(original_data - reconstruction)/M

    def show_E4_error(self, K, L, div=10):
        r = K.shape[0]
        results = []
        for n_inv in range(1, r+1, r//div):
            try:
                st = time.time()
                iL, ipower_V, _, _ = eigensolvers.inverse_iteration_with_integer_power(K, self.basis, n_dim=self.Yr.shape[1], n_inv=n_inv, eigenvalue_approx=L)
                error = self.get_E4_error(iL, ipower_V)
                et = time.time()
                print(f"n_inv={n_inv}, norm={round(error, 2)}, time_taken={et-st}")
                results.append((n_inv, error, et-st))
            except AssertionError:
                print(f"n_inv={n_inv}, eigenvalue assertion error.")
                continue
            except np.linalg.LinAlgError:
                print(f"n_inv={n_inv}, matrix inverse error")
                continue

        df = pd.DataFrame.from_records(results, columns=['n_inv', 'error', 'time'])
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.n_inv, df.error)
        plt.show()
        return df

    def run(self):
        self.Yr, self.Xr = self.get_settings()
        K, V, L = self.representation_func(self.Yr, basis=self.basis)
        _, true_V, _ = eigensolvers.QR_algorithm_with_inverse_iteration(K)
        # _, power_V, n = eigensolvers.QR_algorithm_with_power_vector(K, basis, Xr.shape[1])
        # power_V = self.get_power_vector_with_one_true_eigenfunction(eigenvalues, basis

        mpPh = eigenfunctions.get_eigenfunctions_from_matrix(self.basis, V)
        newPh = eigenfunctions.get_eigenfunctions_from_matrix(self.basis, true_V)
        L_angles = np.angle(L)
        v0 = deepcopy(V[:,0])
        powPh = eigenfunctions.get_eigenfunctions_with_exponential(self.basis, v0, [angle/L_angles[0] for angle in L_angles])
        domain = np.linspace([-np.pi for _ in range(self.n_dim)], [np.pi for _ in range(self.n_dim)], num=500)
        # trPh = lambda x: np.array([self.get_true_eigenfunction(eig)(x) for eig in eigenvalues])
        mp_ef_val = np.array([np.squeeze(mpPh(d)) for d in domain])
        new_ef_val = np.array([np.squeeze(newPh(d)) for d in domain])
        pow_ef_val = np.array([np.squeeze(powPh(d)) for d in domain])
        # tr_ef_val = np.array([trPh(d) for d in domain])
        assert mp_ef_val.shape == pow_ef_val.shape == pow_ef_val.shape

        # show plot on the same graph
        plot_domain = domain[:,0] if domain.ndim > 1 else domain
        vals = [mp_ef_val, pow_ef_val, new_ef_val]
        labels = ['mpEDMD', 'Power', 'QR+EIG']
        self.plot_eigenfunctions(domain=plot_domain, vals=vals, n_row=5, labels=labels)

        # self.compare_on_unit_circle(mp_ef_val, tr_ef_val, new_ef_val, pow_ef_val, n_row=n_row)
        Vs = [V, true_V]
        results = [K, Vs, L, labels, self.n_dim, self.degree, self.frequencies]
        self.save_results_to_pickle(results, self.filename)

        return results
