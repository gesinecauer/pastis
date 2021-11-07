import pytest
import numpy as np
from scipy import sparse
from sklearn.metrics import euclidean_distances


class Simulation:
    def __init__(self, seed, lengths, ploidy):
        self.seed = seed
        self.lengths = lengths
        self.ploidy = ploidy

        self.dis = euclidean_distances(self.struct_true)

    def get_counts(self, alpha, beta, ratio_ua=1, ratio_pa=0, ratio_ambig=0,
                   use_poisson=False):
        dis = self.dis.copy()
        dis[dis == 0] = np.inf
        dis_alpha = dis ** alpha
        n = self.lengths.sum()

        if self.ploidy == 1:
            ratio_ua, ratio_pa, ratio_ambig = 1, 0, 0
        elif ratio_ua + ratio_pa + ratio_ambig != 1:
            raise ValueError("Sum of ratios should be 1.")

        counts = []
        if ratio_ua:
            ua_counts = ratio_ua * beta * dis_alpha
            if use_poisson:
                ua_counts = self.random_state.poisson(ua_counts)
            ua_counts = np.triu(ua_counts, 1)
            ua_counts = sparse.coo_matrix(ua_counts)
            counts.append(ua_counts)
        if ratio_pa:
            # FIXME Is this wrong?? Shouldn't it be ratio_pa/2 or something?
            pa_counts = ratio_pa * beta * dis_alpha
            if use_poisson:
                pa_counts = self.random_state.poisson(pa_counts)
            pa_counts = pa_counts[:, :n] + pa_counts[:, n:]
            np.fill_diagonal(pa_counts[:n, :], 0)
            np.fill_diagonal(pa_counts[n:, :], 0)
            pa_counts = sparse.coo_matrix(pa_counts)
            counts.append(pa_counts)
        if ratio_ambig:
            ambig_counts = ratio_ambig * beta * dis_alpha
            if use_poisson:
                ambig_counts = self.random_state.poisson(ambig_counts)
            ambig_counts = ambig_counts[:n, :n] + ambig_counts[n:, n:] + \
                ambig_counts[:n, n:] + ambig_counts[n:, :n]
            ambig_counts = np.triu(ambig_counts, 1)
            ambig_counts = sparse.coo_matrix(ambig_counts)
            counts.append(ambig_counts)

        return counts


@pytest.fixture(scope="module")
def smtp_connection():
    return True