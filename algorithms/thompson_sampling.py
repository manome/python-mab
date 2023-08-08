# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import random
import numpy as np

class ThompsonSampling():
    '''
    Thompson sampling algorithm.
    ----------
    Parameters
    ----------
    n_arms : int
        number of arms.
    '''
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms # number of arms
        self.counts_alpha = np.zeros(n_arms) # counts alpha
        self.counts_beta = np.zeros(n_arms) # counts beta
        self.alpha = 1 # alpha
        self.beta = 1 # beta
        self.scores = np.zeros(n_arms) # evaluation value for each arm

    def select_arm(self):
        theta = [(arm, random.betavariate(self.counts_alpha[arm] + self.alpha, self.counts_beta[arm] + self.beta)) for arm in range(len(self.counts_alpha))]
        theta = sorted(theta, key=lambda x:x[1])
        return theta[-1][0]

    def update(self, selected_arm, reward):
        if reward == 1:
            self.counts_alpha[selected_arm] += 1
        else:
            self.counts_beta[selected_arm] += 1
        n = float(self.counts_alpha[selected_arm]) + self.counts_beta[selected_arm]
        self.scores[selected_arm] = (n - 1) / n * self.scores[selected_arm] + 1 / n * reward
