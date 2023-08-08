# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import numpy as np

class Softmax():
    '''
    Softmax algorithm.
    ----------
    Parameters
    ----------
    n_arms : int
        number of arms.
    temperature : float, optional (default=0.1)
        0 ≤ temperature ≤ 1.
    '''
    def __init__(self, n_arms, temperature=0.1):
        self.n_arms = n_arms # number of arms
        self.temperature = temperature # param
        self.trials = np.zeros(n_arms) # number of levers pulled
        self.rewards = np.zeros(n_arms) # rewards
        self.softmaxs = np.zeros(n_arms) # evaluation value for each arm

    def select_arm(self):
        idxs = np.where(np.cumsum(self.softmaxs) > np.random.rand())[0]
        if idxs.shape[0] == 0:
            return self.n_arms - 1
        return idxs[0]

    def update(self, selected_arm, reward):
        self.trials[selected_arm] += 1
        self.rewards[selected_arm] += reward
        with np.errstate(all='ignore'):
            e = self.rewards / self.trials
        e = np.nan_to_num(e, nan=0)
        exp_etau = np.exp(e/self.temperature)
        self.softmaxs = exp_etau / np.sum(exp_etau)
