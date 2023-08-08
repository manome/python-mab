# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import numpy as np

class Ucb1Tuned():
    '''
    Ucb1Tuned algorithm.
    ----------
    Parameters
    ----------
    n_arms : int
        number of arms.
    '''
    def __init__(self, n_arms):
        self.n_arms = n_arms # number of arms
        self.trials = np.zeros(n_arms) # number of levers pulled
        self.step = 0 # step
        self.rewards = np.zeros(n_arms) # rewards
        self.scores = np.zeros(n_arms) # evaluation value for each arm

    def select_arm(self):
        if self.step < self.n_arms:
            return self.step
        return np.argmax(self.scores)

    def update(self, selected_arm, reward):
        self.step += 1
        self.trials[selected_arm] += 1
        self.rewards[selected_arm] += reward
        if self.step < self.n_arms:
            return
        e = self.rewards / self.trials
        variances = e - e ** 2
        v = variances + np.sqrt(2 * np.log(self.step) / self.trials)
        self.scores = e + np.sqrt( (np.log(self.step) / self.trials) * np.where(v > 0.25, 0.25, v))
