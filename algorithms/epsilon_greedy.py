# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import numpy as np

class EpsilonGreedy():
    '''
    Epsilon greedy algorithm.
    ----------
    Parameters
    ----------
    n_arms : int
        number of arms.
    epsilon : float, optional (default=0.1)
        0 ≤ epsilon ≤ 1.
    '''
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms # number of arms
        self.epsilon = epsilon # epsilon
        self.trials = np.zeros(n_arms) # number of levers pulled
        self.scores = np.zeros(n_arms) # evaluation value for each arm

    def select_arm(self):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.scores)
        return np.random.randint(0, self.n_arms)

    def update(self, selected_arm, reward):
        self.trials[selected_arm] += 1
        n_selected_arm = self.trials[selected_arm]
        score_selected_arm = self.scores[selected_arm]
        self.scores[selected_arm] = ((n_selected_arm-1)/n_selected_arm)*score_selected_arm + (1/n_selected_arm)*reward
