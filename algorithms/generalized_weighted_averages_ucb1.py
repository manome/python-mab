# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import numpy as np

def generalized_weighted_averages(val1, val2, alpha=0.5, m=1.0):
    '''
    Calculate generalized weighted averages.
    arithmetic mean : alpha=0.5, m=1.0
    geometric mean  : alpha=0.5, m=0.0
    harmonic mean   : alpha=0.5, m=-1.0
    ----------
    Parameters
    ----------
    val1 : float
        0 < val1.
    val2 : float
        0 < val2.
    alpha : float, optional (default=0.5)
        0 ≤ alpha ≤ 1.
    m : float, optional (default=1.0)
        −∞ ≤ m ≤ ∞.
    '''
    with np.errstate(all='ignore'):
        if m == 0:
            return ( val1**(1-alpha) ) * (val2**alpha)
        return ( (1-alpha)*(val1**m) + alpha*(val2**m) ) ** (1/m)

class GeneralizedWeightedAveragesUCB1():
    '''
    Generalized weighted average algorithm.
    ----------
    Parameters
    ----------
    n_arms : int
        number of arms.
    alpha : float, optional (default=0.5)
        0 ≤ alpha ≤ 1.
    m : float, optional (default=1.0)
        −∞ ≤ m ≤ ∞.
    '''
    def __init__(self, n_arms, alpha=0.5, m=1.0):
        self.n_arms = n_arms # number of arms
        self.trials = np.zeros(n_arms) # number of levers pulled
        self.step = 0 # step
        self.rewards = np.zeros(n_arms) # rewards
        self.scores = np.zeros(n_arms) # evaluation value for each arm
        self.alpha = alpha # param for generalized weighted average
        self.m = m # param for generalized weighted average

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
        self.scores = generalized_weighted_averages(e, np.sqrt(2*np.log(self.step)/self.trials), alpha=self.alpha, m=self.m)
