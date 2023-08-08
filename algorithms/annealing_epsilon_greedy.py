# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import numpy as np

class AnnealingEpsilonGreedy():
    '''
    Annealing epsilon greedy algorithm.
    ----------
    Parameters
    ----------
    n_arms : int
        number of arms.
    initial_epsilon : float, optional (default=0.1)
        0 ≤ initial_epsilon ≤ 1.
    final_epsilon : float, optional (default=0.01)
        0 ≤ final_epsilon ≤ 1.
    exploration_steps : int, optional (default=10000)
        number of steps from initial to final state.
    '''
    def __init__(self, n_arms, initial_epsilon=0.1, final_epsilon=0.01, exploration_steps=10000):
        self.n_arms = n_arms # number of arms
        self.initial_epsilon = initial_epsilon # initial epsilon
        self.final_epsilon = final_epsilon # final epsilon
        self.exploration_steps = exploration_steps # exploration steps
        self.epsilon_step = (initial_epsilon - final_epsilon) / exploration_steps # epsilon step
        self.step = 0 # step
        self.trials = np.zeros(n_arms) # number of levers pulled
        self.scores = np.zeros(n_arms) # evaluation value for each arm

    def select_arm(self):
        epsilon = self.initial_epsilon - self.step * self.epsilon_step
        if epsilon < self.final_epsilon:
            epsilon = self.final_epsilon
        if np.random.rand() > epsilon:
            return np.argmax(self.scores)
        return np.random.randint(0, self.n_arms)

    def update(self, selected_arm, reward):
        self.step += 1
        self.trials[selected_arm] += 1
        n_selected_arm = self.trials[selected_arm]
        score_selected_arm = self.scores[selected_arm]
        self.scores[selected_arm] = ((n_selected_arm-1)/n_selected_arm)*score_selected_arm + (1/n_selected_arm)*reward
