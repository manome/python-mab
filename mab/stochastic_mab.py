# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import numpy as np

class StochasticMAB():
    '''
    Stochastic multi armed bandits.
    ----------
    Parameters
    ----------
    n_arms : int
        number of arms.
    n_steps_to_switch_rewards_probabilities : int, optional (default=None)
        n_steps_to_switch_rewards_probabilities ≧ 0.
    '''
    def __init__(self, n_arms, n_steps_to_switch_rewards_probabilities=None):
        self.n_arms = n_arms # number of arms
        self.winning_rewards = np.ones(n_arms) # winning rewards
        self.losing_rewards = np.zeros(n_arms) # losing rewards
        self.distribution_of_rewards_probabilities = 'uniform' # distribution of rewards probabilities
        self.rewards_probabilities = np.random.rand(n_arms) # rewards probabilities
        self.regrets = np.amax(self.rewards_probabilities) - self.rewards_probabilities # regrets
        self.best_arms = np.where(self.regrets == 0, 1, 0) # if best arms: 1, else: 0
        self.step = 0 # step
        self.n_steps_to_switch_rewards_probabilities = n_steps_to_switch_rewards_probabilities # 
        self.rewards = np.zeros(n_arms) # rewards after pulling lever

    # Get reward
    def reward(self, idx):
        return self.rewards[idx]

    # Get regret
    def regret(self, idx):
        return self.regrets[idx]

    # Determine if best arm
    def is_best_arm(self, idx):
        return self.best_arms[idx]

    # Update rewards probabilities
    def update_rewards_probabilities(self, rewards_probabilities):
        self.rewards_probabilities = rewards_probabilities
        self.regrets = np.amax(rewards_probabilities) - rewards_probabilities
        self.best_arms = np.where(self.regrets == 0, 1, 0)

    # Update distribution of rewards probabilities
    def update_distribution_of_rewards_probabilities(self, distribution_of_rewards_probabilities):
        self.distribution_of_rewards_probabilities = distribution_of_rewards_probabilities

    # Update rewards probabilities from distribution
    def update_rewards_probabilities_from_distribution(self):
        if self.distribution_of_rewards_probabilities == 'uniform':
            rewards_probabilities = np.random.rand(self.n_arms) # uniform distribution
        elif self.distribution_of_rewards_probabilities == 'normal':
            rewards_probabilities = np.zeros(self.n_arms)
            while np.where(rewards_probabilities <= 0.0)[0].shape[0] > 0 \
                or np.where(rewards_probabilities >= 1.0)[0].shape[0] > 0:
                rewards_probabilities = np.random.normal(0.5, 0.1, self.n_arms) # normal distribution with mean 0.5 and standard deviation 0.1
        else:
            rewards_probabilities = np.random.rand(self.n_arms) # uniform distribution
        self.update_rewards_probabilities(rewards_probabilities)

    # Pull all arms and update rewards results
    def pull_lever(self):
        if self.n_steps_to_switch_rewards_probabilities is not None \
            and self.step % self.n_steps_to_switch_rewards_probabilities == 0:
            self.update_rewards_probabilities_from_distribution()
        rnd = np.full(self.n_arms, np.random.rand())
        self.rewards = np.where(self.rewards_probabilities - rnd >= 0, self.winning_rewards, self.losing_rewards)
        self.step += 1
