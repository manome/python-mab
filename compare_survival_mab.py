# -*- encoding: utf8 -*-

import datetime
import numpy as np
import mab as mab
from mab.utils import savefig_line2D
import algorithms as algorithms

# Define model labels
MODEL_LABELS = [
    'Epsilon Greedy',
    'Annealing Epsilon Greedy',
    'Softmax',
    'KL-Ucb',
    'UCB1',
    'UCB1-Tuned',
    'Thompson Sampling',
    'G-UCB1',
    'GWA-UCB1',
]

# Define models
def init_models(n_arms):
    return [
        algorithms.EpsilonGreedy(n_arms, epsilon=0.1),
        algorithms.AnnealingEpsilonGreedy(n_arms, initial_epsilon=0.1, final_epsilon=0.01, exploration_steps=10000),
        algorithms.Softmax(n_arms, temperature=0.1),
        algorithms.KLUcb(n_arms, c=0, delta=1e-8, eps=1e-12, max_iter=1e2),
        algorithms.Ucb1(n_arms),
        algorithms.Ucb1Tuned(n_arms),
        algorithms.ThompsonSampling(n_arms),
        algorithms.GeneralizedUcb1(n_arms, c=0.30),
        algorithms.GeneralizedWeightedAveragesUCB1(n_arms, alpha=0.21, m=1.30),
    ]

# Main
def main():
    # Define params
    OUTPUT_FILENAME = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    RANDOM_SEED = 0
    N_SIMULATIONS = 1000
    N_STEPS = 10000
    N_STEPS_TO_SWITCH_REWARDS_PROBABILITIES = None
    N_ARMS = 8
    INITIAL_BUDGETS = N_ARMS * 10
    INITIAL_REWARDS_PROBABILITIES = np.array([0.55 if i == 0 else 0.45 for i in range(N_ARMS)])

    # Init
    np.random.seed(RANDOM_SEED)
    n_model = len(init_models(N_ARMS))
    total_budgets = np.zeros([n_model, N_STEPS])
    n_survivals = np.zeros([n_model, N_STEPS])
    print('RANDOM_SEED: {}'.format(RANDOM_SEED))
    print('N_SIMULATIONS: {}'.format(N_SIMULATIONS))
    print('N_STEPS: {}'.format(N_STEPS))
    print('N_STEPS_TO_SWITCH_REWARDS_PROBABILITIES: {}'.format(N_STEPS_TO_SWITCH_REWARDS_PROBABILITIES))
    print('N_ARMS: {}'.format(N_ARMS))
    print('INITIAL_BUDGETS: {}'.format(INITIAL_BUDGETS))
    print('INITIAL_REWARDS_PROBABILITIES: {}'.format(INITIAL_REWARDS_PROBABILITIES))

    # Simulate
    for i in range(N_SIMULATIONS):
        print('{}/{}'.format(i+1, N_SIMULATIONS))
        smab = mab.SurvivalMAB(N_ARMS, n_steps_to_switch_rewards_probabilities=N_STEPS_TO_SWITCH_REWARDS_PROBABILITIES)
        smab.update_rewards_probabilities(INITIAL_REWARDS_PROBABILITIES)
        models = init_models(N_ARMS)
        budgets = np.zeros([n_model, N_STEPS])
        is_survivals = np.zeros([n_model, N_STEPS])
        for j in range(N_STEPS):
            smab.pull_lever()
            for k, model in enumerate(models):
                if j == 0 or budgets[k, j-1] > 0:
                    selected_arm = model.select_arm()
                    reward = smab.reward(selected_arm)
                    model.update(selected_arm, 1.0 if reward == 1.0 else 0.0)
                    budgets[k, j] = INITIAL_BUDGETS + reward if j == 0 else budgets[k, j-1] + reward
                    is_survivals[k, j] = 1 if budgets[k, j] > 0 else 0
        total_budgets += budgets
        n_survivals += is_survivals
    average_budgets = total_budgets / N_SIMULATIONS
    survival_rates = n_survivals / N_SIMULATIONS

    # Save figure
    savefig_line2D(average_budgets, xlim=[0, average_budgets.shape[1]], ylim=None, xscale=None, yscale=None, markevery=N_STEPS/10, labels=MODEL_LABELS, xlabel='Steps', ylabel='Avg. budgets', title='Arm {}'.format(N_ARMS), path='output/', filename='{}-budgets'.format(OUTPUT_FILENAME), extension='png', is_show=False)
    savefig_line2D(survival_rates, xlim=[0, survival_rates.shape[1]], ylim=[-0.05, 1.05], xscale=None, yscale=None, markevery=N_STEPS/10, labels=MODEL_LABELS, xlabel='Steps', ylabel='Survival rate', title='Arm {}'.format(N_ARMS), path='output/', filename='{}-survival_rate'.format(OUTPUT_FILENAME), extension='png', is_show=False)

if __name__ == '__main__':
    main()
