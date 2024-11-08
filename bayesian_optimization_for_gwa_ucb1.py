import time
import numpy as np
import matplotlib.pyplot as plt
import mab as mab
import algorithms as algorithms
from mpl_toolkits.mplot3d import Axes3D
from skopt import gp_minimize
from skopt.plots import plot_convergence
from functools import partial
from mab.utils import plot_bayesian_optimization_progress, generate_3d_bayesian_optimization_plot

# Execute a simulation for a Stochastic MAB using the GWA-UCB1 algorithm
def execute_stochastic_mab_with_gwa_ucb1(
    xy,
    random_seed=0,
    n_simulations=1000,
    n_steps=10000,
    n_steps_to_switch_rewards_probabilities=None,
    n_arms=2,
    distribution_of_rewards_probabilities='uniform'
):
    # Initialize parameters
    alpha, m = xy
    np.random.seed(random_seed)
    n_model = 1
    total_regrets = np.zeros([n_model, n_steps])
    n_best_arms_selected = np.zeros([n_model, n_steps])

    # Simulate the Stochastic MAB process
    for i in range(n_simulations):
        smab = mab.StochasticMAB(n_arms, n_steps_to_switch_rewards_probabilities=n_steps_to_switch_rewards_probabilities)
        smab.update_distribution_of_rewards_probabilities(distribution_of_rewards_probabilities)
        smab.update_rewards_probabilities_from_distribution()
        models = [algorithms.GeneralizedWeightedAveragesUCB1(n_arms, alpha=alpha, m=m)]
        regrets = np.zeros([n_model, n_steps])
        is_best_arms = np.zeros([n_model, n_steps])
        for j in range(n_steps):
            smab.pull_lever()
            for k, model in enumerate(models):
                selected_arm = model.select_arm()
                reward = smab.reward(selected_arm)
                model.update(selected_arm, reward)
                # Update regret and track if the selected arm was the best
                regrets[k, j] = smab.regret(selected_arm) if j == 0 else regrets[k, j-1] + smab.regret(selected_arm)
                is_best_arms[k, j] = smab.is_best_arm(selected_arm)
        total_regrets += regrets
        n_best_arms_selected += is_best_arms
    average_regrets = total_regrets / n_simulations
    accuracy_rates = n_best_arms_selected / n_simulations

    return average_regrets[0, -1]

# Execute a simulation for a Survival MAB using the GWA-UCB1 algorithm
def execute_survival_mab_with_gwa_ucb1(
    xy,
    random_seed=0,
    n_simulations=1000,
    n_steps=10000,
    n_steps_to_switch_rewards_probabilities=None,
    n_arms=8,
    initial_budgets=80,
    initial_rewards_probabilities=[0.55, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]
):
    # Initialize parameters
    alpha, m = xy
    np.random.seed(random_seed)
    n_model = 1
    total_budgets = np.zeros([n_model, n_steps])
    n_survivals = np.zeros([n_model, n_steps])

    # Simulate the Survival MAB process
    for i in range(n_simulations):
        smab = mab.SurvivalMAB(n_arms, n_steps_to_switch_rewards_probabilities=n_steps_to_switch_rewards_probabilities)
        smab.update_rewards_probabilities(initial_rewards_probabilities)
        models = [algorithms.GeneralizedWeightedAveragesUCB1(n_arms, alpha=alpha, m=m)]
        budgets = np.zeros([n_model, n_steps])
        is_survivals = np.zeros([n_model, n_steps])
        for j in range(n_steps):
            smab.pull_lever()
            for k, model in enumerate(models):
                # Check if the budget allows pulling the lever
                if j == 0 or budgets[k, j-1] > 0:
                    selected_arm = model.select_arm()
                    reward = smab.reward(selected_arm)
                    model.update(selected_arm, 1.0 if reward == 1.0 else 0.0)
                    # Update budget based on reward
                    budgets[k, j] = initial_budgets + reward if j == 0 else budgets[k, j-1] + reward
                    is_survivals[k, j] = 1 if budgets[k, j] > 0 else 0
        total_budgets += budgets
        n_survivals += is_survivals
    average_budgets = total_budgets / n_simulations
    survival_rates = n_survivals / n_simulations

    return -average_budgets[0, -1]  # Multiply by -1 for Bayesian optimization

# Main
def main():
    # Experiment 1: Stochastic MAB with GWA-UCB1, 2 Arms
    TARGET_SIMULATION = execute_stochastic_mab_with_gwa_ucb1
    TARGET_SIMULATION_OPTIONS = {
        'random_seed': 0,
        'n_simulations': 1000,
        'n_steps': 10000,
        'n_steps_to_switch_rewards_probabilities': None,
        'n_arms': 2,
        'distribution_of_rewards_probabilities': 'uniform',
    }
    GRAPH_TITLE = 'Experiment 1, Arm 2'
    GRAPH_X_LABEL = 'α'
    GRAPH_Y_LABEL = 'm'
    GRAPH_Z_LABEL = 'Avg. regret at final step'

    '''
    # Experiment 2: Stochastic MAB with GWA-UCB1, 32 Arms
    TARGET_SIMULATION = execute_stochastic_mab_with_gwa_ucb1
    TARGET_SIMULATION_OPTIONS = {
        'random_seed': 0,
        'n_simulations': 1000,
        'n_steps': 50000,
        'n_steps_to_switch_rewards_probabilities': None,
        'n_arms': 32,
        'distribution_of_rewards_probabilities': 'normal',
    }
    GRAPH_TITLE = 'Experiment 2, Arm 32'
    GRAPH_X_LABEL = 'α'
    GRAPH_Y_LABEL = 'm'
    GRAPH_Z_LABEL = 'Avg. regret at final step'
    '''

    '''
    # Experiment 3: Survival MAB with GWA-UCB1, 8 Arms
    TARGET_SIMULATION = execute_survival_mab_with_gwa_ucb1
    TARGET_SIMULATION_OPTIONS = {
        'random_seed': 0,
        'n_simulations': 1000,
        'n_steps': 50000,
        'n_steps_to_switch_rewards_probabilities': None,
        'n_arms': 8,
        'initial_budgets': 8 * 10,
        'initial_rewards_probabilities': np.array([0.55 if i == 0 else 0.45 for i in range(8)]),
    }
    GRAPH_TITLE = 'Experiment 3, Arm 8'
    GRAPH_X_LABEL = 'α'
    GRAPH_Y_LABEL = 'm'
    GRAPH_Z_LABEL = 'Avg. budget at final step'
    '''

    # Define the space for Bayesian optimization (α and m search spaces)
    X_SPACE = (0.0, 1.0)  # α search space
    Y_SPACE = (-2.0, 4.0)  # m search space

    # Set Bayesian optimization parameters
    BAYESIAN_OPTIMIZATION_RANDOM_SEED = 0
    BAYESIAN_OPTIMIZATION_MAX_ITERATIONS = 100

    # Parameters for GWA-UCB1 to compare with Bayesian Optimization results
    COMPARATIVE_GWAUCB1_ALPHA = 0.21
    COMPARATIVE_GWAUCB1_M = 1.30

    # Perform Bayesian optimization and track execution time
    start_time = time.time()  # Start time of the optimization process
    objective = partial(TARGET_SIMULATION, **TARGET_SIMULATION_OPTIONS)
    result = gp_minimize(objective, [X_SPACE, Y_SPACE], n_calls=BAYESIAN_OPTIMIZATION_MAX_ITERATIONS, random_state=BAYESIAN_OPTIMIZATION_RANDOM_SEED, verbose=True)
    end_time = time.time()  # End time of the optimization process

    # Print the results of Bayesian Optimization for the chosen experiment
    z_best = result.fun if TARGET_SIMULATION == execute_stochastic_mab_with_gwa_ucb1 else -result.fun
    z_vals = result.func_vals if TARGET_SIMULATION == execute_stochastic_mab_with_gwa_ucb1 else -result.func_vals

    print('*************************************************')
    print(f'Execution Time: {end_time - start_time:.2f} seconds ({(end_time - start_time) / 60:.2f} minutes)')
    print(f'Best (α, m): {result.x}')
    print(f'Best {GRAPH_Z_LABEL}: {z_best}')

    # Execute GWA-UCB1 for comparison with Bayesian Optimization
    comparative_value = TARGET_SIMULATION([COMPARATIVE_GWAUCB1_ALPHA, COMPARATIVE_GWAUCB1_M], **TARGET_SIMULATION_OPTIONS)
    comparative_value = comparative_value if TARGET_SIMULATION == execute_stochastic_mab_with_gwa_ucb1 else -comparative_value

    print(f'Comparative (α, m): {[COMPARATIVE_GWAUCB1_ALPHA, COMPARATIVE_GWAUCB1_M]}')
    print(f'Comparative {GRAPH_Z_LABEL}: {comparative_value}')

    # Plot the progress of Bayesian optimization
    plot_bayesian_optimization_progress(
        z_vals=z_vals,
        comparative_value=comparative_value,
        graph_title=GRAPH_TITLE,
        graph_z_label=GRAPH_Z_LABEL,
        comparative_gwaucb1_alpha=COMPARATIVE_GWAUCB1_ALPHA,
        comparative_gwaucb1_m=COMPARATIVE_GWAUCB1_M,
        save_path='output/progress_of_bayesian_optimization.png'
    )

    # Generate a 3D plot of Bayesian optimization results
    generate_3d_bayesian_optimization_plot(
        result=result,
        x_iters=result.x_iters,
        z_vals=z_vals,
        graph_title=GRAPH_TITLE,
        graph_x_label=GRAPH_X_LABEL,
        graph_y_label=GRAPH_Y_LABEL,
        graph_z_label=GRAPH_Z_LABEL,
        save_path='output/results_of_bayesian_optimization.png',
        x_space=X_SPACE,
        y_space=Y_SPACE
    )

if __name__ == '__main__':
    main()
