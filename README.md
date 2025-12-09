# Multi-armed bandit problems

This project provides the official implementation of the following paper. Simple Modification of the Upper Confidence Bound Algorithm by Generalized Weighted Averages (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0322757).

## Requirements

Compatible with Python3.6 and above.
The required libraries are listed below.

- NumPy
- matplotlib
- scikit-optimize

## Installation

```
$ pip install numpy
$ pip install matplotlib
$ pip install scikit-optimize
```

## Quickstart StochasticMAB

```
$ python compare_stochastic_mab.py
```

The results are displayed as below.

<p align='center'>
    <img src='img/compare_stochastic_mab_regrets.png' alt='Execution result of compare_stochastic_mab.py'>
</p>

<p align='center'>
    <img src='img/compare_stochastic_mab_accuracy_rate.png' alt='Execution result of compare_stochastic_mab.py'>
</p>

For more information, [compare_stochastic_mab.py](compare_stochastic_mab.py).

## Quickstart SurvivalMAB

```
$ python compare_survival_mab.py
```

The results are displayed as below.

<p align='center'>
    <img src='img/compare_survival_mab_budgets.png' alt='Execution result of compare_survival_mab.py'>
</p>

<p align='center'>
    <img src='img/compare_survival_mab_survival_rate.png' alt='Execution result of compare_survival_mab.py'>
</p>

For more information, [compare_survival_mab.py](compare_survival_mab.py).

## Experiments in the paper

The three experiments described in the paper can be executed with the following commands.

```
$ python experiment1.py
```
```
$ python experiment2.py
```
```
$ python experiment3.py
```

## License
This is free and open-source software licensed under the 3-clause BSD license.
