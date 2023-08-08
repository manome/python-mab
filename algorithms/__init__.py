from .epsilon_greedy import *
from .annealing_epsilon_greedy import *
from .softmax import *
from .ucb1 import *
from .ucb1_tuned import *
from .kl_ucb import *
from .thompson_sampling import *
from .generalized_ucb1 import *
from .generalized_weighted_averages_ucb1 import *

__all__ = [
    'EpsilonGreedy',
    'AnnealingEpsilonGreedy',
    'Softmax',
    'GeneralizedUcb1',
    'Ucb1',
    'Ucb1Tuned',
    'KLUcb',
    'ThompsonSampling',
    'GeneralizedUcb1',
    'GeneralizedWeightedAveragesUCB1',
]
__version__ = '1.0.0'
