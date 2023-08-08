# -*- encoding: utf8 -*-

# Author: Nobuhito Manome
# License: BSD 3 clause

import numpy as np

class KLUcb():
    '''
    KLUcb algorithm.
    ----------
    Parameters
    ----------
    n_arms : int
        number of arms.
    c : float, optional (default=0)
        c ≧ 0.
    delta : float, optional (default=1e-8)
        delta ≧ 0.
    eps : float, optional (default=1e-12)
        eps ≧ 0.
    max_iter : float, optional (default=1e2)
        max_iter ≧ 0.
    '''
    def __init__(self, n_arms, c=0, delta=1e-8, eps=1e-12, max_iter=1e2):
        self.n_arms = n_arms
        self.params = np.zeros(n_arms)
        self.trials = np.zeros(n_arms)
        self.step = 0
        self.c = c
        self.delta = delta
        self.eps = eps
        self.max_iter = max_iter

    def kl(self, p, q):
        with np.errstate(all='ignore'):
            return p *  np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

    def kl_grad(self, p, q):
        return (q-p)  / (q * (1-q))

    def kl_ucb(self, c = 0, delta = 1e-8, eps = 1e-12, max_iter = 1e2):
        upperbounds = (np.log(self.step) + c * np.log(np.log(self.step)) )/ self.trials
        upperbounds = np.maximum(delta, upperbounds)
        klucb_results = np.zeros(self.n_arms)
        for k in range(self.n_arms):
            p = self.params[k]
            if p >= 1:
                klucb_results[k] = 1
                continue
            q = p + delta
            for _ in range(int(max_iter)):
                f = upperbounds[k] - self.kl(p,q)
                if (f * f < eps):
                    break
                df = - self.kl_grad(p,q)
                q = min(1 - delta, max(q - f / df, p + delta))
            klucb_results[k] = q
        return klucb_results

    def select_arm(self):
        if self.step < self.n_arms:
            return self.step
        klucb_results = self.kl_ucb(self.c, self.delta, self.eps, self.max_iter)
        return np.argmax(klucb_results)

    def update(self, selected_arm, reward):
        self.step += 1
        self.trials[selected_arm] += 1
        if self.step < self.n_arms:
            return
        self.params[selected_arm] = (self.trials[selected_arm] * self.params[selected_arm] + reward ) / (self.trials[selected_arm] + 1)
