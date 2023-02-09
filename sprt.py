import numpy as np
import pandas as pd
from bisect import bisect_left
from collections import deque
from copy import deepcopy
from prettytable import PrettyTable
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.style.use('seaborn')


class MarkovChain():
    def __init__(self, params):
        if not params:
            raise Exception('Input parameters are empty.')
        self.lambd = params[0]
        self.q = params[1]
        self.order = len(params[0])
        self.n = len(params[1])
        self.prev = deque(maxlen=self.order)

    def step(self):
        if len(self.prev) != self.order:
            next_state = random.choice([i for i in range(self.n)])
        else:
            p = []
            for i in range(self.n):
                p_i = 0.
                for j in range(len(self.prev)):
                    state = self.prev[j]
                    p_i += self.lambd[j] * self.q[state][i]
                p.append(p_i)
            next_state = self.generate(p)
        cur_step = list(self.prev) + [next_state]
        self.prev.append(next_state)
        return cur_step

    def reset(self):
        self.prev.clear()

    def proba(self, states):
        if not states:
            raise Exception('Input list is empty.')
        p = 0.
        cur_state = states[-1]
        for i in range(len(states) - 1):
            state = states[i]
            p += self.lambd[i] * self.q[state][cur_state]
        return p

    @staticmethod
    def random(n, order=1):
        if order < 1:
            raise Exception('The Markov chain order must be great or equal than 1.')
        lambd = np.random.random(order)
        lambd /= lambd.sum()
        q = np.random.random((n, n))
        q /= q.sum(axis=-1, keepdims=True)
        return (lambd, q)

    @staticmethod
    def generate(p):
        return bisect_left(np.cumsum(p), np.random.random())


def SPRT(observer, chain0, chain1, alpha, beta):
    A = (1 - beta) / alpha
    B = beta / (1 - alpha)
    states = observer.step()
    n = 1
    while len(states) != observer.order + 1:
        states = observer.step()
        n += 1
    if chain0.proba(states) == 0:
        return (1, n)
    z = chain1.proba(states) / chain0.proba(states)
    while B < z and z < A and n < 100000:
        states = observer.step()
        n += 1
        if chain0.proba(states) == 0:
            return (1, n)
        z *= chain1.proba(states) / chain0.proba(states)
    return (int(z >= A), n)


def monte_carlo(observer, chain0, chain1, alpha, beta, iterations):
    n_sum = 0
    decisions = np.array([0, 0])
    for _ in range(iterations):
        d, n = SPRT(observer, chain0, chain1, alpha, beta)
        decisions[d] += 1
        n_sum += n
        observer.reset()
    return (decisions / iterations, n_sum / iterations)


def get_table(chain0, chain1, iterations):
    alpha = [0.01, 0.05, 0.1]
    beta = [0.01, 0.05, 0.1]
    t = PrettyTable()
    t.field_names = ['exact alpha', 'exact beta', 'expected n(H0)', 'expected n(H1)', 'expected alpha', 'expected beta']
    for a in alpha:
        for b in beta:
            error0, n0 = monte_carlo(chain0, chain0, chain1, a, b, iterations)
            error1, n1 = monte_carlo(chain1, chain0, chain1, a, b, iterations)
            t.add_row([a, b, round(n0, 4), round(n1, 4), round(error0[1], 4), round(error1[0], 4)])
    return t


def get_results(chain0, chain1, errors, iterations):
    z0, z1 = [], []
    for a in errors:
        row0, row1 = [], []
        for b in errors:
            _, n0 = monte_carlo(chain0, chain0, chain1, a, b, iterations)
            _, n1 = monte_carlo(chain1, chain0, chain1, a, b, iterations)
            row0.append(n0)
            row1.append(n1)
        z0.append(row0)
        z1.append(row1)
    return (np.array(z0), np.array(z1))


def plot_results_2d(x, y_H0, y_H1, x_label):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlabel(x_label, size=32, labelpad=16)
    ax.set_ylabel('$\hat{n}$', size=32, rotation=0, labelpad=20)
    ax.set_xticks(np.linspace(0., 0.1, 11))
    ax.tick_params(axis='both', which='major', labelsize=24)
    line_0, = ax.plot(x, y_H0, linewidth=8)
    line_1, = ax.plot(x, y_H1, linewidth=8)
    line_0.set_label('$\mathcal{H}_0$ is valid')
    line_1.set_label('$\mathcal{H}_1$ is valid')
    ax.legend(prop={"size": 32})
    return fig


def plot_results_3d(X, Y, Z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 10))
    ax.set_xlabel(chr(945), size=24, labelpad=20)
    ax.set_ylabel(chr(946), size=24, labelpad=20)
    ax.set_zlabel('$\hat{n}$', size=24, rotation=0, labelpad=16)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.plot_surface(X, Y, Z, cmap=cm.viridis)
    ax.view_init(25, 45)
    return fig
