"""Fixed-size feature search using a cross-validated classification objective."""

from functools import lru_cache
from math import gamma, pi, sin

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

POPULATION = 8
ITERATIONS = 8
PSO = "Particle Swarm Optimization (PSO)"
HHO = "Harris Hawks Optimization (HHO)"


def search_subset(method, folds, candidates, count, seed):
    """Decode each position by taking its top-k coordinates, one per raw feature."""
    rng = np.random.default_rng(seed)
    width = len(candidates)

    def decode(position):
        return tuple(sorted(np.argsort(-position, kind="stable")[:count]))

    @lru_cache(maxsize=None)
    def score(subset):
        scores = []
        for x, y, validation, labels, columns in folds:
            indices = np.concatenate([columns[candidates[i]] for i in subset])
            model = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=seed)
            prediction = model.fit(x[:, indices], y).predict_proba(validation[:, indices])[:, 1]
            scores.append(average_precision_score(labels, prediction))
        return np.mean(scores)

    def fitness(position):
        return score(decode(position))

    positions = rng.random((POPULATION, width))
    with threadpool_limits(limits=1):
        if method == PSO:
            best = particle_swarm(positions, fitness, rng)
        else:
            best = harris_hawks(positions, fitness, rng)
    return [candidates[i] for i in decode(best)]


def particle_swarm(positions, fitness, rng):
    velocity = rng.uniform(-0.1, 0.1, positions.shape)
    personal = positions.copy()
    scores = np.array([fitness(position) for position in personal])
    for _ in range(ITERATIONS):
        leader = personal[scores.argmax()]
        velocity = (0.7 * velocity
                    + 1.4 * rng.random(positions.shape) * (personal - positions)
                    + 1.4 * rng.random(positions.shape) * (leader - positions))
        positions = np.clip(positions + velocity, 0, 1)
        values = np.array([fitness(position) for position in positions])
        improved = values > scores
        personal[improved], scores[improved] = positions[improved], values[improved]
    return personal[scores.argmax()]


def harris_hawks(positions, fitness, rng):
    beta = 1.5
    sigma = (gamma(1 + beta) * sin(pi * beta / 2)
             / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    rabbit = max(positions, key=fitness).copy()
    for step in range(ITERATIONS):
        center = positions.mean(axis=0)
        next_positions = []
        for hawk in positions:
            energy = 2 * (1 - step / ITERATIONS) * rng.uniform(-1, 1)
            jump = 2 * (1 - rng.random())
            if abs(energy) >= 1:
                if rng.random() >= 0.5:
                    peer = positions[rng.integers(len(positions))]
                    candidate = peer - rng.random() * abs(peer - 2 * rng.random() * hawk)
                else:
                    candidate = rabbit - center - rng.random() * rng.random()
            elif rng.random() >= 0.5:
                if abs(energy) >= 0.5:
                    candidate = rabbit - hawk - energy * abs(jump * rabbit - hawk)
                else:
                    candidate = rabbit - energy * abs(rabbit - hawk)
            else:
                reference = hawk if abs(energy) >= 0.5 else center
                dive = np.clip(rabbit - energy * abs(jump * rabbit - reference), 0, 1)
                candidate = hawk
                if fitness(dive) > fitness(hawk):
                    candidate = dive
                else:
                    levy = sigma * rng.normal(size=hawk.size) / np.maximum(
                        abs(rng.normal(size=hawk.size)), 1e-12) ** (1 / beta)
                    rapid = np.clip(dive + 0.01 * rng.random(hawk.size) * levy, 0, 1)
                    if fitness(rapid) > fitness(hawk):
                        candidate = rapid
            next_positions.append(np.clip(candidate, 0, 1))
        positions = np.array(next_positions)
        best = max(positions, key=fitness)
        if fitness(best) > fitness(rabbit):
            rabbit = best.copy()
    return rabbit
