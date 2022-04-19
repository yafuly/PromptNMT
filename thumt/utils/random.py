import random
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, seed=1):
        random.seed(seed)
    def __call__(self, data):
        return list(random.sample(data, len(data)))
    def sample_ratio(self, th=0.35):
        return float(random.uniform(0, th))

class RandomGenerator(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, seed=1):
        self.rng = np.random.RandomState(seed)

    def shuffle(self, data):
        shuf = self.rng.choice(data, size=len(data), replace=False)
        return list(shuf)

    def uniform_ratio(self, th=0.35):
        ratio = self.rng.uniform(0, th) 
        return float(ratio)


class RandomShuffler1(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, seed=1):
        random.Random(seed)
        self._random_state = random.getstate()
        
        # if self._random_state is None:
        #     self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        print(self.random_state[1][0])
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))

    def sample_ratio(self, th=0.35):
        with self.use_internal_state():
            return random.uniform(0, th) 

