import v0
import numpy as np


def random_argmax(values, rng, rtol=1e-9, atol=0.0):
    """Argmax with uniform random tie-breaking among (numerically) equal maxima."""
    ties = np.flatnonzero(np.isclose(values, values.max(), rtol=rtol, atol=atol))
    return rng.choice(ties)

rng = np.random.default_rng()
print(random_argmax(np.array([1, 2, 3, 3, 2, 8]), rng))