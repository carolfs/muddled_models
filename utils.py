"""Common Python functions used in many of the analyses."""

import os
import pickle
import random
import pystan
import numpy as np
from numba import jit, vectorize, float64

def get_stan_model(textfn, binfn):
    """Loads a Stan model, compiling it if necessary."""
    assert textfn != binfn
    if os.path.exists(binfn) and \
        os.path.getmtime(binfn) > os.path.getmtime(textfn):
        with open(binfn, 'rb') as arq:
            stan_model = pickle.load(arq)
    else:
        stan_model = pystan.StanModel(textfn)
        with open(binfn, 'wb') as arq:
            pickle.dump(stan_model, arq)
    return stan_model

# The functions and constants below help simulating the two-stage task.

MIN_RWRD_PROB_VALUE = 0.25
MAX_RWRD_PROB_VALUE = 0.75
RWRD_PROB_DIFFUSION_RATE = 0.025
COMMON_PROB = 0.7

@jit
def get_random_reward(fstate, choice2, rwrd_probs):
    "Get a reward (0 or 1) with this probability."
    return int(random.random() < rwrd_probs[2*fstate + choice2])

@vectorize([float64(float64)])
def diffuse_rwrd_probs(prob):
    "Diffuse reward probability and reflect it on boundaries."
    next_value = prob + (random.gauss(0, RWRD_PROB_DIFFUSION_RATE) % 1)
    if next_value > MAX_RWRD_PROB_VALUE:
        next_value = 2*MAX_RWRD_PROB_VALUE - next_value
    if next_value < MIN_RWRD_PROB_VALUE:
        next_value = 2*MIN_RWRD_PROB_VALUE - next_value
    return next_value

@jit
def create_random_rwrd_probs():
    "Create random reward probabilities within the allowed interval."
    return np.random.uniform(MIN_RWRD_PROB_VALUE, MAX_RWRD_PROB_VALUE, 4)

@jit
def get_random_fstate(choice):
    "Return the final state given a first-stage choice (0 or 1)."
    if random.random() < COMMON_PROB:
        return choice
    return 1 - choice
