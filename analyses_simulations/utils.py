# Copyright (C) 2019  Carolina Feher da Silva

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Common Python functions used in many of the analyses."""

import os
import pickle
import random
import io
import pystan
import numpy as np
import pandas as pd
from numba import jit, vectorize, float64
import requests
from scipy.io import loadmat

ANALYSES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(ANALYSES_DIR, '..')

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
def expit(x):
    "The logistic function, using numba"
    if x < -200:
        return np.exp(x) # Approximation
    else:
        return 1/(1 + np.exp(-x))

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

# Link to the data from Kool et al. (2016) we analyzed in this paper
COMMON_INSTR_DATA = 'https://github.com/wkool/tradeoffs/blob/master/data/'\
    'daw%20paradigm/data.mat?raw=true'
COMMON_INSTR_VARS = {
    'stim_1_left': 1,
    'stim_1_right': 2,
    'rt_1': 3,
    'choice1': 4,
    'stim_2_left': 5,
    'stim_2_right': 6,
    'rt_2': 7,
    'choice2': 8,
    'win': 9,
    'state2': 10,
    'common': 11,
    'score': 12,
    'ps1a1': 14,
    'ps1a2': 15,
    'ps2a1': 16,
    'ps2a2': 17,
    'trial': 18,
}
COMMON_INSTR_CSV_FILE = os.path.join(PROJECT_DIR, 'results', 'common_instr_data.csv')

def load_common_instr_data():
    "Loads the data from Kool et al. (2016) as a pandas DataFrame."
    if not os.path.exists(COMMON_INSTR_CSV_FILE):
        fetch_common_instr_data()
    dtype = {
        'participant': int,
        'stim_1_left': int,
        'stim_1_right': int,
        'rt_1': int,
        'choice1': int,
        'stim_2_left': int,
        'stim_2_right': int,
        'rt_2': int,
        'choice2': int,
        'win': int,
        'state2': int,
        'common': int,
        'score': int,
        'trial': int,
    }
    data = pd.read_csv(COMMON_INSTR_CSV_FILE, dtype=dtype)
    return data

def load_magic_carpet_data():
    "Loads the data from the magic carpet experiment as a pandas DataFrame."
    data_dir = os.path.join(PROJECT_DIR, 'results', 'magic_carpet', 'choices')
    return pd.concat((
        pd.read_csv(os.path.join(data_dir, flnm)).assign(participant=flnm.split('_')[0])
        for flnm in os.listdir(data_dir) if flnm.endswith('_game.csv'))).reset_index()

def load_spaceship_data():
    "Loads the data from the spaceship experiment as a pandas DataFrame."
    data_dir = os.path.join(PROJECT_DIR, 'results', 'spaceship', 'choices')
    ssdf = pd.concat((
        pd.read_csv(os.path.join(data_dir, flnm)).assign(participant=flnm.split('.')[0])
        for flnm in os.listdir(data_dir) \
        if flnm.endswith('.csv') and not flnm.endswith('_practice.csv')))
    ssdf['init_state'] = ssdf.symbol0*2 + ssdf.symbol1
    return ssdf.reset_index()

def fetch_common_instr_data():
    "Fetches the data from Kool et al. (2016) off its Github repository."
    data = requests.get(COMMON_INSTR_DATA)
    data = loadmat(io.BytesIO(data.content))['data']
    name_vars = list(COMMON_INSTR_VARS.keys())
    name_vars.sort()
    try:
        with open(COMMON_INSTR_CSV_FILE, 'w') as outf:
            outf.write('participant,{}\n'.format(','.join(name_vars)))
            for trial in data:
                # Exclude practice trials
                if int(trial[13]) == 1:
                    continue
                assert str(trial[0][0]).startswith('subject_')
                participant = int(str(trial[0][0]).split('_')[1]) - 1
                assert participant < 206
                values = []
                for var in name_vars:
                    value = trial[COMMON_INSTR_VARS[var]][0][0]
                    values.append(str(value))
                outf.write('{},{}\n'.format(participant, ','.join(values)))
    except:
        os.remove(COMMON_INSTR_CSV_FILE)
        raise
