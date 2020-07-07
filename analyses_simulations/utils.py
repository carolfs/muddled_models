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
import sys
import io
import re
import tempfile
import math
import pystan
import numpy as np
import pandas as pd
from numba import jit, vectorize, float64
import requests
from scipy.io import loadmat

ANALYSES_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(ANALYSES_DIR, 'plots')
RESULTS_DIR = os.path.join(ANALYSES_DIR, 'results')
PROJECT_DIR = os.path.join(ANALYSES_DIR, '..')
MODELS_DIR = os.path.join(ANALYSES_DIR, 'models')
MODEL_RESULTS_DIR = os.path.join(ANALYSES_DIR, 'model_results')

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
    cminstr = pd.read_csv(COMMON_INSTR_CSV_FILE, dtype=dtype)
    # Create new columns for this data set, to make it compatible with the others
    cminstr['slow'] = (cminstr.win == -1).astype('int') # Mark slow trials
    cminstr['reward'] = cminstr.win
    cminstr['isymbol_lft'] = cminstr.stim_1_left
    cminstr['init_state'] = [1]*len(cminstr)
    cminstr['final_state'] = cminstr['state2']
    return cminstr

def load_magic_carpet_data():
    "Loads the data from the magic carpet experiment as a pandas DataFrame."
    data_dir = os.path.join(PROJECT_DIR, 'results', 'magic_carpet', 'choices')
    flnms = os.listdir(data_dir)
    flnms.sort()
    mcdata = pd.concat((
        pd.read_csv(os.path.join(data_dir, flnm)).assign(participant=flnm.split('_')[0])
        for flnm in flnms if flnm.endswith('_game.csv'))).reset_index()
    mcdata['init_state'] = [1]*len(mcdata)
    return mcdata

def load_spaceship_data():
    "Loads the data from the spaceship experiment as a pandas DataFrame."
    data_dir = os.path.join(PROJECT_DIR, 'results', 'spaceship', 'choices')
    flnms = os.listdir(data_dir)
    flnms.sort()
    ssdf = pd.concat((
        pd.read_csv(os.path.join(data_dir, flnm)).assign(participant=flnm.split('.')[0])
        for flnm in flnms \
        if flnm.endswith('.csv') and not flnm.endswith('_practice.csv'))).reset_index()
    ssdf['init_state'] = ssdf.symbol0*2 + ssdf.symbol1 + 1
    ssdf.choice1 = ssdf.apply(get_spaceship_choice1, axis=1)
    ssdf.choice2 = ssdf.choice2 + 1
    ssdf['final_state'] = ssdf['final_state'] + 1
    return ssdf

def get_spaceship_choice1(trial):
    """Changes choice encoding to relative 1/2 instead of left(0)/right(1)."""
    if trial.common:
        return trial.final_state + 1
    return 2 - trial.final_state

def load_data_sets():
    "Loads all three sets of human data"
    cminstr = load_common_instr_data()
    mcdata = load_magic_carpet_data()
    ssdata = load_spaceship_data()

    data_sets = (
        ('Common instructions', cminstr),
        ('Magic carpet', mcdata),
        ('Spaceship', ssdata),
    )
    return data_sets

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

class CouldNotFitException(Exception):
    "Exception when Stan model could not be fitted by max likelihoood"

def fit_stan_model_maxlik(stan_model, model_dat, num_fits=10):
    """Fits a Stan model to data to get the maximum likelihood parameters."""
    log_lik = -np.inf
    params = None
    errors = 0
    for _ in range(num_fits):
        while True:
            try:
                op_result = stan_model.optimizing(data=model_dat, as_vector=False, iter=5000)
            except RuntimeError:
                errors += 1
                if errors > 100:
                    sys.stderr.write('Could not fit model to data\n')
                    raise CouldNotFitException
                continue
            else:
                break
        if op_result['value'] > log_lik:
            log_lik = op_result['value']
            params = op_result['par']
    return params

def load_stan_csv(csvfn):
    """Loads only post-adaptation Stan samples from a CSV file"""
    with tempfile.TemporaryFile('w+') as outf:
        with open(csvfn) as inpf:
            column_names = []
            adaptation = True
            for line in inpf.readlines():
                if adaptation:
                    if line.startswith('#'):
                        if line.startswith('# Adaptation terminated'):
                            adaptation = False
                    elif not column_names:
                        column_names = line.strip().split(',')
                else:
                    outf.write(line)
        outf.seek(0)
        samples = pd.read_csv(outf, names=column_names, comment='#')
    return samples

def load_stan_csv_chains(csvfn):
    "Loads multiple chains of Stan samples from CSV files"
    dirname = os.path.dirname(os.path.abspath(csvfn))
    sample_fn_re = re.compile(r'{}(_\d+)?\.csv'.format(
        os.path.splitext(os.path.basename(csvfn))[0]))
    samples = None
    if not os.path.exists(dirname):
        raise ValueError("Directory {} does not exist".format(dirname))
    for fn in os.listdir(dirname):
        if sample_fn_re.match(fn):
            if samples is None:
                samples = load_stan_csv(os.path.join(dirname, fn))
            else:
                samples = pd.concat(
                    (samples, load_stan_csv(os.path.join(dirname, fn))))
    return samples

def hpd(sampleVec, credMass=0.95):
    """
    Computes highest density interval from a sample of representative values,
      estimated as shortest credible interval.
    Arguments:
      sampleVec
        is a vector of representative values from a probability distribution.
      credMass
        is a scalar between 0 and 1, indicating the mass within the credible
        interval that is to be estimated.
    Value:
      HDIlim is a vector containing the limits of the HDI
    
    Adapted from:
      Kruschke, J. K. (2015). Doing Bayesian Data Analysis, Second Edition: 
      A Tutorial with R, JAGS, and Stan. Academic Press / Elsevier.
    https://sites.google.com/site/doingbayesiandataanalysis/software-installation
    """
    sortedPts = list(sampleVec)
    sortedPts.sort()
    ciIdxInc = math.ceil(credMass * len(sortedPts))
    nCIs = len(sortedPts) - ciIdxInc
    ciWidth = [0] * nCIs
    for i in range(nCIs):
        ciWidth[i] = sortedPts[i + ciIdxInc] - sortedPts[i]
    j = ciWidth.index(min(ciWidth))
    HDImin = sortedPts[j]
    HDImax = sortedPts[j + ciIdxInc]
    HDIlim = (HDImin, HDImax)
    return HDIlim

# Color for the stay probabilities after rare transitions (red)
COLOR_RARE = (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0)
# Color for the stay probabilities after common transitions (blue)
COLOR_COMMON = (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0)
