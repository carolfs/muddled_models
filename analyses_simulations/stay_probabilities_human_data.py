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

"""Calculates and plots stay probabilities for human participants."""

from os.path import join
import pandas as pd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from utils import load_stan_csv_chains, COLOR_COMMON, COLOR_RARE, PLOTS_DIR, hpd
from logistic_regression_human_data import run_logreg_analyses,\
    get_logreg_results_filename, DATA_SET_NAMES

# Number of samples to use in stay probability calculation
# Not all are necessary (it makes no difference)
NUM_SAMPLES_TO_ANALYZE = 10000

def calc_stay_probs(samples):
    "Calculate stay probabilities from logistic regression samples"
    probs = [[] for i in range(4)]
    for _, sample in samples.sample(NUM_SAMPLES_TO_ANALYZE).iterrows():
        part = 1
        sample_probs = [[] for i in range(4)]
        while True:
            try:
                inter = sample['coefs.{}.1'.format(part)]
                rew = sample['coefs.{}.2'.format(part)]
                trans = sample['coefs.{}.3'.format(part)]
                rewxtrans = sample['coefs.{}.4'.format(part)]
                sample_probs[0].append(expit(inter + rew + trans + rewxtrans))
                sample_probs[1].append(expit(inter + rew - trans - rewxtrans))
                sample_probs[2].append(expit(inter - rew + trans - rewxtrans))
                sample_probs[3].append(expit(inter - rew - trans + rewxtrans))
            except KeyError:
                break
            else:
                part += 1
        for slist, problist in zip(sample_probs, probs):
            problist.append(np.mean(slist))
    return [pd.Series(problist) for problist in probs]

def plot_stay_probs(probs, condition, legend):
    "Plots stay probabilities with error bars"
    y = [prob.median() for prob in probs]
    plt.title(condition)
    plt.ylim(0, 1)
    plt.bar((0, 2), (y[0], y[2]), color=COLOR_COMMON, label='Common')
    plt.bar((1, 3), (y[1], y[3]), color=COLOR_RARE, label='Rare')
    plt.xticks((0.5, 2.5), ('Rewarded', 'Unrewarded'))
    yerr = [[], []]
    for yy, prob_series in zip(y, probs):
        uplim, lolim = hpd(prob_series)
        yerr[0].append(yy - lolim)
        yerr[1].append(uplim - yy)
    plt.errorbar((0, 1, 2, 3), y, yerr, fmt='none', ecolor='black')
    plt.xlabel('Previous outcome')
    plt.ylabel('Stay probability')
    if legend:
        plt.legend(loc='best', title='Previous transition')

def plot_all_stay_probs():
    "Calculates and plots stay probabilities for all human participant data sets"
    run_logreg_analyses()
    plt.figure(figsize=(3*4, 4))
    for num_plot, data_set_name in enumerate(DATA_SET_NAMES):
        samples = load_stan_csv_chains(get_logreg_results_filename(data_set_name, 'all'))
        stay_probs = calc_stay_probs(samples)
        plt.subplot(1, 3, num_plot + 1)
        plot_stay_probs(stay_probs, data_set_name, (num_plot == 0))
    plt.tight_layout()
    plt.savefig(join(PLOTS_DIR, 'human_stay_probabilities.pdf'))
    plt.close()

if __name__ == "__main__":
    plot_all_stay_probs()
