# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Simulations of three types of purely model-based agents: correct,
transition-dependent learning rates (TDLR), and unlucky symbol.
The data are analyzed and the results are plotted.
"""

import random
import pickle
from os.path import join, exists
from numba import jit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import create_random_rwrd_probs, expit, get_random_fstate,\
    get_random_reward, diffuse_rwrd_probs, fit_stan_model_maxlik,\
    CouldNotFitException, get_stan_model, PLOTS_DIR, RESULTS_DIR, MODELS_DIR,\
    COLOR_COMMON, COLOR_RARE

# Number of trials each simulated agent will perform
NUM_TRIALS = 1000
# Model parameters: learning rate, inverse temperature, and
# second-stage value reduction caused by choosing the "unlucky symbol"
ALPHA, BETA, ETA = 0.5, 5.0, 0.5
# Different learning rates for the TDLR model
ALPHA_COMMON, ALPHA_RARE = 0.8, 0.2

def mb_correct(alpha, beta):
    """Simulation of a correct model-based strategy."""
    value = np.zeros((2, 2))
    rwrd_probs = create_random_rwrd_probs()
    choice1 = 0
    for _ in range(NUM_TRIALS):
        r = 0.4*(max(value[1]) - max(value[0]))
        prob1 = expit(beta*r)
        choice1 = int(random.random() < prob1)
        fstate = get_random_fstate(choice1)
        choice2 = int(random.random() < expit(beta*(value[fstate, 1] - value[fstate, 0])))
        reward = get_random_reward(fstate, choice2, rwrd_probs)
        value[fstate, choice2] = (1 - alpha)*value[fstate, choice2] + alpha*reward
        rwrd_probs = diffuse_rwrd_probs(rwrd_probs)
        yield (choice1, fstate, choice2, reward)

def mb_unlucky_symbol(alpha, beta, eta):
    """Simulation of the unlucky symbol algorithm."""
    value = np.zeros((2, 2))
    rwrd_probs = create_random_rwrd_probs()
    choice1 = 0
    for _ in range(NUM_TRIALS):
        r = 0.7*max(value[1]) + 0.3*max(value[0]) - eta*(0.3*max(value[1]) + 0.7*max(value[0]))
        prob1 = expit(beta*r)
        choice1 = int(random.random() < prob1)
        fstate = get_random_fstate(choice1)
        value_reduction = eta if choice1 == 0 else 1
        choice2 = int(random.random() < expit(value_reduction*beta*(value[fstate, 1] - value[fstate, 0])))
        reward = get_random_reward(fstate, choice2, rwrd_probs)
        value[fstate, choice2] = (1 - alpha)*value[fstate, choice2] + alpha*reward
        rwrd_probs = diffuse_rwrd_probs(rwrd_probs)
        yield (choice1, fstate, choice2, reward)

def mb_tdlr(alpha_common, alpha_rare, beta):
    """Simulation of the transition-dependent learning rates model-based strategy."""
    value = np.zeros((2, 2))
    rwrd_probs = create_random_rwrd_probs()
    choice1 = 0
    for _ in range(NUM_TRIALS):
        r = 0.4*(max(value[1]) - max(value[0]))
        prob1 = expit(beta*r)
        choice1 = int(random.random() < prob1)
        fstate = get_random_fstate(choice1)
        choice2 = int(random.random() < expit(beta*(value[fstate, 1] - value[fstate, 0])))
        reward = get_random_reward(fstate, choice2, rwrd_probs)
        if choice1 == fstate: # Common transition
            alpha = alpha_common
        else:
            alpha = alpha_rare
        value[fstate, choice2] = (1 - alpha)*value[fstate, choice2] + alpha*reward
        rwrd_probs = diffuse_rwrd_probs(rwrd_probs)
        yield (choice1, fstate, choice2, reward)

class StayProbabilitiesCalculator:
    """Helper class for the stay probabilities."""
    def __init__(self):
        self.prev_reward = None
        self.prev_transition = None
        self.prev_choice = None
        self.stay = [(0, 0) for _ in range(4)]
    def add_trial_info(self, choice1, fstate, _, reward):
        if self.prev_choice is not None:
            indx = 2*self.prev_reward + self.prev_transition
            num, den = self.stay[indx]
            self.stay[indx] = (num + int(self.prev_choice == choice1), den + 1)
        self.prev_reward = reward
        self.prev_transition = int(choice1 == fstate)
        self.prev_choice = choice1
    def get_stay_prob(self, reward, transition):
        num, den = self.stay[2*reward + transition]
        return num/den

# Parameters of the hybrid model
HYBRID_PARAMS = ('alpha1', 'alpha2', 'lmbd', 'beta1', 'beta2', 'w', 'p')

MODELS = (mb_correct, mb_unlucky_symbol, mb_tdlr)
MODELS_PARAMS = {
    mb_correct: (ALPHA, BETA),
    mb_unlucky_symbol: (ALPHA, BETA, ETA),
    mb_tdlr: (ALPHA_COMMON, ALPHA_RARE, BETA),
}
MODEL_LABELS = {
    mb_correct: 'Correct',
    mb_unlucky_symbol: 'Unlucky symbol',
    mb_tdlr: 'TDLR',
}

MB_AGENTS_FLNM = join(RESULTS_DIR, 'mb_agents.pickle') 

def simulate_mb_agents(num_agents):
    """Simulates model-based agents, then calculates stay probabilities and hybrid model parameters."""
    if exists(MB_AGENTS_FLNM):
        with open(MB_AGENTS_FLNM, 'rb') as inpf:
            return pickle.load(inpf)
    stan_model = get_stan_model(
        join(MODELS_DIR, 'hybrid_no_log_lik.stan'),
        join(MODELS_DIR, 'hybrid_no_log_lik.bin'))
    cols = HYBRID_PARAMS + ('stay_prob_11', 'stay_prob_10', 'stay_prob_01', 'stay_prob_00')
    model_results = {model: None for model in MODELS}
    for model, params in MODELS_PARAMS.items():
        results = []
        agents = 0
        while agents < num_agents:
            model_dat = {}
            s2 = []
            rewards = []
            a1 = []
            a2 = []
            spc = StayProbabilitiesCalculator()
            for choice1, fstate, choice2, reward in model(*params):
                # We sum 1 to final state, choice1 and choice2
                # because the hybrid model expects 1/2 rather than 0/1.
                s2.append(fstate + 1)
                rewards.append(reward)
                a1.append(choice1 + 1)
                a2.append(choice2 + 1)
                spc.add_trial_info(choice1, fstate, choice2, reward)
            model_dat['s2'] = s2
            model_dat['reward'] = rewards
            model_dat['a1'] = a1
            model_dat['a2'] = a2
            model_dat['T'] = NUM_TRIALS
            try:
                fitted_params = fit_stan_model_maxlik(stan_model, model_dat)
            except CouldNotFitException:
                continue
            agents += 1
            results.append([float(v) for v in fitted_params.values()] + [
                spc.get_stay_prob(1, 1),
                spc.get_stay_prob(1, 0),
                spc.get_stay_prob(0, 1),
                spc.get_stay_prob(0, 0),
            ])
        model_results[model] = pd.DataFrame(results, columns=cols)
    with open(MB_AGENTS_FLNM, 'wb') as outf:
        pickle.dump(model_results, outf)
    return model_results

def plot_results(model_results, histogram_bins=None):
    """Plot results from simulated model-based agents in comparison with idealized results."""
    plt.figure(figsize=(3*4, 2*4))
    panel_label = ord('A')
    def draw_label(ax, panel_label):
        ax.text(-0.15, 1.2, chr(panel_label), transform=ax.transAxes, fontsize=16,
                va='top', ha='right')
        return panel_label + 1
    models_coefs = ( # These numbers are from Decker et al. (2016)
        ('Model-free',
         [0.9205614508160216, 0.9205614508160216, 0.7772998611746911, 0.7772998611746911]),
        ('Correct model-based',
         [0.9205614508160216, 0.7772998611746911, 0.7772998611746911, 0.9205614508160216]),
        ('Hybrid',
         [0.9478464369215823, 0.8721384336809187, 0.6899744811276125, 0.8556968659094812]),
    )
    for i, (condition, y) in enumerate(models_coefs):
        # Plot idealized stay probabilities
        ax = plt.subplot(2, 3, i + 1)
        panel_label = draw_label(ax, panel_label)
        plt.title(condition)
        plt.ylim(0.5, 1)
        plt.bar((0, 2), (y[0], y[2]), color=COLOR_COMMON, label='Common')
        plt.bar((1, 3), (y[1], y[3]), color=COLOR_RARE, label='Rare')
        plt.xticks((0.5, 2.5), ('Rewarded', 'Unrewarded'))
        plt.yticks([])
        plt.xlabel('Previous outcome')
        plt.ylabel('Stay probability')
        if i == 0:
            plt.legend(loc='best', title='Previous transition')
    for plotnum, model in enumerate((mb_unlucky_symbol, mb_tdlr)):
        # Plot stay probabilities
        ax = plt.subplot(2, 3, plotnum + 4)
        panel_label = draw_label(ax, panel_label)
        plt.title(MODEL_LABELS[model])
        plt.xlabel('Previous outcome')
        plt.ylabel('Stay probability')
        plt.ylim(0, 1)
        y = [
            model_results[model].stay_prob_11.mean(),
            model_results[model].stay_prob_10.mean(),
            model_results[model].stay_prob_01.mean(),
            model_results[model].stay_prob_00.mean(),
        ]
        plt.bar((0, 2), (y[0], y[2]), color=COLOR_COMMON, label='Common')
        plt.bar((1, 3), (y[1], y[3]), color=COLOR_RARE, label='Rare')
        plt.xticks((0.5, 2.5), ('Rewarded', 'Unrewarded'))
        if plotnum == 0:
            plt.legend(title='Previous transition')
    # Plot histogram of weights
    ax = plt.subplot(2, 3, 6)
    panel_label = draw_label(ax, panel_label)
    plt.title('Hybrid model fitting')
    plt.ylabel('Relative frequency')
    plt.xlabel('Model-based weight')
    plt.xlim(0, 1)
    for plotnum, model in enumerate(MODELS_PARAMS.keys()):
        plt.hist(
            model_results[model].w, bins=histogram_bins, density=True,
            label=MODEL_LABELS[model], alpha=0.75)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(join(PLOTS_DIR, 'plots_wrong_mb_agents.pdf'))
    plt.close()

if __name__ == "__main__":
    plot_results(simulate_mb_agents(num_agents=1000), histogram_bins=50)
