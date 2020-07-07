# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Logistic regression analysis of human data from three data sets: common instructions,
magic carpet, and spaceship.
"""

from os.path import join, exists
import matplotlib.pyplot as plt
import numpy as np
import utils

def get_all_trial_pairs(part_df):
    "Gets all trial pairs from a participant's data"
    for prev_trial, next_trial in zip(part_df[:-1].itertuples(), part_df[1:].itertuples()):
        yield prev_trial, next_trial

def get_same_sides_trial_pairs(part_df):
    "Gets trial pairs where the first-stage symbols are on the same sides"
    for prev_trial, next_trial in zip(part_df[:-1].itertuples(), part_df[1:].itertuples()):
        if prev_trial.isymbol_lft == next_trial.isymbol_lft:
            yield prev_trial, next_trial

def get_different_sides_trial_pairs(part_df):
    "Gets trial pairs where the first-stage symbols are on different sides"
    for prev_trial, next_trial in zip(part_df[:-1].itertuples(), part_df[1:].itertuples()):
        if prev_trial.isymbol_lft != next_trial.isymbol_lft:
            yield prev_trial, next_trial

def get_same_ss_trial_pairs(part_df):
    "Gets trial pairs with the same spaceship and planet at the first-stage board"
    for prev_trial, next_trial in zip(part_df[:-1].itertuples(), part_df[1:].itertuples()):
        if prev_trial.init_state == next_trial.init_state:
            yield prev_trial, next_trial

def get_first_ss_trial_pairs(part_df):
    "Gets trial pairs with the same spaceship but different planets at the first-stage board"
    for prev_trial, next_trial in zip(part_df[:-1].itertuples(), part_df[1:].itertuples()):
        if prev_trial.symbol0 == next_trial.symbol0 and prev_trial.symbol1 != next_trial.symbol1:
            yield prev_trial, next_trial

def get_second_ss_trial_pairs(part_df):
    "Gets trial pairs with different spaceships but the same planet at the first-stage board"
    for prev_trial, next_trial in zip(part_df[:-1].itertuples(), part_df[1:].itertuples()):
        if prev_trial.symbol0 != next_trial.symbol0 and prev_trial.symbol1 == next_trial.symbol1:
            yield prev_trial, next_trial

def get_different_ss_trial_pairs(part_df):
    "Gets trial pairs with different spaceships and planest at the first-stage board"
    for prev_trial, next_trial in zip(part_df[:-1].itertuples(), part_df[1:].itertuples()):
        if prev_trial.symbol0 != next_trial.symbol0 and prev_trial.symbol1 != next_trial.symbol1:
            yield prev_trial, next_trial

DATA_SET_NAMES = ('Common instructions', 'Magic carpet', 'Spaceship')
LOGREG_ANALYSES = {
    'Common instructions': (
        ('all', get_all_trial_pairs),
        ('same_sides', get_same_sides_trial_pairs),
        ('diff_sides', get_different_sides_trial_pairs),
    ),
    'Magic carpet': (
        ('all', get_all_trial_pairs),
        ('same_sides', get_same_sides_trial_pairs),
        ('diff_sides', get_different_sides_trial_pairs),
    ),
    'Spaceship': (
        ('all', get_all_trial_pairs),
        ('same', get_same_ss_trial_pairs),
        ('spaceship', get_first_ss_trial_pairs),
        ('planet', get_second_ss_trial_pairs),
        ('diff', get_different_ss_trial_pairs),
    ),
}

def get_logreg_data(trial_pairs, num_trials):
    "Calculate logistic regression predictors"
    # Intercept, Reward, Transition, Reward x Transition
    x = [
        (1, 2*prev_trial.reward - 1, 2*prev_trial.common - 1,
             (2*prev_trial.reward - 1)*(2*prev_trial.common - 1))
        for prev_trial, _ in trial_pairs
    ]
    # Calculate the dependent variable (stay)
    y = [
        int(prev_trial.choice1 == next_trial.choice1)
        for prev_trial, next_trial in trial_pairs
    ]
    assert len(x) == len(y)
    # Insert rows of zeros to complete the correct number of trials
    # These rows don't make any difference because all predictors (including intercept)
    # are zero
    if len(x) < num_trials:
        zeros = [0, 0, 0, 0]
        dif = (num_trials - len(x))
        x += dif*[zeros]
        y += dif*[0]
    return x, y

ITER = 30000
WARMUP = 15000
CHAINS = 4

def get_logreg_results_filename(data_set_name, analysis_name):
    "Returns the name of the file where the Stan samples are saved, without extension"
    data_set_name = data_set_name.replace(' ', '_').lower()
    return join(utils.RESULTS_DIR, f'logreg_samples_{data_set_name}_{analysis_name}')

def run_logreg_model(data_set_name, data, get_trial_pairs, analysis_name):
    "Returns the results of the logistic regression model fitting."
    logreg_model = utils.get_stan_model(
        join(utils.MODELS_DIR, 'logreg_model.stan'), join(utils.MODELS_DIR, 'logreg_model.bin'))
    # Exclude slow trials
    data = data[data['slow'] == 0]
    # Get data for each participant
    part_dfs = [data[data.participant == part] for part in data.participant.unique()]
    # Calculate maximum number of trials
    num_trials = max([len(part_df) for part_df in part_dfs])
    # Get logistic regression data for each participant
    logreg_data = [
        get_logreg_data(tuple(get_trial_pairs(part_df)), num_trials) for part_df in part_dfs]
    model_dat = {
        'M': len(logreg_data), # Number of participants
        'N': num_trials,
        'K': 4, # Number of predictors
        'y': [y for x, y in logreg_data], # Dependent variable
        'x': [x for x, y in logreg_data], # Predictors
    }
    # Run the model
    filename = get_logreg_results_filename(data_set_name, analysis_name)
    fit = logreg_model.sampling(
        data=model_dat, iter=ITER, chains=CHAINS, warmup=WARMUP,
        sample_file=filename + '.csv')
    with open(filename + '.txt', 'w') as outf:
        outf.write(str(fit))

def run_logreg_analyses():
    "Runs the logistic regression analyses and saves the Stan samples to disk"
    data_sets = utils.load_data_sets()
    for data_set_name, data in data_sets:
        for analysis_name, get_trial_pairs in LOGREG_ANALYSES[data_set_name]:
            if not exists(
                get_logreg_results_filename(data_set_name, analysis_name) + '.txt'):
                run_logreg_model(data_set_name, data, get_trial_pairs, analysis_name)

def plot_coefs(coefs, label, offset, errorbars=True, marker='o'):
    "Plots the coefficients of a logistic regression analysis"
    x = np.array((0, 1, 2, 3))
    if errorbars:
        y = [coef_series.mean() for coef_series in coefs]
        yerr = [[], []]
        for yy, coef_series in zip(y, coefs):
            uplim, lolim = utils.hpd(coef_series)
            yerr[0].append(yy - lolim)
            yerr[1].append(uplim - yy)
        plt.errorbar(x + offset, y, yerr, fmt='none', ecolor='black')
    else:
        y = coefs
    plt.plot(x + offset, y, marker, label=label)

def get_mean_coefs(samples):
    "Returns the series of mean coefficients"
    return [samples[f'grp.{i + 1}'] for i in range(4)]

def plot_logreg_coefs():
    "Plots the coefficients of all logistic regression analyses"
    # Plot results using all trial pairs
    plt.figure()
    plt.title('Logistic regression including all trial pairs')
    for set_num, data_set_name in enumerate(DATA_SET_NAMES):
        samples = utils.load_stan_csv_chains(
            get_logreg_results_filename(data_set_name, 'all'))
        plot_coefs(get_mean_coefs(samples), data_set_name, 0.15*(set_num - 1))
    plt.grid(True, axis='y')
    plt.xticks((0, 1, 2, 3), ('Intercept', 'Reward', 'Transition', 'Reward ×Transition'))
    plt.ylabel('Mean value')
    plt.xlabel('Coefficient')
    plt.axhline(y=0, color='black', zorder=-100)
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.savefig(join(utils.PLOTS_DIR, 'logreg_all_coefs.pdf'))
    plt.close()
    # Plot results from the same sides/different sides categories
    plt.figure(figsize=[2*6.4, 4.8])
    sides_analyses = (
        ('same_sides', 'Same sides'),
        ('diff_sides', 'Different sides'),
    )
    for set_num, data_set_name in enumerate(('Common instructions', 'Magic carpet')):
        plt.subplot(1, 2, set_num + 1)
        plt.title(data_set_name)
        for ana_num, (analysis_name, analysis_label) in enumerate(sides_analyses):
            samples = utils.load_stan_csv_chains(
                get_logreg_results_filename(data_set_name, analysis_name))
            plot_coefs(get_mean_coefs(samples), analysis_label, 0.15*(ana_num - 0.5))
        plt.grid(True, axis='y')
        plt.xticks((0, 1, 2, 3), ('Intercept', 'Reward', 'Transition', 'Reward ×Transition'))
        plt.ylabel('Mean value')
        plt.xlabel('Coefficient')
        plt.axhline(y=0, color='black', zorder=-100)
        plt.legend(loc='upper center', title='Symbol locations')
    plt.tight_layout()
    plt.savefig(join(utils.PLOTS_DIR, 'logreg_sides_coefs.pdf'))
    plt.close()
    # Plot spaceship categories
    plt.figure()
    ss_cats = (
        ('same', 'Same planet and spaceship'),
        ('spaceship', 'Same spaceship'),
        ('planet', 'Same planet'),
        ('diff', 'Different planet and spaceship'),
    )
    plt.title('Spaceship')
    for ana_num, (analysis_name, analysis_label) in enumerate(ss_cats):
        samples = utils.load_stan_csv_chains(
            get_logreg_results_filename('Spaceship', analysis_name))
        plot_coefs(get_mean_coefs(samples), analysis_label, 0.15*(ana_num - 1.5))
    plt.grid(True, axis='y')
    plt.xticks((0, 1, 2, 3), ('Intercept', 'Reward', 'Transition', 'Reward ×Transition'))
    plt.ylabel('Mean value')
    plt.xlabel('Coefficient')
    plt.axhline(y=0, color='black', zorder=-100)
    plt.legend(loc='upper center', title='Subset')
    plt.tight_layout()
    plt.savefig(join(utils.PLOTS_DIR, 'logreg_spaceship_coefs.pdf'))
    plt.close()

if __name__ == "__main__":
    run_logreg_analyses()
    plot_logreg_coefs()
