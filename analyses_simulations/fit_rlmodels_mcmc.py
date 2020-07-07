# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Fits the hybrid and model-based models to the three data sets using MCMC.
"""

from os import mkdir
from os.path import join, exists
import utils

WARMUP = 20000
ITER = 40000
CHAINS = 1

def main():
    "Runs the model fitting for all data sets"
    hybrid_model = utils.get_stan_model(
        join(utils.MODELS_DIR, 'hybrid.stan'), join(utils.MODELS_DIR, 'hybrid.bin'))
    mb_model = utils.get_stan_model(
        join(utils.MODELS_DIR, 'model_based.stan'), join(utils.MODELS_DIR, 'model_based.bin'))
    if not exists(utils.MODEL_RESULTS_DIR):
        mkdir(utils.MODEL_RESULTS_DIR)
    for dsname, data_set in utils.load_data_sets():
        dsname = dsname.replace(' ', '_').lower()
        # Eliminate slow trials
        data_set = data_set[data_set.slow == 0]
        model_dat = {
            's1': [],
            'a1': [],
            'a2': [],
            's2': [],
            'reward': [],
            'N': len(data_set.participant.unique()),
            'num_trials': [
                len(data_set[data_set.participant == part]) \
                    for part in data_set.participant.unique()],
        }
        model_dat['T'] = max(model_dat['num_trials'])
        for i, part in enumerate(data_set.participant.unique()):
            part_info = data_set[data_set.participant == part]
            assert len(part_info) == model_dat['num_trials'][i]
            fill = [1]*(model_dat['T'] - len(part_info)) # Dummy data to fill the array
            model_dat['s1'].append(list(part_info.init_state) + fill)
            model_dat['s2'].append(list(part_info.final_state) + fill)
            model_dat['a1'].append(list(part_info.choice1) + fill)
            model_dat['a2'].append(list(part_info.choice2) + fill)
            model_dat['reward'].append(list(part_info.reward) + fill)
        for model_name, stan_model in (('hybrid', hybrid_model), ('model_based', mb_model)):
            flnm = f'{dsname}_{model_name}'
            results_flnm = join(utils.MODEL_RESULTS_DIR, flnm + '.txt')
            # Do not rerun the model if results already exist
            if not exists(results_flnm):
                fit = stan_model.sampling(
                    data=model_dat, iter=ITER, chains=CHAINS,
                    warmup=WARMUP, sample_file=join(utils.MODEL_RESULTS_DIR, flnm + '.csv'),
                    refresh=10)
                with open(results_flnm, 'w') as fit_results_file:
                    fit_results_file.write(str(fit))

if __name__ == "__main__":
    main()
