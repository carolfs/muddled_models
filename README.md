This project contains the code required to run the spaceship and magic carpet
tasks mentioned in the paper:

Feher da Silva, C., Hare, T.A.
*Humans primarily use model-based inference in the two-stage task.*
Nat Hum Behav (2020). https://doi.org/10.1038/s41562-020-0905-y

The experiments inside the `magic_carpet` and `spaceship` directories
require Python 2.7 and PsychoPy 2 to run.
They should run on standalone PsychoPy version 1.90.3.
Just run the `exp.py` script in each directory.

We used the following fonts to display the instructions for the participants:
* `Noteworthy-Bold.ttf` for the spaceship task
* `OpenSans-SemiBold.ttf` for the magic carpet task

These fonts are not included in this repository because of the license.
To run the tasks, you will need to change the path to these fonts.
Where you have to do this is indicated in the `exp.py` file.

The behavioral data (choices and questionnaires) can be found in the `results`
directory.

The data analyses and simulations are still being added to this repository.
They use Python 3.6 and the `Pipenv` package to manage dependencies.