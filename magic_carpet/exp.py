# -*- coding: utf-8 -*-

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

"""Runs the two-stage task with the magic carpet story."""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import argparse
import sys
import os
import socket
import random
import csv
import io
import time
from itertools import chain
import StringIO
from os.path import join
from psychopy import visual, core, event, data, gui
import wx

# Directories
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ASSETS_DIR = join(CURRENT_DIR, 'assets')
RESULTS_DIR = join(CURRENT_DIR, 'results')
# Money per trial
RWRD = 0.37
# Maximum response time in seconds
RESPONSE_TIME = 2
# CHANGE PARAMETER BELOW BEFORE RUNNING
# Font for displaying the instructions
TTF_FONT = join(CURRENT_DIR, 'OpenSans-SemiBold.ttf')
# Configuration for tutorial and game
class TutorialConfig:
    final_state_colors = ('red', 'black')
    initial_state_symbols = (7, 8)
    final_state_symbols = ((9, 10), (11, 12))
    num_trials = 50
    block = 67
    common_prob = 0.7 # Optimal performance 61%
    @classmethod
    def proceed(cls, trials, slow_trials):
        del slow_trials
        return trials < cls.num_trials
    @classmethod
    def do_break(cls, trials, slow_trials):
        del trials, slow_trials
        return False
    @classmethod
    def get_common(cls, trial):
        if trial == 0 or trial == 1:
            return True
        if trial == 2:
            return False
        return random.random() < cls.common_prob

class GameConfig:
    final_state_colors = ('pink', 'blue')
    initial_state_symbols = (1, 2)
    final_state_symbols = ((3, 4), (5, 6))
    num_trials = 201
    block = 67
    common_prob = 0.7 # Optimal performance 61%
    @classmethod
    def proceed(cls, trials, slow_trials):
        return (trials - slow_trials) < cls.num_trials
    @classmethod
    def do_break(cls, trials, slow_trials):
        return trials % cls.block == 0 and (trials - slow_trials) <= (cls.num_trials - 10)
    @classmethod
    def get_common(cls, trial):
        del trial
        return random.random() < cls.common_prob

# Classes and functions

def main():
    try:
        # Get participant information
        # This was actually not performed
        # We used the computer name instead
        info = {'Participant code': ''}
        dlg = gui.DlgFromDict(info, title='Participant information')
        if not dlg.OK:
            core.quit()
        part_code = info['Participant code'].strip().upper()
        if not part_code:
            print('Empty participant code')
            core.quit()

        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)

        filename = join(
            RESULTS_DIR, '{}_{}'.format(part_code, data.getDateStr()))
        if part_code == 'TEST':
            fullscr = False # Displays small window
            # Decrease number of trials
            TutorialConfig.num_trials = 1
            GameConfig.num_trials = 2
            GameConfig.block = 1
        else:
            fullscr = True # Fullscreen for the participants

        # Create window
        win = visual.Window(
            fullscr=fullscr, size=[1280, 1024], units='pix', color='#404040',
            checkTiming=False)
        win.mouseVisible = False

        # Load all images
        images = load_image_collection(win, ASSETS_DIR)

        # Create a center text stimulus for announcing the reward
        reward_text = visual.TextStim(
            win=win,
            pos=(0, 0),
            fontFiles=[TTF_FONT],
            font='OpenSans',
            color=(1, 1, 1),
            units='height',
            height=0.09,
            name='Reward text',
        )

        instructions = Instructions(win, images)

        # Randomize mountain sides and common transitions for the tutorial and game
        tutorial_mountain_sides = list(TutorialConfig.final_state_colors)
        random.shuffle(tutorial_mountain_sides)
        tutorial_model = Model.create_random(TutorialConfig)
        game_mountain_sides = list(GameConfig.final_state_colors)
        random.shuffle(game_mountain_sides)
        game_model = Model.create_random(GameConfig)
        # Save game configuration
        with io.open('{}_config.txt'.format(filename), 'w', encoding='utf-8') as outf:
            outf.write(str(game_model) + '\n')

        # Tutorial instructions
        string_replacements = get_string_replacements(
            TutorialConfig, tutorial_mountain_sides, tutorial_model)
        instructions.display('tutorial_instructions.txt', string_replacements)

        # Quiz
        run_quiz(win, images, string_replacements)

        # Tutorial flights
        instructions.display('tutorial_flights_instructions.txt', string_replacements)
        with io.open('{}_tutorial.csv'.format(filename), 'wb') as outf:
            csv_writer = csv.DictWriter(outf, fieldnames=CSV_FIELDNAMES)
            csv_writer.writeheader()
            run_trial_sequence(
                TutorialConfig, TutorialDisplay(win, images, tutorial_mountain_sides),
                tutorial_model, csv_writer)

        # Game instructions
        string_replacements = get_string_replacements(
            GameConfig, game_mountain_sides, game_model)
        instructions.display('game_instructions.txt', string_replacements)

        # Game
        with io.open('{}_game.csv'.format(filename), 'wb') as outf:
            csv_writer = csv.DictWriter(outf, fieldnames=CSV_FIELDNAMES)
            csv_writer.writeheader()
            rewards = run_trial_sequence(
                GameConfig, GameDisplay(win, images), game_model, csv_writer)

        # Save payment information in a file
        payment_filename = '{}_{:.2f}'.format(filename, get_money_reward(RWRD, rewards))
        with open(payment_filename, 'w') as _:
            pass

        # Show payment
        money_msg = 'You won CHF {:.2f}.\n'\
            'Press SPACE to fill in the questionnaire.'.format(
                get_money_reward(RWRD, rewards))
        reward_text.text = money_msg
        reward_text.draw()
        win.flip()
        event.waitKeys(keyList=('space',))
        win.close()

        # Show questionnaire
        show_questionnaire(ASSETS_DIR, filename, GameConfig, game_model)
    except Exception as exc:
        print(exc)
        sys.stdout.flush()
        time.sleep(3600)

class RewardProbability(float):
    "Reward probability that drifts within a min and a max value."
    MIN_VALUE = 0.25
    MAX_VALUE = 0.75
    DIFFUSION_RATE = 0.025
    def __new__(cls, value):
        assert value >= cls.MIN_VALUE and value <= cls.MAX_VALUE
        return super(RewardProbability, cls).__new__(cls, value)
    @classmethod
    def create_random(cls):
        "Create a random reward probability within the allowed interval."
        return cls(random.uniform(cls.MIN_VALUE, cls.MAX_VALUE))
    def diffuse(self):
        "Get the next probability by diffusion."
        return self.__class__(
            self.reflect_on_boundaries(random.gauss(0, self.DIFFUSION_RATE)))
    def get_reward(self):
        "Get a reward (0 or 1) with this probability."
        return int(random.random() < self)
    def reflect_on_boundaries(self, incr):
        "Reflect reward probability on boundaries."
        next_value = self + (incr % 1)
        if next_value > self.MAX_VALUE:
            next_value = 2*self.MAX_VALUE - next_value
        if next_value < self.MIN_VALUE:
            next_value = 2*self.MIN_VALUE - next_value
        return next_value

class Symbol(object):
    "A Tibetan symbol for a carpet or lamp."
    def __init__(self, code):
        self.code = code
    def __str__(self):
        return '{:02d}'.format(self.code)

class InitialSymbol(Symbol):
    "An initial state symbol."
    def __init__(self, code, final_state):
        super(InitialSymbol, self).__init__(code)
        self.final_state = final_state

class FinalSymbol(Symbol):
    "A final state symbol."
    def __init__(self, code, reward_probability):
        super(FinalSymbol, self).__init__(code)
        self.reward_probability = reward_probability
        self.reward = self.reward_probability.get_reward()

class State(object):
    "A initial state in the task."
    def __init__(self, symbols):
        assert len(symbols) == 2
        self.symbols = symbols

class FinalState(State):
    "A final state in the task."
    def __init__(self, color, symbols):
        self.color = color
        super(FinalState, self).__init__(symbols)

class Model(object):
    """A transition model and configuration of final states for the task."""
    def __init__(self, isymbol_codes, colors, fsymbol_codes):
        self.isymbol_codes = isymbol_codes
        self.colors = colors
        self.fsymbol_codes = fsymbol_codes
    @classmethod
    def create_random(cls, config):
        """Create a random model for the task from a given configuration."""
        colors = list(config.final_state_colors)
        random.shuffle(colors)
        fsymbol_codes = list(config.final_state_symbols)
        random.shuffle(fsymbol_codes)
        return cls(config.initial_state_symbols, colors, fsymbol_codes)
    def get_paths(self, common):
        "Generator for the paths from initial symbol to final symbols."
        if common:
            for isymbol_code, color, fsymbol_codes in zip(
                    self.isymbol_codes, self.colors, self.fsymbol_codes):
                yield (isymbol_code, color, fsymbol_codes)
        else:
            for isymbol_code, color, fsymbol_codes in zip(
                    self.isymbol_codes, reversed(self.colors), reversed(self.fsymbol_codes)):
                yield (isymbol_code, color, fsymbol_codes)
    def __str__(self):
        output = "Common transitions: "
        for isymbol_code, color, fsymbol_codes in self.get_paths(True):
            output += "{} -> {} -> {}; ".format(isymbol_code, color, fsymbol_codes)
        return output

class Trial(object):
    "A trial in the task."
    def __init__(self, number, initial_state, common):
        self.number = number
        self.initial_state = initial_state
        self.common = common
    @classmethod
    def get_sequence(cls, config, model):
        "Get an infinite sequence of trials with this configuration."
        trials = 0
        reward_probabilities = {
            fsymbol_code: RewardProbability.create_random()
            for fsymbol_code in chain(*config.final_state_symbols)
        }
        while True:
            common = config.get_common(trials)
            isymbols = []
            for isymbol_code, color, fsymbol_codes in model.get_paths(common):
                fsymbols = [
                    FinalSymbol(fsymbol_code, reward_probabilities[fsymbol_code])
                    for fsymbol_code in fsymbol_codes
                ]
                random.shuffle(fsymbols)
                final_state = FinalState(color, tuple(fsymbols))
                isymbols.append(InitialSymbol(isymbol_code, final_state))
            random.shuffle(isymbols)
            initial_state = State(isymbols)
            yield cls(trials, initial_state, common)
            for fsymbol_code, prob in reward_probabilities.items():
                reward_probabilities[fsymbol_code] = prob.diffuse()
            trials += 1

CSV_FIELDNAMES = (
    'trial', 'common', 'reward.1.1', 'reward.1.2', 'reward.2.1',
    'reward.2.2', 'isymbol_lft', 'isymbol_rgt', 'rt1', 'choice1', 'final_state',
    'fsymbol_lft', 'fsymbol_rgt', 'rt2', 'choice2', 'reward', 'slow')

def get_intertrial_interval():
    return random.uniform(0.7, 1.3)

def code_to_bin(code, common=True):
    if common:
        return 2 - code % 2
    else:
        return code % 2 + 1

def run_trial_sequence(config, display, model, csv_writer):
    rewards = 0
    slow_trials = 0
    common_transitions = {
        isymbol_code: {'color': color}
        for isymbol_code, color, fsymbol_codes in model.get_paths(True)
    }
    # Trial loop
    for trial in Trial.get_sequence(config, model):
        completed_trials = trial.number - slow_trials
        row = {'trial': trial.number, 'common': int(trial.common)}
        for isymbol in trial.initial_state.symbols:
            for fsymbol in isymbol.final_state.symbols:
                key = 'reward.{}.{}'.format(
                    code_to_bin(isymbol.code, trial.common), code_to_bin(fsymbol.code))
                row[key] = fsymbol.reward_probability
        row['isymbol_lft'] = code_to_bin(trial.initial_state.symbols[0].code)
        row['isymbol_rgt'] = code_to_bin(trial.initial_state.symbols[1].code)

        display.display_start_of_trial(trial.number)

        # First-stage choice
        isymbols = [symbol.code for symbol in trial.initial_state.symbols]
        display.display_carpets(completed_trials, isymbols, common_transitions)

        event.clearEvents()
        keys_times = event.waitKeys(
            maxWait=2, keyList=('left', 'right'), timeStamped=core.Clock())

        if keys_times is None:
            slow_trials += 1
            display.display_slow1()
            row.update({
                'rt1': -1,
                'choice1': -1,
                'final_state': -1,
                'fsymbol_lft': -1,
                'fsymbol_rgt': -1,
                'rt2': -1,
                'choice2': -1,
                'reward': 0,
                'slow': 1,
            })
        else:
            choice1, rt1 = keys_times[0]
            row['rt1'] = rt1

            display.display_selected_carpet(completed_trials, choice1, isymbols, common_transitions)

            # Transition
            chosen_symbol1 = trial.initial_state.symbols[int(choice1 == 'right')]
            final_state = chosen_symbol1.final_state
            row['choice1'] = code_to_bin(chosen_symbol1.code)
            row['final_state'] = code_to_bin(chosen_symbol1.code, trial.common)

            display.display_transition(completed_trials, final_state.color, trial.common)

            # Second-stage choice
            fsymbols = [symbol.code for symbol in final_state.symbols]
            row['fsymbol_lft'] = code_to_bin(final_state.symbols[0].code)
            row['fsymbol_rgt'] = code_to_bin(final_state.symbols[1].code)

            display.display_lamps(completed_trials, final_state.color, fsymbols)

            event.clearEvents()
            keys_times = event.waitKeys(
                maxWait=2, keyList=('left', 'right'), timeStamped=core.Clock())
            if keys_times is None:
                slow_trials += 1
                display.display_slow2(final_state.color)
                row.update({
                    'rt2': -1,
                    'choice2': -1,
                    'reward': 0,
                    'slow': 1,
                })
            else:
                choice2, rt2 = keys_times[0]
                row['rt2'] = rt2

                display.display_selected_lamp(completed_trials, final_state.color, fsymbols, choice2)

                # Reward
                chosen_symbol2 = final_state.symbols[int(choice2 == 'right')]
                row['choice2'] = code_to_bin(chosen_symbol2.code)
                reward = chosen_symbol2.reward
                row['reward'] = reward
                row['slow'] = 0
                if reward:
                    rewards += 1
                    display.display_reward(completed_trials, final_state.color, chosen_symbol2.code)
                else:
                    display.display_no_reward(completed_trials, final_state.color, chosen_symbol2.code)

        display.display_end_of_trial()

        # Break
        if config.do_break(trial.number + 1, slow_trials):
            display.display_break()
            event.waitKeys(keyList=('space',))
        assert all([fdn in row.keys() for fdn in CSV_FIELDNAMES])
        assert all([key in CSV_FIELDNAMES for key in row.keys()])
        csv_writer.writerow(row)
        # Should we run another trial?
        if not config.proceed(trial.number + 1, slow_trials):
            break
    return rewards

# String replacements
def replace_all(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def get_string_replacements(config, mountain_sides, model):
    string_replacements = {
        '[color1]': mountain_sides[0].capitalize(),
        '[color2]': mountain_sides[1].capitalize(),
        '[num_trials]': str(config.num_trials),
    }
    for num, (isymbol_code, color, fsymbol_codes) in enumerate(model.get_paths(True)):
        string_replacements.update({
            '[color_common{}]'.format(num + 1): color.capitalize(),
            '[isymbol{}]'.format(num + 1): '{:02d}'.format(isymbol_code),
            '[fsymbol{}1]'.format(num + 1): '{:02d}'.format(fsymbol_codes[0]),
            '[fsymbol{}2]'.format(num + 1): '{:02d}'.format(fsymbol_codes[1]),
        })
    return string_replacements

def load_image_collection(win, images_directory):
    image_collection = {
        os.path.splitext(fn)[0]: visual.ImageStim(
            win=win,
            pos=(0, 0),
            image=join(images_directory, fn),
            name=os.path.splitext(fn)[0]
        )
        for fn in os.listdir(images_directory) if os.path.splitext(fn)[1] == '.png'
    }
    return image_collection

def get_random_transition_model(config):
    isymbols = list(config.initial_state_symbols)
    fsymbols = list(config.final_state_symbols)
    colors = list(config.final_state_colors)
    random.shuffle(isymbols)
    random.shuffle(fsymbols)
    random.shuffle(colors)
    return {isymbols[i]: {'color': colors[i], 'symbols': fsymbols[i]} for i in xrange(2)}

def get_money_reward(rwrd, x):
    """Round reward for easier paying"""
    return round(rwrd*x, 1)

def show_questionnaire(assets_dir, filename, config, model):
    class Questionnaire(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(
                self, None, wx.ID_ANY, 'Questionnaire', size=(500, 600), style=wx.CAPTION)
            font = wx.SystemSettings_GetFont(wx.SYS_SYSTEM_FONT)
            font.SetPointSize(14)
            panel = wx.Panel(self)
            vbox = wx.BoxSizer(wx.VERTICAL)
            # Symbol 1
            text_symbol1 = wx.StaticText(
                panel,
                label='What was the meaning of the symbol below?')
            text_symbol1.SetFont(font)
            vbox.Add(text_symbol1, 1, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=10)
            symbol1_png = wx.Image(
                join(assets_dir, 'tibetan.{:02d}.png'.format(config.initial_state_symbols[0])),
                wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            s1bm = wx.StaticBitmap(
                panel, bitmap=symbol1_png, size=(
                    symbol1_png.GetWidth(), symbol1_png.GetHeight()))
            mountains = ['{} Mountain'.format(color.capitalize()) \
                for color in config.final_state_colors]
            self.symbol1_color = [
                wx.RadioButton(panel, label=label, style=(wx.RB_GROUP if i == 0 else 0))
                for i, label in enumerate(mountains)
            ]
            s1_hbox = wx.BoxSizer(wx.HORIZONTAL)
            s1_hbox.Add(s1bm, 1, wx.TOP|wx.LEFT|wx.RIGHT, border=10)
            s1_vbox = wx.BoxSizer(wx.VERTICAL)
            for button in self.symbol1_color:
                s1_vbox.Add(button, 1, wx.TOP|wx.LEFT|wx.RIGHT, border=10)
            s1_hbox.Add(s1_vbox, 1, wx.LEFT|wx.RIGHT|wx.TOP, border=10)
            vbox.Add(s1_hbox, 1, wx.BOTTOM, border=20)
            # Symbol 2
            text_symbol2 = wx.StaticText(
                panel,
                label='What was the meaning of the symbol below?')
            text_symbol2.SetFont(font)
            vbox.Add(text_symbol2, 1, wx.LEFT|wx.RIGHT|wx.TOP, border=10)
            symbol2_png = wx.Image(
                join(assets_dir, 'tibetan.{:02d}.png'.format(config.initial_state_symbols[1])),
                wx.BITMAP_TYPE_ANY).ConvertToBitmap()
            s2bm = wx.StaticBitmap(
                panel, bitmap=symbol2_png, size=(
                    symbol1_png.GetWidth(), symbol2_png.GetHeight()))
            self.symbol2_color = [
                wx.RadioButton(panel, label=label, style=(wx.RB_GROUP if i == 0 else 0))
                for i, label in enumerate(mountains)
            ]
            s2_hbox = wx.BoxSizer(wx.HORIZONTAL)
            s2_hbox.Add(s2bm, 1, wx.TOP|wx.LEFT|wx.RIGHT, border=10)
            s2_vbox = wx.BoxSizer(wx.VERTICAL)
            for button in self.symbol2_color:
                s2_vbox.Add(button, 1, wx.TOP|wx.LEFT|wx.RIGHT, border=10)
            s2_hbox.Add(s2_vbox, 1, wx.LEFT|wx.RIGHT|wx.TOP, border=10)
            vbox.Add(s2_hbox, 1, wx.BOTTOM, border=20)
            # Task difficulty
            text_hard = wx.StaticText(
                panel,
                label='How difficult was the game?')
            text_hard.SetFont(font)
            vbox.Add(text_hard, 1, wx.LEFT|wx.RIGHT|wx.TOP, border=10)
            self.difficulty_labels = ['very easy', 'easy', 'average', 'difficult', 'very difficult']
            self.difficulty_buttons = [
                wx.RadioButton(panel, label=label, style=(wx.RB_GROUP if i == 0 else 0))
                for i, label in enumerate(
                    self.difficulty_labels)
            ]
            diff_box = wx.BoxSizer(wx.HORIZONTAL)
            for button in self.difficulty_buttons:
                diff_box.Add(button, 1, wx.TOP|wx.LEFT|wx.RIGHT, border=10)
            vbox.Add(diff_box, 1, wx.BOTTOM, border=20)
            # Strategy
            question = wx.StaticText(
                panel,
                label='Please describe in detail the strategy you used to make your choices.')
            question.SetFont(font)
            vbox.Add(question, 1, wx.LEFT|wx.RIGHT|wx.BOTTOM, border=10)
            self.answer = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
            self.answer.SetFont(font)
            vbox.Add(self.answer, 10, wx.EXPAND|wx.LEFT|wx.RIGHT, border=10)
            button = wx.Button(panel, wx.ID_OK, 'Send', (80, 220))
            vbox.Add(button, 1, wx.ALIGN_CENTER|wx.ALL, border=10)
            self.Bind(wx.EVT_BUTTON, self.OnClose, button)
            panel.SetSizer(vbox)
            self.Centre()
        def OnClose(self, _):
            # Check if questionnaire was completed
            check = (
                any([button.GetValue() for button in self.symbol1_color]),
                any([button.GetValue() for button in self.symbol2_color]),
                any([button.GetValue() for button in self.difficulty_buttons]),
                bool(self.answer.GetValue()),
            )
            if not all(check):
                msg = wx.MessageDialog(
                    self, 'Please fill in the questionnaire before sending it.', 'Error',
                    style=wx.OK|wx.ICON_ERROR|wx.CENTRE|wx.STAY_ON_TOP)
                msg.ShowModal()
                msg.Destroy()
                return
            # Save questionnaire in a file
            common_transitions = {
                isymbol_code: color for isymbol_code, color, _ in model.get_paths(common=True)}
            with io.open('{}_questionnaire.txt'.format(filename), 'w', encoding='utf-8') as outf:
                if self.symbol1_color[0].GetValue() == \
                    (common_transitions[config.initial_state_symbols[0]] == \
                        config.final_state_colors[0]):
                    outf.write('symbol1: correct\n')
                else:
                    outf.write('symbol1: incorrect\n')
                if self.symbol2_color[0].GetValue() == \
                    (common_transitions[config.initial_state_symbols[1]] == \
                        config.final_state_colors[0]):
                    outf.write('symbol2: correct\n')
                else:
                    outf.write('symbol2: incorrect\n')
                for label, button in zip(self.difficulty_labels, self.difficulty_buttons):
                    if button.GetValue():
                        difficulty = label
                outf.write('Difficulty: {}\n\n'.format(difficulty))
                outf.write(unicode(self.answer.GetValue()))
            self.Close()
            goodbye = Goodbye()
            goodbye.Show()

    class Goodbye(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(
                self, None, wx.ID_ANY, 'Goodbye', size=(500, 100), style=wx.CAPTION)
            font = wx.SystemSettings_GetFont(wx.SYS_SYSTEM_FONT)
            font.SetPointSize(14)
            panel = wx.Panel(self)
            vbox = wx.BoxSizer(wx.VERTICAL)
            msg = 'Thank you for participating in this experiment! '\
                'Please wait for your payment. You will get it soon.'
            msg = wx.StaticText(panel, label=msg)
            msg.SetFont(font)
            vbox.Add(msg, 1, wx.EXPAND|wx.ALL, border=10)
            panel.SetSizer(vbox)
            self.Centre()

    app = wx.App(False)
    questionnaire = Questionnaire()
    questionnaire.Show(True)
    app.MainLoop()

class Instructions(object):
    def __init__(self, win, images):
        self.win = win
        self.images = images
        self.instr_text = visual.TextStim(
            win=win,
            pos=(-527, 359),
            alignHoriz='left',
            height=35,
            fontFiles=[TTF_FONT],
            font='OpenSans',
            color=(1, 1, 1),
            wrapWidth=1077,
            name='Instructions text'
        )
        self.instr_all_text = visual.TextStim(
            win=win,
            pos=(-527, 0),
            alignHoriz='left',
            height=35,
            fontFiles=[TTF_FONT],
            font='OpenSans',
            color=(1, 1, 1),
            wrapWidth=1077,
            name='Introduction text'
        )
        self.instr_image = visual.ImageStim(
            win=win,
            pos=(0, -97),
            name='Instructions image'
        )
    @staticmethod
    def __parse_instructions(instructions_fn, string_replacements):
        with io.open(join(ASSETS_DIR, instructions_fn), encoding='utf-8') as instr_file:
            screens = []
            images = []
            while True:
                line = instr_file.readline()
                if line == "":
                    text = replace_all(text, string_replacements)
                    screens.append({'text': text, 'images': tuple(images)})
                    break
                elif line.startswith('"'):
                    text = line[1:]
                    while True:
                        if len(line.strip()) > 0 and line.strip()[-1] == '"':
                            text = text.strip()[:-1]
                            break
                        line = instr_file.readline()
                        text += line
                elif line == '\n':
                    text = replace_all(text, string_replacements)
                    screens.append({'text': text, 'images': tuple(images)})
                    images = []
                else:
                    image = replace_all(line.strip(), string_replacements)
                    images.append(image)
        return screens
    def display(self, instructions_fn, string_replacements):
        screens = self.__parse_instructions(instructions_fn, string_replacements)
        scrnum = 0
        while True:
            screen = screens[scrnum]
            if screen['text'].find('\n') != -1:
                self.instr_all_text.text = screen['text']
                self.instr_all_text.draw()
            else:
                self.instr_text.text = screen['text']
                self.instr_text.draw()
                for image in screen['images']:
                    self.instr_image.image = join(
                        ASSETS_DIR, 'instructions', image.lower() + '.png')
                    self.instr_image.draw()
            keyList = tuple()
            if scrnum > 0:
                self.images['arrow_left'].draw()
                keyList += ('left',)
            if scrnum < (len(screens) - 1):
                self.images['arrow_right'].draw()
                keyList += ('right',)
            else:
                keyList += ('space',)
            self.win.flip()
            keys = event.waitKeys(keyList=keyList)
            if 'left' in keys:
                scrnum -= 1
                if scrnum < 0:
                    scrnum = 1
            elif 'right' in keys:
                scrnum += 1
            else:
                assert 'space' in keys
                break

def run_quiz(win, images, string_replacements):
    quiz_text = visual.TextStim(
        win=win,
        pos=(-450, +50),
        alignHoriz='left',
        height=35,
        fontFiles=[TTF_FONT],
        font='OpenSans',
        color=(1, 1, 1),
        wrapWidth=900,
        name='Quiz text'
    )
    quiz_help_text = visual.TextStim(
        win=win,
        pos=(-450, -250),
        text='Press the key corresponding to the correct answer.',
        alignHoriz='left',
        height=35,
        fontFiles=[TTF_FONT],
        font='OpenSans',
        color=(1, 1, 1),
        wrapWidth=900,
        name='Quiz help text'
    )
    quiz_feedback = visual.TextStim(
        win=win,
        pos=(-250, -250),
        alignHoriz='left',
        height=35,
        fontFiles=[TTF_FONT],
        font='OpenSans',
        color=(1, 1, 1),
        wrapWidth=900,
        name='Quiz feedback text'
    )
    with io.open(join(ASSETS_DIR, 'quiz.py'), encoding='utf-8') as qf:
        quiz = qf.read()
    quiz = eval(quiz)

    answered = []
    while True:
        for qn, (question, answer, correct, maxanswer) in enumerate(quiz):
            if qn in answered:
                continue
            quiz_text.text = "{}\n\n{}".format(
                replace_all(question, string_replacements),
                replace_all(answer, string_replacements))
            quiz_text.draw()
            quiz_help_text.draw()
            win.flip()
            keyList = [chr(ord('a') + i) for i in xrange(ord(maxanswer) - ord('a') + 1)]
            key = event.waitKeys(keyList=keyList)[0].lower()
            quiz_text.draw()
            if key == correct:
                images['quiz_correct'].draw()
                quiz_feedback.text = 'Correct!'
                answered.append(qn)
            else:
                images['quiz_incorrect'].draw()
                quiz_feedback.text = 'The correct answer is {}.'.format(correct)
            quiz_feedback.draw()
            win.flip()
            core.wait(3)
        if len(answered) == len(quiz):
            break

class GameDisplay(object):
    def __init__(self, win, images):
        self.win = win
        self.images = images
    def display_start_of_trial(self, trial):
        pass
    def display_carpets(self, trial, isymbols, common_transitions):
        isymbols_image = self.images['tibetan.{:02d}{:02d}'.format(*isymbols)]
        self.images['carpets_glow'].draw()
        isymbols_image.draw()
        self.win.flip()
    def display_selected_carpet(self, trial, choice1, isymbols, common_transitions):
        isymbols_image = self.images['tibetan.{:02d}{:02d}'.format(*isymbols)]
        self.images['carpets'].draw()
        self.images['{}_carpet_selected'.format(choice1)].draw()
        isymbols_image.draw()
        self.win.flip()
        core.wait(3)
    def display_transition(self, trial, final_state_color, common):
        self.images['nap'].draw()
        self.win.flip()
        core.wait(1)
    def display_lamps(self, trial, final_state_color, fsymbols):
        fsymbols_image = self.images['tibetan.{:02}{:02}'.format(*fsymbols)]
        self.images['lamps_{}_glow'.format(final_state_color)].draw()
        fsymbols_image.draw()
        self.win.flip()
    def display_selected_lamp(self, trial, final_state_color, fsymbols, choice2):
        fsymbols_image = self.images['tibetan.{:02}{:02}'.format(*fsymbols)]
        self.images['lamps_{}'.format(final_state_color)].draw()
        self.images['{}_lamp_selected'.format(choice2)].draw()
        fsymbols_image.draw()
        self.win.flip()
        core.wait(3)
    def display_reward(self, trial, final_state_color, chosen_symbol2):
        self.images['genie_coin'].draw()
        self.images['reward_{}'.format(final_state_color)].draw()
        self.images['tibetan.{:02}'.format(chosen_symbol2)].draw()
        self.win.flip()
        core.wait(1.5)
    def display_no_reward(self, trial, final_state_color, chosen_symbol2):
        self.images['genie_zero'].draw()
        self.images['reward_{}'.format(final_state_color)].draw()
        self.images['tibetan.{:02}'.format(chosen_symbol2)].draw()
        self.win.flip()
        core.wait(1.5)
    def display_end_of_trial(self):
        self.images['carpet_background'].draw()
        self.win.flip()
        core.wait(get_intertrial_interval())
    def display_slow1(self):
        self.images['slow1'].draw()
        self.win.flip()
        core.wait(3)
    def display_slow2(self, final_state_color):
        self.images['lamps_{}'.format(final_state_color)].draw()
        self.images['slow2'].draw()
        self.win.flip()
        core.wait(3)
    def display_break(self):
        self.images['break'].draw()
        self.win.flip()

class TutorialDisplay(object):
    def __init__(self, win, images, mountain_sides):
        self.win = win
        self.images = images
        self.mountain_sides = mountain_sides
        self.visits_to_mountains = {color: 0 for color in TutorialConfig.final_state_colors}
        # Create frame to display messages
        self.msg_frame = visual.Rect(
            win=self.win,
            pos=(0, 400),
            width=1180,
            height=80,
            fillColor=(1.0, 1.0, 1.0),
            opacity=0.9,
            name='Tutorial message frame',
        )
        self.msg_text = visual.TextStim(
            win=self.win,
            pos=(-560, 405),
            height=30,
            fontFiles=[TTF_FONT],
            font='OpenSans',
            color=(-1, -1, -1),
            wrapWidth=1120,
            alignHoriz='left',
            alignVert='center',
            name='Tutorial message text'
        )
        self.center_text = visual.TextStim(
            win=win,
            pos=(0, 0),
            height=80,
            wrapWidth=980,
            fontFiles=[TTF_FONT],
            font='OpenSans',
            color=(1, 1, 1),
            name='Center text'
        )
    def display_start_of_trial(self, trial):
        self.center_text.text = 'Tutorial flight #{}'.format(trial + 1)
        self.center_text.draw()
        self.win.flip()
        core.wait(3)
    def display_carpets(self, trial, isymbols, common_transitions):
        isymbols_image = self.images['tibetan.{:02d}{:02d}'.format(*isymbols)]
        destination_image = self.images['carpets_to_{}_{}'.format(
            *[common_transitions[symbol]['color'] for symbol in isymbols]
        )]

        def draw_main_images():
            self.images['carpets_tutorial'].draw()
            isymbols_image.draw()
            destination_image.draw()

        if trial < 3:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            self.msg_text.text = 'You took your magic carpets out of the cupboard '\
                'and unrolled them on the floor.'
            self.msg_text.draw()
            self.win.flip()
            core.wait(4.5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            if trial < 2:
                draw_main_images()
                self.images['left_carpet_destination'].draw()
                self.msg_frame.draw()
                self.msg_text.text = "On the left you put the carpet that was enchanted "\
                    "to fly to {} Mountain …".format(
                        common_transitions[isymbols[0]]['color'].capitalize())
                self.msg_text.draw()
                self.win.flip()
                core.wait(4.5)

                draw_main_images()
                self.win.flip()
                core.wait(0.5)

                draw_main_images()
                self.images['right_carpet_destination'].draw()
                self.msg_frame.draw()
                self.msg_text.text = "… and on the right the carpet that was enchanted "\
                    "to fly to {} Mountain.".format(
                        common_transitions[isymbols[1]]['color'].capitalize())
                self.msg_text.draw()
                self.win.flip()
                core.wait(4.5)

                draw_main_images()
                self.win.flip()
                core.wait(0.5)

                draw_main_images()
                self.images['tutorial_carpet_symbols'].draw()
                self.msg_frame.draw()
                self.msg_text.text = "The symbols written on the carpets mean “{} "\
                    "Mountain” and “{} Mountain” in the local language.".format(
                        common_transitions[isymbols[0]]['color'].capitalize(),
                        common_transitions[isymbols[1]]['color'].capitalize(),
                    )
                self.msg_text.draw()
                self.win.flip()
                core.wait(6)

                draw_main_images()
                self.win.flip()
                core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            self.msg_text.text = 'You will soon be able to choose a carpet '\
                'and fly on it by pressing the left or right arrow key.'
            self.msg_text.draw()
            self.win.flip()
            core.wait(6)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            self.msg_text.text = 'When the carpets start to glow, you have 2 seconds '\
                'to press a key, or else they will fly away without you.'
            self.msg_text.draw()
            self.win.flip()
            core.wait(6)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)
        elif trial < 10:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            self.msg_text.text = "Your carpets are out of the cupboard and about "\
                "to glow. Prepare to make your choice."
            self.msg_text.draw()
            self.win.flip()
            core.wait(5)

        # Glow carpets for response
        self.images['carpets_glow_tutorial'].draw()
        isymbols_image.draw()
        destination_image.draw()
        self.win.flip()
    def display_selected_carpet(self, trial, choice1, isymbols, common_transitions):
        isymbols_image = self.images['tibetan.{:02d}{:02d}'.format(*isymbols)]
        destination_image = self.images['carpets_to_{}_{}'.format(
            *[common_transitions[symbol]['color'] for symbol in isymbols]
        )]
        def draw_main_images():
            self.images['carpets_tutorial'].draw()
            isymbols_image.draw()
            destination_image.draw()
            self.images['tutorial_{}_carpet_selected'.format(choice1)].draw()
        if trial < 10:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            self.msg_text.text = 'You chose the carpet on the {}, enchanted to fly to '\
                '{} Mountain. Bon voyage!'.format(
                    choice1,
                    common_transitions[isymbols[int(choice1 == 'right')]]['color'].capitalize(),
                )
            self.msg_text.draw()
            self.win.flip()
            core.wait(4)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)
        else:
            draw_main_images()
            self.win.flip()
            core.wait(3)
    def display_transition(self, trial, final_state_color, common):
        transition_image = self.images['flight_{}-{}_{}{}'.format(
            final_state_color,
            self.mountain_sides[0],
            self.mountain_sides[1],
            '-wind' if not common else '',
        )]

        if common:
            self.msg_text.text = 'Your flight to {} Mountain goes well, without '\
                'any incidents.'.format(final_state_color.capitalize())
        else:
            colors = TutorialConfig.final_state_colors
            self.msg_text.text = 'Oh, no! The wind near {} Mountain is too strong. '\
                'You decide to land your carpet on {} Mountain instead.'.format(
                    colors[1 - colors.index(final_state_color)].capitalize(),
                    final_state_color.capitalize(),
                )
        if trial < 10:
            transition_image.draw()
            self.win.flip()
            core.wait(0.5)

            transition_image.draw()
            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(3 if common else 6)

            transition_image.draw()
            self.win.flip()
            core.wait(0.5)
        else:
            transition_image.draw()
            self.win.flip()
            core.wait(2)
    def display_lamps(self, trial, final_state_color, fsymbols):
        fsymbols_image = self.images['tibetan.{:02}{:02}'.format(*fsymbols)]
        def draw_main_images():
            self.images['lamps_{}'.format(final_state_color)].draw()
            fsymbols_image.draw()

        if self.visits_to_mountains[final_state_color] < 1:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_text.text = 'You have safely landed on {} Mountain.'.format(
                final_state_color.capitalize())
            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(4.5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_text.text = 'Here are the lamps where the {} Mountain genies live.'.format(
                final_state_color.capitalize())
            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(4.5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_text.text = 'The lamp on the left is home to the genie '\
                'whose name is shown below.'
            self.msg_frame.draw()
            self.msg_text.draw()
            self.images['left_lamp_symbol'].draw()
            self.win.flip()
            core.wait(5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_text.text = 'The lamp on the right is home to the genie '\
                'whose name is shown below.'
            self.msg_frame.draw()
            self.msg_text.draw()
            self.images['right_lamp_symbol'].draw()
            self.win.flip()
            core.wait(5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_text.text = 'You will soon be able to choose a lamp and '\
                'rub it by pressing the left or right arrow key.'
            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(6)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_text.text = 'When the lamps start to glow, you have 2 seconds '\
                'to press a key, or else the genies will go to sleep.'
            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(6)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)
        elif trial < 10:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_text.text = 'You have safely landed on {} Mountain, and the '\
                'lamps are about to glow. Prepare to make your choice.'.format(
                    final_state_color.capitalize())
            self.msg_frame.draw()
            self.msg_text.draw()
            self.win.flip()
            core.wait(5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

        self.images['lamps_{}_glow'.format(final_state_color)].draw()
        fsymbols_image.draw()
        self.win.flip()
        self.visits_to_mountains[final_state_color] += 1
    def display_selected_lamp(self, trial, final_state_color, fsymbols, choice2):
        fsymbols_image = self.images['tibetan.{:02}{:02}'.format(*fsymbols)]
        def draw_main_images():
            self.images['lamps_{}'.format(final_state_color)].draw()
            self.images['{}_lamp_selected'.format(choice2)].draw()
            fsymbols_image.draw()
        if trial < 10:
            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            draw_main_images()
            self.msg_frame.draw()
            self.msg_text.text = 'You pick up the lamp on the {} and rub it.'.format(choice2)
            self.msg_text.draw()
            self.win.flip()
            core.wait(3.5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)
        else:
            draw_main_images()
            self.win.flip()
            core.wait(3)
    def display_reward(self, trial, final_state_color, chosen_symbol2):
        def draw_main_images():
            self.images['genie_coin'].draw()
            self.images['reward_{}'.format(final_state_color)].draw()
            self.images['tibetan.{:02}'.format(chosen_symbol2)].draw()

        if trial < 10:
            draw_main_images()
            self.win.flip()
            core.wait(1.5)

            draw_main_images()
            self.msg_frame.draw()
            self.msg_text.text = 'The genie came out of his lamp, listened to a song, '\
                'and gave you a gold coin!'
            self.msg_text.draw()
            self.win.flip()
            core.wait(5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            if trial < 2:
                draw_main_images()
                self.msg_frame.draw()
                self.msg_text.text = 'Remember this genie’s name in case you want to choose his '\
                    'lamp again in the future.'
                self.msg_text.draw()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(5)

                draw_main_images()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(0.5)

                draw_main_images()
                self.msg_frame.draw()
                self.msg_text.text = 'The color of his lamp reminds you he lives '\
                    'on {} Mountain.'.format(final_state_color.capitalize())
                self.msg_text.draw()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(5)

                draw_main_images()
                self.win.flip()
                core.wait(0.5)
        else:
            draw_main_images()
            self.win.flip()
            core.wait(1.5)
    def display_no_reward(self, trial, final_state_color, chosen_symbol2):
        def draw_main_images():
            self.images['genie_zero'].draw()
            self.images['reward_{}'.format(final_state_color)].draw()
            self.images['tibetan.{:02}'.format(chosen_symbol2)].draw()

        if trial < 10:
            draw_main_images()
            self.win.flip()
            core.wait(1.5)

            draw_main_images()
            self.msg_frame.draw()
            self.msg_text.text = 'The genie stayed inside his lamp, and you didn’t get a gold coin.'
            self.msg_text.draw()
            self.win.flip()
            core.wait(5)

            draw_main_images()
            self.win.flip()
            core.wait(0.5)

            if trial < 2:
                draw_main_images()
                self.msg_frame.draw()
                self.msg_text.text = 'Remember this genie’s name in case you want to choose his '\
                    'lamp again in the future.'
                self.msg_text.draw()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(5)

                draw_main_images()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(0.5)

                draw_main_images()
                self.msg_frame.draw()
                self.msg_text.text = 'The color of his lamp reminds you he lives '\
                    'on {} Mountain.'.format(final_state_color.capitalize())
                self.msg_text.draw()
                self.images['rubbed_lamp'].draw()
                self.win.flip()
                core.wait(5)

                draw_main_images()
                self.win.flip()
                core.wait(0.5)
        else:
            draw_main_images()
            self.win.flip()
            core.wait(1.5)
    def display_end_of_trial(self):
        pass
    def display_slow1(self):
        self.images['slow1'].draw()
        self.win.flip()
        core.wait(4)
    def display_slow2(self, final_state_color):
        self.images['lamps_{}'.format(final_state_color)].draw()
        self.images['slow2'].draw()
        self.win.flip()
        core.wait(4)
    def display_break(self):
        self.images['break'].draw()
        self.win.flip()

if __name__ == '__main__':
    main()
