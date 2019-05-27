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

"""Runs the two-stage task with the spaceships story."""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import random
import hashlib
import sys
import time
import os
import re
import io
import socket
from os.path import join
import argparse
import pandas as pd

# ------------------------------------
# Parameters

# Common transition probability
COMMON_PROB = 0.7
# Number of second-stage choices
NUM_2ND_CHOICES = 4
# Number of practice trials
NUM_PRAC_TRIALS = 20
# Number of trials in the experiment
NUM_TRIALS = 250
# Number of trials in a block
BLOCK = 84
# This directory
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
# Assets directory
ASSETS_DIR = join(CURRENT_DIR, 'assets')
# Money per trial
RWRD = 0.29
# Maximum response time in seconds
RESPONSE_TIME = 2
# Results directory
RESULTS_DIR = join(CURRENT_DIR, 'results')

# CHANGE PARAMETER BELOW BEFORE RUNNING
# Font for displaying the instructions
TTF_FONT = join(CURRENT_DIR, 'Noteworthy-Bold.ttf')

# ------------------------------------

def get_money_reward(x):
    """Round reward for easier paying"""
    return round(RWRD*x, 1)

def init_reward_probs():
    return [random.uniform(0.25, 0.75) for _ in range(NUM_2ND_CHOICES)]

def drift_reward_prob(p):
    p += random.gauss(0, 0.025) % 1
    if p > 0.75:
        p = 1.5 - p
    if p < 0.25:
        p = 0.5 - p
    return p

def drift_reward_probs(probs):
    return [drift_reward_prob(prob) for prob in probs]

# String replacements
def replace_all(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

def display_instructions(instructions_fn, end_screen_image):
    image_num = 1
    with io.open(join(CURRENT_DIR, instructions_fn), encoding='utf-8') as tut_instr:
        screens = []
        images = []
        while True:
            line = tut_instr.readline()
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
                    line = tut_instr.readline()
                    text += line
            elif line == '\n':
                text = replace_all(text, string_replacements)
                screens.append({'text': text, 'images': tuple(images)})
                images = []
            else:
                imagefn = replace_all(line.strip(), string_replacements)
                imagefn = join(ASSETS_DIR, 'instructions', '{}.png'.format(imagefn))
                images.append(imagefn)
    scrnum = 0
    while True:
        if scrnum < len(screens):
            screen = screens[scrnum]
        if scrnum == len(screens):
            middle_image.image = join(ASSETS_DIR, end_screen_image)
            middle_image.draw()
        elif len(screen['images']) > 0:
            instr_text.text = screen['text']
            instr_text.draw()
            for image in screen['images']:
                instr_image.image = image
                instr_image.draw()
        elif screen['text'].find('\n') != -1:
            instr_all_text.text = screen['text']
            instr_all_text.draw()
        else:
            center_text.text = screen['text']
            center_text.draw()
        keyList = tuple()
        if scrnum > 0:
            arrow_left.draw()
            keyList += ('left',)
        if scrnum < len(screens):
            keyList += ('right',)
            arrow_right.draw()
        else:
            keyList += ('space',)
        win.flip()
        keys = event.waitKeys(keyList=keyList)
        if 'left' in keys:
            scrnum -= 1
        elif 'right' in keys:
            scrnum += 1
        else:
            assert 'space' in keys
            break

if __name__ == '__main__':
    # Start PsychoPy
    from psychopy import visual, core, event, gui, logging, data

    # Get computer name
    computer_name = socket.gethostname()

    # We were initially tested in different presentations of the flight information.
    # But we only ever ran this task in the simultaneous condition.
    condition = 'simultaneous'

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    # Get participant information
    info = {'Participant code': ''}
    dlg = gui.DlgFromDict(info, title='Participant information')
    if not dlg.OK:
        core.quit()
    part_code = info['Participant code'].strip().upper()
    if not part_code:
        print('Empty participant code')
        core.quit()
    filename = join(
        RESULTS_DIR, '{}_{}_{}'.format(part_code, condition, data.getDateStr()))
    if part_code == 'TEST':
        fullscr = False
        NUM_PRAC_TRIALS = 1
        RWRD *= NUM_TRIALS/2
        NUM_TRIALS = 2
        BLOCK = NUM_TRIALS // 2
    else:
        fullscr = True

    # Settings
    simultaneous = (condition == 'simultaneous')
    planets = random.choice((('Red', 'Black'), ('Black', 'Red')))
    spaceships = random.choice((('X', 'Y'), ('Y', 'X')))

    # String replacements for the instructions
    string_replacements = {
        '[planet0]': planets[0],
        '[planet1]': planets[1],
        '[spaceship0]': spaceships[0],
        '[spaceship1]': spaceships[1],
    }

    info = {
        'part_code': part_code,
        'computer_name': computer_name,
        'simultaneous': simultaneous,
        'planets': planets,
        'spaceships': spaceships,
    }
    with open(filename + '_info.txt', 'w') as info_file:
        info_file.write(str(info))

    # Create window
    win = visual.Window(
        fullscr=fullscr, size=[1280, 1024], units='pix', color='#121321',
        checkTiming=False)
    win.mouseVisible = False

    try:
        # Stimuli
        middle_image = visual.ImageStim(win=win,
            pos=(0, 0),
            name='Middle image'
        )
        instr_image = visual.ImageStim(win=win,
            pos=(0, -97),
            name='Instructions image'
        )
        instr_text = visual.TextStim(win=win,
            pos=(-527, 359),
            alignHoriz='left',
            height=28,
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(1, 1, 1),
            wrapWidth=1077,
            name='Instructions text'
        )
        instr_all_text = visual.TextStim(win=win,
            pos=(-527, 0),
            alignHoriz='left',
            height=28,
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(1, 1, 1),
            wrapWidth=1077,
            name='Instructions text'
        )
        center_text = visual.TextStim(win=win,
            pos=(0, 0),
            height=60,
            wrapWidth=980,
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(1, 1, 1),
            name='Center text'
        )
        arrow_left = visual.ImageStim(win=win,
            pos=(0, 0),
            name='Left arrow image',
            image=join(ASSETS_DIR, 'instructions', 'arrow_left.png'),
        )
        arrow_right = visual.ImageStim(win=win,
            pos=(0, 0),
            name='Right arrow image',
            image=join(ASSETS_DIR, 'instructions', 'arrow_right.png'),
        )
        tutorial_frame = visual.Rect(win=win,
            pos=(0, 150),
            width=1000,
            height=140,
            fillColor=(1.0, 1.0, 1.0),
            opacity=0.9,
            name='Tutorial frame',
        )
        tutorial_text = visual.TextStim(win=win,
            pos=(0, 150),
            height=40,
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(-1, -1, -1),
            wrapWidth=980,
            name='Tutorial text'
        )
        tutorial_drawing = visual.ImageStim(win=win,
            pos=(0, 0),
            name='Tutorial drawing image'
        )
        sold_out = visual.ImageStim(win=win,
            pos=(0, 0),
            image=join(ASSETS_DIR, 'sold_out.png'),
            name='Sold out'
        )
        center_text = visual.TextStim(win=win,
            pos=(0, 0),
            height=60,
            wrapWidth=980,
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(1, 1, 1),
            name='Center text'
        )
        spaceship_blank = visual.ImageStim(win=win,
            pos=(0, 0),
            image=join(ASSETS_DIR, 'spaceship_blank.png'),
            name='Spaceship blank'
        )
        planet_blank = visual.ImageStim(win=win,
            pos=(0, 0),
            image=join(ASSETS_DIR, 'planet_blank.png'),
            name='Planet blank'
        )
        ticket_blank = visual.ImageStim(win=win,
            pos=(0, 0),
            image=join(ASSETS_DIR, 'Ticket_EMPTY.png'),
            name='Blank ticketmachine'
        )
        tickets_available = visual.ImageStim(win=win,
            pos=(0, 0),
            image=join(ASSETS_DIR, 'tickets_available.png'),
            name='Tickets available'
        )
        choice2_available = visual.ImageStim(win=win,
            pos=(0, 0),
            image=join(ASSETS_DIR, '2choice_green_arrows.png'),
            name='Second-stage choice available'
        )
        instructions_text = visual.TextStim(win=win,
            pos=(-527, 359),
            alignHoriz='left',
            height=28,
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(1, 1, 1),
            wrapWidth=1077,
            name='Instructions text'
        )
        planet_images = [
            visual.ImageStim(win=win,
                image=join(ASSETS_DIR, 'choice2_{}.png'.format(planets[p])),
                pos=(0, 0),
                name='Second choice image',
            )
            for p in range(2)
        ]
        common_images = {
            (ss, p): visual.ImageStim(win=win,
                image=join(ASSETS_DIR, 'SS_{}_{}_{}.png'.format(spaceships[ss],
                    planets[p], 'Left' if p == 0 else 'Right')),
                pos=(0, 0),
                name='Common flight image',
            )
            for ss in range(2)
            for p in range(2)
        }
        rare_images = {
            (ss, p): visual.ImageStim(win=win,
                image=join(ASSETS_DIR, 'SS_{}_{}_{}_EM.png'.format(spaceships[ss],
                    planets[p], 'Left' if p == 0 else 'Right')),
                pos=(0, 0),
                name='Rare flight image',
            )
            for ss in range(2)
            for p in range(2)
        }
        ticketmachine_images = {
            (ss, p): visual.ImageStim(win=win,
                image=join(ASSETS_DIR, 'Ticket_{}{}_{}-{}.png'.format(
                    spaceships[0], spaceships[1], spaceships[ss], planets[p])),
                pos=(0, 0),
                name='Ticket machine image',
            )
            for ss in range(2)
            for p in range(2)
        }
        obelisks = ['Left', 'Right']
        crystal_images = {
            (p, o): visual.ImageStim(win=win,
                image=join(ASSETS_DIR, 'Pillar_Crystal_{}_{}.png'.format(planets[p], obelisks[o])),
                pos=(0, 0),
                name='Obelisk with crystal image',
            )
            for p in range(2)
            for o in range(2)
        }
        empty_images = {
            (p, o): visual.ImageStim(win=win,
                image=join(ASSETS_DIR, 'Pillar_Empty_{}_{}.png'.format(planets[p], obelisks[o])),
                pos=(0, 0),
                name='Empty obeslisk image',
            )
            for p in range(2)
            for o in range(2)
        }
        
        # Tutorial instructions
        display_instructions('tutorial_instructions.txt', 'practice_flights.png')

        # Practice flights
        rwrd_probs = init_reward_probs()
        slow1 = True
        slow2 = True
        trial = 0
        slow_trials = 0
        with open(filename + '_practice.csv', 'w', newline='') as practice_file:
            practice_file.write(
                'trial,rwrd_prob0,rwrd_prob1,rwrd_prob2,rwrd_prob3,symbol0,symbol1,common,choice1,'
                'rt1,final_state,choice2,rt2,reward,slow\n')
            while trial - slow_trials < NUM_PRAC_TRIALS:
                common = int(random.random() < COMMON_PROB)
                center_text.text = 'Practice flight #{}'.format(trial + 1)
                center_text.draw()
                win.flip()
                core.wait(3)
                # Ticket buying tutorial
                anss = random.choice((0, 1))
                anpl = random.choice((0, 1))
                inist = (anss, anpl)
                ticketmachine_images[inist].draw()
                ticket_blank.draw()
                win.flip()
                core.wait(1)
                if trial == 0:
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    tutorial_frame.draw()
                    tutorial_text.text = 'To start your space journey, you must wait until the next departure is announced on the board above.'
                    tutorial_text.draw()
                    win.flip()
                    core.wait(6)
                    ticketmachine_images[inist].draw()
                    spaceship_blank.draw()
                    planet_blank.draw()
                    win.flip()
                    core.wait(1)
                if simultaneous:
                    ticketmachine_images[inist].draw()
                    planet_blank.draw()
                    win.flip()
                    core.wait(3)
                    ticketmachine_images[inist].draw()
                    win.flip()
                    core.wait(3)
                else:
                    ticketmachine_images[inist].draw()
                    planet_blank.draw()
                    win.flip()
                    core.wait(2.5)
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    win.flip()
                    core.wait(0.5)
                    ticketmachine_images[inist].draw()
                    spaceship_blank.draw()
                    win.flip()
                    core.wait(2.5)
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    win.flip()
                    core.wait(0.5)
                ticketmachine_images[inist].draw()
                ticket_blank.draw()
                tutorial_frame.draw()
                tutorial_text.text = 'Spaceship {} will fly to planet {}. '\
                    'This means spaceship {} will fly to planet {}.'.format(
                        spaceships[anss], planets[anpl], spaceships[1 - anss], planets[1 - anpl])
                tutorial_text.draw()
                win.flip()
                core.wait(7)
                ticketmachine_images[inist].draw()
                ticket_blank.draw()
                win.flip()
                core.wait(0.5)

                if slow1:
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    tutorial_drawing.image = join(ASSETS_DIR, 'tutorial.first.png')
                    tutorial_drawing.draw()
                    win.flip()
                    core.wait(5)
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    win.flip()
                    core.wait(0.5)
                    
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    tutorial_drawing.image = join(ASSETS_DIR, 'tutorial.Left_{}.png'.format(spaceships[0]))
                    tutorial_drawing.draw()
                    win.flip()
                    core.wait(4)
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    win.flip()
                    core.wait(0.5)

                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    tutorial_drawing.image = join(ASSETS_DIR, 'tutorial.Right_{}.png'.format(spaceships[1]))
                    tutorial_drawing.draw()
                    win.flip()
                    core.wait(4)
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    win.flip()
                    core.wait(0.5)

                ticketmachine_images[inist].draw()
                ticket_blank.draw()
                tutorial_drawing.image = join(ASSETS_DIR, 'tutorial.last.png')
                tutorial_drawing.draw()
                win.flip()
                core.wait(5)
                ticketmachine_images[inist].draw()
                ticket_blank.draw()
                win.flip()
                core.wait(0.5)
                ticketmachine_images[inist].draw()
                ticket_blank.draw()
                tickets_available.draw()
                win.flip()
                event.clearEvents()
                keys_times = event.waitKeys(maxWait=2, keyList=('left', 'right'),
                        timeStamped=core.Clock())
                if keys_times is None:
                    # Draw "sold out" message
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    sold_out.draw()
                    win.flip()
                    core.wait(4)
                    choice1 = -1
                    rt1 = -1
                    finalst = -1
                    choice2 = -1
                    rt2 = -1
                    reward = 0,
                    slow1 = True
                    slow2 = True
                    slow_trials += 1
                else:
                    choice1, rt1 = keys_times[0]
                    slow1 = False
                    # Space flight
                    choice1 = int(choice1 == 'right')
                    spaceship = choice1
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    tutorial_frame.draw()
                    tutorial_text.text = "You have bought a ticket for spaceship {}. Bon voyage!".format(spaceships[spaceship])
                    tutorial_text.draw()
                    win.flip()
                    core.wait(4)
                    
                    finalst = anpl if spaceship == anss else 1 - anpl
                    if common:
                        common_images[(spaceship, finalst)].draw()
                        win.flip()
                        core.wait(1)
                        common_images[(spaceship, finalst)].draw()
                        tutorial_frame.draw()
                        tutorial_text.text = 'Your trip to planet {} proceeds without incidents.'.format(planets[finalst])
                        tutorial_text.draw()
                        win.flip()
                        core.wait(5)
                        common_images[(spaceship, finalst)].draw()
                        win.flip()
                        core.wait(0.5)
                    else:
                        finalst = 1 - finalst
                        rare_images[(spaceship, finalst)].draw()
                        win.flip()
                        core.wait(1)
                        rare_images[(spaceship, finalst)].draw()
                        tutorial_frame.draw()
                        tutorial_text.text = 'Oh, no! An asteroid cloud has forced your spaceship to land on planet {}.'.format(planets[finalst])
                        tutorial_text.draw()
                        win.flip()
                        core.wait(5)
                        rare_images[(spaceship, finalst)].draw()
                        win.flip()
                        core.wait(0.5)
                    
                    # On the planet
                    planet_images[finalst].draw()
                    win.flip()
                    core.wait(0.5)
                    planet_images[finalst].draw()
                    tutorial_frame.draw()
                    tutorial_text.text = 'You have arrived on planet {}.'.format(planets[finalst])
                    tutorial_text.draw()
                    win.flip()
                    core.wait(3)
                    planet_images[finalst].draw()
                    win.flip()
                    core.wait(0.5)
                    if slow2:
                        planet_images[finalst].draw()
                        tutorial_drawing.image = join(ASSETS_DIR, '2choice_msg.png')
                        tutorial_drawing.draw()
                        win.flip()
                        core.wait(9)
                    planet_images[finalst].draw()
                    choice2_available.draw()
                    win.flip()
                    event.clearEvents()
                    keys_times = event.waitKeys(maxWait=2, keyList=('left', 'right'),
                            timeStamped=core.Clock())
                    if keys_times is None:
                        # Draw "out of oxygen" message
                        planet_images[finalst].draw()
                        tutorial_drawing.image = join(ASSETS_DIR, '2choice_slow_tutorial.png')
                        tutorial_drawing.draw()
                        win.flip()
                        core.wait(7)
                        choice2 = -1
                        rt2 = -1
                        reward = 0
                        slow2 = True
                        slow_trials += 1
                    else:
                        choice2, rt2 = keys_times[0]
                        slow2 = False
                        obelisk = choice2
                        choice2 = int(choice2 == 'right')
                        if random.random() < rwrd_probs[2*finalst + choice2]:
                            reward = 1
                            crystal_images[(finalst, choice2)].draw()
                            win.flip()
                            core.wait(1)
                            crystal_images[(finalst, choice2)].draw()
                            tutorial_frame.draw()
                            tutorial_text.text = 'The {} obelisk has a crystal inside!'.format(obelisk)
                            tutorial_text.draw()
                            win.flip()
                        else:
                            reward = 0
                            empty_images[(finalst, choice2)].draw()
                            win.flip()
                            core.wait(1)
                            empty_images[(finalst, choice2)].draw()
                            tutorial_frame.draw()
                            tutorial_text.text = 'The {} obelisk is empty.'.format(obelisk)
                            tutorial_text.draw()
                            win.flip()
                        core.wait(5)
                trial += 1
                practice_file.write(
                    '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                        trial, rwrd_probs[0], rwrd_probs[1], rwrd_probs[2],
                        rwrd_probs[3], inist[0], inist[1], int(common),
                        choice1, rt1, finalst, choice2, rt2, reward,
                        int(rt1 < 0 or rt2 < 0)))
                rwrd_probs = drift_reward_probs(rwrd_probs)

        del tutorial_drawing
        del tutorial_frame
        del tutorial_text
        # Game instructions
        display_instructions('game_instructions.txt', 'quiz.png')
        
        # Quiz
        quiz_text = visual.TextStim(win=win,
            pos=(-450, +50),
            alignHoriz='left',
            height=35,
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(1, 1, 1),
            wrapWidth=900,
            name='Quiz text'
        )
        quiz_help_text = visual.TextStim(win=win,
            pos=(-450, -250),
            text='Press the key corresponding to the correct answer.',
            alignHoriz='left',
            height=35,
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(1, 1, 1),
            wrapWidth=900,
            name='Quiz help text'
        )
        quiz_correct = visual.ImageStim(win=win,
            pos=(0, 0),
            image=join(ASSETS_DIR, 'instructions', 'quiz_correct.png'),
            name='Correct quiz answer'
        )
        quiz_incorrect = visual.ImageStim(win=win,
            pos=(0, 0),
            image=join(ASSETS_DIR, 'instructions', 'quiz_incorrect.png'),
            name='Incorrect quiz answer'
        )
        quiz_feedback = visual.TextStim(win=win,
            pos=(-250, -250),
            alignHoriz='left',
            height=35,
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(1, 1, 1),
            wrapWidth=900,
            name='Quiz feedback text'
        )
        with io.open(join(CURRENT_DIR, 'quiz.py'), encoding='utf-8') as qf:
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
                keyList = [chr(ord('a') + i) for i in range(ord(maxanswer) - ord('a') + 1)]
                key = event.waitKeys(keyList=keyList)[0].lower()
                quiz_text.draw()
                if key == correct:
                    quiz_correct.draw()
                    quiz_feedback.text = 'Correct!'
                    answered.append(qn)
                else:
                    quiz_incorrect.draw()
                    quiz_feedback.text = 'The correct answer is {}.'.format(correct)
                quiz_feedback.draw()
                win.flip()
                core.wait(3)
            if len(answered) == len(quiz):
                break
        
        del quiz_correct
        del quiz_feedback
        del quiz_incorrect
        del quiz_help_text

        middle_image.image = join(ASSETS_DIR, 'game_start.png')
        middle_image.draw()
        win.flip()
        key = event.waitKeys(keyList=['space'])[0]
        del middle_image
        
        # Game
        # Message
        msg = visual.TextStim(
            win=win,
            pos=(0, 0),
            fontFiles=[TTF_FONT],
            font='Noteworthy',
            color=(1, 1, 1),
            units='height',
            height=0.06,
            name='Message',
        )
        choice2_slow = visual.ImageStim(win=win,
            pos=(0, 0),
            image=join(ASSETS_DIR, '2choice_slow.png'),
            name='Second choice slow'
        )

        rwrd_probs = init_reward_probs()
        trial = 0
        slow_trials = 0
        rewards = 0
        with open(filename + '.csv', 'w', newline='') as results_file:
            results_file.write(
                'trial,rwrd_prob0,rwrd_prob1,rwrd_prob2,rwrd_prob3,symbol0,symbol1,common,choice1,'
                'rt1,final_state,choice2,rt2,reward,slow\n')
            while (trial - slow_trials) < NUM_TRIALS:
                common = int(random.random() < COMMON_PROB)
                anss = random.choice((0, 1))
                anpl = random.choice((0, 1))
                inist = (anss, anpl)
                ticketmachine_images[inist].draw()
                ticket_blank.draw()
                win.flip()
                core.wait(1)
                if simultaneous:
                    ticketmachine_images[inist].draw()
                    planet_blank.draw()
                    win.flip()
                    core.wait(2)
                    ticketmachine_images[inist].draw()
                    win.flip()
                    core.wait(2)
                else:
                    ticketmachine_images[inist].draw()
                    planet_blank.draw()
                    win.flip()
                    core.wait(1.5)
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    win.flip()
                    core.wait(0.5)
                    ticketmachine_images[inist].draw()
                    spaceship_blank.draw()
                    win.flip()
                    core.wait(1.5)
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    win.flip()
                    core.wait(0.5)
                ticketmachine_images[inist].draw()
                ticket_blank.draw()
                tickets_available.draw()
                win.flip()
                event.clearEvents()
                keys_times = event.waitKeys(maxWait=2, keyList=('left', 'right'),
                        timeStamped=core.Clock())
                if keys_times is None:
                    # Draw "sold out" message
                    ticketmachine_images[inist].draw()
                    ticket_blank.draw()
                    sold_out.draw()
                    win.flip()
                    core.wait(2)
                    choice1 = -1
                    rt1 = -1
                    finalst = -1
                    choice2 = -1
                    rt2 = -1
                    reward = 0
                    slow_trials += 1
                else:
                    choice1, rt1 = keys_times[0]
                    slow1 = False
                    # Space flight
                    choice1 = int(choice1 == 'right')
                    spaceship = choice1
                    finalst = anpl if spaceship == anss else 1 - anpl
                    if common:
                        common_images[(spaceship, finalst)].draw()
                    else:
                        finalst = 1 - finalst
                        rare_images[(spaceship, finalst)].draw()
                    win.flip()
                    core.wait(1)
                        
                    # On the planet
                    planet_images[finalst].draw()
                    choice2_available.draw()
                    win.flip()
                    event.clearEvents()
                    keys_times = event.waitKeys(maxWait=2, keyList=('left', 'right'),
                            timeStamped=core.Clock())
                    if keys_times is None:
                        # Draw "out of oxygen" message
                        planet_images[finalst].draw()
                        choice2_slow.draw()
                        win.flip()
                        core.wait(2)
                        choice2 = -1
                        rt2 = -1
                        reward = 0
                        slow_trials += 1
                    else:
                        choice2, rt2 = keys_times[0]
                        slow2 = False
                        choice2 = int(choice2 == 'right')
                        if random.random() < rwrd_probs[2*finalst + choice2]:
                            reward = 1
                            rewards += 1
                            crystal_images[(finalst, choice2)].draw()
                        else:
                            reward = 0
                            empty_images[(finalst, choice2)].draw()
                        win.flip()
                        core.wait(2)
                trial += 1
                results_file.write(
                    '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                        trial, rwrd_probs[0], rwrd_probs[1], rwrd_probs[2],
                        rwrd_probs[3], inist[0], inist[1], int(common),
                        choice1, rt1, finalst, choice2, rt2, reward,
                        int(rt1 < 0 or rt2 < 0)))
                rwrd_probs = drift_reward_probs(rwrd_probs)
                # Interval
                if trial % BLOCK == 0 and (trial - slow_trials) < (NUM_TRIALS - 10):
                    msg.text = 'BREAK\nYou can take a break now\nPress SPACE to continue'
                    msg.draw()
                    win.flip()
                    keys = event.waitKeys(keyList=('space',))

        # Save payment information in a file
        filename = join(
            RESULTS_DIR,
            'participant-{}_condition-{}_computer-{}_payment-{:.2f}'.format(
                part_code, condition, computer_name, get_money_reward(rewards)))
        with open(filename, 'w') as _:
            pass

        # Final screen with money reward
        money_msg = 'You won CHF {:.2f}.\n'\
            'Press SPACE to fill in the questionnaire.'.format(
                get_money_reward(rewards))
        msg.text = money_msg
        msg.draw()
        win.flip()
        event.waitKeys(keyList=('space',))
        win.close()

        # Questionnaire
        
        import wx
        class Questionnaire(wx.Frame):
            def __init__(self, parent, id, title): 
                font = wx.SystemSettings_GetFont(wx.SYS_SYSTEM_FONT)
                font.SetPointSize(14)
                wx.Frame.__init__(self, parent, id, title, size=(500, 600), style=wx.CAPTION)
                panel = wx.Panel(self)
                vbox = wx.BoxSizer(wx.VERTICAL)
                question = wx.StaticText(panel,
                    label='Please describe in detail the strategy you used to make your choices.')
                question.SetFont(font)
                vbox.Add(question, 1, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.TOP, border=15)
                self.answer = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
                self.answer.SetFont(font)
                vbox.Add(self.answer, 10, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, border=15)
                button = wx.Button(panel, 1, 'Send', (80, 220))
                vbox.Add(button, 1, wx.ALIGN_CENTER, border=15)
                self.Bind(wx.EVT_BUTTON, self.OnClose, id=1)
                panel.SetSizer(vbox)
                self.Centre() 
            def OnClose(self, event):
                # Save questionnaire in a file
                with open(join(RESULTS_DIR, '{}_{}_{}_questionnaire.txt'.format(part_code, condition, computer_name)), 'w') as outf:
                    outf.write(self.answer.GetValue())
                self.Close()
                goodbye = Goodbye(None, -1, 'Goodbye')
                goodbye.Show()
        
        class Goodbye(wx.Frame):
            def __init__(self, parent, id, title): 
                font = wx.SystemSettings_GetFont(wx.SYS_SYSTEM_FONT)
                font.SetPointSize(14)
                wx.Frame.__init__(self, parent, id, title, size=(500, 100), style=wx.CAPTION)
                panel = wx.Panel(self)
                vbox = wx.BoxSizer(wx.VERTICAL)
                msg = 'Thank you for participating in this experiment! Please just wait for your payment. You will get it soon.'
                msg = wx.StaticText(panel, label=msg)
                msg.SetFont(font)
                vbox.Add(msg, 1, wx.EXPAND|wx.ALL, border=15)
                panel.SetSizer(vbox)
                self.Centre() 

        class MyApp(wx.App): 
            def OnInit(self): 
                dlg = Questionnaire(None, -1, 'Questionnaire')
                dlg.Show()
                return True 

        app = MyApp(0) 
        app.MainLoop()
        
    except Exception as exc:
        print('An exception occurred:', str(exc))
        sys.stdout.flush()
        try:
            win.close()
        except:
            pass
        time.sleep(3600)
