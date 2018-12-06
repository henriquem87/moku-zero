###############################################################################
#
# Copyright (c) 2018, Henrique Morimitsu,
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# #############################################################################

import numpy as np
import os.path as osp
import tensorflow as tf

from game_manager_io import GameManagerIO

tfe = tf.contrib.eager


class MokuManagerIO(GameManagerIO):
    """ Implements functions for handling inputs and outputs related to the
    Moku game. For more information about the Moku game, read the comments in
    moku_manager.py.

    Args:
      drop_mode: boolean: indicating if the game is played in drop mode.
      board_size: tuple: two values indicating number of rows and number of
        columns of the board.
    """
    def __init__(self, drop_mode, board_size):
        self.drop_mode = drop_mode
        self.board_size = board_size

    def get_input(self, state):
        """ Hub for calling the correct input function.

        Args:
        state: State: the current state of the game.

        Returns:
        int: the move index representing the typed input.
        """
        if self.drop_mode:
            return self._get_input_drop(state)
        else:
            return self._get_input_non_drop(state)

    def _get_input_drop(self, state):
        """ Manages user input for game with drop mode (user only needs to type
        the column index).

        Args:
        state: State: the current state of the game.

        Returns:
        int: the move index representing the typed input.
        """
        valid = False
        while not valid:
            col_str = input(
                'Type one of the following:\n' +
                '  - the column number\n' +
                '  - c - to print computer predictions for move\n' +
                '  - q - to quit the game\n')
            if col_str == 'c':
                imove = type(self).ICOMPUTER_MOVE
                valid = True
            elif col_str == 'q':
                imove = type(self).IEXIT
                valid = True
            else:
                try:
                    col = int(col_str)-1
                except ValueError:
                    print('Type a number')
                    continue

                if col < 0 or col >= self.board_size[1]:
                    print('Typed column is out of bounds')
                    continue
                if np.sum(state.state[:2], axis=(0, 1))[col] == \
                        state.state.shape[1]:
                    print('This column is already full')
                    continue

                imove = col
                valid = True
        return imove

    def _get_input_non_drop(self, state):
        """ Manages user input for game without drop mode (user needs to type
        the row and column index).

        Args:
        state: State: the current state of the game.

        Returns:
        int: the move index representing the typed input.
        """
        valid = False
        while not valid:
            coords_str = input(
                'Type one of the following:\n' +
                '  - move coordinates (row number and columns number ' +
                'respectively, separated by a space)\n' +
                '  - c - to print computer predictions for move\n' +
                '  - q - to quit the game\n')
            if coords_str == 'c':
                imove = type(self).ICOMPUTER_MOVE
                valid = True
            elif coords_str == 'q':
                imove = type(self).IEXIT
                valid = True
            else:
                coords = coords_str.strip().split(' ')
                if len(coords) != 2:
                    print('Invalid input, type again')
                    continue
                try:
                    coords = [int(x)-1 for x in coords]
                except ValueError:
                    print('Type two numbers separated by space')
                    continue
                if coords[0] < 0 or coords[0] >= self.board_size[0] or \
                        coords[1] < 0 or coords[1] >= self.board_size[1]:
                    print('Typed coordinates are out of bounds')
                    continue
                if np.max(state.state[:2, coords[0], coords[1]]) > 0:
                    print('This position is already occupied')
                    continue

                imove = self.board_size[1] * coords[0] + coords[1]
                valid = True
        return imove

    def print_board(self, state, last_imove=None):
        """ Hub for calling the correct print board function.

        Args:
        state: State: the current state of the game.
        last_imove: int or None: index of the last played move or None if no
            move has been played yet.
        """
        if self.drop_mode:
            self._print_board_drop(state, last_imove)
        else:
            self._print_board_non_drop(state, last_imove)

    def _print_board_drop(self, state, last_imove):
        """ Prints the board when playing in drop mode.

        Args:
        state: State: the current state of the game.
        last_imove: int or None: index of the last played move or None if no
            move has been played yet.
        """
        last_move_coord = None
        if last_imove is not None:
            last_move_coord = (
                np.sum(state.state[:2], axis=(0, 1))[last_imove]-1, last_imove)
        for i in range(state.state.shape[1], -2, -1):
            for j in range(state.state.shape[2]):
                if i < 0 or i >= state.state.shape[1]:
                    print(str(j+1).rjust(2), end=' ')
                else:
                    if last_move_coord is not None and \
                            i == last_move_coord[0] and \
                            j == last_move_coord[1]:
                        if state.state[0, i, j] == 1:
                            print('(X)', end='')
                        elif state.state[1, i, j] == 1:
                            print('(O)', end='')
                    else:
                        if state.state[0, i, j] == 1:
                            print(' X ', end='')
                        elif state.state[1, i, j] == 1:
                            print(' O ', end='')
                        else:
                            print(' . ', end='')
            print()
        print()

    def _print_board_non_drop(self, state, last_imove):
        """ Prints the board when playing without drop mode.

        Args:
        state: State: the current state of the game.
        last_imove: int or None: index of the last played move or None if no
            move has been played yet.
        """
        last_move_coord = None
        if last_imove is not None:
            last_move_coord = [last_imove // self.board_size[1],
                               last_imove % self.board_size[1]]
        for i in range(-1, state.state.shape[1]+1):
            for j in range(-1, state.state.shape[2]+1):
                if i < 0 or i >= state.state.shape[1]:
                    if j < 0 or j >= state.state.shape[2]:
                        print('   ', end='')
                    else:
                        print(str(j+1).rjust(2), end=' ')
                else:
                    if j < 0:
                        print(str(i+1).rjust(2), end=' ')
                    elif j >= state.state.shape[2]:
                        print(i+1, end='')
                    else:
                        if last_move_coord is not None and \
                                i == last_move_coord[0] and \
                                j == last_move_coord[1]:
                            if state.state[0, i, j] == 1:
                                print('(X)', end='')
                            elif state.state[1, i, j] == 1:
                                print('(O)', end='')
                        else:
                            if state.state[0, i, j] == 1:
                                print(' X ', end='')
                            elif state.state[1, i, j] == 1:
                                print(' O ', end='')
                            else:
                                print(' . ', end='')
            print()
        print()

    def print_stats(self, stats):
        """ Prints the table of MCTS parameter values.

        Args:
        stats: np.array: table of MCTS parameter values for all children of the
            current node.
        """
        s = 'N\tW\tQ\tP'
        if stats is None:
            s += '\nNone'
        else:
            for i in range(stats.shape[0]):
                s += '\n%.0f\t%.02f\t%.02f\t%.02f' % \
                    (stats['N'][i], stats['W'][i],
                     stats['Q'][i], stats['P'][i])
        print(s)

    def print_stats_on_board(self, stats, move_temperature):
        """ Prints the probability of playing at each position.

        Args:
        stats: np.array: table of MCTS parameter values for all children of the
            current node.
        move_temperature: float: a value to exponentially adjust the raw move
            probabilities.
        """
        exponent = 1.0 / move_temperature
        exp_N = np.power(stats['N']/np.max(stats['N']), exponent)
        probs = (exp_N / np.sum(exp_N) * 100).astype(np.int32)
        if self.drop_mode:
            for j in range(self.board_size[1]):
                print(str(probs[j]).rjust(2), end=' ')
            print()
        else:
            for i in range(-1, self.board_size[0]+1):
                for j in range(-1, self.board_size[1]+1):
                    if i < 0 or i >= self.board_size[0]:
                        if j < 0 or j >= self.board_size[1]:
                            print('   ', end='')
                        else:
                            print(str(j+1).rjust(2), end=' ')
                    else:
                        if j < 0:
                            print(str(i+1).rjust(2), end=' ')
                        elif j >= self.board_size[1]:
                            print(i+1, end='')
                        else:
                            print(str(probs[i*self.board_size[1]+j]).rjust(2),
                                  end=' ')
                print()
            print()
