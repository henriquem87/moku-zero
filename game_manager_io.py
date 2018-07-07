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

from abc import (ABC, abstractmethod)


class GameManagerIO(ABC):
    """ Handles inputs and outputs for the game. """
    ICOMPUTER_MOVE = -10
    IEXIT = -11

    @abstractmethod
    def get_input(self, state):
        """ Asks for input from user returns the typed move index.

        Args:
        state: State: the current state of the game.

        Returns:
        int: the move index representing the typed input.
        """
        pass

    @abstractmethod
    def print_board(self, state, last_imove=None):
        """ Prints the current state of the board.

        Args:
        state: State: the current state of the game.
        last_imove: int or None: index of the last played move or None if no
            move has been played yet.
        """
        pass

    @abstractmethod
    def print_stats(self, stats):
        """ Prints the table of MCTS parameter values.

        Args:
        stats: np.array: table of MCTS parameter values for all children of the
            current node.
        """
        pass

    @abstractmethod
    def print_stats_on_board(self, stats, move_temperature):
        """ Prints the probability of playing at each position at the given
        move temperature.

        Args:
        stats: np.array: table of MCTS parameter values for all children of the
            current node.
        move_temperature: float: a value to exponentially adjust the raw move
            probabilities.
        """
        pass
