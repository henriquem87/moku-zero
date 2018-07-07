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

from state import State


class MokuState(State):
    """ Handles a state of the moku game (see moku_manager.py).

    Args:
      board_size: tuple: two values indicating number of rows and number of
        columns of the board.
    """
    def __init__(self, board_size):
        self.state = np.zeros((3, board_size[0], board_size[1]), np.uint8)

    def board_size(self):
        """ Returns a tuple with size of the playing board.

        Returns:
          tuple: the size of the board
        """
        return (self.state.shape[1], self.state.shape[2])

    def copy(self):
        """ Returns a deep copy of this state.

        Returns:
          State: a copy of this state.
        """
        cp_state = MokuState((self.state.shape[1], self.state.shape[2]))
        cp_state.state = self.state.copy()
        return cp_state

    def iplayer_to_move(self):
        """ Returns the index of the player who will make the next move.

        Returns:
          int: the index of the player to move.
        """
        return self.state[-1, 0, 0]
