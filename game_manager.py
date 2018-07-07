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


class GameManager(ABC):
    """ Implements functions for handling logic related to the rules of the
    game.
    """
    DECIDED_MULTIPLIER = 10

    @abstractmethod
    def augment_samples(self,
                        state_batch_tf,
                        policy_batch_tf,
                        value_prior_batch_tf):
        """ Performs data augmentation on the training samples (e.g. flipping).

        Args:
          state_batch_tf: tf.Tensor: batch of states of the game.
          policy_batch_tf: tf.Tensor: batch of policy labels.
          value_prior_batch_tf: tf.Tensor: batch of value labels.

        Returns:
          tf.Tensor: the same states transformed by the augmentation.
          tf.Tensor: the same policies transformed by the augmentation.
          tf.Tensor: the same values transformed by the augmentation.
        """
        pass

    @abstractmethod
    def get_iwinner(self, state_batch):
        """ Returns the index of the winner in each state of the batch.

        Args:
          state_batch: np.array: a batch of game states.

        Returns:
          np.array: a ternary vector whose values are:
            -1 if there is no winner (it is a draw, or the game is unfinished),
            0 if player 1 won,
            1 if player 2 won.
        """
        pass

    @abstractmethod
    def has_moves_remaining(self, state_batch):
        """ Indicates if each state from the batch has remaining valid moves.

        Args:
          state_batch: np.array: a batch of game states.

        Returns:
          np.array: a binary vector indicating whether there is at least
            one valid move to be played at each state.
        """
        pass

    @abstractmethod
    def initial_state(self):
        """ Returns a state representing the initial configuration of the board
        before the game starts.

        Returns:
          State: the initial state of the game
        """
        pass

    @abstractmethod
    def initialize_resnet(self,
                          ckpt_path,
                          num_res_layers=None,
                          num_channels=None,
                          replace_unloaded_resnet_by_naivenet=True):
        """ Initializes a residual network.

        Args:
          ckpt_path: string of None: path to the checkpoint to be loaded.
          tf_device: string: tensorflow string for tf.device to be used.
          num_res_layers: int: number of residual blocks in the CNN.
          num_channels: int: number of channels in the CNN layers.
          replace_unloaded_resnet_by_naivenet: boolean: if True, when ckpt_path
            is None, a NaiveNet is used instead of a ResNet with random
            weights.

        Returns:
          tf.keras.Model: the initialized network.
        """
        pass

    @abstractmethod
    def invalid_moves(self, state_batch):
        """ Indicates which of the possible moves at each state from the batch
        are illegal according to the rules of the game.

        Args:
          state_batch: np.array: a batch of game states.

        Returns:
          np.array: a binary tensor indicating if each possible move is
            invalid.
        """
        pass

    @abstractmethod
    def is_over(self, state_batch):
        """ Indicates if each state from the batch represents a finished game.

        Args:
          state_batch: np.array: a batch of game states.

        Returns:
          np.array: a binary vector indicating whether the game at each
          state has reached a finished state.
        """
        pass

    @abstractmethod
    def max_num_moves(self):
        """ Returns the maximum number of moves that can be played from the start
        the end of the game.

        Returns:
          int: maximum number of moves
        """
        pass

    def predict(self, state_batch):
        """ Predict the policies and values for each state in the batch.

        Args:
          state_batch: np.array NCHW: a batch of game states.

        Returns:
          np.array: a tensor with the predicted policies.
          np.array: a vector with the predicted values.
        """
        pass

    @abstractmethod
    def update_state(self, state, imove):
        """ Given a state and a valid move index, plays the move at the given
        state and update it according to the rules of the game.

        Args:
          state: np.array: a game state.

        Returns:
          np.array, the new state after the move imove is played.
        """
        pass

    @abstractmethod
    def valid_moves(self, state_batch):
        """ Indicates which of the possible moves at each state from the batch
        are legal according to the rules of the game.

        Args:
          state_batch: np.array: a batch of game states.

        Returns:
          np.array: a binary tensor indicating if each possible move is valid.
        """
        pass
